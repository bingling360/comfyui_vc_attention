"""Probe 4: discriminate fp32->fp8 register cast vs fp8x fp8 MMA as the NaN source.

Variants (all share the same prepared tensors and softmax skeleton):
  C  fp8-cast P, then back to fp32 for the PV dot  -> isolates the cast
  D  fp8 rowsum r only, PV entirely fp32           -> isolates r-from-p8
  E  P fp8 -> bf16, V fp8 -> bf16, bf16 MMA        -> practical fallback check
  F  exact copy of full kernel but PV dot on bf16 upcast operands
"""
import os
import sys

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/tcache4")
import torch
import triton
import triton.language as tl

sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention")
from vc_attention.kernels.triton_attn import prepare

torch.manual_seed(0)
T, H, D = 2048, 56, 128
dev = "cuda"
q = torch.randn(1, H, T, D, device=dev, dtype=torch.bfloat16)
k = torch.randn(1, H, T, D, device=dev, dtype=torch.bfloat16)
v = torch.randn(1, H, T, D, device=dev, dtype=torch.bfloat16)
ref_sdpa = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
perm = torch.arange(T, device=dev, dtype=torch.int32).unsqueeze(0)


def psnr(out, ref):
    mse = ((out.float() - ref.float()) ** 2).mean()
    if not torch.isfinite(out.float()).all():
        return float("nan")
    return float(10 * torch.log10(ref.float().pow(2).mean() / mse.clamp(min=1e-30)))


p = prepare(q, k, v, perm, block_rows=128, hadamard=True)

LOG2E = tl.constexpr(1.4426950408889634)
PS = tl.constexpr(448.0)


def _skeleton(fp8_p_mode: str):
    """fp8_p_mode: 'cast_only' | 'rowsum_only' | 'bf16_dot' | 'none'"""

    @triton.jit
    def _kern(Q, QS, K, KS, V, VS, MU, Out, sm_scale, N, N_PAD, D: tl.constexpr,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
              MODE: tl.constexpr):
        start_m = tl.program_id(0)
        off_bh = tl.program_id(1)
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        nb = N_PAD // BLOCK_N
        base = off_bh.to(tl.int64) * N_PAD * D
        q = tl.load(Q + base + offs_m[:, None] * D + offs_d[None, :], mask=offs_m[:, None] < N, other=0.0)
        qs = tl.load(QS + off_bh * N_PAD + offs_m, mask=offs_m < N, other=0.0)
        m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
        for start_n in range(0, N_PAD, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            nmask = offs_n < N
            k = tl.load(K + base + offs_n[:, None] * D + offs_d[None, :], mask=nmask[:, None], other=0.0)
            ks = tl.load(KS + off_bh * N_PAD + offs_n, mask=nmask, other=0.0)
            v = tl.load(V + base + offs_n[:, None] * D + offs_d[None, :], mask=nmask[:, None], other=0.0)
            s = tl.dot(q, tl.trans(k))
            s = s * (qs[:, None] * ks[None, :]) * sm_scale
            s = tl.where(nmask[None, :], s, -1.0e30)
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_new = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.math.exp2((m_i - m_new) * LOG2E)
            alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
            pf = tl.math.exp2((s - m_new[:, None]) * LOG2E)
            blk = start_n // BLOCK_N
            vs = tl.load(VS + off_bh * nb * D + blk * D + offs_d)
            mu = tl.load(MU + off_bh * nb * D + blk * D + offs_d)
            if MODE == 0:      # cast_only: P -> fp8, back to fp32 for the dot
                p8 = (pf * PS).to(tl.float8e4nv)
                r = tl.sum(p8.to(tl.float32), 1) / PS
                tile = tl.dot(p8.to(tl.float32), v.to(tl.float32))
                acc = acc * alpha[:, None]
                acc += tile * (vs[None, :] / PS)
                acc += r[:, None] * mu[None, :]
                l_i = l_i * alpha + r
            elif MODE == 1:    # rowsum_only: r from fp8, PV pure fp32
                p8 = (pf * PS).to(tl.float8e4nv)
                r = tl.sum(p8.to(tl.float32), 1) / PS
                tile = tl.dot(pf.to(tl.float32), v.to(tl.float32))
                acc = acc * alpha[:, None]
                acc += tile * (vs[None, :] / PS)
                acc += r[:, None] * mu[None, :]
                l_i = l_i * alpha + r
            elif MODE == 2:    # bf16_dot: P and V upcast to bf16 for the MMA
                p8 = (pf * PS).to(tl.float8e4nv)
                r = tl.sum(p8.to(tl.float32), 1) / PS
                tile = tl.dot(p8.to(tl.bfloat16), v.to(tl.bfloat16))
                acc = acc * alpha[:, None]
                acc += tile * (vs[None, :] / PS)
                acc += r[:, None] * mu[None, :]
                l_i = l_i * alpha + r
            else:              # fp32 reference-in-kernel
                r = tl.sum(pf, 1)
                tile = tl.dot(pf.to(tl.float32), v.to(tl.float32))
                acc = acc * alpha[:, None]
                acc += tile * vs[None, :]
                acc += r[:, None] * (mu[None, :] * vs[None, :])
                l_i = l_i * alpha + r
            m_i = m_new
        acc = acc / l_i[:, None]
        tl.store(Out + base + offs_m[:, None] * D + offs_d[None, :], acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N)

    return _kern


kern = _skeleton("x")
out = torch.empty_like(q)
names = {0: "C fp8-cast-only", 1: "D fp8-rowsum-only", 2: "E bf16-dot", 3: "F all-fp32"}
for mode in (0, 1, 2, 3):
    out.zero_()
    kern[(triton.cdiv(p.n_pad, 128), H)](
        p.q, p.q_scale, p.k, p.k_scale, p.v, p.v_scale, p.mu, out,
        D ** -0.5, T, p.n_pad, D, BLOCK_M=128, BLOCK_N=128, MODE=mode,
        num_warps=8, num_stages=3)
    torch.cuda.synchronize()
    print(f"{names[mode]:20s}: PSNR {psnr(out, ref_sdpa):8.2f} dB  finite={bool(torch.isfinite(out.float()).all())}")
