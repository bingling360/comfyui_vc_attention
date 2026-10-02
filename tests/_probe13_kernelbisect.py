"""Probe 13: bisect the kernel's partial-tile NaN at n=6067, n_pad=6144.

Variants all use ORACLE-prepared tensors (known good):
  0  N arg = 6067 (truth)          -> NaN (known)
  1  N arg = 6144 (= N_PAD, lie)   -> isolates the column-mask path
  2  N arg = 6067, no r*mu*vs term -> isolates the mean restore
  3  N arg = 6067, l_i from fp32 exp rowsum, not from p8 -> isolates r
  4  N arg = 6067, p8 kept fp32 (no fp8 cast), rest identical
"""
import os
import sys

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/tcache13")
import torch
import triton
import triton.language as tl

sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention")
from vc_attention.kernels.triton_attn import prepare

torch.manual_seed(0)
N, H, D, BR = 6067, 56, 128, 128
dev = "cuda"
q = torch.randn(1, H, N, D, device=dev, dtype=torch.bfloat16)
k = torch.randn(1, H, N, D, device=dev, dtype=torch.bfloat16)
v = torch.randn(1, H, N, D, device=dev, dtype=torch.bfloat16)
ref_sdpa = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
p = prepare(q, k, v, perm=None, block_rows=BR, hadamard=True)

LOG2E = tl.constexpr(1.4426950408889634)
PS = tl.constexpr(448.0)


def psnr(out, ref):
    mse = ((out.float() - ref.float()) ** 2).mean()
    if not torch.isfinite(out.float()).all():
        return float("nan")
    return float(10 * torch.log10(ref.float().pow(2).mean() / mse.clamp(min=1e-30)))


@triton.jit
def _kern(Q, QS, K, KS, V, VS, MU, Out, sm_scale, N, N_PAD,
          D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
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
        p8 = (pf * PS).to(tl.float8e4nv)
        blk = start_n // BLOCK_N
        vs = tl.load(VS + off_bh * nb * D + blk * D + offs_d)
        mu = tl.load(MU + off_bh * nb * D + blk * D + offs_d)
        if MODE == 0:
            r = tl.sum(p8.to(tl.float32), 1) / PS
            tile = tl.dot(p8.to(tl.bfloat16), v.to(tl.bfloat16))
            acc = acc * alpha[:, None]
            acc += tile * (vs[None, :] / PS)
            acc += r[:, None] * (mu[None, :] * vs[None, :])
            l_i = l_i * alpha + r
        elif MODE == 1:   # N arg lies = N_PAD (no effective column mask)
            r = tl.sum(p8.to(tl.float32), 1) / PS
            tile = tl.dot(p8.to(tl.bfloat16), v.to(tl.bfloat16))
            acc = acc * alpha[:, None]
            acc += tile * (vs[None, :] / PS)
            acc += r[:, None] * (mu[None, :] * vs[None, :])
            l_i = l_i * alpha + r
        elif MODE == 2:   # no mean restore
            r = tl.sum(p8.to(tl.float32), 1) / PS
            tile = tl.dot(p8.to(tl.bfloat16), v.to(tl.bfloat16))
            acc = acc * alpha[:, None]
            acc += tile * (vs[None, :] / PS)
            l_i = l_i * alpha + r
        elif MODE == 3:   # l_i from fp32 rowsum, PV from p8
            r_true = tl.sum(pf, 1)
            tile = tl.dot(p8.to(tl.bfloat16), v.to(tl.bfloat16))
            acc = acc * alpha[:, None]
            acc += tile * (vs[None, :] / PS)
            acc += r_true[:, None] * (mu[None, :] * vs[None, :])
            l_i = l_i * alpha + r_true
        else:             # MODE 4: P never quantized (fp32 path)
            r = tl.sum(pf, 1)
            tile = tl.dot(pf.to(tl.float32), v.to(tl.float32))
            acc = acc * alpha[:, None]
            acc += tile * vs[None, :]
            acc += r[:, None] * (mu[None, :] * vs[None, :])
            l_i = l_i * alpha + r
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(Out + base + offs_m[:, None] * D + offs_d[None, :], acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N)


out = torch.empty(1, H, N, D, device=dev, dtype=torch.bfloat16)
names = ["0 truth-N full", "1 N=N_PAD lie", "2 no mean restore", "3 fp32 rowsum l", "4 fp32 P"]
for mode in (0, 1, 2, 3, 4):
    out.zero_()
    n_arg = p.n_pad if mode == 1 else N
    _kern[(triton.cdiv(p.n_pad, 128), H)](
        p.q, p.q_scale, p.k, p.k_scale, p.v, p.v_scale, p.mu, out,
        D ** -0.5, n_arg, p.n_pad, D, BLOCK_M=128, BLOCK_N=128, MODE=mode,
        num_warps=8, num_stages=3)
    torch.cuda.synchronize()
    print(f"{names[mode]:18s}: PSNR {psnr(out, ref_sdpa):8.2f} dB  "
          f"finite={bool(torch.isfinite(out.float()).all())}")
