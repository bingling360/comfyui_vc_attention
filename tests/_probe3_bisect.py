"""Bisect the NaN in _vc_attn_fwd: run kernel variants of increasing fidelity.

Temporary diagnostic script (remote GPU probe); not part of the test suite.
"""
import os
import sys

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/tcache3")
import torch
import triton
import triton.language as tl

sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention")
from vc_attention.kernels.triton_attn import prepare
from vc_attention.kernels.reference import RefConfig, vc_attention_reference

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


orc = vc_attention_reference(
    q, k, v,
    RefConfig(backend="fp8", enable_vsmooth=True, block_rows=128, enable_expcast=False),
    perm=perm, scale=None,
)
print(f"oracle reference  : PSNR {psnr(orc, ref_sdpa):.2f} dB  finite={bool(torch.isfinite(orc.float()).all())}")

p = prepare(q, k, v, perm, block_rows=128, hadamard=True)
print(f"prepared: mu finite={bool(torch.isfinite(p.mu).all())} v_scale finite={bool(torch.isfinite(p.v_scale).all())} "
      f"q_scale finite={bool(torch.isfinite(p.q_scale).all())} k_scale finite={bool(torch.isfinite(p.k_scale).all())}")

LOG2E = tl.constexpr(1.4426950408889634)
PS = tl.constexpr(448.0)


@triton.jit
def _v_full(Q, QS, K, KS, V, VS, MU, Out, sm_scale, N, N_PAD, D: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
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
        p8 = (tl.math.exp2((s - m_new[:, None]) * LOG2E) * PS).to(tl.float8e4nv)
        r = tl.sum(p8.to(tl.float32), 1) / PS
        tile = tl.dot(p8, v)
        blk = start_n // BLOCK_N
        vs = tl.load(VS + off_bh * nb * D + blk * D + offs_d)
        mu = tl.load(MU + off_bh * nb * D + blk * D + offs_d)
        acc = acc * alpha[:, None]
        acc += tile * (vs[None, :] / PS)
        acc += r[:, None] * mu[None, :]
        l_i = l_i * alpha + r
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(Out + base + offs_m[:, None] * D + offs_d[None, :], acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N)


@triton.jit
def _v_fp32p(Q, QS, K, KS, V, VS, MU, Out, sm_scale, N, N_PAD, D: tl.constexpr,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
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
        r = tl.sum(pf, 1)
        tile = tl.dot(pf.to(tl.float32), v.to(tl.float32))
        blk = start_n // BLOCK_N
        vs = tl.load(VS + off_bh * nb * D + blk * D + offs_d)
        mu = tl.load(MU + off_bh * nb * D + blk * D + offs_d)
        acc = acc * alpha[:, None]
        acc += tile * vs[None, :]
        acc += r[:, None] * (mu[None, :] * vs[None, :])
        l_i = l_i * alpha + r
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(Out + base + offs_m[:, None] * D + offs_d[None, :], acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N)


@triton.jit
def _v_p8nomean(Q, QS, K, KS, V, VS, MU, Out, sm_scale, N, N_PAD, D: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
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
        p8 = (tl.math.exp2((s - m_new[:, None]) * LOG2E) * PS).to(tl.float8e4nv)
        r = tl.sum(p8.to(tl.float32), 1) / PS
        tile = tl.dot(p8, v)
        blk = start_n // BLOCK_N
        vs = tl.load(VS + off_bh * nb * D + blk * D + offs_d)
        acc = acc * alpha[:, None]
        acc += tile * (vs[None, :] / PS)
        l_i = l_i * alpha + r
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(Out + base + offs_m[:, None] * D + offs_d[None, :], acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N)


out = torch.empty_like(q)
for name, kern in [("full", _v_full), ("A fp32-P", _v_fp32p), ("B p8-nomean", _v_p8nomean)]:
    out.zero_()
    kern[(triton.cdiv(p.n_pad, 128), H)](
        p.q, p.q_scale, p.k, p.k_scale, p.v, p.v_scale, p.mu, out,
        D ** -0.5, T, p.n_pad, D, BLOCK_M=128, BLOCK_N=128, num_warps=8, num_stages=3)
    torch.cuda.synchronize()
    print(f"kernel {name:12s}: PSNR {psnr(out, ref_sdpa):8.2f} dB  finite={bool(torch.isfinite(out.float()).all())}")
