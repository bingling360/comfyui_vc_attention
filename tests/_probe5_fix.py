"""Probe 5: confirm fp8-MVA NaN scope and validate the candidate fix.

  0. torch._scaled_mm e4m3 sanity (is the HW/driver fp8 path sane at all?)
  1. full fp8 PV dot, corrected mean term, config sweep (warps/stages/out_dtype)
  2. p8 via scratch memory round-trip before the dot (layout-conversion probe)
  3. FIX CANDIDATE: QK fp8 MMA + PV bf16 MMA + corrected mean term
"""
import os
import sys
import time

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/tcache5")
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

# --- 0. torch fp8 sanity -----------------------------------------------------
a8 = torch.randn(128, 128, device=dev).to(torch.float8_e4m3fn)
b8 = torch.randn(128, 128, device=dev).to(torch.float8_e4m3fn)
sa = torch.ones(1, device=dev)
sb = torch.ones(1, device=dev)
try:
    c = torch._scaled_mm(a8, b8.t(), scale_a=sa, scale_b=sb, out_dtype=torch.float32)
    ok = bool(torch.isfinite(c).all())
except Exception as e:
    ok = f"exception: {e}"
print(f"torch._scaled_mm e4m3: finite={ok}")

LOG2E = tl.constexpr(1.4426950408889634)
PS = tl.constexpr(448.0)


@triton.jit
def _kern(Q, QS, K, KS, V, VS, MU, SCRATCH, Out, sm_scale, N, N_PAD,
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
        r = tl.sum(p8.to(tl.float32), 1) / PS
        blk = start_n // BLOCK_N
        vs = tl.load(VS + off_bh * nb * D + blk * D + offs_d)
        mu = tl.load(MU + off_bh * nb * D + blk * D + offs_d)
        if MODE == 0:      # fp8 x fp8 MMA (original path), corrected mean term
            tile = tl.dot(p8, v)
        elif MODE == 1:    # fp8 round-trip through global memory, then MMA
            sp = start_m * 1000 + start_n
            tl.store(SCRATCH + (off_bh.to(tl.int64) * (N_PAD // BLOCK_N) + blk) * BLOCK_M * BLOCK_N
                     + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :], p8)
            p8m = tl.load(SCRATCH + (off_bh.to(tl.int64) * (N_PAD // BLOCK_N) + blk) * BLOCK_M * BLOCK_N
                          + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :])
            tile = tl.dot(p8m, v)
        else:              # bf16 MMA (candidate fix)
            tile = tl.dot(p8.to(tl.bfloat16), v.to(tl.bfloat16))
        acc = acc * alpha[:, None]
        acc += tile * (vs[None, :] / PS)
        acc += r[:, None] * (mu[None, :] * vs[None, :])   # CORRECTED mean restore
        l_i = l_i * alpha + r
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(Out + base + offs_m[:, None] * D + offs_d[None, :], acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N)


out = torch.empty_like(q)
scratch = torch.empty(H * (p.n_pad // 128) * 128 * 128, device=dev, dtype=torch.float8_e4m3fn)

print("\n-- MODE 0: fp8x fp8 PV MMA, corrected mean, config sweep --")
for warps, stages in [(8, 3), (4, 3), (8, 2), (4, 2)]:
    out.zero_()
    _kern[(triton.cdiv(p.n_pad, 128), H)](
        p.q, p.q_scale, p.k, p.k_scale, p.v, p.v_scale, p.mu, scratch, out,
        D ** -0.5, T, p.n_pad, D, BLOCK_M=128, BLOCK_N=128, MODE=0,
        num_warps=warps, num_stages=stages)
    torch.cuda.synchronize()
    print(f"  warps={warps} stages={stages}: PSNR {psnr(out, ref_sdpa):8.2f} dB  "
          f"finite={bool(torch.isfinite(out.float()).all())}")

print("\n-- MODE 1: fp8 via scratch round-trip --")
out.zero_()
_kern[(triton.cdiv(p.n_pad, 128), H)](
    p.q, p.q_scale, p.k, p.k_scale, p.v, p.v_scale, p.mu, scratch, out,
    D ** -0.5, T, p.n_pad, D, BLOCK_M=128, BLOCK_N=128, MODE=1, num_warps=8, num_stages=3)
torch.cuda.synchronize()
print(f"  scratch round-trip: PSNR {psnr(out, ref_sdpa):8.2f} dB  "
      f"finite={bool(torch.isfinite(out.float()).all())}")

print("\n-- MODE 2: bf16 PV MMA (fix candidate) --")
out.zero_()
_kern[(triton.cdiv(p.n_pad, 128), H)](
    p.q, p.q_scale, p.k, p.k_scale, p.v, p.v_scale, p.mu, scratch, out,
    D ** -0.5, T, p.n_pad, D, BLOCK_M=128, BLOCK_N=128, MODE=2, num_warps=8, num_stages=3)
torch.cuda.synchronize()
print(f"  bf16 PV: PSNR {psnr(out, ref_sdpa):8.2f} dB  "
      f"finite={bool(torch.isfinite(out.float()).all())}")

t0 = time.perf_counter()
for _ in range(20):
    _kern[(triton.cdiv(p.n_pad, 128), H)](
        p.q, p.q_scale, p.k, p.k_scale, p.v, p.v_scale, p.mu, scratch, out,
        D ** -0.5, T, p.n_pad, D, BLOCK_M=128, BLOCK_N=128, MODE=2, num_warps=8, num_stages=3)
torch.cuda.synchronize()
print(f"  bf16 PV steady: {(time.perf_counter()-t0)/20*1000:.2f} ms  (2048 tok, 56 heads)")
t0 = time.perf_counter()
for _ in range(20):
    torch.nn.functional.scaled_dot_product_attention(q, k, v)
torch.cuda.synchronize()
print(f"  bf16 SDPA     : {(time.perf_counter()-t0)/20*1000:.2f} ms")
