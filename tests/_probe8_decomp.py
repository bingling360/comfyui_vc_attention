"""Probe 8: decompose VC-Attention cost at 16384 tokens — prepare vs kernel,
fp8 QK dot vs bf16 QK dot, and a BLOCK_M/num_warps sweep."""
import os
import sys
import time

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/tcache8")
import torch
import triton
import triton.language as tl

sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention")
from vc_attention.kernels.triton_attn import TritonConfig, prepare, _vc_attn_fwd
from vc_attention.grouping import GroupingConfig, build_permutation

torch.manual_seed(0)
T, H, D = 16384, 56, 128
dev = "cuda"
q = torch.randn(1, H, T, D, device=dev, dtype=torch.bfloat16)
k = torch.randn(1, H, T, D, device=dev, dtype=torch.bfloat16)
v = torch.randn(1, H, T, D, device=dev, dtype=torch.bfloat16)


def timeit(fn, warmup=3, repeat=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat * 1000


t_sdpa = timeit(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))
print(f"bf16 SDPA                 : {t_sdpa:8.2f} ms")

perm = build_permutation(v.reshape(H, T, D), GroupingConfig(block_rows=128, iters=3)).perm

t_prep = timeit(lambda: prepare(q, k, v, perm, block_rows=128, hadamard=True), repeat=5)
print(f"prepare (quant+fwht+gather): {t_prep:8.2f} ms")

p = prepare(q, k, v, perm, block_rows=128, hadamard=True)
out = torch.empty_like(q)
cfg = TritonConfig(block_m=128, block_n=128)


def launch(bm, warps, stages=3):
    grid = (triton.cdiv(p.n_pad, bm), H)
    _vc_attn_fwd[grid](p.q, p.q_scale, p.k, p.k_scale, p.v, p.v_scale, p.mu, out,
                       D ** -0.5, T, p.n_pad, D, BLOCK_M=bm, BLOCK_N=128,
                       EXPCAST=False, BETA=-0.35, num_warps=warps, num_stages=stages)


print("\nkernel-only sweep (fixed kernel: fp8 QK + bf16 PV + mean fix):")
for bm, warps in [(128, 8), (128, 4), (64, 4), (64, 8), (256, 8)]:
    try:
        t = timeit(lambda: launch(bm, warps), repeat=5)
        print(f"  BLOCK_M={bm:3d} warps={warps}: {t:8.2f} ms   ({t_sdpa/t:5.2f}x vs SDPA)")
    except Exception as e:
        print(f"  BLOCK_M={bm:3d} warps={warps}: FAILED {type(e).__name__}")

# bf16-QK variant to test whether the fp8 QK dot is emulated
LOG2E = tl.constexpr(1.4426950408889634)
PS = tl.constexpr(448.0)


@triton.jit
def _kern_bf16qk(Q, QS, K, KS, V, VS, MU, Out, sm_scale, N, N_PAD,
                 D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
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
        s = tl.dot(q.to(tl.bfloat16), tl.trans(k.to(tl.bfloat16)))
        s = s * (qs[:, None] * ks[None, :]) * sm_scale
        s = tl.where(nmask[None, :], s, -1.0e30)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_new = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.math.exp2((m_i - m_new) * LOG2E)
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
        p8 = (tl.math.exp2((s - m_new[:, None]) * LOG2E) * PS).to(tl.float8e4nv)
        r = tl.sum(p8.to(tl.float32), 1) / PS
        tile = tl.dot(p8.to(tl.bfloat16), v.to(tl.bfloat16))
        blk = start_n // BLOCK_N
        vs = tl.load(VS + off_bh * nb * D + blk * D + offs_d)
        mu = tl.load(MU + off_bh * nb * D + blk * D + offs_d)
        acc = acc * alpha[:, None]
        acc += tile * (vs[None, :] / PS)
        acc += r[:, None] * (mu[None, :] * vs[None, :])
        l_i = l_i * alpha + r
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(Out + base + offs_m[:, None] * D + offs_d[None, :], acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N)


for bm, warps in [(128, 8), (64, 4)]:
    grid = (triton.cdiv(p.n_pad, bm), H)
    t = timeit(lambda: _kern_bf16qk[grid](p.q, p.q_scale, p.k, p.k_scale, p.v, p.v_scale, p.mu, out,
                                          D ** -0.5, T, p.n_pad, D, BLOCK_M=bm, BLOCK_N=128,
                                          num_warps=warps, num_stages=3), repeat=5)
    print(f"bf16-QK variant BM={bm:3d} warps={warps}: {t:8.2f} ms   ({t_sdpa/t:5.2f}x vs SDPA)")
