"""Probe 16: exact in-kernel e4m3 encode (bit math + bitcast) as the PV fix.

  A. torch-level: bit-math encoder vs quant.py e4m3_encode — byte match on
     positives [0, 448] + edge values
  B. kernel with bitcast-P (fp8 PV MMA) vs fp32 SDPA: PSNR at 2048/6067/16384
  C. timing at 16384: fp8-PV kernel vs current bf16-PV kernel vs bf16 SDPA
"""
import os
import sys
import time

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/tcache16")
import torch
import triton
import triton.language as tl

sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention")
from vc_attention import quant as Q
from vc_attention.kernels.triton_attn import prepare

dev = "cuda"

# --- A. encoder byte-match ---------------------------------------------------
x = torch.rand(8_000_000, device=dev) * 448.0
edges = torch.tensor([0.0, 1e-6, 2 ** -9 * 0.5, 2 ** -9, 2 ** -6 * 0.999, 2 ** -6, 0.25,
                      448.0, 448.0 - 2 ** -6, 417.999, 1.0, 2.0, 16.0, 0.0625], device=dev)
x = torch.cat([x, edges])


def encode_bits(a):
    a = a.float().clamp(max=448.0)
    bits = a.view(torch.int32)
    ef = ((bits >> 23) & 0xFF) - 120
    m3 = (bits >> 20) & 7
    rem = bits & 0xFFFFF
    round_up = (rem > 0x80000) | ((rem == 0x80000) & ((m3 & 1) == 1))
    m3b = m3 + round_up.to(torch.int32)
    m3 = m3b & 7
    ef = ef + (m3b >> 3)
    is_sub = ef < 1
    sub_code = (a * 512.0 + 0.5).to(torch.int32).clamp(max=8)
    code_n = (ef.clamp(min=1, max=15) << 3) | m3
    code = torch.where(is_sub, sub_code, code_n) & 0x7F
    return code.to(torch.uint8)


codes_bits = encode_bits(x)
codes_ref = Q.e4m3_encode(x)
mismatch = int((codes_bits != codes_ref).sum())
print(f"A torch encoder byte-match: {mismatch} / {x.numel()} differ")
if mismatch:
    bad = (codes_bits != codes_ref).nonzero().flatten()[:5].tolist()
    for i in bad:
        print(f"   x={x[i].item():.6g}  bits={codes_bits[i].item():#04x}  ref={codes_ref[i].item():#04x}")

# --- B/C. kernel with bitcast-P ----------------------------------------------
LOG2E = tl.constexpr(1.4426950408889634)
PS = tl.constexpr(448.0)


@triton.jit
def _kern_fp8pv(Q, QS, K, KS, V, VS, MU, Out, sm_scale, N, N_PAD,
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
        s = tl.dot(q, tl.trans(k))
        s = s * (qs[:, None] * ks[None, :]) * sm_scale
        s = tl.where(nmask[None, :], s, -1.0e30)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_new = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.math.exp2((m_i - m_new) * LOG2E)
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
        a = tl.math.exp2((s - m_new[:, None]) * LOG2E) * PS
        a = tl.minimum(tl.maximum(a, 0.0), PS)
        ab = a.to(tl.int32, bitcast=True)
        ef = ((ab >> 23) & 0xFF) - 120
        m3 = (ab >> 20) & 7
        rem = ab & 0xFFFFF
        round_up = (rem > 0x80000) | ((rem == 0x80000) & ((m3 & 1) == 1))
        m3b = m3 + round_up.to(tl.int32)
        m3 = m3b & 7
        ef = ef + (m3b >> 3)
        is_sub = ef < 1
        sub_code = tl.minimum((a * 512.0 + 0.5).to(tl.int32), 8)
        code_n = (tl.minimum(tl.maximum(ef, 1), 15) << 3) | m3
        code = tl.where(is_sub, sub_code, code_n) & 0x7F
        p8 = code.to(tl.uint8).to(tl.float8e4nv, bitcast=True)
        r = tl.sum(p8.to(tl.float32), 1) / PS
        tile = tl.dot(p8, v)
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


def timeit(fn, warmup=3, repeat=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat * 1000


def psnr(out, ref):
    mse = ((out.float() - ref.float()) ** 2).mean()
    if not torch.isfinite(out.float()).all():
        return float("nan")
    return float(10 * torch.log10(ref.float().pow(2).mean() / mse.clamp(min=1e-30)))


T, H, D, BR = 16384, 56, 128, 128
q = torch.randn(1, H, T, D, device=dev, dtype=torch.bfloat16)
k = torch.randn(1, H, T, D, device=dev, dtype=torch.bfloat16)
v = torch.randn(1, H, T, D, device=dev, dtype=torch.bfloat16)
ref_sdpa = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
p = prepare(q, k, v, perm=None, block_rows=BR, hadamard=True)
out = torch.empty(1, H, p.n_pad, D, device=dev, dtype=torch.bfloat16)

for n_test in (2048, 6067, 16384):
    qn = torch.randn(1, H, n_test, D, device=dev, dtype=torch.bfloat16)
    kn = torch.randn(1, H, n_test, D, device=dev, dtype=torch.bfloat16)
    vn = torch.randn(1, H, n_test, D, device=dev, dtype=torch.bfloat16)
    refn = torch.nn.functional.scaled_dot_product_attention(qn.float(), kn.float(), vn.float())
    pn = prepare(qn, kn, vn, perm=None, block_rows=BR, hadamard=True)
    outn = torch.empty(1, H, pn.n_pad, D, device=dev, dtype=torch.bfloat16)
    _kern_fp8pv[(triton.cdiv(pn.n_pad, 128), H)](
        pn.q, pn.q_scale, pn.k, pn.k_scale, pn.v, pn.v_scale, pn.mu, outn,
        D ** -0.5, n_test, pn.n_pad, D, BLOCK_M=128, BLOCK_N=BR, num_warps=8, num_stages=3)
    torch.cuda.synchronize()
    print(f"B n={n_test:6d}: fp8-PV kernel PSNR {psnr(outn[:, :, :n_test], refn):7.2f} dB  "
          f"finite={bool(torch.isfinite(outn[:, :, :n_test].float()).all())}")

t_fp8 = timeit(lambda: _kern_fp8pv[(triton.cdiv(p.n_pad, 128), H)](
    p.q, p.q_scale, p.k, p.k_scale, p.v, p.v_scale, p.mu, out,
    D ** -0.5, T, p.n_pad, D, BLOCK_M=128, BLOCK_N=BR, num_warps=8, num_stages=3))
from vc_attention.kernels.triton_attn import _vc_attn_fwd  # current bf16-PV kernel
t_bf16 = timeit(lambda: _vc_attn_fwd[(triton.cdiv(p.n_pad, 128), H)](
    p.q, p.q_scale, p.k, p.k_scale, p.v, p.v_scale, p.mu, out,
    D ** -0.5, T, p.n_pad, D, BLOCK_M=128, BLOCK_N=BR, EXPCAST=False, BETA=-0.35,
    num_warps=8, num_stages=3))
t_sdpa = timeit(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))
print(f"C timing @16384: fp8-PV {t_fp8:7.2f} ms ({t_sdpa/t_fp8:.2f}x SDPA)  "
      f"bf16-PV {t_bf16:7.2f} ms ({t_sdpa/t_bf16:.2f}x)  SDPA {t_sdpa:7.2f} ms")
