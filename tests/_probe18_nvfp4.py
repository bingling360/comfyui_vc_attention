"""Probe 18: NVFP4 (e2m1 + per-16 e4m3 microscales) via tl.dot_scaled on sm_120.

Layouts per triton 3.6 semantic.py:
  lhs data (M, K//2) uint8 packed (low nibble first), lhs scale (M, K//16) float8e4nv
  rhs data (K//2, N) uint8 packed (K along dim0!),          rhs scale (N, K//16) float8e4nv

  A. correctness vs exact dequant
  B. speed at QK shape vs bf16 dot -> native fp4 MMA or bf16 emulation?
  C. register-fp8 .to() dot on sm_120 (is the sm_89 cvt bug present here?)
"""
import os
import sys
import time

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/tcache18")
import torch
import triton
import triton.language as tl

sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention")
from vc_attention import quant as Q

print(f"triton {triton.__version__}  {torch.cuda.get_device_name(0)}")
GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")


def nvfp4_quant(x):
    """(R, K) -> (packed (R, K//2) uint8, micro e4m3 codes (R, K//16) uint8)."""
    R, K = x.shape
    xb = x.float().reshape(R, K // 16, 16)
    amax = xb.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    micro_code = Q.e4m3_encode(amax / 6.0)            # uint8 (R, K//16)
    micro_val = Q.e4m3_decode(micro_code).clamp(min=1e-30)
    q = (xb / micro_val.unsqueeze(-1)).clamp(-6, 6)
    codes = Q.fp4_encode(q).to(torch.uint8)           # (R, K) nibbles
    packed = (codes[:, 0::2] & 0xF) | ((codes[:, 1::2] & 0xF) << 4)
    return packed, micro_code


def nvfp4_dequant(packed, micro_code):
    R, half = packed.shape
    lo = (packed & 0xF).long()
    hi = ((packed >> 4) & 0xF).long()
    v = torch.stack([GRID[lo], GRID[hi]], dim=-1).reshape(R, half * 2)
    micro = Q.e4m3_decode(micro_code).repeat_interleave(16, dim=1)
    return v * micro[:, : v.shape[1]]


@triton.jit
def _dot_nvfp4(A, AS, B, BS, OUT,
               M, N, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_kp = tl.arange(0, K // 2)
    offs_kg = tl.arange(0, K // 16)
    a = tl.load(A + offs_m[:, None] * (K // 2) + offs_kp[None, :])
    b = tl.load(B + offs_kp[:, None] * N + offs_n[None, :])          # (K//2, N) K-first!
    asc = tl.load(AS + offs_m[:, None] * (K // 16) + offs_kg[None, :]).to(tl.float8e4nv, bitcast=True)
    bsc = tl.load(BS + offs_n[:, None] * (K // 16) + offs_kg[None, :]).to(tl.float8e4nv, bitcast=True)
    c = tl.dot_scaled(a, asc, "e2m1", b, bsc, "e2m1", out_dtype=tl.float32)
    tl.store(OUT + offs_m[:, None] * N + offs_n[None, :], c)


@triton.jit
def _dot_bf16(A, B, OUT, M, N, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, K)
    a = tl.load(A + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(B + offs_k[:, None] * N + offs_n[None, :])
    c = tl.dot(a, b, out_dtype=tl.float32)
    tl.store(OUT + offs_m[:, None] * N + offs_n[None, :], c)


@triton.jit
def _reg_fp8(A32, B8, OUT, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr):
    om = tl.arange(0, M)
    ok = tl.arange(0, K)
    on = tl.arange(0, N)
    a = tl.load(A32 + om[:, None] * K + ok[None, :])
    p8 = (a * 448.0).to(tl.float8e4nv)                # register fp8 (cvt path)
    b = tl.load(B8 + ok[:, None] * N + on[None, :])
    c = tl.dot(p8, b, out_dtype=tl.float32)
    tl.store(OUT + om[:, None] * N + on[None, :], c)


def run_nvfp4(ac, asc, bc, bsc, M, N, K, BM=128, BN=128):
    out = torch.empty(M, N, device="cuda", dtype=torch.float32)
    _dot_nvfp4[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        ac, asc, bc, bsc, out, M, N, K, BM=BM, BN=BN, num_warps=8)
    return out


def bench(fn, repeat=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat * 1000


# A. correctness (small)
M0, N0, K0 = 256, 256, 128
x = torch.randn(M0, K0, device="cuda") * 0.5
w = torch.randn(N0, K0, device="cuda") * 0.5          # (N, K) row-major tokens
ac, asc = nvfp4_quant(x)
bc, bsc = nvfp4_quant(w.t().contiguous())              # (K, N) -> K-first packing
out = run_nvfp4(ac, asc, bc, bsc, M0, N0, K0)
ref = nvfp4_dequant(ac, asc) @ nvfp4_dequant(bc, bsc).T
rel = float((out - ref).abs().max() / ref.abs().mean().clamp(min=1e-9))
print(f"A NVFP4 correctness: max|diff|={float((out-ref).abs().max()):.4f} rel={rel:.4f} "
      f"finite={bool(torch.isfinite(out).all())}")

# B. speed at QK shape
M, N, K = 8192, 8192, 128
x = torch.randn(M, K, device="cuda") * 0.5
w = torch.randn(N, K, device="cuda") * 0.5
ac, asc = nvfp4_quant(x)
bc, bsc = nvfp4_quant(w.t().contiguous())
xd = nvfp4_dequant(ac, asc).to(torch.bfloat16)
wd = nvfp4_dequant(bc, bsc).to(torch.bfloat16).T.contiguous()

t4 = bench(lambda: run_nvfp4(ac, asc, bc, bsc, M, N, K))
outb = torch.empty(M, N, device="cuda", dtype=torch.float32)
t16 = bench(lambda: _dot_bf16[(triton.cdiv(M, 128), triton.cdiv(N, 128))](
    xd, wd, outb, M, N, K, BM=128, BN=128, num_warps=8))
fl = 2 * M * N * K / 1e12
print(f"B speed ({M}x{K}x{N}): NVFP4 {t4:7.2f} ms ({fl/t4*1000:6.1f} TFLOP/s)  "
      f"bf16 {t16:7.2f} ms ({fl/t16*1000:6.1f} TFLOP/s)  ratio {t16/t4:.2f}x")

# C. register-fp8 dot on sm_120 (sm_89 had the cvt bug)
a32 = torch.rand(128, 128, device="cuda")
b8 = (torch.randn(128, 128, device="cuda") * 0.05).to(torch.float8_e4m3fn)
a8 = (a32 * 448.0).to(torch.float8_e4m3fn)
outc = torch.empty(128, 128, device="cuda", dtype=torch.float32)
_reg_fp8[(1,)](a32, b8, outc, M=128, K=128, N=128)
torch.cuda.synchronize()
refc = a8.float() @ b8.float()
err = float((outc - refc).abs().max())
print(f"C register-fp8 dot on sm_120: max|diff|={err:.4f} -> "
      f"{'BROKEN (keep bf16 PV)' if err > 1.0 else 'OK (fp8 PV possible!)'}")
