"""Probe 19: pin down tl.dot_scaled e2m1 packing on sm_120, then benchmark
in a COMPUTE-BOUND regime (probe18's 8192x8192x128 was dominated by the 268 MB
output store, so both paths looked equal).

  A. packing convention: which nibble of a packed byte is element 2i?
     Tested both ways, with unit scales (isolates the e2m1 path) and real scales.
  B. compute-bound speed: PV-shaped  M=16384, N=128, K=4096  (small output)
     - tl.dot_scaled e2m1 (fp4) vs bf16 tl.dot vs fp8 tl.dot
  C. register-fp8 dot on sm_120 (re-confirm)
"""
import os
import sys
import time

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/tcache19")
import torch
import triton
import triton.language as tl

sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention")
from vc_attention import quant as Q

print(f"triton {triton.__version__}  torch {torch.__version__}  {torch.cuda.get_device_name(0)}", flush=True)
GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")
E4M3_ONE = 0x38  # e4m3 code for 1.0


def _nib_val(codes):
    c = codes.to(torch.int64) & 0xF
    sign = torch.where((c & 0x8) != 0, -1.0, 1.0)
    return GRID[c & 0x7] * sign


def pack_even_low(codes, axis):
    """codes: (..., K) 4-bit. low nibble = even index along `axis`."""
    if axis == -1:
        return (codes[..., 0::2] & 0xF) | ((codes[..., 1::2] & 0xF) << 4)
    return (codes[:, 0::2, :] & 0xF) | ((codes[:, 1::2, :] & 0xF) << 4)


def pack_odd_low(codes, axis):
    if axis == -1:
        return (codes[..., 1::2] & 0xF) | ((codes[..., 0::2] & 0xF) << 4)
    return (codes[:, 1::2, :] & 0xF) | ((codes[:, 0::2, :] & 0xF) << 4)


def quant_row(x, unit=False, swap=False):
    """(R, K) -> packed (R, K//2), micro (R, K//16)."""
    R, K = x.shape
    xb = x.float().reshape(R, K // 16, 16)
    amax = xb.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    mc = torch.full((R, K // 16), E4M3_ONE, dtype=torch.uint8, device=x.device) if unit \
        else Q.e4m3_encode((amax / 6.0).reshape(R, K // 16))
    mv = Q.e4m3_decode(mc).clamp(min=1e-30)
    q = (xb / mv.unsqueeze(-1)).clamp(-6, 6)
    codes = Q.fp4_encode(q).to(torch.uint8).reshape(R, K)
    return (pack_odd_low if swap else pack_even_low)(codes, -1), mc


def quant_col(w, unit=False, swap=False):
    """(K, N) -> packed (K//2, N), micro (K//16, N)."""
    K, N = w.shape
    xb = w.float().reshape(K // 16, 16, N)
    amax = xb.abs().amax(dim=1, keepdim=True).clamp(min=1e-30)
    mc = torch.full((K // 16, N), E4M3_ONE, dtype=torch.uint8, device=w.device) if unit \
        else Q.e4m3_encode((amax / 6.0).reshape(K // 16, N))
    mv = Q.e4m3_decode(mc).reshape(K // 16, 1, N).clamp(min=1e-30)
    q = (xb / mv).clamp(-6, 6)
    codes = Q.fp4_encode(q).to(torch.uint8).reshape(K // 16, 16, N)
    pk = (pack_odd_low if swap else pack_even_low)(codes, 1)
    return pk.reshape(K // 2, N), mc


def deq_row(packed, mc, swap=False):
    R, half = packed.shape
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    a, b = (_nib_val(hi), _nib_val(lo)) if swap else (_nib_val(lo), _nib_val(hi))
    v = torch.stack([a, b], dim=2).reshape(R, half * 2)
    return v * Q.e4m3_decode(mc).repeat_interleave(16, dim=1)[:, : v.shape[1]]


def deq_col(packed, mc, swap=False):
    half, N = packed.shape
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    a, b = (_nib_val(hi), _nib_val(lo)) if swap else (_nib_val(lo), _nib_val(hi))
    v = torch.stack([a, b], dim=1).reshape(half * 2, N)
    return v * Q.e4m3_decode(mc).repeat_interleave(16, dim=0)


@triton.jit
def _dot_fp4(A, AS, B, BS, OUT, M, N, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_kp = k0 // 2 + tl.arange(0, BK // 2)
        offs_kg = k0 // 16 + tl.arange(0, BK // 16)
        a = tl.load(A + offs_m[:, None] * (K // 2) + offs_kp[None, :])
        b = tl.load(B + offs_kp[:, None] * N + offs_n[None, :])
        asc = tl.load(AS + offs_m[:, None] * (K // 16) + offs_kg[None, :]).to(tl.float8e4nv, bitcast=True)
        bsc = tl.load(BS + offs_n[:, None] * (K // 16) + offs_kg[None, :]).to(tl.float8e4nv, bitcast=True)
        acc = tl.dot_scaled(a, asc, "e2m1", b, bsc, "e2m1", acc=acc, out_dtype=tl.float32)
    tl.store(OUT + offs_m[:, None] * N + offs_n[None, :], acc)


@triton.jit
def _dot_bf16(A, B, OUT, M, N, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        a = tl.load(A + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(B + offs_k[:, None] * N + offs_n[None, :])
        acc = tl.dot(a, b, acc=acc, out_dtype=tl.float32)
    tl.store(OUT + offs_m[:, None] * N + offs_n[None, :], acc)


@triton.jit
def _dot_fp8(A, B, OUT, M, N, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        a = tl.load(A + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(B + offs_k[:, None] * N + offs_n[None, :])
        acc = tl.dot(a, b, acc=acc, out_dtype=tl.float32)
    tl.store(OUT + offs_m[:, None] * N + offs_n[None, :], acc)


def run_fp4(ac, asc, bc, bsc, M, N, K, BK=128):
    out = torch.empty(M, N, device="cuda", dtype=torch.float32)
    _dot_fp4[(triton.cdiv(M, 128), triton.cdiv(N, 128))](
        ac, asc, bc, bsc, out, M, N, K, BM=128, BN=128, BK=BK, num_warps=8)
    return out


def bench(fn, repeat=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat * 1000


# ---------------- A. packing convention ----------------
M0, N0, K0 = 256, 256, 128
torch.manual_seed(0)
x = torch.randn(M0, K0, device="cuda") * 0.5
w = torch.randn(N0, K0, device="cuda") * 0.5
wt = w.t().contiguous()
print("--- A. packing convention (rel err, lower = correct) ---", flush=True)
for unit in (True, False):
    for swap in (False, True):
        ac, asc = quant_row(x, unit=unit, swap=swap)
        bc, bsc = quant_col(wt, unit=unit, swap=swap)          # bsc: (K//16, N)
        out = run_fp4(ac, asc, bc, bsc.t().contiguous(), M0, N0, K0)   # kernel wants (N, K//16)
        ref = deq_row(ac, asc, swap=swap) @ deq_col(bc, bsc, swap=swap)
        rel = float((out - ref).norm() / ref.norm())
        print(f"  unit_scales={unit!s:5} swap_nibbles={swap!s:5} -> rel={rel:.5f}  "
              f"finite={bool(torch.isfinite(out).all())}", flush=True)

# ---------------- B. compute-bound speed ----------------
M, N, K = 8192, 1024, 4096    # long reduction, wide enough to fill 170 SMs
x = torch.randn(M, K, device="cuda") * 0.5
w = torch.randn(N, K, device="cuda") * 0.5
wt = w.t().contiguous()
ac, asc = quant_row(x)
bc, bsc = quant_col(wt)
bsc_k = bsc.t().contiguous()                       # (N, K//16) for the kernel
xd = deq_row(ac, asc).to(torch.bfloat16)
wd = deq_col(bc, bsc).to(torch.bfloat16).contiguous()
x8 = (xd.float() / 8.0).to(torch.float8_e4m3fn)
w8 = (wd.float() / 8.0).to(torch.float8_e4m3fn)

GRID2 = (triton.cdiv(M, 128), triton.cdiv(N, 128))
t4 = bench(lambda: run_fp4(ac, asc, bc, bsc_k, M, N, K))
ob = torch.empty(M, N, device="cuda", dtype=torch.float32)
t16 = bench(lambda: _dot_bf16[GRID2](xd, wd, ob, M, N, K, BM=128, BN=128, BK=128, num_warps=8, num_stages=1))
t8 = bench(lambda: _dot_fp8[GRID2](x8, w8, ob, M, N, K, BM=128, BN=128, BK=128, num_warps=8, num_stages=1))
fl = 2 * M * N * K / 1e12
print(f"--- B. compute-bound ({M}x{K}x{N}), {fl*1000:.1f} GFLOP ---", flush=True)
print(f"  fp4 dot_scaled : {t4:7.3f} ms  {fl/t4*1000:8.1f} TFLOP/s", flush=True)
print(f"  bf16 dot       : {t16:7.3f} ms  {fl/t16*1000:8.1f} TFLOP/s   (fp4/bf16 speedup {t16/t4:.2f}x)", flush=True)
print(f"  fp8  dot       : {t8:7.3f} ms  {fl/t8*1000:8.1f} TFLOP/s   (fp8/bf16 speedup {t16/t8:.2f}x)", flush=True)

# ---------------- C. register-fp8 (sm_89 bug present?) ----------------
@triton.jit
def _reg_fp8(A32, B8, OUT, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr):
    om = tl.arange(0, M); ok = tl.arange(0, K); on = tl.arange(0, N)
    a = tl.load(A32 + om[:, None] * K + ok[None, :])
    p8 = (a * 448.0).to(tl.float8e4nv)
    b = tl.load(B8 + ok[:, None] * N + on[None, :])
    c = tl.dot(p8, b, out_dtype=tl.float32)
    tl.store(OUT + om[:, None] * N + on[None, :], c)

a32 = torch.rand(128, 128, device="cuda")
b8 = (torch.randn(128, 128, device="cuda") * 0.05).to(torch.float8_e4m3fn)
a8 = (a32 * 448.0).to(torch.float8_e4m3fn)
outc = torch.empty(128, 128, device="cuda", dtype=torch.float32)
_reg_fp8[(1,)](a32, b8, outc, M=128, K=128, N=128)
torch.cuda.synchronize()
err = float((outc - a8.float() @ b8.float()).abs().max())
print(f"--- C. register-fp8 dot sm_120: max|diff|={err:.4f} -> "
      f"{'BROKEN' if err > 1.0 else 'OK'}", flush=True)
