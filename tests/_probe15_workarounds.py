"""Probe 15: workarounds for the register-fp8 dot corruption on sm_89.

  V0  A quantized via .to(float8e4nv) in registers      (known broken)
  V1  A codes bitcast from uint8 in registers           (bypasses cvt path)
  V2  A stored to global scratch, reloaded, then dot    (memory operand)
  V3  control: A loaded from memory                     (known good)
Each variant computes dot(A, B8) with B8 from memory; reference = exact fp32
of the quantized values.
"""
import os
import sys

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/tcache15")
import torch
import triton
import triton.language as tl

print(f"triton {triton.__version__}")
M, K, N = 128, 128, 128
dev = "cuda"
torch.manual_seed(0)
a32 = torch.rand(M, K, device=dev)
a8_mem = (a32 * 448.0).to(torch.float8_e4m3fn)
a8_codes = a8_mem.view(torch.uint8)
b8 = (torch.randn(K, N, device=dev) * 0.05).to(torch.float8_e4m3fn)
ref = a8_mem.float() @ b8.float()
ref_mag = float(ref.abs().mean())
scratch = torch.empty(M * K, device=dev, dtype=torch.float8_e4m3fn)


@triton.jit
def _k(A32, A8, A_CODES, B8, SCRATCH, OUT,
       M: tl.constexpr, K: tl.constexpr, N: tl.constexpr, MODE: tl.constexpr):
    om = tl.arange(0, M)
    ok = tl.arange(0, K)
    on = tl.arange(0, N)
    a_offs = om[:, None] * K + ok[None, :]
    b_offs = ok[:, None] * N + on[None, :]
    b = tl.load(B8 + b_offs)
    if MODE == 0:      # .to() conversion in registers
        a = tl.load(A32 + a_offs)
        p8 = (a * 448.0).to(tl.float8e4nv)
    elif MODE == 1:    # bitcast from uint8 codes
        codes = tl.load(A_CODES + a_offs)
        p8 = codes.to(tl.float8e4nv, bitcast=True)
    elif MODE == 2:    # global scratch round-trip
        a = tl.load(A32 + a_offs)
        p8 = (a * 448.0).to(tl.float8e4nv)
        tl.store(SCRATCH + om[:, None] * K + ok[None, :], p8)
        p8 = tl.load(SCRATCH + om[:, None] * K + ok[None, :])
    else:              # control: from memory
        p8 = tl.load(A8 + a_offs)
    c = tl.dot(p8, b)
    tl.store(OUT + om[:, None] * N + on[None, :], c)


out = torch.empty(M, N, device=dev, dtype=torch.float32)
names = {0: "V0 .to() registers ", 1: "V1 uint8 bitcast  ", 2: "V2 scratch trip   ", 3: "V3 memory control "}
for mode in (0, 1, 2, 3):
    out.zero_()
    _k[(1,)](a32, a8_mem, a8_codes, b8, scratch, out, M=M, K=K, N=N, MODE=mode)
    torch.cuda.synchronize()
    nan = not torch.isfinite(out).all()
    err = float((out - ref).abs().max())
    rel = err / max(ref_mag, 1e-9)
    ok = (not nan) and rel < 0.02
    print(f"{names[mode]}: NaN={nan}  max|diff|={err:9.4f}  (ref |mean|={ref_mag:8.2f})  -> {'OK' if ok else 'BROKEN'}")
