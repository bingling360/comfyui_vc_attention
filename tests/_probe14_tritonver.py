"""Probe 14: minimal repro of the sm_89 register-fp8 MMA NaN, version-sensitive.

Kernel: A = fp8 quantized IN REGISTERS (from fp32 probabilities), B = fp8 from
memory, C = dot(A, B). On Triton 3.6 + sm_89 this yields NaN; both-operands-
from-memory dots are fine. Run under different Triton versions to check for a
fix.
"""
import os
import sys

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/tcache_repro")
import torch
import triton
import triton.language as tl

print(f"triton {triton.__version__}  torch {torch.__version__}  {torch.cuda.get_device_name(0)}")

M, K, N = 128, 128, 128
dev = "cuda"
torch.manual_seed(0)
a32 = torch.rand(M, K, device=dev)                     # probabilities in [0, 1]
b8 = (torch.randn(K, N, device=dev) * 0.05).to(torch.float8_e4m3fn)
a8_mem = (a32 * 448.0).to(torch.float8_e4m3fn)         # same values, quantized host-side

# reference in fp32 with exact dequant
a_ref = a8_mem.float()
b_ref = b8.float()
ref = a_ref @ b_ref


@triton.jit
def _reg_dot(A32, B8, OUT, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_k = tl.arange(0, K)
    offs_n = tl.arange(0, N)
    a = tl.load(A32 + offs_m[:, None] * K + offs_k[None, :])
    p8 = (a * 448.0).to(tl.float8e4nv)          # computed in registers
    b = tl.load(B8 + offs_k[:, None] * N + offs_n[None, :])
    c = tl.dot(p8, b)                            # register-fp8 A x memory-fp8 B
    tl.store(OUT + offs_m[:, None] * N + offs_n[None, :], c)


@triton.jit
def _mem_dot(A8, B8, OUT, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_k = tl.arange(0, K)
    offs_n = tl.arange(0, N)
    a = tl.load(A8 + offs_m[:, None] * K + offs_k[None, :])   # fp8 from memory
    b = tl.load(B8 + offs_k[:, None] * N + offs_n[None, :])
    c = tl.dot(a, b)
    tl.store(OUT + offs_m[:, None] * N + offs_n[None, :], c)


out = torch.empty(M, N, device=dev, dtype=torch.float32)
_reg_dot[(1,)](a32, b8, out, M=M, K=K, N=N)
torch.cuda.synchronize()
reg_nan = not torch.isfinite(out).all()
reg_err = float((out - ref).abs().max())
print(f"register-fp8 A dot : NaN={reg_nan}  max|diff|={reg_err:.4e}")

out2 = torch.empty(M, N, device=dev, dtype=torch.float32)
_mem_dot[(1,)](a8_mem, b8, out2, M=M, K=K, N=N)
torch.cuda.synchronize()
mem_nan = not torch.isfinite(out2).all()
mem_err = float((out2 - ref).abs().max())
print(f"memory-fp8 A dot   : NaN={mem_nan}  max|diff|={mem_err:.4e}")

verdict = "FIXED" if (not reg_nan and reg_err < 1.0) else "STILL BROKEN"
print(f"[{verdict} on triton {triton.__version__}]")
