"""Numerical validation of the low-bit formats and ExpCast-FP8.

Runs on CPU. Every assertion here is a property the CUDA/Triton kernels rely on.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vc_attention.expcast import expcast_codes, standard_codes  # noqa: E402
from vc_attention.quant import (  # noqa: E402
    E4M3_MAX,
    FP4_VALUES,
    e4m3_decode,
    e4m3_encode,
    dequantize_nvfp4,
    fp4_decode,
    fp4_encode,
    quantize_e4m3,
    quantize_nvfp4,
)

torch.manual_seed(0)
ok = 0
fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name}  {detail}")
    else:
        fail += 1
        print(f"  FAIL  {name}  {detail}")


print("\n[1] E4M3 byte encoding against hand-computed values")
# byte(v) ~= 8*log2(v) + 56 ; the encoder must agree with the closed form.
known = {1.0: 56, 2.0: 64, 0.5: 48, 4.0: 72, 256.0: 120, E4M3_MAX: 126, 0.0: 0}
for val, byte in known.items():
    got = int(e4m3_encode(torch.tensor([val]))[0])
    check(f"byte({val})", got == byte, f"expected {byte}, got {got}")

print("\n[2] E4M3 decode matches its own encode (round trip)")
x = torch.cat(
    [
        torch.randn(20000) * 3.0,
        torch.randn(5000) * 0.001,          # subnormal region
        torch.tensor([0.0, E4M3_MAX, -E4M3_MAX, 1e-4, -1e-4]),
    ]
)
q = e4m3_encode(x)
d = e4m3_decode(q)
# Above the min normal the format is 3-bit mantissa: <= 2^-4 relative error.
normal = x.abs() >= 2.0 ** -6
rel = ((d - x).abs() / x.abs())[normal]
check("rel err <= 1 ULP (normal range)", float(rel.max()) <= 2.0 ** -4 + 1e-6,
      f"max rel err {float(rel.max()):.5f} (bound {2.0**-4:.5f})")
# In the subnormal range the grid is absolute (quantum 2^-9), not relative.
sub = (x.abs() < 2.0 ** -6) & (x.abs() > 0)
abs_err = (d - x).abs()[sub]
check("abs err <= half quantum (subnormal)", float(abs_err.max()) <= 2.0 ** -10 + 1e-9,
      f"max abs err {float(abs_err.max()):.3e} (bound {2.0**-10:.3e})")
check("no NaN produced", not bool(torch.isnan(d).any()))

print("\n[3] E4M3 encoder agrees with PyTorch's native fp8 conversion")
native = x.to(torch.float8_e4m3fn).to(torch.float32)
ours = e4m3_decode(e4m3_encode(x))
mism = int((native != ours).sum())
check("byte-identical to torch e4m3fn", mism == 0, f"{mism} / {x.numel()} differ")

print("\n[4] FP4 (E2M1) magnitude table")
for i, v in enumerate(FP4_VALUES):
    got = float(fp4_decode(torch.tensor([i], dtype=torch.uint8))[0])
    check(f"fp4 code {i}", abs(got - v) < 1e-6, f"expected {v}, got {got}")

print("\n[5] FP4 RNE midpoint behaviour (ties go to the even code)")
# midpoints: 0.25 -> 0.0, 0.75 -> 1.0, 1.25 -> 1.0, 1.75 -> 2.0, 2.5 -> 2.0,
#            3.5 -> 4.0, 5.0 -> 4.0
ties = {0.25: 0.0, 0.75: 1.0, 1.25: 1.0, 1.75: 2.0, 2.5: 2.0, 3.5: 4.0, 5.0: 4.0}
for t, want in ties.items():
    got = float(fp4_decode(fp4_encode(torch.tensor([t])))[0])
    check(f"tie {t}", abs(got - want) < 1e-6, f"expected {want}, got {got}")

print("\n[6] NVFP4 round trip")
v = torch.randn(1, 4, 4096, 128) * 0.7
codes, micro, gscale = quantize_nvfp4(v.reshape(1, -1), block=16)
deq = dequantize_nvfp4(codes, micro, gscale, block=16).reshape(v.shape)
snr = 20 * torch.log10(v.norm() / (v - deq).norm())
check("NVFP4 SNR sane", float(snr) > 10.0, f"{float(snr):.2f} dB (4-bit, expect ~12-18)")

print("\n[7] ExpCast-FP8 vs exp-then-cast")
# The paper evaluates on softmax scores from a real attention tile, i.e.
# s = q.k / sqrt(d) with q,k ~ N(0,1): score std ~1. Using a much wider score
# distribution pushes most probabilities into E4M3's subnormal range, where the
# affine byte map degrades and the reported statistics no longer apply.
q = torch.randn(256, 128)
kk = torch.randn(4096, 128)
s = (q @ kk.T) / (128 ** 0.5)
check("score std ~ 1 (realistic tile)", 0.8 < float(s.std()) < 1.2, f"std {float(s.std()):.3f}")
m = s.amax(dim=-1)
c_exp = expcast_codes(s, m).to(torch.int16)
c_std = standard_codes(s, m).to(torch.int16)
agree = float((c_exp == c_std).float().mean())
check("byte agreement ~79.6%", abs(agree - 0.796) < 0.01, f"{agree:.4f}")

from vc_attention.quant import e4m3_decode  # noqa: E402


def tv_distance(a, b):
    """TV between two softmax rows: normalise first, then 0.5 * L1."""
    a = a / a.sum(-1, keepdim=True)
    b = b / b.sum(-1, keepdim=True)
    return 0.5 * (a - b).abs().sum(-1)


p_exp = e4m3_decode(c_exp.to(torch.uint8)) / 256.0
p_std = e4m3_decode(c_std.to(torch.uint8)) / 256.0
p_exact = torch.exp2((s - m.unsqueeze(-1)) * 1.4426950408889634)
tv = tv_distance(p_exp, p_exact)
tv_ref = tv_distance(p_std, p_exact)
check("total variation < 3.64% bound", float(tv.max()) < 0.0364, f"max TV {float(tv.max()):.5f}")
check("ExpCast close to the exact cast it replaces", float(tv.mean()) < float(tv_ref.mean()) + 0.01,
      f"expcast {float(tv.mean()):.5f} vs exp+cast {float(tv_ref.mean()):.5f}")

print("\n[8] ExpCast row maximum lands on 2^8 = 256")
rowmax = e4m3_decode(expcast_codes(s, m))[0].max()
check("row max == 256", float(rowmax) == 256.0, f"got {float(rowmax)}")

print(f"\n=== {ok} passed, {fail} failed ===")
sys.exit(1 if fail else 0)
