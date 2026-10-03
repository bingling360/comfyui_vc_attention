"""Probe 28: is the real blocker for 4-bit P a hardware limit, or just a
missing encoder?

Earlier claim (mine): "FP4 PV is impossible in principle because P is a register
operand". That is too strong. tl.dot_scaled needs P *packed* as e2m1, which is
just a conversion -- and Blackwell has a single-instruction converter:

    cvt.rn.satfinite.e2m1x2.f32 d, a, b;      // 2 x f32 -> 1 byte of 2 x e2m1

PyTorch does not expose it, which is why the portable encoder needs 7
comparisons. Triton can reach it via tl.inline_asm_elementwise. If that works,
the encode cost collapses and 4-bit PV becomes a real option; if it does not,
the "encode costs 600x the MMA" argument stands.
"""
import os
import sys
import time

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/tcache28")
import torch
import triton
import triton.language as tl

sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention")
from vc_attention import quant as Q  # noqa: E402

GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")


def decode(packed):
    lo = (packed & 0xF).long()
    hi = ((packed >> 4) & 0xF).long()
    v = torch.stack([GRID[lo], GRID[hi]], dim=-1)
    return v.reshape(-1)


@triton.jit
def _pack_cvt(X, OUT, N, BLOCK: tl.constexpr):
    """pack=1, x passed twice: exercises cvt.rn.satfinite.e2m1x2.f32 and lets us
    read the low nibble. d must be a .b8 register, not .b32 -- that was the
    ptxas 'Arguments mismatch'."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    x = tl.load(X + offs, mask=m, other=0.0)
    b = tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b8 t;
        cvt.rn.satfinite.e2m1x2.f32 t, $1, $2;
        cvt.u32.u8 $0, t;
        }
        """,
        constraints="=r,r,r",
        args=[x, x],
        dtype=tl.uint8,
        is_pure=True,
        pack=1,
    )
    tl.store(OUT + offs, b, mask=m)


def run_cvt(x, out, block=4096):
    n = x.numel()
    _pack_cvt[(triton.cdiv(n, block),)](x, out, n, BLOCK=block, num_warps=4)


N = 1 << 20
x = (torch.randn(N, device="cuda") * 1.5).clamp(-6, 6)
out = torch.zeros(N, device="cuda", dtype=torch.uint8)
try:
    run_cvt(x, out)
    torch.cuda.synchronize()
    print("cvt.rn.satfinite.e2m1x2.f32 COMPILED AND RAN", flush=True)
    lo = (out & 0xF).to(torch.int64)
    sign = torch.where((lo & 0x8) != 0, -1.0, 1.0)
    got = GRID[lo & 0x7] * sign          # GRID has 8 entries: strip the sign bit
    ref = Q.fp4_decode(Q.fp4_encode(x[: 1 << 16].clamp(-6, 6))).to(torch.float32)
    print(f"  low-nibble agree with quant.fp4_encode: "
          f"{float((got[: 1 << 16] == ref).float().mean()):.4f}", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"cvt inline-asm FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)


def timeit(fn, rep=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / rep * 1000


# encoder cost: portable (7 comparisons) vs PTX cvt, over a PV-shaped tile count
BIG = 16384 * 128  # elements in one 128x128 tile x 128 tiles
xb = (torch.randn(BIG, device="cuda") * 1.5).clamp(-6, 6)
ob = torch.zeros(BIG, device="cuda", dtype=torch.uint8)


def portable():
    a = xb.abs()
    c = (a > 0.25).to(torch.uint8)
    for b in (0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        c = c + (a > b).to(torch.uint8)
    return c


def ptx():
    run_cvt(xb, ob)


try:
    print(f"\nportable encode ({BIG/1e6:.1f}M elems): {timeit(portable):7.3f} ms", flush=True)
    print(f"PTX cvt encode  ({BIG/1e6:.1f}M elems): {timeit(ptx):7.3f} ms", flush=True)
except Exception as e:  # noqa: BLE001
    print("timing failed:", type(e).__name__, str(e)[:200], flush=True)
