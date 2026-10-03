"""Probe 29: full NVFP4 pack, portable vs PTX-cvt.

probe28 showed cvt.rn.satfinite.e2m1x2.f32 works on sm_120 in Triton and is
bit-exact vs quant.fp4_encode. That undermines my earlier claim that the encode
cost is what blocks 4-bit. This measures the WHOLE pack (per-16 microscale +
e4m1 codes + 2-per-byte packing) on a realistic (16384*56, 128) tensor, which is
the number that decides whether the host-side 4-bit path is viable.
"""
import os
import sys
import time

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/tcache29")
import torch
import triton
import triton.language as tl

sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention")
from vc_attention import quant as Q  # noqa: E402
from vc_attention.kernels.triton_attn import nvfp4_pack  # noqa: E402

GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")


@triton.jit
def _pack_pair(A, B, OUT, N, BLOCK: tl.constexpr):
    """Two fp32 tensors -> one packed byte each: a in low nibble, b in high."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    a = tl.load(A + offs, mask=m, other=0.0)
    b = tl.load(B + offs, mask=m, other=0.0)
    r = tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b8 t;
        cvt.rn.satfinite.e2m1x2.f32 t, $1, $2;
        cvt.u32.u8 $0, t;
        }
        """,
        constraints="=r,r,r",
        args=[a, b],
        dtype=tl.uint8,
        is_pure=True,
        pack=1,
    )
    tl.store(OUT + offs, r, mask=m)


def nvfp4_pack_ptx(x, group=16, block=4096, swap=False):
    d = x.shape[-1]
    xb = x.reshape(-1, d // group, group)
    amax = xb.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    micro_fp8 = (amax / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn)
    micro = micro_fp8.view(torch.uint8).squeeze(-1)
    mv = micro_fp8.float().clamp(min=1e-30)
    q = (xb / mv).reshape(-1, d).contiguous()
    first = 1 if swap else 0
    a = q[:, first::2].contiguous()
    b = q[:, (1 - first)::2].contiguous()
    out = torch.empty(a.shape, dtype=torch.uint8, device=x.device)
    n = a.numel()
    _pack_pair[(triton.cdiv(n, block),)](a, b, out, n, BLOCK=block, num_warps=4)
    return out, micro


def deq(packed, micro):
    lo = (packed & 0xF).long()
    hi = ((packed >> 4) & 0xF).long()
    s = lambda c: GRID[c & 0x7] * torch.where((c & 0x8) != 0, -1.0, 1.0)
    v = torch.stack([s(lo), s(hi)], dim=-1).reshape(packed.shape[0], -1)
    return v * Q.e4m3_decode(micro).repeat_interleave(16, dim=1)


def timeit(fn, rep=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / rep * 1000


R, D = 16384 * 56, 128          # one Q (or K) at 16K tokens, 56 heads
x = torch.randn(R, D, device="cuda") * 0.5
print(f"tensor {R}x{D} = {x.numel()/1e6:.1f}M elements", flush=True)

tp, mp = nvfp4_pack(x)
for sw in (False, True):
    tq, mq = nvfp4_pack_ptx(x, swap=sw)
    torch.cuda.synchronize()
    print(f"swap={sw!s:5} packed bytes identical to portable: "
          f"{float((tp == tq).float().mean()):.6f}   micro identical: "
          f"{float((mp == mq).float().mean()):.6f}", flush=True)

print(f"\nportable nvfp4_pack : {timeit(lambda: nvfp4_pack(x)):7.2f} ms", flush=True)
print(f"PTX      nvfp4_pack : {timeit(lambda: nvfp4_pack_ptx(x, swap=True)):7.2f} ms", flush=True)

# reference: what the fp8 path costs (a single fused cast, for scale)
print(f"fp8 cast (one pass) : {timeit(lambda: x.to(torch.float8_e4m3fn)):7.2f} ms", flush=True)
