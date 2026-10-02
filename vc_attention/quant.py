"""Low-bit quantizers used by VC-Attention.

Everything here is written in portable PyTorch integer/bit ops rather than
relying on hardware fp8/fp4 conversion instructions, so the same code path
produces identical codes on CPU (for tests), Ada, Hopper and Blackwell.
The CUDA/Triton kernels re-derive these formats natively inside the kernel;
this module is the *oracle* they are checked against.

Formats
-------
E4M3 (OCP, e4m3fn)
    bias 7, 3 mantissa bits. Max finite 448 (code 0x7E). 0x7F / 0xFF are NaN.
    Subnormals: e==0 -> value = m * 2^-9, which tile continuously into the
    min normal 2^-6 (code 8).

E2M1 (OCP MX/NV FP4)
    1 sign, 2 exponent (bias 1), 1 mantissa. Magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}.
    NVFP4 = E2M1 element under a per-16 E4M3 microscale plus one tensor-level
    FP32 scale.

INT4
    Symmetric signed 4-bit, scale per group. Only used as the Ada fallback,
    where no FP4 MMA exists.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

__all__ = [
    "E4M3_MAX", "E4M3_MIN_NORMAL", "FP4_VALUES",
    "e4m3_encode", "e4m3_decode",
    "fp4_encode", "fp4_decode",
    "quantize_e4m3", "dequantize_e4m3",
    "quantize_nvfp4", "dequantize_nvfp4",
    "quantize_int4", "dequantize_int4",
    "amax_reduce",
]

E4M3_MAX = 448.0
E4M3_MIN_NORMAL = 2.0 ** -6
_E4M3_NAN_CODES = (0x7F, 0xFF)

FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_FP4_MAX = 6.0
# Midpoints between adjacent FP4 magnitudes; ties go to the even code (RNE).
_FP4_BREAKS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


# ---------------------------------------------------------------------------
# E4M3
# ---------------------------------------------------------------------------

def e4m3_encode(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even encode ``x`` to E4M3 codes (uint8, 0..126).

    ``torch.round`` is round-half-to-even, which is exactly the RNE the
    hardware cvt instruction implements.
    """
    a = x.detach().to(torch.float32)
    sign = (a < 0).to(torch.uint8)
    a = a.abs().clamp(max=E4M3_MAX)

    frac, exp = torch.frexp(a)          # a = frac * 2**exp, frac in [0.5, 1)
    is_sub = a < E4M3_MIN_NORMAL

    # --- subnormal / zero: one quantum is 2^-9, so the code is round(a * 2^9).
    code_sub = torch.round(a * 512.0).clamp_(0, 8)

    # --- normal: mantissa = round((frac * 2 - 1) * 8) in [0, 8], 8 carries.
    mant = torch.round(frac * 16.0 - 8.0)
    carry = mant >= 8.0
    mant = torch.where(carry, mant - 8.0, mant)
    exp = torch.where(carry, exp + 1, exp)
    exp = torch.where(is_sub, torch.zeros_like(exp), exp)
    e_field = (exp - 1) + 7             # unbiased exponent + bias
    e_field = e_field.clamp(1, 15)
    code_n = (e_field.to(torch.int64) << 3) | mant.to(torch.int64).clamp(0, 7)

    code = torch.where(is_sub, code_sub.to(torch.int64), code_n)
    code = (code & 0x7F) | (sign.to(torch.int64) << 7)
    return code.to(torch.uint8)


def e4m3_decode(code: torch.Tensor) -> torch.Tensor:
    """Decode E4M3 codes (uint8) back to float32."""
    c = code.to(torch.int64) & 0xFF
    sign = ((c >> 7) & 1) * -2.0 + 1.0
    e = (c >> 3) & 0xF
    m = c & 0x7
    is_sub = e == 0
    val_sub = m.to(torch.float32) * (2.0 ** -9)
    val_norm = (1.0 + m.to(torch.float32) / 8.0) * torch.pow(2.0, (e - 7).to(torch.float32))
    val = torch.where(is_sub, val_sub, val_norm)
    # 0x7F / 0xFF are NaN in e4m3fn; clamp them to max finite for safety.
    nan_mask = ((e == 15) & (m == 7))
    val = torch.where(nan_mask, torch.full_like(val, E4M3_MAX), val)
    return (val * sign).to(torch.float32)


def amax_reduce(x: torch.Tensor, dim: int, keepdim: bool = True) -> torch.Tensor:
    return x.detach().abs().amax(dim=dim, keepdim=keepdim)


def quantize_e4m3(
    x: torch.Tensor,
    dim: int = -1,
    scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric E4M3 quantization.

    Args:
        x: tensor to quantize.
        dim: reduction axis for the scale (-1 -> per row/token, -2 -> per channel).
        scale: precomputed scale; if given, ``dim`` is ignored.

    Returns:
        (codes uint8, scale float32). Dequantized value = e4m3_decode(codes) * scale.
    """
    if scale is None:
        amp = amax_reduce(x, dim=dim, keepdim=True)
        scale = amp.clamp(min=1e-30) / E4M3_MAX
    q = e4m3_encode(x.detach() / scale)
    return q, scale.to(torch.float32)


def dequantize_e4m3(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return e4m3_decode(codes) * scale


def _pad_along(x: torch.Tensor, dim: int, pad: int) -> torch.Tensor:
    if dim < 0:
        dim += x.dim()
    shape = [0] * ((x.dim() - 1 - dim) * 2) + [0, pad]
    return torch.nn.functional.pad(x, shape)


# ---------------------------------------------------------------------------
# FP4 / NVFP4
# ---------------------------------------------------------------------------

def fp4_encode(x: torch.Tensor) -> torch.Tensor:
    """RNE encode to E2M1 codes 0..15 (bit 3 is the sign).

    ``idx = count(breaks < |a|)`` floors to the enclosing magnitude. When |a|
    lands exactly on a midpoint the hardware picks the *even* code, which for
    the midpoint between codes i and i+1 means "round up only if i is odd".
    """
    a = x.detach().to(torch.float32)
    sign = (a < 0).to(torch.int64)
    a = a.abs()
    breaks = a.new_tensor(_FP4_BREAKS, dtype=torch.float32)
    idx = torch.searchsorted(breaks, a, right=False).to(torch.int64)
    nb = len(_FP4_BREAKS)
    in_range = idx < nb
    safe = idx.clamp(max=nb - 1)
    tie = in_range & (a == breaks[safe]) & ((safe & 1) != 0)
    idx = torch.where(tie, idx + 1, idx).clamp_(0, 7)
    return ((sign << 3) | idx).to(torch.uint8)


def fp4_decode(code: torch.Tensor) -> torch.Tensor:
    c = code.to(torch.int64) & 0xF
    sign = ((c >> 3) & 1) * -2.0 + 1.0
    mag = c & 0x7
    table = torch.tensor(FP4_VALUES, dtype=torch.float32, device=code.device)
    return table[mag] * sign.to(torch.float32)


def quantize_nvfp4(
    x: torch.Tensor,
    block: int = 16,
    global_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NVFP4: E2M1 payload + per-``block`` E4M3 microscale + tensor FP32 scale.

    Returns (codes uint8 packed 2-per-byte is NOT done here -- codes are
    returned as one uint8 per element for clarity, microscales as E4M3 codes,
    and the tensor-level FP32 scale).
    """
    a = x.detach()
    flat = a.reshape(-1)
    n = flat.numel()
    pad = (-n) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.reshape(-1, block)

    if global_scale is None:
        # Keep the per-block amax inside E4M3's finite range after division.
        g = blocks.abs().amax().clamp(min=1e-30) / (_FP4_MAX * E4M3_MAX)
    else:
        g = global_scale
    g = g.to(torch.float32).clamp(min=1e-30)

    amax_b = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-30)
    micro = (amax_b / _FP4_MAX) / g                       # target microscale
    micro_code = e4m3_encode(micro.clamp(max=E4M3_MAX))   # -> E4M3
    micro_val = e4m3_decode(micro_code).to(torch.float32)

    q = fp4_encode(blocks / (g * micro_val).clamp(min=1e-30))
    return q.reshape(-1)[:n], micro_code, g


def dequantize_nvfp4(
    codes: torch.Tensor, micro_code: torch.Tensor, global_scale: torch.Tensor, block: int = 16
) -> torch.Tensor:
    micro_val = e4m3_decode(micro_code).to(torch.float32)
    vals = fp4_decode(codes) * (micro_val.repeat_interleave(block)[: codes.numel()] * global_scale.to(torch.float32))
    return vals


# ---------------------------------------------------------------------------
# INT4 (Ada fallback)
# ---------------------------------------------------------------------------

def quantize_e4m3_blocks(x: torch.Tensor, block_rows: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-(block of rows) x per-channel E4M3: the scale granularity V-Smooth needs.

    This is the granularity that makes grouping pay off. A quantisation block
    gets one scale per channel, so the block's largest entries set it and
    everything else in the block wastes code range. Homogeneous blocks
    (which is what the k-means permutation buys) keep that waste small.

    Args:
        x: (G, N, D) with N a multiple of block_rows.
    Returns:
        (codes (G, N, D) uint8, scale (G, NB, D) float32).
    """
    g, n, d = x.shape
    xb = x.reshape(g, n // block_rows, block_rows, d)
    amp = xb.detach().abs().amax(dim=2).clamp(min=1e-30)
    scale = amp / E4M3_MAX
    q = e4m3_encode(xb.detach() / scale.unsqueeze(2))
    return q.reshape(g, n, d), scale.to(torch.float32)


def dequantize_e4m3_blocks(codes: torch.Tensor, scale: torch.Tensor, block_rows: int) -> torch.Tensor:
    g, n, d = codes.shape
    s = scale.unsqueeze(2).expand(g, n // block_rows, block_rows, d).reshape(g, n, d)
    return e4m3_decode(codes) * s


def quantize_nvfp4_grouped(
    x: torch.Tensor, group_rows: int = 16
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NVFP4 with per-(group_rows tokens) x per-channel E4M3 microscales.

    In the PV product the contraction axis is the token axis, so NVFP4's
    1-scale-per-16-elements lands on 16 tokens of one output channel. This
    is the granularity a Blackwell ``tcgen05.mma.block_scale`` uses; we expose
    it as (group_rows, channel) so the reference implementation can apply the
    microscale in its epilogue.

    Returns:
        (codes (G,N,D) uint8, microscale E4M3 codes (G,NB,D), tensor scale g).
        Effective per-(group, channel) scale = g * decode(micro).
    """
    g, n, d = x.shape
    xb = x.reshape(g, n // group_rows, group_rows, d)
    gscale = xb.detach().abs().amax().clamp(min=1e-30) / (_FP4_MAX * E4M3_MAX)
    amax_g = xb.detach().abs().amax(dim=2).clamp(min=1e-30)
    micro_code = e4m3_encode((amax_g / _FP4_MAX / gscale).clamp(max=E4M3_MAX))
    micro_val = e4m3_decode(micro_code)
    q = fp4_encode(xb.detach() / (gscale * micro_val).unsqueeze(2).clamp(min=1e-30))
    return q.reshape(g, n, d), micro_code, gscale.to(torch.float32)


def dequantize_nvfp4_grouped(
    codes: torch.Tensor, micro_code: torch.Tensor, gscale: torch.Tensor, group_rows: int = 16
) -> Tuple[torch.Tensor, torch.Tensor]:
    """-> (dequantised (G,N,D), effective scale (G, NB, D))."""
    g, n, d = codes.shape
    micro_val = e4m3_decode(micro_code)
    scale = (gscale.to(torch.float32) * micro_val).to(torch.float32)
    s = scale.unsqueeze(2).expand(g, n // group_rows, group_rows, d).reshape(g, n, d)
    return fp4_decode(codes) * s, scale


def quantize_int4(x: torch.Tensor, dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    amp = amax_reduce(x, dim=dim, keepdim=True).clamp(min=1e-30)
    scale = amp / 7.0
    q = torch.round(x.detach() / scale).clamp_(-8, 7).to(torch.int8)
    return q, scale.to(torch.float32)


def dequantize_int4(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scale
