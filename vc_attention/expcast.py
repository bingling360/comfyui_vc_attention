"""ExpCast-FP8: skip the FP32 exponential by writing the E4M3 byte directly.

The observation (VC-Attention, Sec 4.2 / Appendix B) is that an E4M3 byte is an
almost-affine function of log2 of the value it stores:

    byte(v) ~= 8 * log2(v) + 56        for v in [2^-9, 448]

    check: v=0.5  -> 48 -> e=6,m=0 -> 2^-1      = 0.5
           v=1    -> 56 -> e=7,m=0 -> 2^0       = 1
           v=2    -> 64 -> e=8,m=0 -> 2^1       = 2
           v=448  -> 126.5 -> e=15,m=6 -> 448

So instead of

    p     = exp2(u)          # MUFU.EX2, FP32
    p_fp8 = to_e4m3(p)       # cvt with saturation, FP32 -> FP8

we can compute the byte with one FMA and a rounding instruction:

    c     = 8 * u + 56 + beta
    code  = clip(round_rne(c), 0, 120)
    p_fp8 = bitcast_u8_to_e4m3(code)   # no-op

where u = (s_ij - m_i) * log2(e) + 8 is the log2-domain score of the *scaled*
probability p * 256: the +8 shifts the row maximum to log2(256), so the row max
lands on byte 120 = 2^8 = 256 and the whole E4M3 exponent range is used. The
256 factor is shared by the numerator and the softmax normaliser, so it cancels
in O = A / l.

beta = -0.35 removes the systematic bias introduced by rounding the *byte*
(a uniform grid in log space) rather than the value.

Applicability
-------------
ExpCast-FP8 is an 8-bit feature. At 4 bits the codes are NVFP4, whose value is
not an affine function of any log-domain score because the per-16 E4M3
microscale jumps between binades, so no linear map exists (Appendix A). The
workstation configuration (RTX 5090 / RTX PRO 6000) therefore leaves it off.
"""

from __future__ import annotations

import torch

__all__ = ["LOG2E", "DEFAULT_BETA", "expcast_codes", "standard_codes", "expcast_probabilities"]

LOG2E = 1.4426950408889634
DEFAULT_BETA = -0.35
_CODE_MAX = 120  # byte 120 = e15,m0 = 2^8 = 256


def expcast_codes(
    s: torch.Tensor,
    m: torch.Tensor,
    beta: float = DEFAULT_BETA,
) -> torch.Tensor:
    """E4M3 byte codes for the unnormalised probabilities of a score tile.

    Args:
        s: (M, N) raw scores, already divided by sqrt(d) but *not* shifted.
        m: (M,) running row maximum over the whole KV sequence so far.
        beta: rounding bias, see module docstring.

    Returns:
        (M, N) uint8 codes. Decode with ``quant.e4m3_decode`` to get p * 256.
    """
    u = (s - m.unsqueeze(-1)) * LOG2E + 8.0
    c = u * 8.0 + (56.0 + beta)
    c = torch.round(c).clamp_(0, _CODE_MAX)
    return c.to(torch.uint8)


def standard_codes(s: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """The path ExpCast replaces: exp2 in FP32, then cast to E4M3 at scale 256."""
    from .quant import e4m3_encode

    u = (s - m.unsqueeze(-1)) * LOG2E + 8.0
    p = torch.exp2(u)
    return e4m3_encode(p)


def expcast_probabilities(
    s: torch.Tensor, m: torch.Tensor, beta: float = DEFAULT_BETA
) -> torch.Tensor:
    """Convenience: decoded (p * 256) values, ready for an FP8 PV product."""
    from .quant import e4m3_decode

    return e4m3_decode(expcast_codes(s, m, beta))
