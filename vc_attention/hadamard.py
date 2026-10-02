"""Orthonormal Hadamard rotation for Q and K.

VC-Attention assumes the QK product is already accurate at low bit width, which
in practice means the standard Hadamard rotation (SageAttention2, FlashAttention-3
FP8). Without it the QK term dominates everything else: measured on an
H3-shaped tile, per-token FP8 Q/K leaves 7.3% relative output error while V and
P each leave ~0.3%, so no amount of V smoothing is visible end to end.

The Walsh-Hadamard matrix H_d (d a power of two) satisfies H H^T = d I, so the
normalised H / sqrt(d) is orthonormal and

    (Q R) (K R)^T = Q R R^T K^T = Q K^T      with R = H / sqrt(d)

for *any* Q and K. Nothing about RoPE matters here: R is applied after the
rotation is baked in, and the identity holds for the full head_dim. The point of
applying it is distributional: it spreads an outlier channel across all d
channels, so a per-token or per-channel scale is no longer set by a single spike.

H3 note: MM-RoPE rotates 96 of the 128 head channels; the remaining 32 are
carried through unrotated. The transform is still applied to all 128 channels of
both Q and K -- it has to be, or the identity above does not hold.
"""

from __future__ import annotations

import math

import torch

__all__ = ["fwht", "hadamard_matrix", "is_power_of_two"]


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def fwht(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """In-place-free Walsh-Hadamard transform along the last axis.

    Args:
        x: (..., d) with d a power of two.
        normalize: divide by sqrt(d) so the transform is orthonormal.
    """
    d = x.shape[-1]
    if not is_power_of_two(d):
        raise ValueError(f"Hadamard transform needs a power-of-two dim, got {d}")
    if d == 1:
        return x

    batch = x.shape[:-1]
    out = x
    h = 1
    while h < d:
        # Butterfly stage h: pair up runs of h elements. The leading shape is
        # taken from the *input*, not from ``out``, whose trailing axis has
        # already been widened to 2h by the previous stage.
        out = out.reshape(*batch, -1, 2, h)
        a = out[..., 0, :]
        b = out[..., 1, :]
        out = torch.cat((a + b, a - b), dim=-1)
        h *= 2
    out = out.reshape(x.shape)
    if normalize:
        out = out / math.sqrt(d)
    return out


def hadamard_matrix(d: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Explicit normalised Hadamard matrix, for tests and for kernel constants."""
    if not is_power_of_two(d):
        raise ValueError(f"need a power of two, got {d}")
    h = torch.ones((1, 1), device=device, dtype=dtype)
    while h.shape[0] < d:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0
        )
    return h / math.sqrt(d)
