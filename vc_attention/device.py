"""Hardware capability detection and backend selection for VC-Attention.

VC-Attention has two deployment configurations (see arXiv:2609.15810, Sec 4.1):

  8-bit  : QK and PV both E4M3, V-Smooth + ExpCast-FP8.
           Deployed on datacenter parts (B200/B300/H200) where the FP32
           exponential in softmax is the longest pipeline stage.

  4-bit  : QK and PV both NVFP4 (E2M1 + per-16 E4M3 microscale), V-Smooth only.
           Deployed on workstation Blackwell (RTX 5090 / RTX PRO 6000), where
           softmax is *not* the bottleneck and ExpCast-FP8 does not apply
           (Appendix A: no affine log-domain -> code map exists for NVFP4).

This module decides which one the current GPU can actually run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

# (major, minor) -> marketing name, only for the parts we care about.
_KNOWN_SM = {
    (8, 9): "Ada Lovelace (RTX 40xx)",
    (9, 0): "Hopper (H100/H200)",
    (10, 0): "Blackwell datacenter (B200)",
    (10, 3): "Blackwell datacenter (B300)",
    (12, 0): "Blackwell workstation (RTX 50xx / RTX PRO)",
}


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    sm: Tuple[int, int]
    has_fp8: bool          # E4M3 tensor cores
    has_fp4: bool          # NVFP4 / MXFP4 tensor cores
    softmax_bound: bool    # FP32 exp is the critical stage -> ExpCast-FP8 pays
    recommended_bits: int  # 8 or 4
    recommended_expcast: bool

    def describe(self) -> str:
        return (
            f"{self.name} (sm_{self.sm[0]}{self.sm[1]}): "
            f"fp8={self.has_fp8}, fp4={self.has_fp4}, "
            f"recommended={self.recommended_bits}-bit, "
            f"expcast={self.recommended_expcast}"
        )


def detect_profile(device: Optional[torch.device] = None) -> DeviceProfile:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        return DeviceProfile(
            name="CPU", sm=(0, 0), has_fp8=False, has_fp4=False,
            softmax_bound=False, recommended_bits=8, recommended_expcast=False,
        )

    major, minor = torch.cuda.get_device_capability(device)
    sm = (major, minor)
    name = _KNOWN_SM.get(sm, f"sm_{major}{minor}")

    # FP8 E4M3 tensor cores: Hopper (9.0) and newer, Ada (8.9) and Blackwell.
    has_fp8 = sm >= (8, 9)
    # NVFP4 / MXFP4 tensor cores: Blackwell only (datacenter 10.x, workstation 12.0).
    # Ada has INT4 IMMA but no FP4 MMA, so the 4-bit kernel cannot run there.
    has_fp4 = (major == 10) or sm == (12, 0)
    # Blackwell doubled Tensor Core throughput without doubling MUFU exp
    # throughput, so the softmax stage dominates on 10.x and 9.0 at 8 bits.
    # On workstation Blackwell the 4-bit MMA is slow enough that softmax
    # is no longer the longest stage, and ExpCast-FP8 is disabled.
    softmax_bound = has_fp8 and not has_fp4

    if has_fp4:
        return DeviceProfile(name, sm, has_fp8, True, False, 4, False)
    if has_fp8:
        return DeviceProfile(name, sm, True, False, True, 8, True)
    return DeviceProfile(name, sm, False, False, False, 8, False)


def resolve_backend(requested: str, profile: Optional[DeviceProfile] = None) -> str:
    """Map a user-facing backend request onto something the GPU can run.

    ``requested`` is one of: auto, fp8, nvfp4, int4, reference.
    Returns one of: fp8, nvfp4, int4, reference.
    """
    profile = profile or detect_profile()
    requested = (requested or "auto").lower()

    if requested == "reference":
        return "reference"
    if requested == "fp8":
        if not profile.has_fp8:
            return "reference"
        return "fp8"
    if requested in ("nvfp4", "fp4"):
        if not profile.has_fp4:
            # Ada and older: fall back rather than silently emulating FP4 in
            # software, which is slower than the bf16 path it replaces.
            return "fp8" if profile.has_fp8 else "reference"
        return "nvfp4"
    if requested == "int4":
        # Blackwell datacenter dropped INT4/INT8 MMA; Ada and older keep it.
        if profile.sm >= (10, 0):
            return "fp8" if profile.has_fp8 else "reference"
        return "int4" if profile.sm >= (7, 5) else "reference"

    # auto
    if profile.recommended_bits == 4 and profile.has_fp4:
        return "nvfp4"
    if profile.has_fp8:
        return "fp8"
    return "reference"
