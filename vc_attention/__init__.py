"""VC-Attention for MiniMax-H3.

Nunchux AI's VC-Attention (arXiv:2609.15810) = V-Smooth (group value tokens
with k-means, subtract the block mean, restore it from the online softmax row
sum) + ExpCast-FP8 (write the E4M3 probability byte with one FMA instead of an
FP32 exponential).

Adapted here to MiniMax-H3's attention: 56 heads, head_dim 128, no GQA, full
self-attention over one packed sequence holding text, video and audio rows.

Quick start on a torch model::

    from vc_attention.patch import install, VCAttentionConfig
    install(VCAttentionConfig(backend="auto"))

or use the ComfyUI node.
"""

from .device import DeviceProfile, detect_profile, resolve_backend  # noqa: F401
from .grouping import GroupingConfig, build_permutation  # noqa: F401
from .h3 import H3, H3Profile, TAG, estimate_tokens, recommended_config  # noqa: F401
from .schedule import GroupSchedule, PermutationCache, StepTracker  # noqa: F401

__all__ = [
    "DeviceProfile", "detect_profile", "resolve_backend",
    "GroupingConfig", "build_permutation",
    "H3", "H3Profile", "TAG", "estimate_tokens", "recommended_config",
    "GroupSchedule", "PermutationCache", "StepTracker",
]

try:  # Triton is optional
    from .kernels.triton_attn import vc_attention_triton  # noqa: F401
    __all__.append("vc_attention_triton")
except Exception:  # pragma: no cover
    pass
