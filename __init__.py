"""ComfyUI custom node package: VC-Attention for MiniMax-H3.

Install by copying this folder into ``ComfyUI/custom_nodes/``.
"""

from .nodes import (  # noqa: F401
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    VCAttentionDisable,
    VCAttentionMiniMaxH3,
)

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "VCAttentionMiniMaxH3",
    "VCAttentionDisable",
]

print("[VC-Attention] MiniMax-H3 node loaded")
