"""ComfyUI nodes for VC-Attention on MiniMax-H3.

Two nodes:

``VC Attention (MiniMax-H3)``
    Patches the model's attention. Everything is set from the H3 defaults
    (56 heads, head_dim 128, 8-step distilled LoRA), and the preset is chosen
    from the GPU: NVFP4 on workstation Blackwell, FP8 everywhere else.

``VC Attention Disable``
    Restores the original attention.

The patch is process-global (see ``vc_attention.patch``), so installing it
affects every sampling run in the same process until it is disabled.
"""

from __future__ import annotations

import torch

from .vc_attention.device import detect_profile, resolve_backend
from .vc_attention.h3 import recommended_config
from .vc_attention.patch import (
    VCAttentionConfig,
    get_runtime,
    install,
    kernel_status,
    uninstall,
)

CATEGORY = "vc_attention"


class VCAttentionMiniMaxH3:
    """Drop-in low-bit attention for MiniMax-H3, following Nunchux VC-Attention."""

    @classmethod
    def INPUT_TYPES(cls):
        h3 = recommended_config(total_steps=8)
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model to patch."}),
                "backend": (["auto", "nvfp4", "fp8", "int4", "reference"],
                            {"default": "auto",
                             "tooltip": "auto picks NVFP4 on RTX 50xx / PRO 6000, "
                                        "FP8 on other fp8-capable cards. 'reference' is the "
                                        "slow but exact PyTorch path, useful for checking."}),
                "enable_vsmooth": ("BOOLEAN",
                                   {"default": True,
                                    "tooltip": "V-Smooth: k-means grouping + block demean. "
                                               "This is the accuracy half of VC-Attention."}),
                "enable_expcast": ("BOOLEAN",
                                   {"default": False,
                                    "tooltip": "ExpCast-FP8: replace the FP32 exponential with "
                                               "one FMA. 8-bit datacenter path only; off on "
                                               "workstation cards, where softmax is not the "
                                               "bottleneck and NVFP4 has no affine log map."}),
                "total_steps": ("INT", {"default": 8, "min": 1, "max": 200,
                                        "tooltip": "Denoising steps. 8 for the H3 distilled "
                                                   "LoRA. Grouping runs on the first 25%."}),
                "block_rows": ("INT", {"default": h3["block_rows"], "min": 16, "max": 256,
                                       "step": 16,
                                       "tooltip": "Value rows per quantisation block. 128 "
                                                  "matches H3's head_dim."}),
                "min_tokens": ("INT", {"default": 8192, "min": 512, "max": 1 << 20, "step": 512,
                                       "tooltip": "Skip VC-Attention below this sequence "
                                                  "length; SDPA is cheaper for short ones."}),
            },
            "optional": {
                "group_fraction": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0,
                                             "step": 0.05,
                                             "tooltip": "Fraction of steps that run k-means."}),
                "reuse_every": ("INT", {"default": 4, "min": 1, "max": 64,
                                        "tooltip": "Reuse one permutation for this many steps."}),
                "kmeans_iters": ("INT", {"default": 3, "min": 1, "max": 16,
                                         "tooltip": "Lloyd iterations on a cold start."}),
                "modality_aware": ("BOOLEAN",
                                   {"default": False,
                                    "tooltip": "Use the H3 modality tag as the primary sort "
                                               "key. Measured slightly worse than plain label "
                                               "sorting; left off by default."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Nunchux VC-Attention adapted to MiniMax-H3: V-Smooth value grouping plus "
        "low-bit QK/PV. Training-free; no checkpoint changes."
    )

    def apply(
        self,
        model,
        backend="auto",
        enable_vsmooth=True,
        enable_expcast=False,
        total_steps=8,
        block_rows=128,
        min_tokens=8192,
        group_fraction=0.25,
        reuse_every=4,
        kmeans_iters=3,
        modality_aware=False,
    ):
        profile = detect_profile()
        resolved = resolve_backend(backend, profile)

        cfg = VCAttentionConfig(
            enabled=True,
            backend=backend,
            enable_vsmooth=enable_vsmooth,
            # ExpCast needs the 8-bit path; NVFP4 codes have no affine log map.
            enable_expcast=bool(enable_expcast and resolved == "fp8"),
            block_rows=int(block_rows),
            min_tokens=int(min_tokens),
            kmeans_iters=int(kmeans_iters),
            modality_aware=bool(modality_aware),
            group_fraction=float(group_fraction),
            reuse_every=int(reuse_every),
            total_steps_hint=int(total_steps),
        )
        runtime = install(cfg, model=model)
        runtime.reset()
        status = kernel_status()

        print(
            f"[VC-Attention] {profile.describe()}\n"
            f"[VC-Attention] backend={resolved} vsmooth={cfg.enable_vsmooth} "
            f"expcast={cfg.enable_expcast} block_rows={cfg.block_rows} "
            f"grouping on steps 0..{max(0, runtime.schedule.group_steps(total_steps) - 1)} "
            f"of {total_steps}\n"
            f"[VC-Attention] {status}"
        )
        if "inactive" in status:
            print(
                "[VC-Attention] Nothing will be accelerated; sampling runs at native "
                "speed. See README 'Install' for the Triton step."
            )
        return (model,)


class VCAttentionDisable:
    """Restore the original attention implementation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = CATEGORY

    def apply(self, model):
        uninstall()
        print("[VC-Attention] restored the original attention")
        return (model,)


NODE_CLASS_MAPPINGS = {
    "VCAttentionMiniMaxH3": VCAttentionMiniMaxH3,
    "VCAttentionDisable": VCAttentionDisable,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VCAttentionMiniMaxH3": "VC Attention (MiniMax-H3)",
    "VCAttentionDisable": "VC Attention Disable",
}
