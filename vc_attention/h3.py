"""MiniMax-H3 specific facts the VC-Attention node is adapted to.

Sources
-------
* ``MiniMax-AI/MiniMax-H3`` -> ``transformer/config.json`` (architecture)
* ``MiniMax-AI/MiniMax-H3`` README (VAE f16t4d24, patch 1x2x2, audio @40 Hz)
* diffusers ``MiniMaxH3Transformer3DModel`` docs (packed sequence, modality tags)
* VC-Attention paper, Sec 5.2 (the H3 workload: 1344x768, 243 frames,
  ~73.5K tokens, 8-step distilled LoRA, Bv = 128)

Why H3 needs adaptation at all
------------------------------
1. **One packed sequence, several modalities.** H3 runs a single stack of 50
   blocks over one packed 1-D sequence holding text (tag 1), conditioning image
   / video rows (tag 0), audio rows (tag 2) and the target video rows (tag 0).
   There is no cross-attention, so the permutation V-Smooth applies to K and V
   is unconditionally safe: with no mask and a single attention document,
   P'V' = PV for *any* permutation.

   The modality tag is exposed as an optional sort key (``modality_aware``) but
   is **off by default**: using it as the primary key splits clusters along
   modality boundaries and measured ~0.4 dB worse than plain label sorting on
   H3-shaped synthetic values. The magnitudes differ enough that k-means
   separates the modalities on its own.
2. **No GQA, head_dim 128, 56 heads** (56 * 128 = 7168 > hidden_size 5376, i.e.
   Q/K/V are separate projections). Bv = 128 rows matches head_dim 128, which
   is what the paper's Blackwell kernel tile assumes.
3. **MM-RoPE touches only 96 of the 128 head channels** (2 * 3 * rope_freq_dim
   = 96); the top 32 are not rotated. This does not affect V-Smooth (it acts on
   V, which carries no RoPE) nor the K channel-mean smoothing, but it is why
   the Hadamard rotation used by some low-bit kernels must be applied to the
   full 128 channels of both Q and K to stay exact.
4. **8-step distilled LoRA** is the deployed setting, so grouping's
   "first 25% of steps" is 2 steps and a 4-step reuse window means the
   permutation is computed exactly once and reused once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

__all__ = ["H3Profile", "H3", "TAG", "estimate_tokens", "recommended_config"]


@dataclass(frozen=True)
class H3Profile:
    """Values taken verbatim from MiniMax-H3 ``transformer/config.json``."""

    num_attention_heads: int = 56
    attention_head_dim: int = 128
    hidden_size: int = 5376
    num_layers: int = 50
    num_refiner_layers: int = 2
    ffn_dim: int = 14336
    in_channels: int = 24
    audio_in_channels: int = 32
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    freq_dim: int = 256
    time_embed_hidden_dim: int = 5376
    time_embed_dim: int = 2688
    rope_freq_dim: int = 16
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5

    # --- VAE / tokeniser geometry (README) ---
    vae_spatial: int = 16          # f16
    vae_temporal: int = 4          # t4
    vae_channels: int = 24         # d24
    audio_latent_rate_hz: float = 40.0
    video_fps: float = 24.0

    @property
    def qkv_dim(self) -> int:
        """56 * 128 = 7168, larger than hidden_size: Q/K/V are separate projs."""
        return self.num_attention_heads * self.attention_head_dim

    @property
    def rope_dims(self) -> int:
        """2 * 3 * rope_freq_dim channels of each head are rotated by MM-RoPE."""
        return 2 * 3 * self.rope_freq_dim

    @property
    def unrotated_dims(self) -> int:
        return self.attention_head_dim - self.rope_dims


H3 = H3Profile()


class TAG:
    """Per-row modality tags in the packed H3 sequence."""

    VIDEO = 0
    TEXT = 1
    AUDIO = 2
    PAD = -1


def estimate_tokens(
    frames: int,
    height: int,
    width: int,
    audio_seconds: float = 0.0,
    text_tokens: int = 1024,
    conditioning_rows: int = 0,
    profile: H3Profile = H3,
) -> Dict[str, int]:
    """Row counts in the packed sequence for one clip.

    Video path: VAE compresses space by 16 and time by 4 (temporal causal, so
    latent frames = ceil(frames / 4)), then patchify 1x2x2 gives an effective
    32x spatial and 4x temporal reduction.

    Note: the paper reports ~73.5K tokens for 1344x768 x 243 frames. This
    estimator returns the *target* rows only; add ``conditioning_rows`` for the
    keyframe / reference rows that the omni task also packs in.
    """
    lt = -(-frames // profile.vae_temporal)
    lh = height // (profile.vae_spatial * profile.patch_size[1])
    lw = width // (profile.vae_spatial * profile.patch_size[2])
    video = lt * lh * lw
    audio = int(round(audio_seconds * profile.audio_latent_rate_hz)) if audio_seconds else 0
    total = video + audio + text_tokens + conditioning_rows
    return {
        "video": video,
        "audio": audio,
        "text": text_tokens,
        "conditioning": conditioning_rows,
        "total": total,
        "latent_grid": (lt, lh, lw),
    }


def recommended_config(profile: H3Profile = H3, total_steps: int = 8) -> Dict[str, object]:
    """The settings the paper deploys for H3, expressed as node defaults.

    Grouping runs on the first 25% of denoising steps and the permutation is
    reused for groups of 4 steps. With the 8-step distilled LoRA that is
    ceil(8/4) = 2 grouping steps, i.e. one k-means pass and one reuse.
    """
    group_steps = max(1, -(-int(total_steps * 0.25) // 1))
    return {
        "block_rows": 128,                 # = attention_head_dim
        "heads": profile.num_attention_heads,
        "head_dim": profile.attention_head_dim,
        "group_fraction": 0.25,
        "group_steps": group_steps,
        "reuse_every": 4,
        "modality_aware": True,
        "min_tokens": 8192,                # below this, SDPA is cheaper
    }
