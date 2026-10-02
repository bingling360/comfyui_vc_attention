"""When to run grouping, and how long a permutation stays valid.

V-Smooth's k-means is not free, so the paper runs it on a subset of the
denoising steps and reuses the result. Two knobs:

  * **group_fraction** (0.25): grouping runs on the first quarter of the steps.
    Early steps carry the coarse layout; later steps refine it, and by then the
    value distribution has settled enough that one permutation still fits.
  * **reuse_every** (4): the permutation is recomputed every 4 steps and reused
    in between.

For MiniMax-H3's deployed 8-step distilled LoRA that multiplies out to
ceil(8 * 0.25) = 2 grouping steps with a 4-step reuse window: the k-means runs
exactly once, at step 0, and step 1 reuses it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch

__all__ = ["GroupSchedule", "StepTracker", "PermutationCache"]


@dataclass
class GroupSchedule:
    group_fraction: float = 0.25
    reuse_every: int = 4
    total_steps_hint: int = 8      # H3 distilled LoRA
    min_tokens: int = 8192         # below this, plain SDPA is cheaper

    def group_steps(self, total_steps: Optional[int] = None) -> int:
        total = total_steps or self.total_steps_hint
        return max(1, int(-(-round(total * self.group_fraction) // 1)))

    def should_group(self, step_index: int, total_steps: Optional[int] = None) -> bool:
        return step_index < self.group_steps(total_steps)

    def should_recompute(self, step_index: int) -> bool:
        return step_index % max(1, self.reuse_every) == 0

    def window_index(self, step_index: int) -> int:
        """Bucket used to key the permutation cache."""
        return step_index // max(1, self.reuse_every)


@dataclass
class StepTracker:
    """Counts denoiser forward passes and exposes the current step index.

    ComfyUI does not hand the attention layer a step number, so the count is
    derived from calls to the diffusion model: one call per denoising step for
    the DiT-style models this targets (CFG is a batch duplication inside the
    call, not a second call).
    """

    total_steps_hint: int = 8
    index: int = -1
    total_steps: Optional[int] = None
    _installed: bool = field(default=False, repr=False)

    def reset(self) -> None:
        self.index = -1
        self.total_steps = None

    def begin_step(self) -> int:
        self.index += 1
        return self.index

    def observe_total(self, total: int) -> None:
        if total and total > 0:
            self.total_steps = int(total)

    @property
    def current(self) -> int:
        return max(0, self.index)


class PermutationCache:
    """Stores one permutation (and the centroids that produced it) per bucket.

    ``scope='per_head'`` keeps a (B*H, N) index tensor per bucket. For H3 that
    is 56 * 73.5K * 4 B ~= 16 MB, which is why the same permutation is shared
    across all 50 layers of a step: permutations cannot change the attention
    result, so reusing one costs nothing mathematically and saves ~50x the
    memory and compute.
    """

    def __init__(self, max_entries: int = 8):
        self.max_entries = max_entries
        self._store: Dict[Tuple[int, int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self.hits = 0
        self.misses = 0

    def key(self, bucket: int, n_tokens: int, n_heads: int) -> Tuple[int, int, int]:
        return (bucket, n_tokens, n_heads)

    def get(self, bucket: int, n_tokens: int, n_heads: int):
        item = self._store.get(self.key(bucket, n_tokens, n_heads))
        if item is None:
            self.misses += 1
            return None
        self.hits += 1
        return item

    def put(self, bucket: int, n_tokens: int, n_heads: int, perm, centroids) -> None:
        if len(self._store) >= self.max_entries:
            self._store.pop(next(iter(self._store)))
        self._store[self.key(bucket, n_tokens, n_heads)] = (perm, centroids)

    def clear(self) -> None:
        self._store.clear()
        self.hits = self.misses = 0
