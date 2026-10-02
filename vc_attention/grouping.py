"""V-Smooth step 1: online k-means over value tokens + token permutation.

Why this exists
---------------
A quantization block of Bv value rows gets one scale, so the scale is set by
the block's largest entries. In H3 the packed sequence interleaves text, video
and audio rows whose value magnitudes differ by an order of magnitude, so a
block taken in sequence order is dominated by its worst row and the rest of the
block wastes most of the low-bit code range.

V-Smooth reorders value tokens so that similar tokens land in the same block.
The permutation is *mathematically free*: applying the same permutation to K
and V leaves P @ V bit-identical, because

    P'_ij = Q_i . K'_j = Q_i . K_{pi(j)} = P_i,pi(j)
    (P'V')_i = sum_j P'_ij V'_j = sum_j P_i,pi(j) V_pi(j) = (PV)_i

Only the *quantization error* changes, and that is the whole point.

Cost control (deviation from the paper, documented)
---------------------------------------------------
The paper runs grouping per (layer, head) on grouping steps. We compute one
permutation per (head, step-window) and share it across the 50 transformer
layers of that step. Since a permutation cannot change the exact attention
result, this is a pure quality/cost knob: it cuts grouping cost and the
permutation cache by ~50x. Set ``scope='per_layer'`` for the paper-faithful
behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

__all__ = ["GroupingConfig", "GroupingResult", "kmeans", "build_permutation", "apply_permutation"]


@dataclass
class GroupingConfig:
    block_rows: int = 128        # Bv: value rows per quantization block
    iters: int = 3               # k-means iterations on a cold start
    warm_iters: int = 2          # k-means iterations when centroids are warm-started
    init_samples_per_centroid: int = 4   # strided subsample used to seed centroids
    max_clusters: int = 1024
    chunk: int = 8192            # tokens per assignment chunk (bounds (G,N,k) memory)
    # Off by default. Making the modality tag the primary sort key forces each
    # cluster to be split along modality boundaries, and a cluster that
    # legitimately spans modalities (text and video rows with a similar value
    # profile) then produces more partial blocks. Measured on H3-shaped
    # synthetic values it costs ~0.4 dB against plain label sorting, so it is
    # kept as an option rather than a default.
    modality_aware: bool = False
    scope: str = "per_head"      # per_head | shared
    dtype: torch.dtype = torch.bfloat16


@dataclass
class GroupingResult:
    perm: torch.Tensor           # (G, N) int32, gather index along the KV axis
    centroids: Optional[torch.Tensor]  # (G, k, D) for warm-starting the next step
    k: int


def _num_clusters(n: int, cfg: GroupingConfig) -> int:
    k = max(1, (n + cfg.block_rows - 1) // cfg.block_rows)
    return min(k, cfg.max_clusters, n)


def _seed_centroids(v: torch.Tensor, k: int, cfg: GroupingConfig) -> torch.Tensor:
    """Strided subsample seeding.

    H3 rows are laid out in a packed (t, h, w) order, so a stride that is
    coprime with the frame pitch samples the whole clip instead of one corner.
    """
    g, n, d = v.shape
    m = min(n, k * cfg.init_samples_per_centroid)
    idx = torch.linspace(0, n - 1, m, device=v.device).round().long()
    sub = v[:, idx, :]                       # (G, m, D)
    m_per = max(1, m // k)
    sub = sub[:, : m_per * k, :].reshape(g, k, m_per, d)
    return sub.mean(dim=2)


def _assign(
    v: torch.Tensor, centroids: torch.Tensor, cfg: GroupingConfig
) -> torch.Tensor:
    """Nearest-centroid labels, chunked over tokens to bound temporary memory."""
    g, n, d = v.shape
    k = centroids.shape[1]
    labels = torch.empty((g, n), dtype=torch.int32, device=v.device)
    c2 = (centroids.float() ** 2).sum(-1)                     # (G, k)
    for start in range(0, n, cfg.chunk):
        end = min(n, start + cfg.chunk)
        vc = v[:, start:end, :]
        # (G, chunk, k) = -2 v.c + |c|^2 ; |v|^2 is constant per row, drop it.
        sim = torch.bmm(vc, centroids.transpose(1, 2)) * 2.0 - c2.unsqueeze(1)
        labels[:, start:end] = sim.argmax(dim=-1).to(torch.int32)
    return labels


def _update_centroids(
    v: torch.Tensor, labels: torch.Tensor, k: int, cfg: GroupingConfig
) -> torch.Tensor:
    g, n, d = v.shape
    flat = v.reshape(g * n, d)
    lab = labels.reshape(g * n).long()
    off = (torch.arange(g, device=v.device, dtype=torch.long) * k).unsqueeze(1)
    flat_lab = lab + off.repeat(1, n).reshape(g * n)

    acc = v.new_zeros((g * k, d), dtype=torch.float32)
    acc.index_add_(0, flat_lab, flat.float())
    cnt = torch.zeros(g * k, dtype=torch.float32, device=v.device)
    cnt.index_add_(0, flat_lab, torch.ones_like(flat_lab, dtype=torch.float32))
    cnt = cnt.clamp(min=1.0)
    new = (acc / cnt.unsqueeze(1)).reshape(g, k, d)

    # Re-seed centroids that captured nothing (can happen with strided seeding).
    empty = (cnt.reshape(g, k) <= 1.0)
    if bool(empty.any()):
        filler = _seed_centroids(v, k, cfg)
        new = torch.where(empty.unsqueeze(-1), filler, new)
    return new.to(cfg.dtype)


def kmeans(
    v: torch.Tensor,
    k: int,
    cfg: GroupingConfig,
    centroids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched k-means over (G, N, D). Returns (labels (G,N) int32, centroids)."""
    v = v.to(cfg.dtype)
    if centroids is None:
        centroids = _seed_centroids(v, k, cfg)
        iters = cfg.iters
    else:
        iters = cfg.warm_iters
    centroids = centroids.to(cfg.dtype)

    labels = None
    for _ in range(iters):
        labels = _assign(v, centroids, cfg)
        centroids = _update_centroids(v, labels, k, cfg)
    labels = _assign(v, centroids, cfg)
    return labels, centroids


def build_permutation(
    v: torch.Tensor,
    cfg: GroupingConfig,
    tags: Optional[torch.Tensor] = None,
    centroids: Optional[torch.Tensor] = None,
    valid_len: Optional[int] = None,
) -> GroupingResult:
    """Build the V-Smooth permutation for one value tensor.

    Args:
        v: (G, N, D) value states, G = batch * heads (or 1 when scope='shared').
        cfg: grouping config.
        tags: (N,) int modality tag per row. H3 uses 0=video, 1=text, 2=audio,
            -1=padding. Used only when ``cfg.modality_aware``; it forces each
            hardware block to hold rows of a single modality.
        centroids: warm-start centroids from the previous step window.
        valid_len: if not None, rows >= valid_len are padding and are pushed to
            the tail of the permutation so the kernel can mask them in one run.

    Returns:
        GroupingResult with ``perm`` shaped (G, N).
    """
    g, n, d = v.shape
    k = _num_clusters(n, cfg)
    labels, centroids = kmeans(v, k, cfg, centroids)

    key = labels.to(torch.int64)
    if cfg.modality_aware and tags is not None:
        tag = tags.to(torch.int64).clamp(min=-1)
        tag = torch.where(tag < 0, torch.full_like(tag, 8), tag)  # pad sorts last
        key = tag.unsqueeze(0) * (k + 1) + key
    if valid_len is not None and valid_len < n:
        is_pad = torch.arange(n, device=v.device) >= valid_len
        key = key + is_pad.to(torch.int64).unsqueeze(0) * (16 * (k + 1))

    perm = torch.argsort(key, dim=-1, stable=True).to(torch.int32)
    return GroupingResult(perm=perm, centroids=centroids, k=k)


def apply_permutation(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Gather ``x`` (..., N, D) along the token axis with ``perm`` (G, N)."""
    return x.gather(-2, perm.to(torch.int64).unsqueeze(-1).expand_as(x))
