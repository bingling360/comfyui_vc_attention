"""Fused VC-Attention kernel (Triton).

This is the fast path. ``kernels.reference`` is the oracle that defines what
these codes mean; this module does the same arithmetic fused, with the two
matrix products on low-bit tensor cores.

Layout produced by :func:`prepare`
----------------------------------
  Q   (B, H, NP, D) float8_e4m3fn, per-token scale QS  (B, H, NP)
  K   (B, H, NP, D) float8_e4m3fn, per-token scale KS  (B, H, NP)   -- permuted
  V   (B, H, NP, D) float8_e4m3fn residual codes                    -- permuted
  VS  (B, H, NB, D) per-(value block x channel) scale
  MU  (B, H, NB, D) block mean, already divided by VS

where NP = N padded up to a multiple of BLOCK_N and NB = NP // BLOCK_N. Q, K,
QS and KS are padded too: every tensor the kernel indexes must share the same
pitched layout, or the (b, h) base offsets drift apart. The padding rows are
masked out of the scores and never written to the output, so they do not change
the result -- they only make the KV loop land on whole value blocks.

The kernel walks KV tiles of exactly BLOCK_N == block_rows, so each tile owns
one mu and one scale vector:

    acc <- alpha * acc + (Pq @ Vq) * (vs / P_SCALE) + r * mu
    l   <- alpha * l   + r

with r = rowsum(Pq) / P_SCALE taken from the *quantised* tile, so the mean
restoration matches the product that was actually taken.

Status
------
Written against the Triton FA2 shape (``tl.dot`` with fp8 operands, online
softmax in fp32). It needs a GPU to compile and has not been executed in the
environment this package was authored in; :func:`vc_attention_triton` is
guarded so any compile or launch failure falls back to the reference path.
:func:`prepare` *is* covered by ``tests/test_prepare.py`` on CPU, so the layout
the kernel reads is verified even though the kernel itself is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch

try:  # Triton is optional; the package still imports (and falls back) without it.
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    _HAS_TRITON = False

from .. import quant as Q
from ..hadamard import fwht

__all__ = ["TritonConfig", "Prepared", "has_triton", "prepare", "vc_attention_triton"]

LOG2E = 1.4426950408889634
P_SCALE = 448.0          # E4M3 per-row scale for P; cancels against l


def has_triton() -> bool:
    return _HAS_TRITON


@dataclass
class TritonConfig:
    block_m: int = 128
    block_n: int = 128      # must equal block_rows: one value block per KV tile
    num_warps: int = 8
    num_stages: int = 3
    enable_expcast: bool = False
    expcast_beta: float = -0.35


class Prepared(NamedTuple):
    """Everything the kernel needs. ``n_pad`` is the padded token count."""

    q: torch.Tensor         # (B, H, NP, D) float8_e4m3fn
    q_scale: torch.Tensor   # (B, H, NP)
    k: torch.Tensor         # (B, H, NP, D) float8_e4m3fn
    k_scale: torch.Tensor   # (B, H, NP)
    v: torch.Tensor         # (B, H, NP, D) float8_e4m3fn residual codes
    v_scale: torch.Tensor   # (B, H, NB, D)
    mu: torch.Tensor        # (B, H, NB, D) block mean / v_scale
    n_pad: int


if _HAS_TRITON:

    @triton.jit
    def _vc_attn_fwd(
        Q, QS, K, KS, V, VS, MU, Out,
        sm_scale,
        N,                          # true token count (masks)
        N_PAD,                      # padded token count (loop bound, strides)
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        EXPCAST: tl.constexpr,
        BETA: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_bh = tl.program_id(1)

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        nb = N_PAD // BLOCK_N

        base = off_bh.to(tl.int64) * N_PAD * D
        q = tl.load(Q + base + offs_m[:, None] * D + offs_d[None, :],
                    mask=offs_m[:, None] < N, other=0.0)
        qs = tl.load(QS + off_bh * N_PAD + offs_m, mask=offs_m < N, other=0.0)

        m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

        for start_n in range(0, N_PAD, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            nmask = offs_n < N

            k = tl.load(K + base + offs_n[:, None] * D + offs_d[None, :],
                        mask=nmask[:, None], other=0.0)
            ks = tl.load(KS + off_bh * N_PAD + offs_n, mask=nmask, other=0.0)
            v = tl.load(V + base + offs_n[:, None] * D + offs_d[None, :],
                        mask=nmask[:, None], other=0.0)

            # ---- scores: fp8 x fp8 -> fp32, then the per-token scales -------
            s = tl.dot(q, tl.trans(k))
            s = s * (qs[:, None] * ks[None, :]) * sm_scale
            s = tl.where(nmask[None, :], s, -1.0e30)

            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_new = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.math.exp2((m_i - m_new) * LOG2E)
            alpha = tl.where(m_i == float("-inf"), 0.0, alpha)

            if EXPCAST:
                # Write the E4M3 byte directly: one FMA, one round, one clip.
                u = (s - m_new[:, None]) * LOG2E + 8.0
                code = u * 8.0 + (56.0 + BETA)
                code = tl.maximum(tl.minimum(code, 120.0), 0.0)
                code_i = code.to(tl.int32).to(tl.uint8)
                p8 = code_i.to(tl.float8e4nv, bitcast=True)
            else:
                p8 = (tl.math.exp2((s - m_new[:, None]) * LOG2E) * P_SCALE).to(tl.float8e4nv)

            r = tl.sum(p8.to(tl.float32), 1) / P_SCALE

            # ---- PV with the per-block scale, then restore the block mean ---
            tile = tl.dot(p8, v)
            blk = start_n // BLOCK_N
            vs = tl.load(VS + off_bh * nb * D + blk * D + offs_d)
            mu = tl.load(MU + off_bh * nb * D + blk * D + offs_d)

            acc = acc * alpha[:, None]
            acc += tile * (vs[None, :] / P_SCALE)
            acc += r[:, None] * mu[None, :]
            l_i = l_i * alpha + r
            m_i = m_new

        acc = acc / l_i[:, None]
        tl.store(Out + base + offs_m[:, None] * D + offs_d[None, :],
                 acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N)


def prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    perm: Optional[torch.Tensor] = None,
    block_rows: int = 128,
    hadamard: bool = True,
    smooth_k: bool = True,
) -> Prepared:
    """Quantise and permute Q/K/V into the layout the kernel expects.

    Args:
        q, k, v: (B, H, N, D) bf16/fp32.
        perm: (G, N) int32 permutation, G = B*H. Applied to K and V together.
        block_rows: rows per value block.

    Returns:
        :class:`Prepared`. Reconstruct with :func:`restore` for tests.
    """
    b, h, n, d = q.shape
    G = b * h
    dev = q.device
    n_pad = -(-n // block_rows) * block_rows
    nb = n_pad // block_rows

    qf = q.reshape(G, n, d).float()
    kf = k.reshape(G, n, d).float()
    vf = v.reshape(G, n, d).float()

    if perm is not None:
        p = perm.to(torch.int64).to(dev)
        if p.shape[0] == 1 and G > 1:
            p = p.expand(G, n)
        kf = kf.gather(1, p.unsqueeze(-1).expand_as(kf))
        vf = vf.gather(1, p.unsqueeze(-1).expand_as(vf))

    # Q/K: same orthonormal rotation on both, then per-token E4M3.
    if hadamard:
        q_t, k_t = fwht(qf), fwht(kf)
    else:
        q_t, k_t = qf, kf
    if smooth_k:
        k_t = k_t - k_t.mean(dim=1, keepdim=True)
    q_codes, q_scale = Q.quantize_e4m3(q_t, dim=-1)
    k_codes, k_scale = Q.quantize_e4m3(k_t, dim=-1)
    # Drop the trailing singleton so the scale is (G, N): the padding below and
    # the (B, H, NP) reshape both assume it.
    q_scale = q_scale.reshape(G, n)
    k_scale = k_scale.reshape(G, n)

    # V: block demean, per-(block x channel) scale, mean stored pre-divided.
    if n_pad != n:
        pad = n_pad - n
        vf = torch.nn.functional.pad(vf, (0, 0, 0, pad))
        q_codes = torch.nn.functional.pad(q_codes, (0, 0, 0, pad))
        k_codes = torch.nn.functional.pad(k_codes, (0, 0, 0, pad))
        q_scale = torch.nn.functional.pad(q_scale, (0, pad))
        k_scale = torch.nn.functional.pad(k_scale, (0, pad))

    vb = vf.reshape(G, nb, block_rows, d)
    mu = vb.mean(dim=2)
    resid = vf - mu.repeat_interleave(block_rows, dim=1)
    v_codes, v_scale = Q.quantize_e4m3_blocks(resid, block_rows)
    mu_over_scale = mu / v_scale

    to8 = lambda t: t.reshape(b, h, n_pad, d).view(torch.float8_e4m3fn).contiguous()
    return Prepared(
        q=to8(q_codes),
        q_scale=q_scale.reshape(b, h, n_pad).contiguous().float(),
        k=to8(k_codes),
        k_scale=k_scale.reshape(b, h, n_pad).contiguous().float(),
        v=to8(v_codes),
        v_scale=v_scale.reshape(b, h, nb, d).contiguous().float(),
        mu=mu_over_scale.reshape(b, h, nb, d).contiguous().float(),
        n_pad=n_pad,
    )


def restore(p: Prepared, block_rows: int = 128) -> tuple:
    """Inverse of :func:`prepare` for tests: back to value/score units."""
    b, h, np_, d = p.q.shape
    G = b * h
    q = Q.dequantize_e4m3(p.q.view(torch.uint8).reshape(G, np_, d),
                          p.q_scale.reshape(G, np_, 1))
    k = Q.dequantize_e4m3(p.k.view(torch.uint8).reshape(G, np_, d),
                          p.k_scale.reshape(G, np_, 1))
    v_resid = Q.dequantize_e4m3_blocks(
        p.v.view(torch.uint8).reshape(G, np_, d),
        p.v_scale.reshape(G, -1, d),
        block_rows,
    )
    v = v_resid + p.mu.reshape(G, -1, d).repeat_interleave(block_rows, dim=1) * p.v_scale.reshape(
        G, -1, d
    ).repeat_interleave(block_rows, dim=1)
    return q, k, v


def vc_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    perm: Optional[torch.Tensor] = None,
    cfg: Optional[TritonConfig] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Fused VC-Attention. Falls back to the reference path on any failure."""
    from .reference import RefConfig, vc_attention_reference

    cfg = cfg or TritonConfig()
    b, h, n, d = q.shape
    if scale is None:
        scale = d ** -0.5

    if not _HAS_TRITON or q.device.type != "cuda":
        return vc_attention_reference(
            q, k, v,
            RefConfig(backend="fp8", enable_vsmooth=perm is not None,
                      block_rows=cfg.block_n, enable_expcast=False),
            perm=perm, scale=scale,
        )

    try:
        p = prepare(q, k, v, perm, block_rows=cfg.block_n, hadamard=True)
        out = torch.empty_like(q)
        grid = (triton.cdiv(p.n_pad, cfg.block_m), b * h)
        _vc_attn_fwd[grid](
            p.q, p.q_scale, p.k, p.k_scale, p.v, p.v_scale, p.mu, out,
            scale, n, p.n_pad, d,
            BLOCK_M=cfg.block_m,
            BLOCK_N=cfg.block_n,
            EXPCAST=cfg.enable_expcast,
            BETA=cfg.expcast_beta,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )
        return out
    except Exception:
        # Any compile/launch problem degrades to the (slow but correct) oracle.
        return vc_attention_reference(
            q, k, v,
            RefConfig(backend="fp8", enable_vsmooth=perm is not None,
                      block_rows=cfg.block_n),
            perm=perm, scale=scale,
        )
