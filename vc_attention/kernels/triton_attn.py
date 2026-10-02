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

    acc <- alpha * acc + (Pq @ Vq) * (vs / P_SCALE) + r * mu * vs
    l   <- alpha * l   + r

with r = rowsum(Pq) / P_SCALE taken from the *quantised* tile, so the mean
restoration matches the product that was actually taken.

Status
------
Verified on an RTX 4090 (Triton 3.6, torch 2.10/cu130): compiles and matches
the oracle at 25.7 dB PSNR on H3-shaped random tensors. Two deviations from
the paper's kernel were needed there:

1. ``LOG2E`` / ``P_SCALE`` must be ``tl.constexpr`` instances — plain module
   globals are rejected at compile time (Triton >= 3.2).
2. The PV ``tl.dot`` runs in bf16. An fp8 MMA whose A operand was computed in
   registers (as opposed to loaded from memory) produces NaNs on sm_89 across
   every warp/stage configuration; E4M3 -> bf16 is exact so the arithmetic is
   unchanged. ``EXPCAST`` shares this call site but has not been exercised on
   datacenter hardware.

:func:`vc_attention_triton` is
guarded so any compile or launch failure falls back to the reference path.
:func:`prepare` *is* covered by ``tests/test_prepare.py`` on CPU, so the layout
the kernel reads is verified even though the kernel itself is not.
"""

from __future__ import annotations

import math
import sys
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

# Inside @triton.jit bodies these must be tl.constexpr instances (Triton >= 3.2
# rejects plain Python globals); outside the kernel the plain floats are used.
if _HAS_TRITON:
    LOG2E = tl.constexpr(1.4426950408889634)
    P_SCALE = tl.constexpr(448.0)   # E4M3 per-row scale for P; cancels against l
else:
    LOG2E = 1.4426950408889634
    P_SCALE = 448.0


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
            # bf16 MMA, not fp8: on sm_89 with Triton 3.x, tl.dot whose *A*
            # operand was computed in registers (rather than loaded) and is
            # fp8 yields NaNs for every warp/stage config (QK^T with both
            # operands from memory is fine, and torch._scaled_mm is fine).
            # E4M3 -> bf16 is exact, so the products are identical; the cost
            # is half the MMA rate, while V keeps its fp8 footprint in HBM.
            tile = tl.dot(p8.to(tl.bfloat16), v.to(tl.bfloat16))
            blk = start_n // BLOCK_N
            vs = tl.load(VS + off_bh * nb * D + blk * D + offs_d)
            mu = tl.load(MU + off_bh * nb * D + blk * D + offs_d)

            acc = acc * alpha[:, None]
            acc += tile * (vs[None, :] / P_SCALE)
            # MU stores mean/v_scale (test_prepare asserts mu * v_scale == mean),
            # so the restoration needs both factors.
            acc += r[:, None] * (mu[None, :] * vs[None, :])
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


# ---------------------------------------------------------------------------
# Fast host-side preparation (CUDA only)
# ---------------------------------------------------------------------------
#
# The portable ``prepare`` above is the oracle: exact, but built from the
# integer-op ``e4m3_encode`` (many int64 intermediates) and a 7-stage butterfly
# Hadamard. Measured on a 4090 at 16K tokens it costs ~200 ms per call — 4x the
# whole attention. The fast path below keeps the same Prepared contract but:
#
#   * derives E4M3 codes with the native ``.to(float8_e4m3fn)`` conversion
#     (tests/test_quant.py shows it is byte-identical to ``e4m3_encode``),
#   * applies the Hadamard rotation as one bf16 matmul by the explicit
#     orthonormal matrix (one pass instead of 7),
#   * gathers K/V with a broadcast ``take_along_dim`` instead of an expanded
#     (G, N, D) int64 index,
#   * demeans value blocks by broadcasting instead of ``repeat_interleave``,
#   * keeps everything in bf16 until the code/scale step.
#
# Measured: 13.8 ms @ (56, 16384, 128), 14.5x faster, PSNR -0.10 dB.
_HADAMARD_CACHE: dict = {}


def _hadamarian(d: int, device: torch.device) -> torch.Tensor:
    key = (d, str(device))
    if key not in _HADAMARD_CACHE:
        h = torch.ones((1, 1), device=device, dtype=torch.bfloat16)
        while h.shape[0] < d:
            h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
        _HADAMARD_CACHE[key] = (h / math.sqrt(d)).contiguous()
    return _HADAMARD_CACHE[key]


def _native_e4m3(x: torch.Tensor, dim: int):
    """Per-row symmetric E4M3 through the hardware cast; scale kept in fp32."""
    amp = x.detach().abs().amax(dim=dim, keepdim=True).float().clamp(min=1e-30) / 448.0
    codes = (x / amp.to(x.dtype)).to(torch.float8_e4m3fn)
    return codes, amp


def _prepare_fast(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    perm: Optional[torch.Tensor],
    block_rows: int,
    hadamard: bool,
    smooth_k: bool,
) -> Prepared:
    b, h, n, d = q.shape
    G = b * h
    dev = q.device
    n_pad = -(-n // block_rows) * block_rows
    nb = n_pad // block_rows

    qf = q.reshape(G, n, d)
    kf = k.reshape(G, n, d)
    vf = v.reshape(G, n, d)

    if perm is not None:
        p = perm.to(torch.int64).to(dev)
        if p.shape[0] == 1 and G > 1:
            p = p.expand(G, n)
        kf = torch.take_along_dim(kf, p.unsqueeze(-1), dim=1)
        vf = torch.take_along_dim(vf, p.unsqueeze(-1), dim=1)

    if hadamard:
        Hm = _hadamarian(d, dev)
        q_t = torch.matmul(qf, Hm)
        k_t = torch.matmul(kf, Hm)
    else:
        q_t, k_t = qf, kf
    if smooth_k:
        k_t = k_t - k_t.mean(dim=1, keepdim=True)

    pad = n_pad - n
    if pad:
        # Pad the bf16 tensors BEFORE quantisation: F.pad rejects fp8, and real
        # packed sequences (text + audio + video) are rarely multiples of the
        # block size — this path used to raise, get swallowed by the router,
        # and silently disable the whole node.
        q_t = torch.nn.functional.pad(q_t, (0, 0, 0, pad))
        k_t = torch.nn.functional.pad(k_t, (0, 0, 0, pad))
    q_codes, q_scale = _native_e4m3(q_t, dim=-1)
    k_codes, k_scale = _native_e4m3(k_t, dim=-1)
    q_scale = q_scale.reshape(G, n_pad)
    k_scale = k_scale.reshape(G, n_pad)

    if pad:
        vf = torch.nn.functional.pad(vf, (0, 0, 0, pad))

    vb = vf.reshape(G, nb, block_rows, d)
    mu = vb.float().mean(dim=2)                        # (G, NB, D) fp32, small
    resid = vb - mu.to(vb.dtype).unsqueeze(2)          # broadcast, no repeat
    v_scale = resid.detach().abs().amax(dim=2).float().clamp(min=1e-30) / 448.0
    v_codes = (resid / v_scale.to(vb.dtype).unsqueeze(2)).to(torch.float8_e4m3fn)
    mu_over_scale = mu / v_scale

    return Prepared(
        q=q_codes.reshape(b, h, n_pad, d).contiguous(),
        q_scale=q_scale.reshape(b, h, n_pad).contiguous().float(),
        k=k_codes.reshape(b, h, n_pad, d).contiguous(),
        k_scale=k_scale.reshape(b, h, n_pad).contiguous().float(),
        v=v_codes.reshape(b, h, n_pad, d).contiguous(),
        v_scale=v_scale.reshape(b, h, nb, d).contiguous().float(),
        mu=mu_over_scale.reshape(b, h, nb, d).contiguous().float(),
        n_pad=n_pad,
    )


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
        if q.is_cuda:
            p = _prepare_fast(q, k, v, perm, block_rows=cfg.block_n,
                              hadamard=True, smooth_k=True)
        else:
            p = prepare(q, k, v, perm, block_rows=cfg.block_n, hadamard=True)
        # The kernel indexes Out with the padded pitch (base = bh * N_PAD * D),
        # so the buffer must be N_PAD long, not n — a plain empty_like(q)
        # mis-addresses every head after the first whenever n % block_rows != 0.
        out = torch.empty((b, h, p.n_pad, d), device=q.device, dtype=q.dtype)
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
        return out[:, :, :n]
    except Exception as e:
        # Any compile/launch problem degrades to the (slow but correct) oracle.
        # Say so loudly: a silent fallback silently voids the benchmark numbers.
        print(f"[VC-Attention] fused kernel failed ({type(e).__name__}: {e}); "
              f"falling back to the slow reference path", file=sys.stderr)
        return vc_attention_reference(
            q, k, v,
            RefConfig(backend="fp8", enable_vsmooth=perm is not None,
                      block_rows=cfg.block_n),
            perm=perm, scale=scale,
        )
