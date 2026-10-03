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
Verified on an RTX 4090 (Triton 3.6) and an RTX 5090 (sm_120, Triton 3.8):
compiles and matches the oracle at 25.7 dB PSNR on H3-shaped random tensors.
Deviations from the paper's kernel:

1. ``LOG2E`` / ``P_SCALE`` must be ``tl.constexpr`` instances — plain module
   globals are rejected at compile time (Triton >= 3.2).
2. On **sm_89** the PV ``tl.dot`` runs in bf16: an fp8 MMA whose A operand was
   computed in registers produces wrong values on sm_89 across every
   warp/stage configuration (Triton's fp32->e4m3 register conversion is
   broken). E4M3 -> bf16 is exact so the arithmetic is unchanged.
   On **sm_120** that bug is absent (probe19), so ``PV_FP8`` defaults on there
   and the PV MMA runs at 2x the bf16 rate at identical precision.
3. ``QK_FP4`` (NVFP4 Q/K via ``tl.dot_scaled``) is implemented and correct, but
   opt-in: on sm_120 it is a net loss for attention (the QK reduction is only
   D=128, while host-side NVFP4 quantization costs ~23 ms/layer and QK error
   rises 3.6% -> 13.4%). See ``tests/bench_pv.py``.

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
    # Tuned on an RTX 5090 (sm_120) at 16384 tokens / 56 heads / D=128 with the
    # fp8 PV path: BM=64, 4 warps, 2 stages -> 24.4 ms vs 31.0 ms for the old
    # 128/8/3 default (bf16 SDPA: 35.3 ms). BLOCK_M=256 spills catastrophically.
    block_m: int = 64
    block_n: int = 128      # must equal block_rows: one value block per KV tile
    num_warps: int = 4
    num_stages: int = 2
    enable_expcast: bool = False
    expcast_beta: float = -0.35
    # PV in fp8 (E4M3 x E4M3 -> fp32) instead of bf16. Doubles the PV MMA rate,
    # but only where the register-operand fp8 dot is correct: sm_120 (verified
    # here, max|diff|=0) and Hopper/datacenter Blackwell. On sm_89 Triton's
    # fp32->e4m3 register conversion is broken (see _probe14/15/16), so this
    # stays off there. ``None`` -> decide from the device capability.
    pv_fp8: Optional[bool] = None
    # QK^T in NVFP4 (e2m1 + per-16 e4m3 microscale) via tl.dot_scaled instead of
    # per-token E4M3. Both operands come from memory, so this is the one place
    # the FP4 tensor cores are reachable in attention. Measured on sm_120:
    # 530 TFLOP/s fp4 vs 277 fp8 vs 139 bf16. But NVFP4 has a 2-bit mantissa:
    # QK^T rel-err 0.134 vs 0.036 for fp8, and attention PSNR 44.9 dB vs 56.8
    # (probe20). Hence opt-in, not default.
    qk_fp4: bool = False
    # Sol-Attn-style block sparsity fused into the kernel. See
    # PLAN_sparse_fusion.md. `tau` is the routing threshold in std-devs above
    # the mean block proxy; larger = fewer blocks kept = faster, lower quality.
    sparse: bool = False
    tau: float = 1.3
    local_blocks: int = 1     # |q_block - kv_block| <= this stays exact
    sink_blocks: int = 0      # first N KV blocks stay exact (H3 conditioning)
    # KV blocks routed together. One tensor-core matmul yields the per-row
    # proxy for the whole group, and one small matmul folds every skipped
    # block's mean-value column in -- that is what keeps the skipped path off
    # the fp32 elementwise units (see PLAN_sparse_fusion.md section 4). tl.dot
    # needs every dim >= 16, so this cannot go below 16.
    group: int = 16


def _default_pv_fp8() -> bool:
    """Register-fp8 PV is safe on sm_90 and sm_10x/sm_120, not on sm_89."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(0)
    return (major, minor) >= (9, 0) and (major, minor) != (8, 9)


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
    # NVFP4 Q/K, only populated when prepare(qk_fp4=True). Q packs along D as
    # (B,H,NP,D//2); K must be K-first for dot_scaled, i.e. (B,H,D//2,NP).
    q4: Optional[torch.Tensor] = None       # (B, H, NP, D//2) uint8
    q4_scale: Optional[torch.Tensor] = None  # (B, H, NP, D//16) uint8 e4m3
    k4: Optional[torch.Tensor] = None       # (B, H, D//2, NP) uint8
    k4_scale: Optional[torch.Tensor] = None  # (B, H, NP, D//16) uint8 e4m3
    # Block-routing tensors, only populated when prepare(sparse=True).
    # Sol-Attn's rule: proxy = <q, mean_k(block)> * scale, threshold =
    # mean + tau*std over KV blocks; a block is kept if its proxy exceeds the
    # threshold, or it is local / a sink. Skipped blocks are approximated from
    # their proxy rather than dropped. See PLAN_sparse_fusion.md.
    k_mean: Optional[torch.Tensor] = None    # (B, H, NB, D) fp32
    v_mean: Optional[torch.Tensor] = None    # (B, H, NB, D) fp32
    thresh: Optional[torch.Tensor] = None    # (B, H, NQB) fp32
    n_qblk: int = 0                          # NQB = NP // q_block


def nvfp4_pack(x: torch.Tensor, group: int = 16):
    """(..., D) float -> (packed (..., D//2) uint8, e4m3 micro codes (..., D//16)).

    Mirrors what ``tl.dot_scaled`` expects: low nibble = even element, e4m3
    microscale per ``group`` elements along the last dim. Verified on sm_120
    against the exact dequant (probe19, rel err 0.0).

    Speed note: PyTorch 2.10 has no float->float4_e2m1fn_x2 cast, so the e2m1
    codes are built from 7 comparisons (RNE ties handled explicitly to match
    quant.fp4_encode). The e4m3 microscale uses the *native* cast, which is
    ~30x faster than the portable integer encoder. Even so this costs ~6 ms per
    (16K x 128) tensor -- against ~0.03 ms for the fp8 path's single fused cast,
    which is why NVFP4 Q/K is opt-in (see TritonConfig.qk_fp4).
    """
    d = x.shape[-1]
    xb = x.reshape(*x.shape[:-1], d // group, group)
    amax = xb.detach().abs().amax(-1, keepdim=True).clamp(min=1e-30)
    micro_fp8 = (amax / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn)
    micro = micro_fp8.view(torch.uint8).squeeze(-1)          # e4m3 codes
    mv = micro_fp8.float().clamp(min=1e-30)
    a = (xb / mv).abs().clamp(max=6.0)
    code = (a > 0.25).to(torch.uint8)
    for b in (0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        code = code + (a > b).to(torch.uint8)
    # RNE: midpoints between code j and j+1 round to the even code, i.e. the odd
    # breakpoints (0.75, 1.75, 3.5) round *up* on an exact tie.
    code = code + ((a == 0.75) | (a == 1.75) | (a == 3.5)).to(torch.uint8)
    sign = (xb < 0).to(torch.uint8) << 3
    codes = (code | sign).reshape(*x.shape[:-1], d)
    packed = (codes[..., 0::2] & 0xF) | ((codes[..., 1::2] & 0xF) << 4)
    return packed.contiguous(), micro.contiguous()


if _HAS_TRITON:

    @triton.jit
    def _vc_attn_fwd(
        Q, QS, K, KS, V, VS, MU, Out,
        Q4, Q4S, K4, K4S,           # NVFP4 Q/K (only read when QK_FP4)
        sm_scale,
        N,                          # true token count (masks)
        N_PAD,                      # padded token count (loop bound, strides)
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        EXPCAST: tl.constexpr,
        BETA: tl.constexpr,
        PV_FP8: tl.constexpr,
        QK_FP4: tl.constexpr,
        TILE_SKIP: tl.constexpr = 1,
    ):
        start_m = tl.program_id(0)
        off_bh = tl.program_id(1)

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        nb = N_PAD // BLOCK_N

        base = off_bh.to(tl.int64) * N_PAD * D
        if QK_FP4:
            # NVFP4 Q/K: packed along D. Q is (NP, D//2) token-major; K is stored
            # K-first as (D//2, NP) because dot_scaled wants the reduction dim on
            # axis 0 of the rhs. The microscales carry the per-16-group scale, so
            # no per-token scale multiply is needed afterwards.
            offs_dp = tl.arange(0, D // 2)
            offs_dg = tl.arange(0, D // 16)
            qpk = tl.load(Q4 + off_bh.to(tl.int64) * N_PAD * (D // 2)
                          + offs_m[:, None] * (D // 2) + offs_dp[None, :],
                          mask=offs_m[:, None] < N, other=0)
            qsc = tl.load(Q4S + off_bh.to(tl.int64) * N_PAD * (D // 16)
                          + offs_m[:, None] * (D // 16) + offs_dg[None, :],
                          mask=offs_m[:, None] < N, other=0).to(tl.float8e4nv, bitcast=True)
        else:
            q = tl.load(Q + base + offs_m[:, None] * D + offs_d[None, :],
                        mask=offs_m[:, None] < N, other=0.0)
            qs = tl.load(QS + off_bh * N_PAD + offs_m, mask=offs_m < N, other=0.0)

        m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

        # TILE_SKIP > 1 walks only every TILE_SKIP-th KV tile. It produces a
        # WRONG result -- it exists to measure how kernel time scales with the
        # number of tiles, i.e. the ceiling for Sol-Attn-style block sparsity
        # (tests/_probe30_tilescale.py). Block sparsity itself lives in the
        # separate _vc_attn_fwd_sparse kernel below.
        for start_n in range(0, N_PAD, BLOCK_N * TILE_SKIP):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            nmask = offs_n < N
            blk = start_n // BLOCK_N

            v = tl.load(V + base + offs_n[:, None] * D + offs_d[None, :],
                        mask=nmask[:, None], other=0.0)

            # ---- scores -----------------------------------------------------
            if QK_FP4:
                kpk = tl.load(K4 + off_bh.to(tl.int64) * (D // 2) * N_PAD
                              + offs_dp[:, None] * N_PAD + offs_n[None, :],
                              mask=nmask[None, :], other=0)
                ksc = tl.load(K4S + off_bh.to(tl.int64) * N_PAD * (D // 16)
                              + offs_n[:, None] * (D // 16) + offs_dg[None, :],
                              mask=nmask[:, None], other=0).to(tl.float8e4nv, bitcast=True)
                s = tl.dot_scaled(qpk, qsc, "e2m1", kpk, ksc, "e2m1", out_dtype=tl.float32)
                s = s * sm_scale
            else:
                # fp8 x fp8 -> fp32, then the per-token scales
                k = tl.load(K + base + offs_n[:, None] * D + offs_d[None, :],
                            mask=nmask[:, None], other=0.0)
                ks = tl.load(KS + off_bh * N_PAD + offs_n, mask=nmask, other=0.0)
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

            # ---- PV with the per-block scale, then restore the block mean
            # fp8 PV (E4M3 x E4M3 -> fp32) on parts where the register-
            # operand fp8 dot is correct: that doubles the MMA rate vs bf16
            # at identical precision, since both operands are already E4M3.
            # On sm_89 the register fp32->e4m3 conversion is broken
            # (probe14/15/16), so there the bf16 dot is used (E4M3 -> bf16
            # is exact, so the arithmetic is unchanged, at half the rate).
            if PV_FP8:
                tile = tl.dot(p8, v)
            else:
                tile = tl.dot(p8.to(tl.bfloat16), v.to(tl.bfloat16))
            vs = tl.load(VS + off_bh * nb * D + blk * D + offs_d)
            mu = tl.load(MU + off_bh * nb * D + blk * D + offs_d)

            acc = acc * alpha[:, None]
            acc += tile * (vs[None, :] / P_SCALE)
            # MU stores mean/v_scale (test_prepare asserts mu * v_scale ==
            # mean), so the restoration needs both factors.
            acc += r[:, None] * (mu[None, :] * vs[None, :])
            l_i = l_i * alpha + r
            m_i = m_new

        acc = acc / l_i[:, None]
        tl.store(Out + base + offs_m[:, None] * D + offs_d[None, :],
                 acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N)

    @triton.jit
    def _vc_exact_tile(
        q, qs, off_bh, start_n, base, n_pad, nb,
        K, KS, V, VS, MU,
        acc, l_i, m_i,
        N, sm_scale,
        D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        EXPCAST: tl.constexpr,
        BETA: tl.constexpr,
        PV_FP8: tl.constexpr,
    ):
        """One exact (kept) KV tile, quantised fp8 QK + PV.

        Byte-for-byte the body of the dense loop's kept branch; factored out so
        the grouped sparse kernel and the dense kernel cannot drift.
        """
        offs_d = tl.arange(0, D)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        nmask = offs_n < N
        blk = start_n // BLOCK_N

        v = tl.load(V + base + offs_n[:, None] * D + offs_d[None, :],
                    mask=nmask[:, None], other=0.0)
        k = tl.load(K + base + offs_n[:, None] * D + offs_d[None, :],
                    mask=nmask[:, None], other=0.0)
        ks = tl.load(KS + off_bh * n_pad + offs_n, mask=nmask, other=0.0)
        s = tl.dot(q, tl.trans(k))
        s = s * (qs[:, None] * ks[None, :]) * sm_scale
        s = tl.where(nmask[None, :], s, -1.0e30)

        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_new = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.math.exp2((m_i - m_new) * LOG2E)
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)

        if EXPCAST:
            u = (s - m_new[:, None]) * LOG2E + 8.0
            code = u * 8.0 + (56.0 + BETA)
            code = tl.maximum(tl.minimum(code, 120.0), 0.0)
            code_i = code.to(tl.int32).to(tl.uint8)
            p8 = code_i.to(tl.float8e4nv, bitcast=True)
        else:
            p8 = (tl.math.exp2((s - m_new[:, None]) * LOG2E) * P_SCALE).to(tl.float8e4nv)
        r = tl.sum(p8.to(tl.float32), 1) / P_SCALE

        if PV_FP8:
            tile = tl.dot(p8, v)
        else:
            tile = tl.dot(p8.to(tl.bfloat16), v.to(tl.bfloat16))
        vs = tl.load(VS + off_bh * nb * D + blk * D + offs_d)
        mu = tl.load(MU + off_bh * nb * D + blk * D + offs_d)

        acc = acc * alpha[:, None]
        acc += tile * (vs[None, :] / P_SCALE)
        acc += r[:, None] * (mu[None, :] * vs[None, :])
        l_i = l_i * alpha + r
        m_i = m_new
        return acc, l_i, m_i

    @triton.jit
    def _vc_attn_fwd_sparse(
        Q, QS, K, KS, V, VS, MU, Out,
        KMEAN, VMEAN, THRESH,
        sm_scale,
        N,
        N_PAD,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        EXPCAST: tl.constexpr,
        BETA: tl.constexpr,
        PV_FP8: tl.constexpr,
        LOCAL: tl.constexpr = 1,
        SINK: tl.constexpr = 0,
        GROUP: tl.constexpr = 16,
    ):
        """VC-Attention with Sol-Attn-style block routing fused in.

        Structure follows ``sol_kernel/fwd.py``: KV blocks are walked GROUP at a
        time so that

          * the routing proxy is ONE tensor-core matmul ``q @ kc^T`` producing a
            per-row score per block (never a per-tile fp32 elementwise reduce --
            that was the 27% overhead the first attempt paid), and
          * every skipped block is folded in with ONE small matmul
            ``p_approx @ vc`` over its block-mean value, instead of a per-tile
            ``(BLOCK_M, BLOCK_N)`` fp32 outer product (which cost ~45% of the
            dense kernel and made the skip pointless).

        Blocks below the threshold are *approximated*, not dropped: their proxy
        score is reused as the score of the block, so the softmax normaliser
        stays right. Kept blocks run the exact quantised tile.

        The all-kept case (tau very negative) is bit-identical to the dense
        kernel: the approximate update is skipped when no block is approximate,
        and exact blocks are visited in increasing order, so the online-softmax
        sequence is unchanged (tests/_probe33_sparse_isolate.py).
        """
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

        thr = tl.load(THRESH + off_bh * (N_PAD // BLOCK_M) + start_m)
        # The query block's index in KV-block units (BLOCK_M may be < BLOCK_N),
        # so the local window is compared in one consistent unit.
        qb_kv = (start_m * BLOCK_M) // BLOCK_N

        for group_start in range(0, N_PAD, BLOCK_N * GROUP):
            blk0 = group_start // BLOCK_N
            offs_blk = blk0 + tl.arange(0, GROUP)
            valid = offs_blk < nb

            # ---- routing proxy: one matmul for the whole group ---------------
            kc = tl.load(KMEAN + off_bh.to(tl.int64) * nb * D
                         + offs_blk[:, None] * D + offs_d[None, :],
                         mask=valid[:, None], other=0.0)
            proxy = tl.dot(q.to(tl.bfloat16), tl.trans(kc.to(tl.bfloat16)))
            proxy = proxy * (qs[:, None] * sm_scale)      # (BLOCK_M, GROUP)

            route = (tl.sum(proxy, axis=0) / BLOCK_M > thr) \
                | (tl.abs(qb_kv - offs_blk) <= LOCAL) | (offs_blk < SINK)
            exact = valid & route
            approx = valid & ~route

            # ---- skipped blocks: one rank-GROUP matmul ----------------------
            if tl.sum(approx.to(tl.int32)) > 0:
                vc = tl.load(VMEAN + off_bh.to(tl.int64) * nb * D
                             + offs_blk[:, None] * D + offs_d[None, :],
                             mask=valid[:, None], other=0.0)
                s_ap = tl.where(approx[None, :], proxy, -float("inf"))
                m_new = tl.maximum(m_i, tl.max(s_ap, axis=1))
                m_new = tl.where(m_new == float("-inf"), 0.0, m_new)
                alpha = tl.math.exp2((m_i - m_new) * LOG2E)
                alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
                p_ap = tl.math.exp2((s_ap - m_new[:, None]) * LOG2E)
                p_ap = tl.where(approx[None, :], p_ap, 0.0)
                # A skipped block stands for its real (possibly tail) length:
                # every one of those tokens carries the same proxy score and the
                # same mean value, so the block contributes length * p * v_mean
                # to the numerator and length * p to the normaliser. Folding the
                # length in here (rather than only in l_i) is what keeps the
                # approximation on the same scale as the exact blocks.
                lengths = tl.where(
                    valid, tl.minimum(BLOCK_N, N - offs_blk * BLOCK_N), 0
                ).to(tl.float32)
                p_ap = p_ap * lengths[None, :]
                acc = acc * alpha[:, None] + tl.dot(p_ap.to(tl.bfloat16),
                                                    vc.to(tl.bfloat16))
                l_i = l_i * alpha + tl.sum(p_ap, axis=1)
                m_i = m_new

            # ---- kept blocks: exact, in increasing block order --------------
            exact_offs = tl.where(exact, tl.arange(0, GROUP), GROUP)
            for _ in range(tl.sum(exact.to(tl.int32))):
                off = tl.min(exact_offs)
                exact_offs = tl.where(tl.arange(0, GROUP) == off, GROUP, exact_offs)
                acc, l_i, m_i = _vc_exact_tile(
                    q, qs, off_bh, (blk0 + off) * BLOCK_N, base, N_PAD, nb,
                    K, KS, V, VS, MU, acc, l_i, m_i, N, sm_scale,
                    D=D, BLOCK_N=BLOCK_N, EXPCAST=EXPCAST, BETA=BETA, PV_FP8=PV_FP8,
                )

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
    qk_fp4: bool = False,
    sparse: bool = False,
    tau: float = 1.3,
    q_block: int = 64,
    scale: Optional[float] = None,
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

    q4 = q4_scale = k4 = k4_scale = None
    if qk_fp4:
        # NVFP4 Q/K. Q packs along D (token-major); K is transposed to K-first
        # (D//2, NP) because tl.dot_scaled wants the reduction dim on axis 0.
        q4p, q4_scale = nvfp4_pack(q_t)
        k4p, k4_scale = nvfp4_pack(k_t)
        q4 = q4p.reshape(b, h, n_pad, d // 2).contiguous()
        q4_scale = q4_scale.reshape(b, h, n_pad, d // 16).contiguous()
        k4 = k4p.transpose(-2, -1).reshape(b, h, d // 2, n_pad).contiguous()
        k4_scale = k4_scale.reshape(b, h, n_pad, d // 16).contiguous()

    if pad:
        vf = torch.nn.functional.pad(vf, (0, 0, 0, pad))

    vb = vf.reshape(G, nb, block_rows, d)
    mu = vb.float().mean(dim=2)                        # (G, NB, D) fp32, small
    resid = vb - mu.to(vb.dtype).unsqueeze(2)          # broadcast, no repeat
    v_scale = resid.detach().abs().amax(dim=2).float().clamp(min=1e-30) / 448.0
    v_codes = (resid / v_scale.to(vb.dtype).unsqueeze(2)).to(torch.float8_e4m3fn)
    mu_over_scale = mu / v_scale

    k_mean = v_mean = thresh = None
    n_qblk = 0
    if sparse:
        # Sol-Attn's routing statistics, computed on exactly the tensors the
        # kernel sees: q_t (hadamard'd, padded) and k_t (hadamard'd, smoothed,
        # permuted, padded). Hadamard is orthonormal so <Hq, Hk> = <q, k>, and
        # the K smoothing shifts every score of a query by the same constant,
        # which softmax ignores -- so the proxy is consistent with the scores.
        sc = scale if scale is not None else d ** -0.5
        k_mean = k_t.reshape(G, nb, block_rows, d).float().mean(dim=2)     # (G, NB, D)
        v_mean = mu                                                        # (G, NB, D)
        n_qblk = n_pad // q_block
        q_cent = q_t.reshape(G, n_qblk, q_block, d).float().mean(dim=2)    # (G, NQB, D)
        proxy = torch.matmul(q_cent, k_mean.transpose(-1, -2)) * sc        # (G, NQB, NB)
        # Population std, matching Sol-Attn's E[x^2] - E[x]^2 form.
        thr = proxy.mean(dim=-1, keepdim=True) \
            + tau * proxy.std(dim=-1, keepdim=True, unbiased=False)
        thresh = thr.reshape(G, n_qblk).contiguous()
        del proxy, q_cent
        k_mean = k_mean.reshape(b, h, nb, d).contiguous()
        v_mean = v_mean.reshape(b, h, nb, d).contiguous()
        thresh = thresh.reshape(b, h, n_qblk).contiguous()

    return Prepared(
        q=q_codes.reshape(b, h, n_pad, d).contiguous(),
        q_scale=q_scale.reshape(b, h, n_pad).contiguous().float(),
        k=k_codes.reshape(b, h, n_pad, d).contiguous(),
        k_scale=k_scale.reshape(b, h, n_pad).contiguous().float(),
        v=v_codes.reshape(b, h, n_pad, d).contiguous(),
        v_scale=v_scale.reshape(b, h, nb, d).contiguous().float(),
        mu=mu_over_scale.reshape(b, h, nb, d).contiguous().float(),
        n_pad=n_pad,
        q4=q4,
        q4_scale=q4_scale,
        k4=k4,
        k4_scale=k4_scale,
        k_mean=k_mean,
        v_mean=v_mean,
        thresh=thresh,
        n_qblk=n_qblk,
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
        qk_fp4 = bool(cfg.qk_fp4) and q.is_cuda
        sparse = bool(cfg.sparse) and q.is_cuda
        if sparse and qk_fp4:
            # The sparse kernel only carries the fp8 QK path; NVFP4 QK is an
            # opt-in side branch that is a measured net loss anyway (see
            # TritonConfig.qk_fp4), so sparsity wins when both are asked for.
            print("[VC-Attention] sparse=True disables qk_fp4 (not combined)",
                  file=sys.stderr)
            qk_fp4 = False
        if q.is_cuda:
            p = _prepare_fast(q, k, v, perm, block_rows=cfg.block_n,
                              hadamard=True, smooth_k=True, qk_fp4=qk_fp4,
                              sparse=sparse, tau=cfg.tau,
                              q_block=cfg.block_m, scale=scale)
        else:
            p = prepare(q, k, v, perm, block_rows=cfg.block_n, hadamard=True)
        if sparse and p.thresh is None:
            sparse = False      # CPU/reference prepare has no routing stats
        # The kernel indexes Out with the padded pitch (base = bh * N_PAD * D),
        # so the buffer must be N_PAD long, not n — a plain empty_like(q)
        # mis-addresses every head after the first whenever n % block_rows != 0.
        out = torch.empty((b, h, p.n_pad, d), device=q.device, dtype=q.dtype)
        grid = (triton.cdiv(p.n_pad, cfg.block_m), b * h)
        pv_fp8 = cfg.pv_fp8 if cfg.pv_fp8 is not None else _default_pv_fp8()
        if sparse:
            # Grouped sparse kernel: fp8 QK/PV plus block routing. Kept as a
            # separate kernel so the dense path stays byte-identical (gate G2).
            _vc_attn_fwd_sparse[grid](
                p.q, p.q_scale, p.k, p.k_scale, p.v, p.v_scale, p.mu, out,
                p.k_mean, p.v_mean, p.thresh,
                scale, n, p.n_pad, d,
                BLOCK_M=cfg.block_m,
                BLOCK_N=cfg.block_n,
                EXPCAST=cfg.enable_expcast,
                BETA=cfg.expcast_beta,
                PV_FP8=pv_fp8,
                LOCAL=cfg.local_blocks,
                SINK=cfg.sink_blocks,
                GROUP=cfg.group,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
            )
            return out[:, :, :n]
        # When QK_FP4 is off the four fp4 slots are unused; pass the fp8 tensors
        # so the argument list stays uniform (the loads are dead-code-eliminated).
        q4 = p.q4 if qk_fp4 else p.q
        q4s = p.q4_scale if qk_fp4 else p.q_scale
        k4 = p.k4 if qk_fp4 else p.k
        k4s = p.k4_scale if qk_fp4 else p.k_scale
        _vc_attn_fwd[grid](
            p.q, p.q_scale, p.k, p.k_scale, p.v, p.v_scale, p.mu, out,
            q4, q4s, k4, k4s,
            scale, n, p.n_pad, d,
            BLOCK_M=cfg.block_m,
            BLOCK_N=cfg.block_n,
            EXPCAST=cfg.enable_expcast,
            BETA=cfg.expcast_beta,
            PV_FP8=pv_fp8,
            QK_FP4=qk_fp4,
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
