"""Blocked reference implementation of VC-Attention.

This is the *oracle*: it consumes exactly the low-bit codes a fused kernel
would produce and finishes the arithmetic in fp32. It is used for

  * correctness tests on CPU (no GPU needed),
  * the fallback path when no low-bit kernel is available,
  * ablation: flip V-Smooth / ExpCast / grouping on and off independently.

It is NOT fast. On a GPU the real path is ``kernels.triton_attn``.

Mathematical outline (per (batch, head), head_dim = D)
------------------------------------------------------
1. Permute K and V with the V-Smooth permutation pi:  K' = K[pi], V' = V[pi].
   The attention result is unchanged; only the quantisation error moves.
2. Smooth K: c = mean_over_tokens(K'), K~ = K' - c.  Dropping c shifts every
   logit in a row by the same Q.c, so softmax is invariant.
3. Quantise K~ per channel with scale s_k and fold s_k into Q:
   Q~ = Q * s_k (elementwise over channels), then quantise Q~ per row.
   A per-row scale on Q is also softmax-invariant, so it is never materialised.
4. Block-demean V': with blocks of Bv rows, mu_j = mean_rows(V'_j),
   R = V' - mu_j. Quantise R with one scale per (value block x channel), and
   store mu~_j = mu_j / s_j.
5. Flash-style online softmax. Each KV tile is one value block, so it owns a
   single mu~_j and the per-tile epilogue multiply s_j covers both terms:

       A <- alpha A + s_j * (Ph_j Rq_j) + r_j mu~_j^T
       l <- alpha l + r_j

   where r_j = rowsum(Ph_j) is taken from the *quantised* probability tile, so
   the mean correction is consistent with the product actually taken.
6. O = A / l.

Scale granularity is what makes V-Smooth pay
---------------------------------------------
The value scale must be per *block*, not per channel over the whole tensor.
With a global per-channel scale the permutation cannot help: the scale is set
by the tensor's largest entry either way, and grouping changes nothing
(measured: 1.01x). With one scale per value block the block's own range sets
it, so homogeneous blocks -- which is exactly what the k-means produces --
quantise tightly. This is the "a block's scale is set by its largest entries"
framing of the paper, and it is why step 5 applies the scale per tile.

At 4 bits the element format is FP4 (E2M1) under per-16-token microscales,
which the hardware applies inside the MMA. The mean therefore cannot be
divided by a microscale; it is divided by the tensor-level scale g instead and
the whole accumulator is multiplied by g at the end ("means are added
unscaled" in the paper).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from .. import quant as Q
from ..expcast import LOG2E, expcast_codes
from ..hadamard import fwht

__all__ = ["RefConfig", "vc_attention_reference"]


@dataclass
class RefConfig:
    backend: str = "fp8"          # fp8 | nvfp4 | int4 | bf16
    block_rows: int = 128         # Bv: rows per value block (== KV tile size)
    block_m: int = 64             # rows per query tile
    micro_rows: int = 0           # 4-bit scale granularity; 0 -> 16 for nvfp4
    enable_vsmooth: bool = True
    enable_expcast: bool = False  # 8-bit only, see expcast.py
    expcast_beta: float = -0.35
    smooth_k: bool = True         # subtract the per-channel K mean
    hadamard: bool = True         # orthonormal Q/K rotation, see hadamard.py
    p_scale: float = 448.0        # E4M3 per-row scale for P (cancels in A/l)


def _pad_tokens(x: torch.Tensor, multiple: int) -> torch.Tensor:
    n = x.shape[-2]
    pad = (-n) % multiple
    if pad == 0:
        return x
    return torch.nn.functional.pad(x, (0, 0, 0, pad))


def _block_demean(v: torch.Tensor, block_rows: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """v: (G, N, D), N multiple of block_rows -> (residual (G,N,D), mu (G,NB,D))."""
    g, n, d = v.shape
    vb = v.reshape(g, n // block_rows, block_rows, d)
    mu = vb.mean(dim=2)
    resid = v - mu.repeat_interleave(block_rows, dim=1)
    return resid, mu


def _quantize_v(resid: torch.Tensor, cfg: RefConfig) -> Tuple[torch.Tensor, int]:
    """Quantise-and-dequantise the (demeaned) value residual.

    The oracle accumulates in *value* units, so the scale never appears in the
    arithmetic -- it only determines how much error this tensor carries. A
    fused kernel instead keeps P and R in code units, applies the per-tile
    scale in that tile's epilogue, and stores the mean pre-divided by it:

        s_j * (P_j Rq_j + r_j (mu_j / s_j)^T)  ==  P_j dequant(Rq_j) + r_j mu_j

    Returns (v_hat (G,N,D), scale_rows).
    """
    g, n, d = resid.shape
    if cfg.backend == "bf16":
        return resid, cfg.block_rows

    if cfg.backend == "nvfp4":
        rows = cfg.micro_rows or 16
        q, micro, gscale = Q.quantize_nvfp4_grouped(resid, group_rows=rows)
        v_hat, _ = Q.dequantize_nvfp4_grouped(q, micro, gscale, group_rows=rows)
        return v_hat, rows

    rows = cfg.block_rows
    if cfg.backend == "int4":
        xb = resid.reshape(g, n // rows, rows, d)
        amp = xb.detach().abs().amax(dim=2).clamp(min=1e-30)
        scale = amp / 7.0
        q = torch.round(xb.detach() / scale.unsqueeze(2)).clamp_(-8, 7)
        v_hat = (q.float() * scale.unsqueeze(2)).reshape(g, n, d)
        return v_hat, rows

    q, scale = Q.quantize_e4m3_blocks(resid, rows)
    return Q.dequantize_e4m3_blocks(q, scale, rows), rows


def _fake_quant(x: torch.Tensor, dim: int, cfg: RefConfig) -> torch.Tensor:
    """Quantise-and-dequantise. dim=-1 -> per row, dim=1 -> per channel."""
    if cfg.backend == "bf16":
        return x
    if cfg.backend == "nvfp4":
        codes, micro, gscale = Q.quantize_nvfp4_grouped(x, group_rows=cfg.micro_rows or 16)
        deq, _ = Q.dequantize_nvfp4_grouped(codes, micro, gscale, group_rows=cfg.micro_rows or 16)
        return deq
    if cfg.backend == "int4":
        q, s = Q.quantize_int4(x, dim=dim)
        return Q.dequantize_int4(q, s)
    q, s = Q.quantize_e4m3(x, dim=dim)
    return Q.dequantize_e4m3(q, s)


def _quantize_p(p: torch.Tensor, cfg: RefConfig) -> torch.Tensor:
    if cfg.backend == "bf16":
        return p
    if cfg.backend == "nvfp4":
        codes, micro, gscale = Q.quantize_nvfp4_grouped(p, group_rows=cfg.micro_rows or 16)
        deq, _ = Q.dequantize_nvfp4_grouped(codes, micro, gscale, group_rows=cfg.micro_rows or 16)
        return deq
    if cfg.backend == "int4":
        q, s = Q.quantize_int4(p, dim=-1)
        return Q.dequantize_int4(q, s)
    # Per-row scale of 1/448: p in (0,1] -> code space (0,448]. The scale is a
    # per-row constant, so it cancels between the numerator and l.
    q, s = Q.quantize_e4m3(p, dim=-1, scale=torch.full_like(p[..., :1], 1.0 / cfg.p_scale))
    return Q.dequantize_e4m3(q, s)


def vc_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cfg: Optional[RefConfig] = None,
    perm: Optional[torch.Tensor] = None,
    valid_len: Optional[int] = None,
    scale: Optional[float] = None,
    return_debug: bool = False,
):
    """VC-Attention forward.

    Args:
        q, k, v: (B, H, N, D) in any float dtype. Full (non-causal) attention.
        cfg: RefConfig.
        perm: (G, N) int32 V-Smooth permutation with G = B*H. None -> no grouping.
        valid_len: rows >= valid_len are padding (masked, sorted to the tail).
        scale: softmax scale; defaults to 1/sqrt(D).
        return_debug: also return a dict of intermediate statistics.

    Returns:
        out (B, H, N, D), or (out, debug).
    """
    cfg = cfg or RefConfig()
    b, h, n, d = q.shape
    device, dtype = q.device, q.dtype
    G = b * h
    if scale is None:
        scale = d ** -0.5
    if cfg.enable_expcast and cfg.backend != "fp8":
        raise ValueError("ExpCast-FP8 is defined for the 8-bit path only")

    qf = q.reshape(G, n, d).to(torch.float32)
    kf = k.reshape(G, n, d).to(torch.float32)
    vf = v.reshape(G, n, d).to(torch.float32)

    # 1. permute K and V together -- the output is invariant under this.
    if perm is not None:
        pidx = perm.to(torch.int64)
        if pidx.shape[0] == 1 and G > 1:
            pidx = pidx.expand(G, n)
        kf = kf.gather(1, pidx.unsqueeze(-1).expand_as(kf))
        vf = vf.gather(1, pidx.unsqueeze(-1).expand_as(vf))

    # 2. pad to a whole number of value blocks.
    n_valid = valid_len if valid_len is not None else n
    kf = _pad_tokens(kf, cfg.block_rows)
    vf = _pad_tokens(vf, cfg.block_rows)
    n_pad = vf.shape[1]
    nb = n_pad // cfg.block_rows

    # 3. value: block demean, then low-bit residual with per-block scales.
    if cfg.enable_vsmooth:
        resid, mu = _block_demean(vf, cfg.block_rows)
    else:
        resid, mu = vf, None
    v_hat, scale_rows = _quantize_v(resid, cfg)
    mu_bar = None if mu is None else mu.float()

    # 4. Q/K: rotate both with the same orthonormal transform (exact), then
    #    quantise *per token* (one scale per row), not per channel.
    #
    #    Per-token matters more than it looks: measured on an H3-shaped tile,
    #    per-channel key quantisation leaves 55% relative output error while
    #    per-token leaves 3.6%. A per-channel scale is set by the largest token
    #    in the sequence, so a single outlier token wrecks every other row.
    #    The Hadamard rotation spreads per-channel spikes and is worth another
    #    ~0.5 dB on top. A fused kernel folds the per-row scales into the score
    #    tile instead of dequantising, exactly as SageAttention2 does.
    if cfg.hadamard and cfg.backend != "bf16":
        q_t, k_t = fwht(qf), fwht(kf)
    else:
        q_t, k_t = qf, kf
    if cfg.smooth_k:
        k_t = k_t - k_t.mean(dim=1, keepdim=True)
    q_hat = _fake_quant(q_t, dim=-1, cfg=cfg)
    k_hat = _fake_quant(k_t, dim=-1, cfg=cfg)

    # 5. flash-style online softmax.
    out = torch.zeros((G, n, d), dtype=torch.float32, device=device)
    arange_pad = torch.arange(n_pad, device=device)

    for start_m in range(0, n, cfg.block_m):
        end_m = min(n, start_m + cfg.block_m)
        qb = q_hat[:, start_m:end_m, :] * scale
        mb = torch.full((G, end_m - start_m), -float("inf"), device=device)
        lb = torch.zeros((G, end_m - start_m), device=device)
        ab = torch.zeros((G, end_m - start_m, d), device=device)

        for j in range(nb):
            js, je = j * cfg.block_rows, (j + 1) * cfg.block_rows
            s = torch.bmm(qb, k_hat[:, js:je, :].transpose(1, 2))
            if n_valid < n_pad:
                s = torch.where(
                    (arange_pad[js:je] >= n_valid).unsqueeze(0).unsqueeze(0),
                    torch.full_like(s, -1e30),
                    s,
                )

            m_new = torch.maximum(mb, s.amax(dim=-1))
            m_new = torch.where(torch.isneginf(m_new), torch.zeros_like(m_new), m_new)

            if cfg.enable_expcast:
                p = Q.e4m3_decode(expcast_codes(s, m_new, cfg.expcast_beta)) / 256.0
            else:
                p = _quantize_p(torch.exp2((s - m_new.unsqueeze(-1)) * LOG2E), cfg)
            p = torch.where(p < 0, torch.zeros_like(p), p)

            alpha = torch.exp2((mb - m_new) * LOG2E)
            alpha = torch.where(torch.isneginf(mb), torch.zeros_like(alpha), alpha)
            r = p.sum(dim=-1)

            ab = ab * alpha.unsqueeze(-1) + torch.bmm(p, v_hat[:, js:je, :])
            if mu_bar is not None:
                ab = ab + r.unsqueeze(-1) * mu_bar[:, j, :].unsqueeze(1)
            lb = lb * alpha + r
            mb = m_new

        out[:, start_m:end_m, :] = ab / lb.unsqueeze(-1).clamp(min=1e-30)

    out = out.reshape(b, h, n, d).to(dtype)

    if return_debug:
        debug: Dict[str, float] = {
            "n_tokens": n,
            "n_pad": n_pad,
            "value_blocks": nb,
            "scale_rows": scale_rows,
            "v_resid_rmse": float(((v_hat - resid) ** 2).mean().sqrt()),
            "v_amax": float(vf[..., :n_valid, :].abs().amax()),
        }
        return out, debug
    return out
