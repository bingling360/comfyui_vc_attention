"""Probe 22: can Sol-Attn (sparse routing) and VC-Attention (quantisation) stack?

Sol-Attn's routing rule is taken verbatim from the ComfyUI integration
(github.com/designloves2/ComfyUI-sol-attn, sol_kernel/preprocess.py):

    BLOCK_SIZE = 64
    kc_mean    = mean of K over each 64-token block
    proxy      = <q, kc_mean> * scale                    (per query row, per KV block)
    pq         = mean of proxy over the 64 queries of a q-block
    threshold  = mean(pq) + tau * std(pq)   over KV blocks
    keep       = (pq > threshold) OR |q_block - k_block| <= 1

Skipped blocks are not dropped: their proxy score is reused to approximate
their contribution (here: every row of the block is given the proxy score).

This probe measures, per head, on H3-like structured data:
  * dense fp32                       (reference)
  * VC-Attention quantisation only   (hadamard + per-token fp8 QK, V-Smooth fp8 V)
  * Sol routing only                 (fp32, sweep tau)
  * both combined
and repeats the routing with K/V in V-Smooth (k-means) order, because the
permutation is the suspected conflict: block-level proxies assume a 64-token
block is semantically coherent, and permuting K/V destroys that.

It is a faithful *simulation* of the routing rule, not their Triton kernel.
"""
import os
import sys

import torch

sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention")
sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention/tests")
from bench_attention import h3_like, psnr  # noqa: E402
from vc_attention import quant as Q  # noqa: E402
from vc_attention.grouping import GroupingConfig, build_permutation  # noqa: E402
from vc_attention.kernels.triton_attn import _hadamarian  # noqa: E402

BLOCK = 64


def route_keep(q, k, scale, tau):
    n, d = q.shape
    nb = n // BLOCK
    kc = k.reshape(nb, BLOCK, d).mean(1)
    proxy = (q @ kc.T) * scale
    pq = proxy.reshape(nb, BLOCK, nb).mean(1)
    thr = pq.mean(-1, keepdim=True) + tau * pq.std(-1, keepdim=True)
    keep = pq > thr
    idx = torch.arange(nb, device=q.device)
    keep = keep | ((idx[:, None] - idx[None, :]).abs() <= 1)
    return keep, float(keep.float().mean())


def sparse_attn(q, k, v, scale, keep):
    n, d = q.shape
    nb = n // BLOCK
    s = (q @ k.T) * scale
    kc = k.reshape(nb, BLOCK, d).mean(1)
    proxy = (q @ kc.T) * scale
    keep_e = keep.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    proxy_e = proxy.repeat_interleave(BLOCK, 1)
    s = torch.where(keep_e, s, proxy_e)
    return torch.softmax(s, -1) @ v


def dq_fp8(x):
    c, sc = Q.quantize_e4m3(x, dim=-1)
    return Q.dequantize_e4m3(c, sc)


def vc_quant(q, k, v, block_rows=128):
    """Mirror _prepare_fast: hadamard + per-token fp8 QK, V-Smooth fp8 V."""
    n, d = q.shape
    Hm = _hadamarian(d, q.device).float()
    q8 = dq_fp8(q @ Hm)
    k8 = dq_fp8((k @ Hm) - (k @ Hm).mean(0, keepdim=True))
    nb = n // block_rows
    vb = v.reshape(nb, block_rows, d)
    mu = vb.mean(1, keepdim=True)
    resid = vb - mu
    v8 = dq_fp8(resid.reshape(n, d)).reshape(nb, block_rows, d) + mu
    return q8, k8, v8.reshape(n, d)


def main():
    h, n, d = 8, 8192, 128
    q, k, v = h3_like(n, h, d, "cuda")
    q = q.reshape(h, n, d).float()
    k = k.reshape(h, n, d).float()
    v = v.reshape(h, n, d).float()
    scale = d ** -0.5

    perm = build_permutation(v, GroupingConfig(block_rows=128, iters=3)).perm  # (h, n)
    print(f"heads={h} tokens={n} head_dim={d}  block={BLOCK}", flush=True)

    ref, vc_only = [], []
    sp_fp32 = {t: [] for t in (0.5, 1.0, 1.3, 2.0)}
    sp_perm = {t: [] for t in (0.5, 1.0, 1.3, 2.0)}
    sp_vc = {t: [] for t in (0.5, 1.0, 1.3, 2.0)}
    ratios = {t: [] for t in (0.5, 1.0, 1.3, 2.0)}
    ratios_perm = {t: [] for t in (0.5, 1.0, 1.3, 2.0)}

    for hh in range(h):
        qh, kh, vh = q[hh], k[hh], v[hh]
        p = perm[hh]
        r = torch.softmax((qh @ kh.T) * scale, -1) @ vh
        ref.append(r)

        q8, k8, v8 = vc_quant(qh, kh, vh)
        vc_only.append(torch.softmax((q8 @ k8.T) * scale, -1) @ v8)

        # V-Smooth order
        kp, vp = kh[p], vh[p]

        for tau in sp_fp32:
            keep, kr = route_keep(qh, kh, scale, tau)
            sp_fp32[tau].append(sparse_attn(qh, kh, vh, scale, keep))
            ratios[tau].append(kr)

            keep_p, krp = route_keep(qh, kp, scale, tau)
            sp_perm[tau].append(sparse_attn(qh, kp, vp, scale, keep_p))
            ratios_perm[tau].append(krp)

            # combined: quantised operands, routing computed on them too
            keep_q, _ = route_keep(q8, k8, scale, tau)
            sp_vc[tau].append(sparse_attn(q8, k8, v8, scale, keep_q))

        del qh, kh, vh, q8, k8, v8
        torch.cuda.empty_cache()

    def cat(xs):
        return torch.stack(xs)

    ref = cat(ref)
    print(f"\nVC-Attention quant only      : {psnr(cat(vc_only), ref):6.2f} dB", flush=True)
    print(f"\n{'tau':>5} {'keep':>7} {'Sol(fp32)':>11} {'Sol(permuted K/V)':>18} "
          f"{'Sol+VC quant':>13}", flush=True)
    for tau in sp_fp32:
        print(f"{tau:>5} {sum(ratios[tau])/h:7.3f} "
              f"{psnr(cat(sp_fp32[tau]), ref):8.2f} dB "
              f"{psnr(cat(sp_perm[tau]), ref):13.2f} dB "
              f"{psnr(cat(sp_vc[tau]), ref):11.2f} dB", flush=True)
    print(f"\n(permuted keep ratio: "
          f"{ {t: round(sum(ratios_perm[t])/h, 3) for t in sp_fp32} })", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
