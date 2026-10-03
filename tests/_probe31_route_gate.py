"""Probe 31 -- Phase 1 gate G1: verify the host-side routing statistics.

Cross-checks the tensors _prepare_fast(sparse=True) produces:
  * thresh really is mean + tau*std of the proxy implied by the returned k_mean
  * k_mean really is the per-BLOCK_N mean of the permuted K the kernel will read
  * the keep ratio at each tau matches probe22, which computed the same Sol-Attn
    rule independently in PyTorch (0.322 / 0.177 / 0.117 at tau 0.5 / 1.0 / 1.3)
"""
import os
import sys

import torch

sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention")
sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention/tests")

from bench_attention import h3_like  # noqa: E402
from vc_attention import quant as Q  # noqa: E402
from vc_attention.grouping import GroupingConfig, build_permutation  # noqa: E402
from vc_attention.kernels import triton_attn as T  # noqa: E402

H, D, QB, BN = 56, 128, 64, 128
REF_KEEP = {0.5: 0.322, 1.0: 0.177, 1.3: 0.117}   # from probe22

for n in (16384, 65536):
    q, k, v = h3_like(n, H, D, "cuda")
    perm = build_permutation(v.reshape(H, n, D), GroupingConfig(block_rows=128, iters=3)).perm
    print(f"\n=== tokens={n} ===", flush=True)

    for tau in (0.5, 1.0, 1.3):
        p = T._prepare_fast(q, k, v, perm, 128, True, True,
                            sparse=True, tau=tau, q_block=QB)
        G = H
        n_pad, nb, nqb = p.n_pad, p.n_pad // BN, p.n_qblk
        assert p.k_mean.shape == (1, H, nb, D), p.k_mean.shape
        assert p.v_mean.shape == (1, H, nb, D), p.v_mean.shape
        assert p.thresh.shape == (1, H, nqb), p.thresh.shape
        assert p.n_qblk == n_pad // QB

        # independent reconstruction of q_t (hadamard + pad), then the proxy
        Hm = T._hadamarian(D, q.device).float()
        qf = q.reshape(G, n, D).float()
        q_t = (qf @ Hm)
        pad = n_pad - n
        if pad:
            q_t = torch.nn.functional.pad(q_t, (0, 0, 0, pad))
        q_cent = q_t.reshape(G, nqb, QB, D).mean(dim=2)
        proxy = torch.matmul(q_cent, p.k_mean.reshape(G, nb, D).transpose(-1, -2)) * (D ** -0.5)
        thr_re = (proxy.mean(-1) + tau * proxy.std(-1, unbiased=False)).reshape(1, H, nqb)
        thr_err = float((thr_re - p.thresh).abs().max() / thr_re.abs().max())

        # k_mean vs the block mean of the K the kernel actually reads (dequantised).
        # Only for the small case -- dequantising (56, 65536, 128) to fp32 needs
        # ~4 GiB of temporaries and OOMs.
        if n <= 16384:
            kdq = Q.dequantize_e4m3(p.k.reshape(G, n_pad, D).view(torch.uint8),
                                    p.k_scale.reshape(G, n_pad, 1)).float()
            kmean_ref = kdq.reshape(G, nb, BN, D).mean(dim=2)
            kmean_err = float((kmean_ref - p.k_mean.reshape(G, nb, D)).norm()
                              / kmean_ref.norm())
            del kdq
        else:
            kmean_err = float("nan")

        keep = proxy > p.thresh.reshape(G, nqb)[:, :, None]
        idx = torch.arange(nb, device=q.device)
        keep = keep | ((torch.arange(nqb, device=q.device)[:, None] // (BN // QB)
                        - idx[None, :]).abs() <= 1)
        ratio = float(keep.float().mean())
        print(f"  tau={tau}: thresh rel-err {thr_err:.2e}  k_mean rel-err {kmean_err:.2e}  "
              f"keep {ratio:.3f} (probe22 ref {REF_KEEP[tau]:.3f})", flush=True)
        del p
    del q, k, v
    torch.cuda.empty_cache()
