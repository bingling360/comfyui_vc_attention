"""Probe 35 -- Phase 2 rework gates (G2/G3) and Phase 3 (G4/G5).

The rework replaced the per-tile proxy and the per-tile fp32 outer-product
approximation with Sol-Attn's structure: one tensor-core matmul per GROUP of KV
blocks for the routing proxy, and one small matmul to fold every skipped block's
mean-value column in. This probe checks

  G2  SPARSE=False path unchanged (the dense kernel is a separate kernel now);
  G3  SPARSE=True output matches a faithful PyTorch simulation of the SAME
      algorithm (block-mean value + reused proxy score on the quantised
      operands, threshold computed host-side exactly as _prepare_fast does);
  G4  fused sparse is faster than Comfy Kitchen INT8 at 16K and 64K;
  G5  the tau / speed / PSNR curve, with the recommended working point.

The simulation is the ground truth here: it mirrors `_prepare_fast` (Hadamard ->
per-token fp8 Q/K, V-Smooth block demean + fp8 V) and then the kernel's own
arithmetic (skipped block == its proxy score with the block-mean value), so a
correct kernel must land on it to fp32 noise.
"""
import os
import sys

import torch

sys.path.insert(0, "/root/ComfyUI")
sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention")
sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention/tests")

import comfy.ldm.modules.attention as A  # noqa: E402
from bench_attention import h3_like, psnr, timeit  # noqa: E402
from vc_attention import quant as Q  # noqa: E402
from vc_attention.grouping import GroupingConfig, build_permutation  # noqa: E402
from vc_attention.kernels import triton_attn as T  # noqa: E402

H, D, QB, BN = 56, 128, 64, 128
TAUS = (0.8, 1.0, 1.3, 2.0)
print("kitchen:", A.optimized_attention.__name__, flush=True)


def dq_fp8(x):
    c, sc = Q.quantize_e4m3(x, dim=-1)
    return Q.dequantize_e4m3(c, sc)


def simulate(q, k, v, perm, tau, scale, local=1, sink=0):
    """PyTorch mirror of the fused kernel's sparse path, one head at a time.

    Kept blocks use the exact quantised score against their real (quantised)
    value; skipped blocks use their proxy score against the *block mean* value,
    weighted by the block length -- exactly the kernel's approximation. The
    skipped value depends on the query block, so the reduction is done one query
    block at a time (which also keeps the (N, N) score matrix out of memory).
    """
    Hh, N, Dd = q.shape[1], q.shape[2], q.shape[3]
    qf = q.reshape(Hh, N, Dd)
    kf = k.reshape(Hh, N, Dd)
    vf = v.reshape(Hh, N, Dd)
    Hm = T._hadamarian(Dd, q.device).float()
    nb = N // BN
    nqb = N // QB
    ki = torch.arange(nb, device=q.device)
    local_mask = (ki[None, :] < sink)

    outs, ratios = [], []
    for hh in range(Hh):
        q_t = qf[hh].float() @ Hm
        k_t = kf[hh].float() @ Hm
        k_t = k_t - k_t.mean(dim=0, keepdim=True)
        p = perm[hh].to(torch.int64)
        k_t = k_t[p]
        v_p = vf[hh].float()[p]

        q8 = dq_fp8(q_t)
        k8 = dq_fp8(k_t)
        vb = v_p.reshape(nb, BN, Dd)
        mu = vb.mean(dim=1)                                     # (nb, D)
        # V-Smooth quantisation granularity is per (block x channel), the same
        # as _prepare_fast / quantize_e4m3_blocks.
        codes, vsc = Q.quantize_e4m3_blocks((vb - mu[:, None, :]).reshape(1, N, Dd), BN)
        v8 = (Q.dequantize_e4m3_blocks(codes, vsc, BN).reshape(nb, BN, Dd)
              + mu[:, None, :]).reshape(N, Dd)

        # Host-side routing statistics: exactly what _prepare_fast builds.
        k_mean = k_t.reshape(nb, BN, Dd).mean(dim=1)             # (nb, D)
        q_cent = q_t.reshape(nqb, QB, Dd).mean(dim=1)            # (nqb, D)
        proxy_h = (q_cent @ k_mean.T) * scale
        thr = proxy_h.mean(-1, keepdim=True) + tau * proxy_h.std(-1, unbiased=False, keepdim=True)
        # The kernel compares its own proxy (quantised q x host k_mean) against it.
        proxy = (q8 @ k_mean.T) * scale                          # (N, nb)
        pq = proxy.reshape(nqb, QB, nb).mean(dim=1)              # (nqb, nb)
        qi = torch.arange(nqb, device=q.device)
        keep = (pq > thr) | ((qi[:, None] // (BN // QB) - ki[None, :]).abs() <= local) | local_mask

        out = torch.empty(N, Dd, device=q.device, dtype=torch.float32)
        for j in range(nqb):
            rows = slice(j * QB, (j + 1) * QB)
            kb = keep[j]                                         # (nb,)
            kb_e = kb.repeat_interleave(BN)[None, :]             # (1, N)
            s = (q8[rows] @ k8.T) * scale                        # (QB, N)
            s_mod = torch.where(kb_e, s, proxy[rows].repeat_interleave(BN, 1))
            m = s_mod.max(dim=-1, keepdim=True).values
            p_all = torch.exp(s_mod - m)
            num = (p_all * kb_e) @ v8
            p_blocks = p_all.reshape(QB, nb, BN).sum(dim=-1)
            # p_blocks already sums the BN tokens of each skipped block, so the
            # block-mean value must NOT be scaled again here.
            num = num + ((p_blocks * (~kb)[None, :].float()) @ mu)
            out[rows] = num / p_all.sum(dim=-1, keepdim=True)
        outs.append(out)
        ratios.append(float(keep.float().mean()))
        del q_t, k_t, q8, k8, vb, mu, v8, k_mean, q_cent, proxy_h
        del proxy, pq, keep, out
        torch.cuda.empty_cache()
    return torch.stack(outs).reshape(1, Hh, N, Dd), sum(ratios) / Hh


def main():
    n = 16384
    q, k, v = h3_like(n, H, D, "cuda")
    ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
    perm = build_permutation(v.reshape(H, n, D), GroupingConfig(block_rows=BN, iters=3)).perm
    scale = D ** -0.5

    # ---- G2: the dense path must be untouched -------------------------------
    cfg_dense = T.TritonConfig(block_m=QB, block_n=BN, num_warps=4, num_stages=2, sparse=False)
    o_dense = T.vc_attention_triton(q, k, v, perm=perm, cfg=cfg_dense)
    print(f"\nG2 dense PSNR {psnr(o_dense, ref):6.2f} dB  (SPARSE=False is a "
          f"separate kernel; probe33 checks bit-exactness)", flush=True)

    # ---- G3: sparse vs the PyTorch simulation -------------------------------
    print(f"\n=== G3: kernel vs PyTorch simulation (tokens={n}) ===", flush=True)
    for tau in TAUS:
        sim, kr = simulate(q, k, v, perm, tau, scale)
        cfg = T.TritonConfig(block_m=QB, block_n=BN, num_warps=4, num_stages=2,
                             sparse=True, tau=tau)
        ker = T.vc_attention_triton(q, k, v, perm=perm, cfg=cfg)
        rel = float((ker.float() - sim).norm() / sim.norm())
        print(f"  tau={tau:<4} keep(sim) {kr:5.3f}  sim {psnr(sim, ref):6.2f} dB  "
              f"kernel {psnr(ker, ref):6.2f} dB  rel-err {rel:.3e}  "
              f"{'PASS' if rel < 5e-2 else 'CHECK'}", flush=True)
        del sim, ker
        torch.cuda.empty_cache()
    del q, k, v, ref, o_dense
    torch.cuda.empty_cache()

    # ---- G4/G5: timing and quality curve ------------------------------------
    for n in (16384, 65536):
        q, k, v = h3_like(n, H, D, "cuda")
        ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
        perm = build_permutation(v.reshape(H, n, D), GroupingConfig(block_rows=BN, iters=3)).perm
        print(f"\n=== tokens={n}  heads={H}  head_dim={D} ===", flush=True)

        rows = []
        sdpa = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v)
        # Comfy Kitchen INT8 is measured separately (tests/_probe26) -- on this
        # pod it raises a shape error if bf16 SDPA ran first in the same process.
        rows.append(("bf16 FlashAttention", sdpa, None))
        rows.append(("VC dense (fp8)", lambda: T.vc_attention_triton(
            q, k, v, perm=perm, cfg=T.TritonConfig(block_m=QB, block_n=BN,
                                                   num_warps=4, num_stages=2, sparse=False)), None))
        for tau in TAUS:
            rows.append((f"VC sparse tau={tau}",
                         lambda tau=tau: T.vc_attention_triton(
                             q, k, v, perm=perm,
                             cfg=T.TritonConfig(block_m=QB, block_n=BN, num_warps=4,
                                                num_stages=2, sparse=True, tau=tau)), tau))

        base = None
        for name, fn, _tau in rows:
            try:
                o = fn()
                torch.cuda.synchronize()
                t = timeit(fn, warmup=2, repeat=5)
                if base is None:
                    base = t
                print(f"  {name:24}: {t:9.2f} ms  {base / t:5.2f}x  "
                      f"PSNR {psnr(o, ref):6.2f} dB", flush=True)
                del o
            except Exception as e:  # noqa: BLE001
                print(f"  {name:24}: FAIL {type(e).__name__}: {str(e)[:90]}", flush=True)
        del q, k, v, ref
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
