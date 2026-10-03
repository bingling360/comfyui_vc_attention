"""Probe 30: how does VC-Attention's kernel time scale with the number of KV
tiles? That is the ceiling for fusing Sol-Attn's block sparsity into it.

Stacking the two as separate nodes buys nothing (probe27). But sparsity is a
different axis from quantisation -- Sol-Attn *skips* KV blocks, VC *cheapens*
them -- and probe22 showed the two compose numerically for <=0.03 dB. So the
only combination with a real speedup is a fused kernel.

This measures the speed side without implementing the routing: TILE_SKIP=k walks
only every k-th tile. The output is wrong on purpose; only the timing matters.
If time falls roughly linearly with 1/k, sparsity has room to pay.
"""
import os
import sys

import torch
import triton

sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention")
sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention/tests")

from bench_attention import h3_like, timeit  # noqa: E402
from vc_attention.grouping import GroupingConfig, build_permutation  # noqa: E402
from vc_attention.kernels import triton_attn as T  # noqa: E402

H, D = 56, 128


def main():
    for n in (16384, 65536):
        q, k, v = h3_like(n, H, D, "cuda")
        perm = build_permutation(v.reshape(H, n, D), GroupingConfig(block_rows=128, iters=3)).perm
        p = T._prepare_fast(q, k, v, perm, 128, True, True)
        out = torch.empty((1, H, p.n_pad, D), device="cuda", dtype=q.dtype)
        t_prep = timeit(lambda: T._prepare_fast(q, k, v, perm, 128, True, True), repeat=5)

        def run(skip):
            T._vc_attn_fwd[(triton.cdiv(p.n_pad, 64), H)](
                p.q, p.q_scale, p.k, p.k_scale, p.v, p.v_scale, p.mu, out,
                p.q, p.q_scale, p.k, p.k_scale,
                D ** -0.5, n, p.n_pad, D,
                BLOCK_M=64, BLOCK_N=128, EXPCAST=False, BETA=-0.35,
                PV_FP8=True, QK_FP4=False, TILE_SKIP=skip,
                num_warps=4, num_stages=2)
            return out

        print(f"\n=== tokens={n}   prepare {t_prep:.2f} ms ===", flush=True)
        base = None
        for skip in (1, 2, 4, 8):
            run(skip)
            torch.cuda.synchronize()
            t = timeit(lambda: run(skip), warmup=2, repeat=5)
            if base is None:
                base = t
            frac = 1.0 / skip
            print(f"  tiles {frac*100:5.1f}%  (TILE_SKIP={skip}): {t:8.2f} ms   "
                  f"{t/base:.3f}x of full   (+prepare {t + t_prep:7.2f} ms)", flush=True)
        print(f"  -> Sol-Attn tau=1.0 keeps ~18% of blocks; tau=1.3 ~12%", flush=True)
        del q, k, v, p, out
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
