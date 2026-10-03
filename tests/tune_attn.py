"""Sweep fused-kernel launch configs on the current GPU.

    python tests/tune_attn.py --tokens 16384

BLOCK_N is pinned to block_rows (128) because one KV tile must equal one
V-Smooth value block; BLOCK_M, num_warps and num_stages are free.
"""
import argparse
import os
import sys

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_attention import h3_like, psnr, timeit  # noqa: E402
from vc_attention.grouping import GroupingConfig, build_permutation  # noqa: E402
from vc_attention.kernels import triton_attn as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--heads", type=int, default=56)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--repeat", type=int, default=10)
    args = ap.parse_args()

    b, h, n, d = 1, args.heads, args.tokens, args.head_dim
    q, k, v = h3_like(n, h, d, "cuda")
    ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
    perm = build_permutation(v.reshape(h, n, d), GroupingConfig(block_rows=128, iters=3)).perm
    p = T._prepare_fast(q, k, v, perm, 128, True, True)
    out = torch.empty((b, h, p.n_pad, d), device="cuda", dtype=q.dtype)
    t_sdpa = timeit(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v), repeat=args.repeat)
    print(f"SDPA {t_sdpa:.2f} ms   tokens={n} heads={h}", flush=True)

    rows = []
    for bm in (64, 128, 256):
        for nw in (4, 8):
            for ns in (2, 3, 4):
                def run():
                    T._vc_attn_fwd[(triton.cdiv(p.n_pad, bm), b * h)](
                        p.q, p.q_scale, p.k, p.k_scale, p.v, p.v_scale, p.mu, out,
                        d ** -0.5, n, p.n_pad, d,
                        BLOCK_M=bm, BLOCK_N=128, EXPCAST=False, BETA=-0.35, PV_FP8=True,
                        num_warps=nw, num_stages=ns)
                    return out[:, :, :n]
                try:
                    o = run()
                    torch.cuda.synchronize()
                    t = timeit(run, warmup=2, repeat=args.repeat)
                    rows.append((t, bm, nw, ns, psnr(o, ref)))
                    print(f"  BM={bm:3} warps={nw} stages={ns}: {t:7.2f} ms  "
                          f"{t_sdpa/t:5.2f}x  PSNR {rows[-1][4]:.2f}", flush=True)
                except Exception as e:
                    print(f"  BM={bm:3} warps={nw} stages={ns}: FAIL {type(e).__name__}", flush=True)
    rows.sort()
    t, bm, nw, ns, ps = rows[0]
    print(f"\nBEST: BM={bm} warps={nw} stages={ns}  {t:.2f} ms  {t_sdpa/t:.2f}x  PSNR {ps:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
