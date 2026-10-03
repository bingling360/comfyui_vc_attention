"""Break down VC-Attention time on the current GPU.

    python tests/bench_pv.py --tokens 16384

Compares the backend combinations that exist in the kernel (QK in fp8 or NVFP4,
PV in bf16 or fp8), reporting prepare() and the fused kernel separately so the
cost of each choice is visible, plus PSNR against fp32 SDPA.
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
from vc_attention.kernels.triton_attn import TritonConfig, vc_attention_triton  # noqa: E402
from vc_attention.device import detect_profile  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--heads", type=int, default=56)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--repeat", type=int, default=10)
    args = ap.parse_args()

    print(detect_profile().describe(), flush=True)
    b, h, n, d = 1, args.heads, args.tokens, args.head_dim
    q, k, v = h3_like(n, h, d, "cuda")
    ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
    perm = build_permutation(v.reshape(h, n, d), GroupingConfig(block_rows=128, iters=3)).perm

    t_sdpa = timeit(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v), repeat=args.repeat)
    t_grp = timeit(lambda: build_permutation(v.reshape(h, n, d), GroupingConfig(block_rows=128, iters=3)),
                   warmup=1, repeat=max(2, args.repeat // 2))

    configs = [
        ("fp8 QK + bf16 PV", False, False),
        ("fp8 QK + fp8  PV", False, True),
        ("fp4 QK + fp8  PV", True, True),
    ]
    print(f"\ntokens={n} heads={h} head_dim={d}")
    print(f"  bf16 SDPA            : {t_sdpa:8.2f} ms")
    print(f"  k-means grouping     : {t_grp:8.2f} ms")
    for name, qk_fp4, pv_fp8 in configs:
        cfg = TritonConfig(block_m=64, block_n=128, num_warps=4, num_stages=2,
                           qk_fp4=qk_fp4, pv_fp8=pv_fp8)
        try:
            t_prep = timeit(lambda: T._prepare_fast(q, k, v, perm, cfg.block_n, True, True, qk_fp4=qk_fp4),
                            repeat=args.repeat)
            p = T._prepare_fast(q, k, v, perm, cfg.block_n, True, True, qk_fp4=qk_fp4)
            out = torch.empty((b, h, p.n_pad, d), device="cuda", dtype=q.dtype)
            q4 = p.q4 if qk_fp4 else p.q
            q4s = p.q4_scale if qk_fp4 else p.q_scale
            k4 = p.k4 if qk_fp4 else p.k
            k4s = p.k4_scale if qk_fp4 else p.k_scale

            def run_kernel():
                T._vc_attn_fwd[(triton.cdiv(p.n_pad, cfg.block_m), b * h)](
                    p.q, p.q_scale, p.k, p.k_scale, p.v, p.v_scale, p.mu, out,
                    q4, q4s, k4, k4s, d ** -0.5, n, p.n_pad, d,
                    BLOCK_M=cfg.block_m, BLOCK_N=cfg.block_n, EXPCAST=False, BETA=-0.35,
                    PV_FP8=pv_fp8, QK_FP4=qk_fp4, num_warps=cfg.num_warps, num_stages=cfg.num_stages)
                return out[:, :, :n]

            o = run_kernel()
            torch.cuda.synchronize()
            t_k = timeit(run_kernel, repeat=args.repeat)
            tot = t_prep + t_k
            print(f"  {name}      : prep {t_prep:6.2f} + kernel {t_k:6.2f} = {tot:6.2f} ms "
                  f"-> {t_sdpa/tot:5.2f}x   PSNR {psnr(o, ref):6.2f} dB", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {name}      : FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
