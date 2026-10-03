"""Speed and fidelity benchmark for VC-Attention on your own GPU.

    python tests/bench_attention.py --tokens 32768 --heads 56 --backend auto

The reference workload imitates MiniMax-H3 at 1344x768 x 243 frames: 56 heads,
head_dim 128, one packed sequence of text + audio + video rows. Reduce
``--tokens`` if you are short on memory; the attention cost is quadratic in it.

Two numbers matter:

  * **speedup** of the attention kernel against bf16 SDPA, and
  * **PSNR** of the VC output against the fp32 SDPA output, which is the
    fidelity measure the VC-Attention paper reports (it quotes ~20 dB on H3
    end-to-end at 8 bits, against 19.9 dB for SageAttention2).
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vc_attention.device import detect_profile, resolve_backend  # noqa: E402
from vc_attention.grouping import GroupingConfig, build_permutation  # noqa: E402
from vc_attention.kernels.triton_attn import TritonConfig, vc_attention_triton  # noqa: E402


def h3_like(tokens: int, heads: int, head_dim: int, device: str, dtype=torch.bfloat16):
    """Region-structured values, per-channel Q/K outliers, three modalities."""
    n_text, n_audio = tokens // 16, tokens // 8
    n_video = tokens - n_text - n_audio

    region = torch.randint(0, 8, (heads, tokens), device=device)
    means = torch.randn(heads, 8, head_dim, device=device) * 2.0
    v = means.gather(1, region.unsqueeze(-1).expand(heads, tokens, head_dim))
    v = v + torch.randn(heads, tokens, head_dim, device=device) * 0.3
    v[:, :n_text] *= 0.30
    v[:, n_text : n_text + n_audio] *= 3.00
    v = v + (torch.rand(heads, tokens, 1, device=device) < 0.01).float() * 15.0

    def spikes(x, frac=0.03, width=4, mag=12.0):
        m = (torch.rand(heads, tokens, 1, device=device) < frac).float()
        ch = torch.randint(0, head_dim - width, (heads, tokens, 1), device=device)
        idx = ch.expand(heads, tokens, width) + torch.arange(width, device=device)
        return x + m * torch.zeros_like(x).scatter(2, idx, mag)

    q = spikes(torch.randn(heads, tokens, head_dim, device=device))
    k = spikes(torch.randn(heads, tokens, head_dim, device=device) * 0.8)
    return (q.unsqueeze(0).to(dtype), k.unsqueeze(0).to(dtype), v.unsqueeze(0).to(dtype))


def psnr(out, ref):
    mse = ((out.float() - ref.float()) ** 2).mean()
    return float(10 * torch.log10(ref.float().pow(2).max() / mse.clamp(min=1e-30)))


def timeit(fn, warmup=3, repeat=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=32768)
    ap.add_argument("--heads", type=int, default=56)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--backend", type=str, default="auto")
    ap.add_argument("--block-rows", type=int, default=128)
    ap.add_argument("--repeat", type=int, default=10)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("This benchmark needs CUDA.")
        return 2

    profile = detect_profile()
    backend = resolve_backend(args.backend, profile)
    print(profile.describe())
    print(f"backend={backend}  tokens={args.tokens}  heads={args.heads} "
          f"head_dim={args.head_dim}")

    dev = "cuda"
    q, k, v = h3_like(args.tokens, args.heads, args.head_dim, dev)

    ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
    base = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    print(f"\nbf16 SDPA vs fp32       : {psnr(base, ref):.2f} dB")

    perm = build_permutation(
        v.reshape(args.heads, args.tokens, args.head_dim),
        GroupingConfig(block_rows=args.block_rows, iters=3),
    ).perm

    t_sdpa = timeit(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v),
                    repeat=args.repeat)

    t_group = timeit(
        lambda: build_permutation(
            v.reshape(args.heads, args.tokens, args.head_dim),
            GroupingConfig(block_rows=args.block_rows, iters=3),
        ),
        warmup=1,
        repeat=max(2, args.repeat // 2),
    )

    tcfg = TritonConfig(block_m=64, block_n=args.block_rows)
    out = vc_attention_triton(q, k, v, perm=perm, cfg=tcfg)
    t_vc = timeit(lambda: vc_attention_triton(q, k, v, perm=perm, cfg=tcfg),
                  repeat=args.repeat)

    print(f"\nbf16 SDPA                : {t_sdpa:8.2f} ms")
    print(f"k-means grouping (once)  : {t_group:8.2f} ms  "
          f"({t_group / t_sdpa * 100:.2f}% of one attention)")
    print(f"VC-Attention             : {t_vc:8.2f} ms  -> {t_sdpa / t_vc:.2f}x")
    print(f"VC-Attention PSNR        : {psnr(out, ref):.2f} dB")

    o_nogroup = vc_attention_triton(q, k, v, perm=None, cfg=tcfg)
    print(f"  without V-Smooth       : {psnr(o_nogroup, ref):.2f} dB "
          f"({psnr(out, ref) - psnr(o_nogroup, ref):+.2f} dB from grouping)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
