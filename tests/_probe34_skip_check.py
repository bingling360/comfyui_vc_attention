"""Probe 34: is the skip actually saving time? (isolates routing vs approximation)

probe33 proved the kept branch is exact (forcing keep=True by three routes
reproduces SPARSE=False bit-for-bit). So the bug is either that the routing
decision never skips (nothing saved, and the wrong answer comes from somewhere
else) or that the approximation is wrong.

tau=100 should keep almost nothing. If the kernel then runs at ~12% of the full
time, the skip works and the approximation is at fault. If it stays slow, the
branch is not skipping at all.
"""
import os
import sys

import torch

sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention")
sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention/tests")

from bench_attention import h3_like, psnr, timeit  # noqa: E402
from vc_attention.grouping import GroupingConfig, build_permutation  # noqa: E402
from vc_attention.kernels.triton_attn import TritonConfig, vc_attention_triton  # noqa: E402

H, D, N = 56, 128, 16384
q, k, v = h3_like(N, H, D, "cuda")
ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
perm = build_permutation(v.reshape(H, N, D), GroupingConfig(block_rows=128, iters=3)).perm
base = dict(block_m=64, block_n=128, num_warps=4, num_stages=2)

for label, kw in [
    ("SPARSE=False", dict(sparse=False)),
    ("tau=-1000 (keep all)", dict(sparse=True, tau=-1000.0)),
    ("tau=0.0 (keep ~50%)", dict(sparse=True, tau=0.0)),
    ("tau=1.3 (keep ~12%)", dict(sparse=True, tau=1.3)),
    ("tau=100 (keep ~0%)", dict(sparse=True, tau=100.0)),
]:
    cfg = TritonConfig(**base, **kw)
    try:
        o = vc_attention_triton(q, k, v, perm=perm, cfg=cfg)
        torch.cuda.synchronize()
        t = timeit(lambda: vc_attention_triton(q, k, v, perm=perm, cfg=cfg), warmup=2, repeat=5)
        print(f"  {label:22}: {t:8.2f} ms   PSNR {psnr(o, ref):6.2f} dB", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  {label:22}: FAILED {type(e).__name__}: {str(e)[:140]}", flush=True)
