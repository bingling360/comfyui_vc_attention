"""Probe 32 -- Phase 2 gates.

G2 (no regression): SPARSE=True with a very negative tau keeps every block, so
    it must reproduce SPARSE=False exactly.
G3 (routing correct): SPARSE=True at a real tau must land near the PyTorch
    simulation of the same rule (probe22), and must be FASTER.
"""
import os
import sys

import torch

sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention")
sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention/tests")

from bench_attention import h3_like, psnr, timeit  # noqa: E402
from vc_attention.grouping import GroupingConfig, build_permutation  # noqa: E402
from vc_attention.kernels.triton_attn import TritonConfig, vc_attention_triton  # noqa: E402

H, D = 56, 128

for n in (16384, 65536):
    q, k, v = h3_like(n, H, D, "cuda")
    ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
    perm = build_permutation(v.reshape(H, n, D), GroupingConfig(block_rows=128, iters=3)).perm
    print(f"\n=== tokens={n} ===", flush=True)

    cases = [
        ("SPARSE=False          ", TritonConfig(block_m=64, block_n=128, num_warps=4,
                                                num_stages=2, sparse=False)),
        ("SPARSE=True tau=-10   ", TritonConfig(block_m=64, block_n=128, num_warps=4,
                                                num_stages=2, sparse=True, tau=-10.0)),
        ("SPARSE=True tau=1.3   ", TritonConfig(block_m=64, block_n=128, num_warps=4,
                                                num_stages=2, sparse=True, tau=1.3)),
        ("SPARSE=True tau=1.0   ", TritonConfig(block_m=64, block_n=128, num_warps=4,
                                                num_stages=2, sparse=True, tau=1.0)),
    ]
    outs = {}
    for name, cfg in cases:
        try:
            o = vc_attention_triton(q, k, v, perm=perm, cfg=cfg)
            torch.cuda.synchronize()
            t = timeit(lambda: vc_attention_triton(q, k, v, perm=perm, cfg=cfg),
                       warmup=2, repeat=5)
            outs[name] = o
            print(f"  {name}: {t:8.2f} ms   PSNR {psnr(o, ref):6.2f} dB", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: FAILED {type(e).__name__}: {str(e)[:160]}", flush=True)

    a = outs.get("SPARSE=False          ")
    b = outs.get("SPARSE=True tau=-10   ")
    if a is not None and b is not None:
        rel = float((a.float() - b.float()).norm() / a.float().norm())
        print(f"  G2 keep-all vs SPARSE=False rel-err: {rel:.3e} "
              f"-> {'PASS' if rel < 1e-5 else 'FAIL'}", flush=True)
    del q, k, v, ref
    torch.cuda.empty_cache()
