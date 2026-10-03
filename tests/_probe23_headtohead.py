"""Probe 23: head-to-head on ONE harness -- bf16 FlashAttention vs Comfy
Kitchen INT8 vs VC-Attention fp8. Same data, same timing loop, same shape.

This answers "does VC-Attention need FP4 to be worth anything?": the fp8 path
uses no FP4 at all, so if it loses to Kitchen INT8 here, the node has no point
on sm_120 regardless of the FP4 question.
"""
import os
import sys

import torch

sys.path.insert(0, "/root/ComfyUI")
sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention")
sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention/tests")

import comfy.ldm.modules.attention as A  # noqa: E402
from bench_attention import h3_like, psnr, timeit  # noqa: E402
from vc_attention.grouping import GroupingConfig, build_permutation  # noqa: E402
from vc_attention.kernels.triton_attn import TritonConfig, vc_attention_triton  # noqa: E402

H, D = 56, 128
print("kitchen:", A.optimized_attention.__name__, flush=True)


def run(n, with_psnr):
    q, k, v = h3_like(n, H, D, "cuda")
    ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float()) if with_psnr else None
    out = {}

    def sdpa():
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    def kitchen():
        return A.attention_comfy_kitchen_int8(q, k, v, H, skip_reshape=True)

    perm = build_permutation(v.reshape(H, n, D), GroupingConfig(block_rows=128, iters=3)).perm
    cfg = TritonConfig(block_m=64, block_n=128, num_warps=4, num_stages=2)

    def vc():
        return vc_attention_triton(q, k, v, perm=perm, cfg=cfg)

    rows = [("bf16 FlashAttention", sdpa), ("Comfy Kitchen INT8", kitchen), ("VC-Attention fp8", vc)]
    base = None
    print(f"\n=== tokens={n}  heads={H}  head_dim={D} ===", flush=True)
    for name, fn in rows:
        try:
            o = fn()
            torch.cuda.synchronize()
            t = timeit(fn, warmup=2, repeat=5)
            p = f"{psnr(o, ref):6.2f} dB" if with_psnr else "     -   "
            if base is None:
                base = t
            print(f"  {name:22}: {t:9.2f} ms  {base/t:5.2f}x  PSNR {p}", flush=True)
            del o
        except Exception as e:  # noqa: BLE001
            print(f"  {name:22}: FAIL {type(e).__name__}: {str(e)[:80]}", flush=True)
    del q, k, v, ref
    torch.cuda.empty_cache()


run(16384, True)
run(65536, False)
