"""Probe 26: Comfy Kitchen INT8 timing measured FIRST in a fresh process.

probe23 calls bf16 SDPA before Kitchen and Kitchen then raises a shape error;
probe24 calls Kitchen first and it works. That ordering sensitivity is a
property of this pod's stack (start_comfyui.sh documents a libcublasLt problem
on RTX 5090), not of Kitchen. So measure Kitchen first, cleanly.
"""
import os
import sys

import torch

sys.path.insert(0, "/root/ComfyUI")
sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention")
sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention/tests")

import comfy.ldm.modules.attention as A  # noqa: E402
from bench_attention import h3_like, timeit  # noqa: E402

H, D = 56, 128
for n in (16384, 65536):
    q, k, v = h3_like(n, H, D, "cuda")
    o = A.attention_comfy_kitchen_int8(q, k, v, H, skip_reshape=True)
    torch.cuda.synchronize()
    t = timeit(lambda: A.attention_comfy_kitchen_int8(q, k, v, H, skip_reshape=True),
               warmup=2, repeat=5)
    print(f"kitchen first, tokens={n}: {t:8.2f} ms  finite={bool(torch.isfinite(o).all())}", flush=True)
    del q, k, v, o
    torch.cuda.empty_cache()
