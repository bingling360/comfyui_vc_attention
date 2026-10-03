"""Probe 24: is the Comfy Kitchen INT8 result actually CORRECT (not just fast)?

probe23 showed Kitchen INT8 at 2.75x vs 1.29x for VC fp8 at 64K tokens, but the
16K call raised a shape error. Speed without correctness is meaningless, so this
pins the call convention, gets PSNR, and re-checks 64K.
"""
import os
import sys
import traceback

import torch

sys.path.insert(0, "/root/ComfyUI")
sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention")
sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention/tests")

import comfy.ldm.modules.attention as A  # noqa: E402
from bench_attention import h3_like, psnr  # noqa: E402

H, D = 56, 128

for n in (16384, 65536):
    q, k, v = h3_like(n, H, D, "cuda")
    print(f"\n=== tokens={n} ===", flush=True)
    for label, call in (
        ("skip_reshape=True", lambda: A.attention_comfy_kitchen_int8(q, k, v, H, skip_reshape=True)),
        ("flat 3-D", lambda: A.attention_comfy_kitchen_int8(
            q.transpose(1, 2).reshape(1, n, H * D),
            k.transpose(1, 2).reshape(1, n, H * D),
            v.transpose(1, 2).reshape(1, n, H * D), H)),
    ):
        try:
            o = call()
            torch.cuda.synchronize()
            finite = bool(torch.isfinite(o).all())
            if n <= 16384:
                ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
                p = f"{psnr(o.reshape(1, n, H, D).transpose(1, 2), ref):6.2f} dB"
            else:
                p = "  (skip)"
            print(f"  {label:20}: shape={tuple(o.shape)} finite={finite} PSNR {p}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {label:20}: FAIL {type(e).__name__}: {str(e)[:110]}", flush=True)
            traceback.print_exc(limit=3)
    del q, k, v
    torch.cuda.empty_cache()
