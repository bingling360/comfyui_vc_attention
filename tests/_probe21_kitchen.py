"""Probe 21: does VC-Attention stack with the Comfy Kitchen attention node?

The user's setup is NOT the global `--use-ck-attention` flag: they use the
"model attention backend" node (comfy_extras/nodes_model_advanced.py), which
calls ModelPatcher.set_model_optimized_attention(fn). That does not touch
`optimized_attention` -- it installs

    model_options["transformer_options"]["optimized_attention_override"] = fn

where fn is `def override(_, *args, **kwargs): return fn_impl(*args, **kwargs)`
(i.e. it drops the wrapped function). ComfyUI's `wrap_attn` dispatch checks
that key FIRST, before `preferred_attention.function` and before the wrapped
function itself.

VC-Attention wraps `optimized_attention` from the outside and defers when the
key is present. This probe uses the REAL `attention_comfy_kitchen_int8` (not a
stub that falls back to SDPA -- that would be intercepted by VC's SDPA-level
hook and give a misleading answer).
"""
import os
import sys

import torch

sys.path.insert(0, "/root/ComfyUI")
sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention")

import comfy.ldm.modules.attention as A  # noqa: E402
from vc_attention import patch as P  # noqa: E402

CALLS = {"kitchen": 0}
_real_kitchen = A.attention_comfy_kitchen_int8


def counting_kitchen(*args, **kwargs):
    CALLS["kitchen"] += 1
    return _real_kitchen(*args, **kwargs)


def override(_, *args, **kwargs):
    """Exactly what ModelPatcher.set_model_optimized_attention builds."""
    return counting_kitchen(*args, **kwargs)


print("optimized_attention before install:", A.optimized_attention.__name__, flush=True)
rt = P.install(P.VCAttentionConfig(backend="fp8", block_rows=128, block_m=64))
print("optimized_attention after  install:", A.optimized_attention.__name__,
      "(functools.wraps keeps the name; the patch is in place)", flush=True)
print("fast_path_available:", P.fast_path_available(), flush=True)

n, h, d = 16384, 56, 128
torch.manual_seed(0)
q = torch.randn(1, h, n, d, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, h, n, d, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, h, n, d, device="cuda", dtype=torch.bfloat16)


def delta(before):
    return {kk: rt.stats[kk] - before[kk] for kk in rt.stats}


before = dict(rt.stats)
out_a = A.optimized_attention(q, k, v, h, skip_reshape=True)
torch.cuda.synchronize()
print(f"\nA no backend node : out={tuple(out_a.shape)}  vc={delta(before)}  "
      f"kitchen={CALLS['kitchen']}", flush=True)

before = dict(rt.stats)
t_opts = {"optimized_attention_override": override}
out_b = A.optimized_attention(q, k, v, h, skip_reshape=True, transformer_options=t_opts)
torch.cuda.synchronize()
dB = delta(before)
print(f"B kitchen node    : out={tuple(out_b.shape)}  vc={dB}  "
      f"kitchen={CALLS['kitchen']}", flush=True)

print("\n--- verdict ---", flush=True)
if CALLS["kitchen"] == 1 and dB["used"] == 0:
    print("Kitchen override WINS. VC-Attention is bypassed entirely (it defers "
          "on `optimized_attention_override`), so the two do NOT stack -- with "
          "the Kitchen node active, VC-Attention is silently inactive.")
elif CALLS["kitchen"] == 1 and dB["used"] == 1:
    print("BOTH ran in the same call -- investigate the call path.")
else:
    print(f"unexpected: calls={CALLS} vc={dB}")

P.uninstall()
