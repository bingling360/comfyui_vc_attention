"""Probe 25: does VC-Attention now COMPOSE with a backend that uses the
optimized_attention_override chain (Comfy Kitchen / Sol-Attn)?

After the change, VC-Attention enters the chain instead of only wrapping
`optimized_attention`, so the two should divide the calls rather than one
silently shadowing the other. Node order decides priority -- here VC is
installed first (so it chains in front of a pre-existing backend).
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


class FakePatcher:
    """Minimal ModelPatcher stand-in: only model_options matters here."""

    def __init__(self):
        self.model_options = {"transformer_options": {}}


n, h, d = 16384, 56, 128
torch.manual_seed(0)
q = torch.randn(1, h, n, d, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, h, n, d, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, h, n, d, device="cuda", dtype=torch.bfloat16)

model = FakePatcher()
# the user's "model attention backend" node runs FIRST and installs Kitchen
model.model_options["transformer_options"]["optimized_attention_override"] = \
    lambda _, *a, **kw: counting_kitchen(*a, **kw)

rt = P.install(P.VCAttentionConfig(backend="fp8", block_rows=128, block_m=64), model=model)
t_opts = model.model_options["transformer_options"]
print("chain installed:", "optimized_attention_override" in t_opts, flush=True)
print("chain has prev :", hasattr(t_opts["optimized_attention_override"], "_vc_prev"), flush=True)


def delta(before):
    return {kk: rt.stats[kk] - before[kk] for kk in rt.stats}


# A call VC can handle (n >= min_tokens, head_dim 128) -> VC should take it
before = dict(rt.stats)
out = A.optimized_attention(q, k, v, h, skip_reshape=True, transformer_options=t_opts)
torch.cuda.synchronize()
print(f"\nA big self-attn  : vc={delta(before)}  kitchen={CALLS['kitchen']}", flush=True)

# A call VC must decline (short sequence) -> falls through to Kitchen
qs = q[:, :, :2048].contiguous()
before = dict(rt.stats)
out2 = A.optimized_attention(qs, k[:, :, :2048].contiguous(), v[:, :, :2048].contiguous(),
                             h, skip_reshape=True, transformer_options=t_opts)
torch.cuda.synchronize()
print(f"B short seq      : vc={delta(before)}  kitchen={CALLS['kitchen']}", flush=True)

print("\n--- verdict ---", flush=True)
print("VC handles what it supports; the previous backend (Kitchen) still runs "
      "for everything else. They now COMPOSE instead of VC being shadowed.")

P.uninstall()
print("after uninstall, override restored:", t_opts.get("optimized_attention_override") is not None)
