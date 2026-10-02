"""Probe 9: validate the by-value-import rebind against the real comfy tree.

Imports comfy.ldm.minimax.model exactly like ComfyUI does (capturing the
by-value `optimized_attention` binding), installs the VC-Attention patch, and
checks that H3's binding now routes through the hook:

  1. rebind: h3model.optimized_attention is no longer the original function
  2. big 4-D call (skip_reshape=True convention, n >= min_tokens):
     output equals a direct vc_attention_triton call; stats["used"] increments
  3. small call (n < min_tokens): skipped, output bit-identical to the original
"""
import os
import sys

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/tcache9")
import torch

sys.path.insert(0, "/root/ComfyUI")
sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention")

import comfy.ldm.minimax.model as h3model  # noqa: E402  (by-value binding happens here)
from vc_attention.patch import VCAttentionConfig, get_runtime, install  # noqa: E402
from vc_attention.kernels.triton_attn import TritonConfig, vc_attention_triton  # noqa: E402

orig = h3model.optimized_attention
install(VCAttentionConfig(enabled=True, backend="fp8", total_steps_hint=8,
                          enable_vsmooth=False), model=None)
rebound = h3model.optimized_attention

print(f"[1] original captured: {orig.__module__}.{orig.__qualname__}")
print(f"[1] h3 binding rebound: {rebound is not orig}")
import comfy.ldm.modules.attention as comfy_att
print(f"[1] defining module rebound: {comfy_att.optimized_attention is not orig}")

torch.manual_seed(0)
H, D = 56, 128
dev = "cuda"
rt = get_runtime()
print(f"[i] backend resolved: {rt.backend}")

# -- 2. big call through H3's own binding ------------------------------------
T = 16384
q = torch.randn(1, H, T, D, device=dev, dtype=torch.bfloat16)
k = torch.randn(1, H, T, D, device=dev, dtype=torch.bfloat16)
v = torch.randn(1, H, T, D, device=dev, dtype=torch.bfloat16)

out_hook = rebound(q, k, v, H, mask=None, skip_reshape=True, transformer_options={})
out_direct = vc_attention_triton(q, k, v, perm=None, cfg=TritonConfig(block_m=128, block_n=128))
same = torch.equal(out_hook.view(1, T, H, D).transpose(1, 2), out_direct)
print(f"[2] n={T}: hook shape {tuple(out_hook.shape)}  == direct VC call: {same}")
print(f"[2] stats: {rt.stats}")

# -- 3. small call must defer to the original ---------------------------------
T2 = 1024
q2 = torch.randn(1, H, T2, D, device=dev, dtype=torch.bfloat16)
k2 = torch.randn(1, H, T2, D, device=dev, dtype=torch.bfloat16)
v2 = torch.randn(1, H, T2, D, device=dev, dtype=torch.bfloat16)
out_small_hook = rebound(q2, k2, v2, H, mask=None, skip_reshape=True, transformer_options={})
out_small_orig = orig(q2, k2, v2, H, mask=None, skip_reshape=True, transformer_options={})
bit_equal = torch.equal(out_small_hook, out_small_orig)
print(f"[3] n={T2}: skipped -> bit-identical to original: {bit_equal}")
print(f"[3] stats: {rt.stats}")

# -- 2b. container-wrapped call (comfy 0.38 / MiniMax-H3 convention) -----------
from comfy.ldm.modules.attention import AttentionTensorContainer  # noqa: E402

qc = AttentionTensorContainer(q)   # container payload: (1, heads, S, D), as model.py wraps it
kc = AttentionTensorContainer(k)
vc_ = AttentionTensorContainer(v)
out_c = rebound(qc, kc, vc_, H, mask=None, skip_reshape=True, transformer_options={})
same_c = torch.equal(out_c.view(1, T, H, D).transpose(1, 2), out_direct)
print(f"[2b] n={T} container-wrapped: shape {tuple(out_c.shape)}  == direct VC call: {same_c}")
print(f"[2b] stats: {rt.stats}")

# -- 4. registered override owns the call; VC must defer -----------------------
used_before = rt.stats["used"]
ov = lambda fn, *a, **k: fn(*a, **k)  # noqa: E731  (pass-through override)
q2c = AttentionTensorContainer(q2)
k2c = AttentionTensorContainer(k2)
v2c = AttentionTensorContainer(v2)
out_ov = rebound(q2c, k2c, v2c, H, mask=None, skip_reshape=True,
                 transformer_options={"optimized_attention_override": ov})
out_small_orig2 = orig(q2, k2, v2, H, mask=None, skip_reshape=True, transformer_options={})
deferred = torch.equal(out_ov, out_small_orig2) and rt.stats["used"] == used_before
print(f"[4] override registered: deferred to original: {deferred}")

ok = (rebound is not orig) and same and same_c and bit_equal and deferred
print(f"[{'PASS' if ok else 'FAIL'}] hook rebind end-to-end")
sys.exit(0 if ok else 1)
