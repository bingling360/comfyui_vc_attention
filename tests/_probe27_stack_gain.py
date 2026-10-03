"""Probe 27: does chaining VC-Attention with Kitchen give ANY speedup?

The chain makes VC and Kitchen coexist, but "coexist" means whoever is at the
FRONT of the chain handles the call -- so the answer depends entirely on which
backend is faster, and VC is not.

Measured through the real dispatch path (optimized_attention):
  1. Kitchen alone
  2. VC with override_priority="front"  -> VC wins  (the pessimistic case)
  3. VC with override_priority="defer"  -> VC stays out (the new default)
  4. VC installed first, Kitchen node second -> Kitchen wins
"""
import os
import sys

import torch

sys.path.insert(0, "/root/ComfyUI")
sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention")
sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention/tests")

import comfy.ldm.modules.attention as A  # noqa: E402
from bench_attention import h3_like, timeit  # noqa: E402
from vc_attention import patch as P  # noqa: E402

H, D, N = 56, 128, 16384
_q, _k, _v = h3_like(N, H, D, "cuda")
_KIT = A.attention_comfy_kitchen_int8


class MP:
    def __init__(self):
        self.model_options = {"transformer_options": {}}


def kitchen_override(_, *a, **kw):
    return _KIT(*a, **kw)


def timed(t_opts, label):
    fn = lambda: A.optimized_attention(_q, _k, _v, H, skip_reshape=True, transformer_options=t_opts)
    fn()
    torch.cuda.synchronize()
    t = timeit(fn, warmup=2, repeat=5)
    print(f"  {label:40}: {t:8.2f} ms", flush=True)
    return t


def cfg(prio):
    return P.VCAttentionConfig(backend="fp8", block_rows=128, block_m=64,
                               override_priority=prio)


# 1. Kitchen alone
P.uninstall()
m1 = MP()
m1.model_options["transformer_options"]["optimized_attention_override"] = kitchen_override
t1 = timed(m1.model_options["transformer_options"], "1. Kitchen alone")

# 2. VC chained IN FRONT of Kitchen -> VC wins
m2 = MP()
m2.model_options["transformer_options"]["optimized_attention_override"] = kitchen_override
rt2 = P.install(cfg("front"), model=m2)
t2 = timed(m2.model_options["transformer_options"], "2. VC front  (VC wins)")

# 3. VC with the default "defer" -> stays out of the chain
P.uninstall()
m3 = MP()
m3.model_options["transformer_options"]["optimized_attention_override"] = kitchen_override
rt3 = P.install(cfg("defer"), model=m3)
t3 = timed(m3.model_options["transformer_options"], "3. VC defer  (stays out)")

# 4. VC first, then Kitchen node -> Kitchen ends up in front
P.uninstall()
m4 = MP()
P.install(cfg("defer"), model=m4)
m4.model_options["transformer_options"]["optimized_attention_override"] = kitchen_override
t4 = timed(m4.model_options["transformer_options"], "4. VC then Kitchen (Kitchen wins)")

print(f"\n  VC handled calls: front={rt2.stats['used']}  defer={rt3.stats['used']}")
print(f"\n  Kitchen alone          : {t1:7.2f} ms")
print(f"  VC front (VC wins)     : {t2:7.2f} ms  -> {t1/t2:.2f}x of kitchen alone")
print(f"  VC defer (stays out)   : {t3:7.2f} ms  -> {t1/t3:.2f}x")
print(f"  VC then Kitchen        : {t4:7.2f} ms  -> {t1/t4:.2f}x")
P.uninstall()
