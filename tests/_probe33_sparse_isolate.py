"""Probe 33: isolate the Phase 2 failure.

probe32 showed SPARSE=True gives the SAME (wrong) PSNR for tau=-10, 1.0 and 1.3,
so the routing is not responding to tau. Two candidate causes:
  (a) the kept branch itself broke when I restructured the loop, or
  (b) the routing/approximation is wrong.
Force keep=True everywhere by three independent routes -- if any of them
reproduces SPARSE=False exactly, the kept branch is fine and it is (b).
"""
import os
import sys

import torch

sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention")
sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention/tests")

from bench_attention import h3_like, psnr  # noqa: E402
from vc_attention.grouping import GroupingConfig, build_permutation  # noqa: E402
from vc_attention.kernels.triton_attn import TritonConfig, vc_attention_triton  # noqa: E402

H, D, N = 56, 128, 16384
q, k, v = h3_like(N, H, D, "cuda")
ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
perm = build_permutation(v.reshape(H, N, D), GroupingConfig(block_rows=128, iters=3)).perm

base = dict(block_m=64, block_n=128, num_warps=4, num_stages=2)
cases = [
    ("baseline SPARSE=False      ", dict(sparse=False)),
    ("SPARSE, LOCAL=100000       ", dict(sparse=True, tau=1.3, local_blocks=100000)),
    ("SPARSE, SINK=100000        ", dict(sparse=True, tau=1.3, sink_blocks=100000)),
    ("SPARSE, tau=-1000          ", dict(sparse=True, tau=-1000.0)),
    ("SPARSE, tau=1.3 (real)     ", dict(sparse=True, tau=1.3)),
]
outs = {}
for name, kw in cases:
    cfg = TritonConfig(**base, **kw)
    try:
        o = vc_attention_triton(q, k, v, perm=perm, cfg=cfg)
        torch.cuda.synchronize()
        outs[name] = o
        print(f"  {name}: PSNR {psnr(o, ref):6.2f} dB", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  {name}: FAILED {type(e).__name__}: {str(e)[:140]}", flush=True)

b = outs.get("baseline SPARSE=False      ")
for name, o in outs.items():
    if b is None or name == "baseline SPARSE=False      ":
        continue
    rel = float((o.float() - b.float()).norm() / b.float().norm())
    print(f"  {name} vs baseline rel-err {rel:.3e}  {'MATCH' if rel < 1e-5 else 'differs'}", flush=True)
