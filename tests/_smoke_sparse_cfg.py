"""Smoke test: the node exposes the sparsity knobs and the config wires them."""
import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
# .../comfyui_vc_attention/tests/ -> up two more levels to import the package.
sys.path.insert(0, os.path.dirname(_root))

import comfyui_vc_attention as pkg  # noqa: E402
from vc_attention.kernels.triton_attn import TritonConfig  # noqa: E402
from vc_attention.patch import VCAttentionConfig  # noqa: E402

nodes = pkg  # the package __init__ re-exports the node classes

c = VCAttentionConfig()
print("cfg:", c.enable_sparsity, c.tau, c.sparsity_min_tokens, c.sink_tokens,
      c.local_blocks, c.override_priority)
print("sink_blocks:", max(0, c.sink_tokens // max(1, c.block_rows)))
t = TritonConfig()
print("tcfg:", t.sparse, t.tau, t.local_blocks, t.sink_blocks, t.group)

it = pkg.VCAttentionMiniMaxH3.INPUT_TYPES()
req, opt = sorted(it["required"]), sorted(it["optional"])
print("required:", req)
print("optional:", opt)
assert "enable_sparsity" in req
for k in ("tau", "sparsity_min_tokens", "sink_tokens", "local_blocks"):
    assert k in opt, k
print("INPUT_TYPES OK")
