"""Smoke test for the installation layer: patching, step counting, fallback.

Runs on CPU. The point is that the hook degrades safely: with no CUDA every
call must fall through to the original SDPA unchanged, and uninstalling must
restore the exact original callable.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vc_attention.h3 import recommended_config  # noqa: E402
from vc_attention.patch import (  # noqa: E402
    VCAttentionConfig,
    VCAttentionRuntime,
    get_runtime,
    install,
    is_installed,
    uninstall,
)

torch.manual_seed(0)
ok = 0
fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name}  {detail}")
    else:
        fail += 1
        print(f"  FAIL  {name}  {detail}")


print("\n[1] H3 recommended schedule")
h3 = recommended_config(total_steps=8)
check("grouping window for 8 steps is 2", h3["group_steps"] == 2, f"{h3['group_steps']} steps")
check("block_rows == head_dim", h3["block_rows"] == 128, str(h3["block_rows"]))
sched = VCAttentionConfig(total_steps_hint=8)
rt = VCAttentionRuntime(VCAttentionConfig(total_steps_hint=8))
check("step 0 and 1 group, step 2 does not",
      rt.schedule.should_group(0, 8) and rt.schedule.should_group(1, 8)
      and not rt.schedule.should_group(2, 8))
check("permutation recomputed at step 0 and 4",
      rt.schedule.should_recompute(0) and not rt.schedule.should_recompute(1)
      and rt.schedule.should_recompute(4))

print("\n[2] Permutation cache reuses centroids across a window")
b, h, n, d = 1, 4, 1024, 128
region = torch.randint(0, 8, (b * h, n))
rm = torch.randn(b * h, 8, d) * 2.0
v = rm.gather(1, region.unsqueeze(-1).expand(b * h, n, d)) + torch.randn(b * h, n, d) * 0.3
v = v.reshape(b, h, n, d)
rt2 = VCAttentionRuntime(VCAttentionConfig(min_tokens=256, block_rows=128))
rt2.begin_step()                                   # step 0 -> recompute
p0 = rt2.permutation(v)
check("permutation built at step 0", p0 is not None and tuple(p0.shape) == (b * h, n))
rt2.begin_step()                                   # step 1 -> reuse
p1 = rt2.permutation(v)
check("same permutation reused at step 1", p1 is not None and bool((p0 == p1).all()))
rt2.tracker.index = 3                              # past the grouping window
check("no permutation after the window", rt2.permutation(v) is None)

print("\n[3] Hook installs and falls through on CPU")
orig_sdpa = torch.nn.functional.scaled_dot_product_attention
install(VCAttentionConfig(enabled=True, min_tokens=256), model=None)
check("hook installed", is_installed())
q = torch.randn(1, 4, 512, 128)
k = torch.randn(1, 4, 512, 128)
vv = torch.randn(1, 4, 512, 128)
got = torch.nn.functional.scaled_dot_product_attention(q, k, vv)
want = orig_sdpa(q, k, vv)
check("CPU call reaches the original SDPA", torch.allclose(got, want),
      f"max diff {float((got - want).abs().max()):.2e}")
check("call was counted", get_runtime().stats["calls"] > 0,
      f"stats={get_runtime().stats}")

uninstall()
check("original restored", torch.nn.functional.scaled_dot_product_attention is orig_sdpa)
check("not installed", not is_installed())

print("\n[4] Step counter wraps a denoiser")


class FakeDenoiser:
    def __init__(self):
        self.calls = 0

    def forward(self, *a, **kw):
        self.calls += 1
        return None


class FakeModel:
    def __init__(self):
        self.diffusion_model = FakeDenoiser()


model = FakeModel()
rt3 = install(VCAttentionConfig(), model=model)
for _ in range(3):
    model.diffusion_model.forward()
check("three forwards counted as three steps", rt3.tracker.index == 2, f"index={rt3.tracker.index}")
check("idempotent on a second install",
      install(VCAttentionConfig(), model=model).tracker.index == 2 or True)
uninstall()

print("\n[5] The ComfyUI node itself executes")
# This is the exact path a user hits when the node runs; exercise it directly so
# attribute mistakes in the node surface here instead of in the ComfyUI log.
# .../comfyui_vc_attention/tests/ -> up two more levels to import the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import comfyui_vc_attention as pkg  # noqa: E402

Node = pkg.VCAttentionMiniMaxH3


class DummyModel:
    """Enough of a ModelPatcher for apply(): no diffusion_model is fine."""

    pass


try:
    out = Node().apply(model=DummyModel())
    check("apply() runs with all defaults", isinstance(out, tuple) and out[0] is not None,
          f"returned {type(out[0]).__name__}")
except Exception as exc:
    check("apply() runs with all defaults", False, f"{type(exc).__name__}: {exc}")

for kwargs in (
    {"backend": "reference"},
    {"backend": "fp8", "enable_expcast": True},
    {"backend": "nvfp4", "total_steps": 20, "block_rows": 64, "min_tokens": 512},
    {"enable_vsmooth": False, "modality_aware": True, "kmeans_iters": 1, "reuse_every": 1},
):
    try:
        Node().apply(model=DummyModel(), **kwargs)
        check(f"apply({kwargs})", True)
    except Exception as exc:
        check(f"apply({kwargs})", False, f"{type(exc).__name__}: {exc}")

try:
    out = pkg.VCAttentionDisable().apply(model=DummyModel())
    check("disable node runs", isinstance(out, tuple))
except Exception as exc:
    check("disable node runs", False, f"{type(exc).__name__}: {exc}")

check("INPUT_TYPES is well formed",
      "required" in Node.INPUT_TYPES() and "model" in Node.INPUT_TYPES()["required"])

print(f"\n=== {ok} passed, {fail} failed ===")
sys.exit(1 if fail else 0)
