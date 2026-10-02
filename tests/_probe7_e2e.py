"""Probe 7: end-to-end check of vc_attention_triton with the fallback trap armed
in the right place (the reference module attribute, which the function-local
import reads at call time)."""
import os
import sys
import time

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/tcache7")
import torch

sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention")
import vc_attention.kernels.reference as ref
from vc_attention.kernels.triton_attn import TritonConfig, vc_attention_triton
from vc_attention.grouping import GroupingConfig, build_permutation


def _boom(*a, **k):
    raise RuntimeError("SILENT FALLBACK TO REFERENCE HAPPENED")


ref.vc_attention_reference = _boom  # any fallback now raises loudly

torch.manual_seed(0)
T, H, D = 2048, 56, 128
dev = "cuda"
q = torch.randn(1, H, T, D, device=dev, dtype=torch.bfloat16)
k = torch.randn(1, H, T, D, device=dev, dtype=torch.bfloat16)
v = torch.randn(1, H, T, D, device=dev, dtype=torch.bfloat16)
ref_sdpa = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())


def psnr(out, r):
    mse = ((out.float() - r.float()) ** 2).mean()
    if not torch.isfinite(out.float()).all():
        return float("nan")
    return float(10 * torch.log10(r.float().pow(2).mean() / mse.clamp(min=1e-30)))


perm = build_permutation(
    v.reshape(H, T, D), GroupingConfig(block_rows=128, iters=3),
).perm

cfg = TritonConfig(block_m=128, block_n=128)
out_g = vc_attention_triton(q, k, v, perm=perm, cfg=cfg)
print(f"vc_attention_triton +V-Smooth: PSNR {psnr(out_g, ref_sdpa):.2f} dB  "
      f"finite={bool(torch.isfinite(out_g.float()).all())}")

out_n = vc_attention_triton(q, k, v, perm=None, cfg=cfg)
print(f"vc_attention_triton  no group: PSNR {psnr(out_n, ref_sdpa):.2f} dB  "
      f"finite={bool(torch.isfinite(out_n.float()).all())}")
print(f"grouping gain: {psnr(out_g, ref_sdpa) - psnr(out_n, ref_sdpa):+.2f} dB")
