"""Probe 11: non-multiple-of-block_rows sequences (the real H3 packing case).

Real packed sequences (text + audio + video) are rarely multiples of 128. The
first live run never reached the kernel because F.pad rejected fp8 tensors in
that path and the router swallowed the error. This probe exercises the pad
path end-to-end.
"""
import os
import sys

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/tcache11")
import torch

sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention")
from vc_attention.kernels.triton_attn import TritonConfig, vc_attention_triton  # noqa: E402

torch.manual_seed(0)
H, D = 56, 128
dev = "cuda"


def psnr(out, ref):
    mse = ((out.float() - ref.float()) ** 2).mean()
    if not torch.isfinite(out.float()).all():
        return float("nan")
    return float(10 * torch.log10(ref.float().pow(2).mean() / mse.clamp(min=1e-30)))


cfg = TritonConfig(block_m=128, block_n=128)
ok = True
for n in (6067, 8321, 16384, 50003):   # odd, just-over-8K, multiple, big+odd
    q = torch.randn(1, H, n, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, H, n, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, H, n, D, device=dev, dtype=torch.bfloat16)
    ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
    out = vc_attention_triton(q, k, v, perm=None, cfg=cfg)
    p = psnr(out, ref)
    shape_ok = tuple(out.shape) == (1, H, n, D)
    good = torch.isfinite(out.float()).all() and p > 20 and shape_ok
    ok = ok and bool(good)
    print(f"n={n:6d} (pad={(-n) % 128:3d}): PSNR {p:6.2f} dB  shape_ok={shape_ok}  "
          f"finite={bool(torch.isfinite(out.float()).all())}  -> {'OK' if good else 'BAD'}")

print(f"[{'PASS' if ok else 'FAIL'}] pad path")
sys.exit(0 if ok else 1)
