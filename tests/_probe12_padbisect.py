"""Probe 12: bisect the padded-sequence corruption.

  A. oracle reference @ n=6067       -> prepare semantics vs kernel
  B. oracle prepare + kernel @ 6067  -> _prepare_fast vs kernel masked path
  C. fast prepare restore identity   -> does dequant(prepare_fast(x)) == x?
  D. fast vs oracle field diffs
"""
import os
import sys

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/tcache12")
import torch
import triton

sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention")
from vc_attention.kernels.triton_attn import (Prepared, _vc_attn_fwd, prepare,
                                              restore)
from vc_attention.kernels.reference import RefConfig, vc_attention_reference

torch.manual_seed(0)
N, H, D, BR = 6067, 56, 128, 128
dev = "cuda"
q = torch.randn(1, H, N, D, device=dev, dtype=torch.bfloat16)
k = torch.randn(1, H, N, D, device=dev, dtype=torch.bfloat16)
v = torch.randn(1, H, N, D, device=dev, dtype=torch.bfloat16)
ref_sdpa = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())


def psnr(out, ref):
    mse = ((out.float() - ref.float()) ** 2).mean()
    if not torch.isfinite(out.float()).all():
        return float("nan")
    return float(10 * torch.log10(ref.float().pow(2).mean() / mse.clamp(min=1e-30)))


def run_kernel(p):
    out = torch.empty(1, H, N, D, device=dev, dtype=torch.bfloat16)
    grid = (triton.cdiv(p.n_pad, 128), H)
    _vc_attn_fwd[grid](p.q, p.q_scale, p.k, p.k_scale, p.v, p.v_scale, p.mu, out,
                       D ** -0.5, N, p.n_pad, D, BLOCK_M=128, BLOCK_N=128,
                       EXPCAST=False, BETA=-0.35, num_warps=8, num_stages=3)
    return out


# A. oracle reference
orc = vc_attention_reference(q, k, v, RefConfig(backend="fp8", enable_vsmooth=True,
                                                block_rows=BR, enable_expcast=False),
                             perm=None, scale=None)
print(f"A oracle reference      : PSNR {psnr(orc, ref_sdpa):7.2f} dB  "
      f"finite={bool(torch.isfinite(orc.float()).all())}")

# B. oracle prepare + fixed kernel
p_o = prepare(q, k, v, perm=None, block_rows=BR, hadamard=True)
out = run_kernel(p_o)
print(f"B oracle prep + kernel  : PSNR {psnr(out, ref_sdpa):7.2f} dB  n_pad={p_o.n_pad}")

# C. fast prepare restore identity (import from the installed module)
import importlib
import vc_attention.kernels.triton_attn as ta
importlib.reload(ta)
p_f = ta._prepare_fast(q, k, v, None, BR, True, True)
q_r, k_r, v_r = restore(p_f, BR)
v_in = v.reshape(H, N, D).float()
n_pad = p_f.n_pad
v_r2 = v_r.reshape(H, n_pad, D).float()[:N]
print(f"C fast restore v        : max abs err {float((v_r2 - v_in).abs().max()):.4f}  "
      f"rel {float((v_r2 - v_in).abs().mean() / v_in.abs().mean()):.4f}")
out2 = run_kernel(p_f)
print(f"  fast prep + kernel    : PSNR {psnr(out2, ref_sdpa):7.2f} dB  n_pad={p_f.n_pad}")

# D. field-by-field: oracle vs fast
for name in ("q_scale", "k_scale", "v_scale", "mu"):
    a = getattr(p_o, name).float()
    b = getattr(p_f, name).float()
    print(f"D {name:8s} shapes {tuple(a.shape)} vs {tuple(b.shape)}  "
          f"maxdiff {float((a - b).abs().max()):.3e}" if a.shape == b.shape else
          f"D {name:8s} SHAPE MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}")
for name in ("q", "k", "v"):
    a = getattr(p_o, name).view(torch.uint8).float()
    b = getattr(p_f, name).view(torch.uint8).float()
    frac = float((a != b).float().mean())
    print(f"D {name + ' codes':8s} mismatch fraction: {frac:.4f}")
