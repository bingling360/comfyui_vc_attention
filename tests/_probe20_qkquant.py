"""Probe 20: is NVFP4 accurate enough for Q/K, or does it wreck fidelity?

The paper's 4-bit config uses NVFP4 for QK as well as PV. Q/K are the sensitive
operands (VC-Attention's notes: per-token fp8 + Hadamard gets QK error to 3.6%;
without it, 55%). NVFP4 has only 2 mantissa bits (magnitudes 0,.5,1,1.5,2,3,4,6)
vs e4m3's 3, so before writing an NVFP4 QK kernel we measure the end-to-end
attention PSNR of fp8-QK vs nvfp4-QK, with the PV fixed.

Pure PyTorch (no Triton): quantise Q/K, run the attention in fp32.
"""
import os
import sys

import torch

sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention")
sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention/tests")
from bench_attention import h3_like, psnr  # noqa: E402
from vc_attention import quant as Q  # noqa: E402
from vc_attention.kernels.triton_attn import _hadamarian  # noqa: E402


def dq_fp8(x):
    codes, sc = Q.quantize_e4m3(x, dim=-1)
    return Q.dequantize_e4m3(codes, sc)


def dq_nvfp4(x):
    """per-16-group e2m1 + e4m3 microscale, along the last dim."""
    D = x.shape[-1]
    xb = x.reshape(*x.shape[:-1], D // 16, 16)
    amax = xb.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    micro = Q.e4m3_encode((amax / 6.0).squeeze(-1))
    mv = Q.e4m3_decode(micro).unsqueeze(-1).clamp(min=1e-30)
    codes = Q.fp4_encode((xb / mv).clamp(-6, 6))
    return (Q.fp4_decode(codes) * mv).reshape(*x.shape[:-1], D)


def dq_mxfp4(x):
    """per-32-group e2m1 + e8m0 (power-of-two) scale, along the last dim."""
    D = x.shape[-1]
    xb = x.reshape(*x.shape[:-1], D // 32, 32)
    amax = xb.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    e = torch.ceil(torch.log2(amax / 6.0))
    mv = torch.pow(2.0, e).clamp(min=1e-30)
    codes = Q.fp4_encode((xb / mv).clamp(-6, 6))
    return (Q.fp4_decode(codes) * mv).reshape(*x.shape[:-1], D)


def attn(q, k, v, scale):
    s = torch.matmul(q, k.transpose(-1, -2)) * scale
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, v)


def main():
    h, n, d = 8, 4096, 128
    q, k, v = h3_like(n, h, d, "cuda")
    qf, kf, vf = q.reshape(h, n, d).float(), k.reshape(h, n, d).float(), v.reshape(h, n, d).float()
    scale = d ** -0.5

    ref = attn(qf, kf, vf, scale)

    Hm = _hadamarian(d, torch.device("cuda")).float()
    qh = qf @ Hm
    kh = kf @ Hm
    kh = kh - kh.mean(dim=1, keepdim=True)          # K smoothing, as prepare does

    variants = {
        "fp32 (no quant)": (qh, kh),
        "fp8  per-token": (dq_fp8(qh), dq_fp8(kh)),
        "nvfp4 per-16 ": (dq_nvfp4(qh), dq_nvfp4(kh)),
        "mxfp4 per-32 ": (dq_mxfp4(qh), dq_mxfp4(kh)),
    }
    print(f"heads={h} tokens={n} head_dim={d}", flush=True)
    for name, (qq, kk) in variants.items():
        o = attn(qq, kk, vf, scale)
        qk_err = float((qq @ kk.transpose(-1, -2) - qh @ kh.transpose(-1, -2)).norm()
                       / (qh @ kh.transpose(-1, -2)).norm())
        print(f"  {name}:  attn PSNR {psnr(o, ref):6.2f} dB   QK^T rel-err {qk_err:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
