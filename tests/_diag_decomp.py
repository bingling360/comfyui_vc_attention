"""Diagnostic: where low-bit attention error comes from, and which Q/K scheme
puts V back in the lead (which is the premise of V-Smooth)."""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vc_attention import quant as Q
from vc_attention.grouping import GroupingConfig, build_permutation
from vc_attention.hadamard import fwht

torch.manual_seed(3)
H, N, D = 2, 2048, 128


def channel_spikes(x, frac=0.03, width=4, mag=12.0):
    """Outliers in a few channels of a few tokens -- the shape real activations take."""
    m = (torch.rand(x.shape[0], x.shape[1], 1) < frac).float()
    ch = torch.randint(0, D - width, (x.shape[0], x.shape[1], 1))
    idx = ch.expand(x.shape[0], x.shape[1], width) + torch.arange(width)
    return x + m * torch.zeros_like(x).scatter(2, idx, mag)


region = torch.randint(0, 8, (H, N))
rm = torch.randn(H, 8, D) * 2.0
v = rm.gather(1, region.unsqueeze(-1).expand(H, N, D)) + torch.randn(H, N, D) * 0.3
v[:, : N // 16] *= 0.3
v[:, N // 16 : N // 16 + N // 8] *= 3.0
v = v + (torch.rand(H, N, 1) < 0.01).float() * 15.0

q = channel_spikes(torch.randn(H, N, D))
k = channel_spikes(torch.randn(H, N, D) * 0.8)

P = torch.softmax((q @ k.transpose(-1, -2)) / D ** 0.5, -1)
O = P @ v


def rel(e):
    return float(e.pow(2).mean().sqrt() / O.pow(2).mean().sqrt())


def qk_error(use_h, k_per_token):
    qh, kh = (fwht(q), fwht(k)) if use_h else (q, k)
    if k_per_token:
        qc, sq = Q.quantize_e4m3(qh, dim=-1)
        kc, sk = Q.quantize_e4m3(kh, dim=-1)
        qd, kd = Q.dequantize_e4m3(qc, sq), Q.dequantize_e4m3(kc, sk)
        S = (qd @ kd.transpose(-1, -2)) / D ** 0.5
        return rel(torch.softmax(S, -1) @ v - O)
    kt = kh - kh.mean(1, keepdim=True)
    kc, sk = Q.quantize_e4m3(kt, dim=1)
    qd = Q.dequantize_e4m3(*Q.quantize_e4m3(qh * sk, dim=-1))
    S = (qd @ Q.dequantize_e4m3(kc, sk).transpose(-1, -2)) / D ** 0.5
    return rel(torch.softmax(S, -1) @ v - O)


def v_error(perm=None, demean=True):
    """Permute K and V together -- only then is P'V' == PV."""
    Px = P if perm is None else P.gather(-1, perm.unsqueeze(1).expand(H, N, N))
    x = v if perm is None else v.gather(1, perm.unsqueeze(-1).expand_as(v))
    if not demean:
        qa, sa = Q.quantize_e4m3(x, dim=1)
        return rel(Px @ Q.dequantize_e4m3(qa, sa) - O)
    mu = x.reshape(H, -1, 128, D).mean(dim=2).repeat_interleave(128, dim=1)
    qc, sc = Q.quantize_e4m3_blocks(x - mu, 128)
    return rel(Px @ (Q.dequantize_e4m3_blocks(qc, sc, 128) + mu) - O)


def p_error():
    Pq, Ps = Q.quantize_e4m3(P, dim=-1, scale=torch.full_like(P[..., :1], 1 / 448.0))
    return rel(Q.dequantize_e4m3(Pq, Ps) @ v - O)


print("relative output error by component (lower is better)")
for use_h in (False, True):
    for per_tok in (False, True):
        e = qk_error(use_h, per_tok)
        tag = f"QK {'Hadamard' if use_h else 'raw     '} + K {'per-token' if per_tok else 'per-chan'}"
        print(f"  {tag} : {e*100:7.3f} %   {-20*math.log10(e):6.2f} dB")
print()
e_g, e_s = v_error(demean=False), v_error()
print(f"  V per-channel global   : {e_g*100:7.3f} %   {-20*math.log10(e_g):6.2f} dB")
print(f"  V block demean (seq)   : {e_s*100:7.3f} %   {-20*math.log10(e_s):6.2f} dB")
print(f"  P per-row E4M3         : {p_error()*100:7.3f} %   {-20*math.log10(p_error()):6.2f} dB")
perm = build_permutation(v, GroupingConfig(block_rows=128, iters=6)).perm.to(torch.int64)
e_p = v_error(perm)
print(f"  V block demean + group : {e_p*100:7.3f} %   {-20*math.log10(e_p):6.2f} dB")
print(f"  -> grouping gain on V  : {20*math.log10(e_s / e_p):+.2f} dB")
