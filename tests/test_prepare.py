"""Validate the layout the Triton kernel reads.

The kernel itself needs a GPU to compile, but everything it *consumes* is
produced on the host by :func:`prepare`, and that can be checked on CPU. If
these assertions hold, a wrong result from the kernel is a kernel bug, not a
layout bug.

This also pins the padded-token invariant that bit once already: V is padded to
a whole number of value blocks, so Q, K and their scales must be padded by the
same amount, or the per-(b, h) base offsets inside the kernel drift apart.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vc_attention.hadamard import fwht  # noqa: E402
from vc_attention.kernels.triton_attn import prepare, restore  # noqa: E402

torch.manual_seed(5)
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


def relerr(a, b):
    return float((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp(min=1e-12))


B, H, N, D = 2, 4, 1000, 128       # 1000 is deliberately not a multiple of 128
BR = 128
G = B * H

q = torch.randn(B, H, N, D)
k = torch.randn(B, H, N, D) * 0.8
region = torch.randint(0, 8, (G, N))
means = torch.randn(G, 8, D) * 2.0
v = means.gather(1, region.unsqueeze(-1).expand(G, N, D))
v = (v + torch.randn(G, N, D) * 0.3).reshape(B, H, N, D)

perm = torch.argsort(region, dim=-1, stable=True).to(torch.int32)   # any permutation
pidx = perm.to(torch.int64).unsqueeze(-1).expand(G, N, D)
v_perm = v.reshape(G, N, D).gather(1, pidx)
k_perm = k.reshape(G, N, D).gather(1, pidx)

print(f"\nprepare(): B={B} H={H} N={N} D={D} block_rows={BR}")
p = prepare(q, k, v, perm=perm, block_rows=BR)

print("\n[1] Shape and dtype contract")
check("n_pad is a multiple of block_rows", p.n_pad % BR == 0 and p.n_pad >= N,
      f"N={N} -> n_pad={p.n_pad}")
check("q/k/v are float8_e4m3fn",
      p.q.dtype == torch.float8_e4m3fn and p.k.dtype == torch.float8_e4m3fn
      and p.v.dtype == torch.float8_e4m3fn)
check("q/k/v padded to n_pad",
      p.q.shape == (B, H, p.n_pad, D) and p.k.shape == (B, H, p.n_pad, D)
      and p.v.shape == (B, H, p.n_pad, D), str(tuple(p.q.shape)))
check("scale tensors share the padded length",
      p.q_scale.shape == (B, H, p.n_pad) and p.k_scale.shape == (B, H, p.n_pad))
nb = p.n_pad // BR
check("v_scale and mu are per (block, channel)",
      p.v_scale.shape == (B, H, nb, D) and p.mu.shape == (B, H, nb, D), f"nb={nb}")
check("tensors are contiguous", p.q.is_contiguous() and p.v_scale.is_contiguous())

print("\n[2] Padding does not corrupt the real rows")
qh, kh, vh = restore(p, block_rows=BR)
e_q = relerr(qh[:, :N], fwht(q.reshape(G, N, D)))
check("restored q == rotated q on the first N rows", e_q < 0.03, f"rel err {e_q:.4f}")
e_v = relerr(vh[:, :N], v_perm)
check("restored v == permuted v on the first N rows", e_v < 0.03, f"rel err {e_v:.4f}")

print("\n[3] The value scale is per block, and mu is stored pre-divided by it")
spread = float(p.v_scale.reshape(-1, D).std() / p.v_scale.reshape(-1, D).mean())
check("v_scale varies across blocks", spread > 0.05, f"std/mean = {spread:.3f}")
v_pad = torch.nn.functional.pad(v_perm, (0, 0, 0, p.n_pad - N))
mu_true = v_pad.reshape(G, nb, BR, D).mean(dim=2)
mu_back = p.mu.reshape(G, nb, D) * p.v_scale.reshape(G, nb, D)
check("mu * v_scale == the true block means",
      float((mu_back - mu_true).abs().max()) < 1e-4,
      f"max diff {float((mu_back - mu_true).abs().max()):.2e}")

print("\n[4] No-op permutation keeps everything identical")
p0 = prepare(q, k, v, perm=None, block_rows=BR)
e0 = relerr(restore(p0, BR)[2][:, :N], v.reshape(G, N, D))
check("identity-prepared v reconstructs v", e0 < 0.03, f"rel err {e0:.4f}")

print("\n[5] Permuting K and V leaves the attention result untouched")
sc = D ** -0.5
qq = q.reshape(G, N, D)
o_ref = torch.softmax(qq @ k.reshape(G, N, D).transpose(-1, -2) * sc, -1) @ v.reshape(G, N, D)
o_perm = torch.softmax(qq @ k_perm.transpose(-1, -2) * sc, -1) @ v_perm
check("P'V' == PV", float((o_ref - o_perm).abs().max()) < 1e-3,
      f"max diff {float((o_ref - o_perm).abs().max()):.2e}")

print(f"\n=== {ok} passed, {fail} failed ===")
sys.exit(1 if fail else 0)
