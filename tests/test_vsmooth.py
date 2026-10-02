"""Does V-Smooth actually help? End-to-end check on an H3-shaped workload.

How the synthetic sequence is built
-----------------------------------
Real DiT value tensors are not i.i.d. noise. They cluster: tokens belonging to
the same content region (a face, a wall, the sky) share a value profile, and
those regions are interleaved once the (t, h, w) latent grid is flattened into
the packed 1-D sequence. Three properties are modelled:

  * **Semantic clusters** whose members are scattered along the sequence. This
    is what k-means recovers, and it is why the block mean then fits.
  * **Modality magnitudes**: H3 packs text, audio and video rows into one
    sequence, and their value scales differ by an order of magnitude.
  * **Scattered outliers**, which follow no fixed channel or position.

The mechanism being tested: E4M3 error is *relative* (~3-6% of the value), so
error scales with the magnitude of what is quantised. V-Smooth quantises the
residual after subtracting the block mean, so a block whose members share a
profile leaves a tiny residual and therefore a tiny error. A block that mixes
profiles leaves a residual as large as the values themselves and gains nothing.

Note on what grouping does NOT do: it does not help on i.i.d. value tensors.
For high-dimensional random vectors the k-means objective is nearly flat, so
the resulting groups are arbitrary and can be worse than sequence order.
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vc_attention.grouping import GroupingConfig, build_permutation  # noqa: E402
from vc_attention.h3 import TAG, estimate_tokens  # noqa: E402
from vc_attention.kernels.reference import RefConfig, vc_attention_reference  # noqa: E402
from vc_attention.quant import (  # noqa: E402
    dequantize_e4m3,
    dequantize_e4m3_blocks,
    quantize_e4m3,
    quantize_e4m3_blocks,
)

torch.manual_seed(1234)
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


def db(a, b):
    return 20 * math.log10(a / b)


def psnr(out, ref):
    mse = ((out.float() - ref.float()) ** 2).mean()
    return 10 * torch.log10(ref.float().pow(2).max() / mse.clamp(min=1e-30))


def make_h3_like(n=2048, heads=2, head_dim=128, n_regions=8, seed=3):
    """Packed H3-style sequence with region structure, modalities and outliers."""
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    n_text, n_audio = n // 16, n // 8

    tags = torch.full((n,), TAG.VIDEO, dtype=torch.long)
    tags[:n_text] = TAG.TEXT
    tags[n_text : n_text + n_audio] = TAG.AUDIO

    # Content regions, scattered along the sequence (not contiguous).
    region = torch.randint(0, n_regions, (heads, n))
    region_mean = torch.randn(heads, n_regions, head_dim) * 2.0
    v = region_mean.gather(1, region.unsqueeze(-1).expand(heads, n, head_dim))
    v = v + torch.randn(heads, n, head_dim) * 0.3

    # Modality magnitudes: text small, audio large.
    v[:, :n_text, :] *= 0.30
    v[:, n_text : n_text + n_audio, :] *= 3.00

    # Scattered outliers.
    out_mask = torch.rand(heads, n, 1) < 0.01
    v = v + out_mask.float() * 15.0

    q = torch.randn(heads, n, head_dim)
    k = torch.randn(heads, n, head_dim) * 0.8
    torch.random.set_rng_state(state)
    return q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), tags


N, H, D = 2048, 2, 128
q, k, v, tags = make_h3_like(N, H, D)
ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
v4 = v.reshape(H, N, D)

print(f"\nH3-shaped workload: {N} tokens, {H} heads, head_dim {D}")
est = estimate_tokens(243, 768, 1344, audio_seconds=10.1)
print(f"  real H3 @1344x768x243f -> {est['total']:,} target rows "
      f"(the paper's ~73.5K also counts conditioning rows)")

cfg = GroupingConfig(block_rows=128, iters=6, modality_aware=True)
res = build_permutation(v4, cfg, tags=tags)
res_plain = build_permutation(v4, GroupingConfig(block_rows=128, iters=6, modality_aware=False))
print(f"  k-means: k={res.k} clusters over {N} rows")
print(f"  V per-token norm spread (std/mean): "
      f"{float(v.norm(dim=-1).std() / v.norm(dim=-1).mean()):.2f}")


def demean_quant_error(x, block_rows=128):
    """Block-demean, quantise the residual with per-block scales, restore."""
    gb, nb_, d = x.shape[0], x.shape[1] // block_rows, x.shape[2]
    xb = x.reshape(gb, nb_, block_rows, d)
    mu = xb.mean(dim=2)
    mu_full = mu.repeat_interleave(block_rows, dim=1)
    resid = x - mu_full
    qc, sc = quantize_e4m3_blocks(resid, block_rows)
    return float((dequantize_e4m3_blocks(qc, sc, block_rows) + mu_full - x).pow(2).mean().sqrt())


# ---------------------------------------------------------------------------
print("\n[1] The permutation and the block mean are mathematically free")
out = vc_attention_reference(q, k, v, RefConfig(backend="bf16", enable_vsmooth=False))
check("bf16 reference path == SDPA", float((out - ref).abs().max()) < 2e-2,
      f"max abs diff {float((out - ref).abs().max()):.2e}")
out_p = vc_attention_reference(q, k, v, RefConfig(backend="bf16", enable_vsmooth=False), perm=res.perm)
check("grouping changes nothing in exact math", float((out_p - out).abs().max()) < 2e-2,
      f"max abs diff {float((out_p - out).abs().max()):.2e}")
out_d = vc_attention_reference(q, k, v, RefConfig(backend="bf16", enable_vsmooth=True), perm=res.perm)
check("block demean + restore is exact", float((out_d - out).abs().max()) < 2e-2,
      f"max abs diff {float((out_d - out).abs().max()):.2e}")

# ---------------------------------------------------------------------------
print("\n[2] Value quantisation error in isolation (Q, K and P left exact)")
perm_g = res.perm.to(torch.int64)
v_g = v4.gather(1, perm_g.unsqueeze(-1).expand_as(v4))
v_p = v4.gather(1, res_plain.perm.to(torch.int64).unsqueeze(-1).expand_as(v4))

qa, sa = quantize_e4m3(v4, dim=1)                       # SageAttention2 style
e_global = float((dequantize_e4m3(qa, sa) - v4).pow(2).mean().sqrt())
e_seq = demean_quant_error(v4)
e_grp = demean_quant_error(v_g)
e_plain = demean_quant_error(v_p)

print(f"  per-channel global scale (SageAttention2) : RMSE {e_global:.5f}")
print(f"  + block demean, sequence order            : RMSE {e_seq:.5f}  {db(e_global, e_seq):+.2f} dB")
print(f"  + V-Smooth grouping                       : RMSE {e_grp:.5f}  {db(e_global, e_grp):+.2f} dB")
check("V-Smooth beats the SageAttention2-style baseline", db(e_global, e_grp) > 1.0,
      f"+{db(e_global, e_grp):.2f} dB (paper reports +1.1 to +2.8 dB)")
check("grouping is what does the work", db(e_seq, e_grp) > 1.0,
      f"+{db(e_seq, e_grp):.2f} dB over sequence order")

# ---------------------------------------------------------------------------
print("\n[3] End-to-end attention PSNR (Q, K, P and V all low-bit)")
o_base = vc_attention_reference(q, k, v, RefConfig(backend="fp8", enable_vsmooth=False))
o_seq = vc_attention_reference(q, k, v, RefConfig(backend="fp8", enable_vsmooth=True))
o_vs = vc_attention_reference(q, k, v, RefConfig(backend="fp8", enable_vsmooth=True), perm=res.perm)
p_base, p_seq, p_vs = psnr(o_base, ref), psnr(o_seq, ref), psnr(o_vs, ref)
print(f"  low-bit baseline (no V-Smooth) : {p_base:.2f} dB")
print(f"  block demean, sequence order   : {p_seq:.2f} dB")
print(f"  full V-Smooth                  : {p_vs:.2f} dB")
check("V-Smooth raises end-to-end PSNR", float(p_vs) > float(p_base) + 0.3,
      f"+{float(p_vs - p_base):.2f} dB (V error is partly masked by QK/P quantisation)")

# ---------------------------------------------------------------------------
print("\n[4] Modality-aware sort key (optional, off by default)")
print(f"  label sorting only  : RMSE {e_plain:.5f}  ({db(e_global, e_plain):+.2f} dB vs baseline)")
print(f"  modality-major sort : RMSE {e_grp:.5f}  ({db(e_global, e_grp):+.2f} dB vs baseline)")
check("both sort orders beat the baseline",
      db(e_global, e_plain) > 1.0 and db(e_global, e_grp) > 1.0,
      f"modality-major is {db(e_plain, e_grp):+.2f} dB vs label sorting, hence off by default")

# ---------------------------------------------------------------------------
print("\n[5] 4-bit (NVFP4) path, the workstation configuration")
nv_base = RefConfig(backend="nvfp4", enable_vsmooth=False)
nv_vs = RefConfig(backend="nvfp4", enable_vsmooth=True)
p_nv_base = psnr(vc_attention_reference(q, k, v, nv_base), ref)
p_nv_vs = psnr(vc_attention_reference(q, k, v, nv_vs, perm=res.perm), ref)
print(f"  NVFP4 baseline   : {p_nv_base:.2f} dB")
print(f"  NVFP4 + V-Smooth : {p_nv_vs:.2f} dB")
check("V-Smooth helps at 4 bit too", float(p_nv_vs) > float(p_nv_base),
      f"+{float(p_nv_vs - p_nv_base):.2f} dB")

print(f"\n=== {ok} passed, {fail} failed ===")
sys.exit(1 if fail else 0)
