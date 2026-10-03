# VC-Attention for MiniMax-H3

ComfyUI node + PyTorch/Triton implementation of **VC-Attention** (Nunchux AI,
[arXiv:2609.15810](https://arxiv.org/abs/2609.15810)), adapted to
**MiniMax-H3**'s attention.

Training-free. No checkpoint changes. Two ideas from the paper:

- **V-Smooth** — group value tokens with a lightweight k-means so each
  quantisation block holds similar tokens, subtract the block mean, quantise the
  residual, and restore the mean from the row sum the online softmax already
  maintains.
- **ExpCast-FP8** — write the E4M3 probability *byte* with one FMA instead of an
  FP32 exponential plus a format conversion. (Off on workstation cards; see
  below.)

---

## Install

**1. Copy the folder.** The whole `comfyui_vc_attention` directory goes into
`ComfyUI/custom_nodes/`:

```
ComfyUI/custom_nodes/comfyui_vc_attention/
    __init__.py          <- ComfyUI reads this
    nodes.py
    vc_attention/
    ...
```

Restart ComfyUI. No build step, no config file.

**2. Install Triton (optional but this is what makes it fast).** The fused
kernel is written in Triton. Many ComfyUI installs — especially the Windows
portable build — do not ship it:

```bash
# Windows, ComfyUI portable
ComfyUI\python_embeded\python.exe -m pip install triton-windows

# Linux
pip install triton
```

Check with:
```bash
ComfyUI\python_embeded\python.exe -c "import triton; print(triton.__version__)"
```

**If Triton is missing the node does nothing.** It prints
`CUDA present but Triton is missing -> VC-Attention inactive` and leaves native
SDPA in place. It will not silently make sampling slower: the only other
implementation available is the portable PyTorch reference, which is a Python
loop over tiles and loses to SDPA. (`allow_slow_fallback=True` forces it on if
you want to exercise the algorithm anyway.)

## Use in ComfyUI

Add **VC Attention (MiniMax-H3)** between the model loader and the sampler.
Defaults are already set for H3; `backend=auto` picks FP8 on every card this
port has measured — on RTX 50xx that is a deliberate choice, not a limitation:
the FP4 path works but loses end to end there (see "The FP4 question" below).
`backend="nvfp4"` opts into NVFP4 QK^T anyway. Add **VC Attention Disable** after
sampling to restore the original attention.

`enable_sparsity` is **on by default**: Sol-Attn-style block skipping is fused
into the same kernel and is what makes this node faster than Comfy Kitchen INT8
(1.80× at 64K tokens). It is the one setting with an unverified quality cost —
read "Fused block sparsity" below before using it for real work, and set
`enable_sparsity=false` for the pure quantisation path.

Watch the console on install — it prints the detected GPU, the resolved backend,
the sparsity settings and whether the fused kernel is actually live.

---

## ComfyUI 0.38 already ships a faster sparse backend — read this first

ComfyUI 0.38 has two native experimental attention nodes:

- **Model Attention Backend** (`ModelAttentionBackend`) selects the *dense*
  implementation: `pytorch attention` or `comfy kitchen attention` (INT8).
- **Model Sparse Attention** (`BlockSparseAttention`) overlays block sparsity. It
  runs on **comfy_kitchen's own sparse kernels** (`ck.sol_attn`,
  `ck.sol_attn_chunked`) and offers `sol-attn` (Sol-Attn's adaptive τ threshold —
  the same algorithm this repo implements), `sla` (SLA-style top-k) and `vsa`
  (FastVideo's 3D video-cube VSA, needs FastH3 weights). On MiniMax-H3 it patches
  the blocks directly and projects QKV in 4096-token chunks straight into the
  kernel's int8 carriers, so full Q/K/V are never materialised.

These two are **complementary by design, not stackable**: the sparse node handles
the calls it can and falls back to whatever dense backend the other node
selected. "Kitchen + official block-sparse" is therefore *one* sparse backend,
not two — and the third-party Sol-Attn node is a third implementation of the
same τ rule.

Measured on the same harness as everything else here (RTX 5090, sm_120, 56
heads, D=128, `h3_like`, `tests/_probe37_official_sparse.py`):

| backend | 16384 tokens | 65536 tokens | PSNR @16K / @64K |
|---|---|---|---|
| bf16 FlashAttention | 35.4 ms (1.00×) | 556 ms (1.00×) | 86.2 / 86.2 dB |
| Kitchen dense INT8 | 13.2 ms (2.68×) | 202 ms (2.76×) | 69.6 / 71.2 dB |
| **official `ck.sol_attn` τ=1.3** | **3.02 ms (11.7×)** | **30.7 ms (18.2×)** | **36.0 / 38.5 dB** |
| official τ=1.3, `extra_tokens=256` | 3.95 ms (8.9×) | 39.0 ms (14.3×) | 41.3 / 41.9 dB |
| official τ=2.0 | 1.96 ms (18.0×) | 12.5 ms (44.5×) | 35.0 / 36.1 dB |
| this repo, fused sparse τ=1.3 | 15.0 ms (2.4×) | 112 ms (5.0×) | 35.4 / 37.3 dB |

So the official kernel is **3.7–5.0× faster than this port's fused sparse kernel
at equal-or-better fidelity**. It also avoids this port's host-side prepare
entirely (8.2 ms @16K / 32.6 ms @64K), reuses pooled K/V statistics across steps,
and is a C++/CUDA kernel rather than Triton.

**Conclusion for sm_120 + ComfyUI 0.38: use the native nodes.** Set
`Model Attention Backend = comfy kitchen attention` and
`Model Sparse Attention = sol-attn, tau≈1.3`. This port's fused sparsity is
dominated. Worse, the native sparse node **re-installs its override on every
prepare step** (so "a node applied later cannot silently replace it"), which
means this node is silently shadowed whenever that node is present — it cannot
win the slot even in principle. In that chain, leave this node out, or at least
set `enable_sparsity=false` and `override_priority="defer"`.

The numbers below are what this port achieves *on its own*; they remain valid
only when the native sparse node is not in the graph.

---

## Use from Python

```python
from vc_attention.patch import install, VCAttentionConfig

install(VCAttentionConfig(
    backend="auto",        # nvfp4 on Blackwell workstation, fp8 elsewhere
    enable_vsmooth=True,
    enable_expcast=False,  # 8-bit datacenter only
    enable_sparsity=True,  # fused Sol-Attn-style block skipping
    tau=1.3,               # sparsity threshold (larger = faster, lower quality)
    block_rows=128,        # == H3 attention_head_dim
    total_steps_hint=8,    # H3 distilled LoRA
), model=model_patcher)
```

---

## What is adapted for MiniMax-H3

From `MiniMax-AI/MiniMax-H3` → `transformer/config.json` and the diffusers
`MiniMaxH3Transformer3DModel` docs:

| Property | Value | Why it matters here |
|---|---|---|
| heads × head_dim | 56 × 128 (no GQA; 7168 > hidden 5376) | `block_rows = 128` matches head_dim, the tile the paper assumes |
| layers | 50 (+2 refiner) | one permutation is shared across layers within a step — see below |
| attention | full self-attention, one packed 1-D sequence, **no cross-attention** | permuting K and V together is unconditionally safe: `P'V' = PV` |
| packed rows | text (tag 1), video (tag 0), audio (tag 2) | three very different value magnitudes in one sequence |
| RoPE | MM-RoPE over (t, h, w), `rope_freq_dim=16` → 96 of 128 channels rotated | the Q/K Hadamard rotation is still applied to all 128 channels, or the identity it relies on does not hold |
| qk-norm | RMSNorm, eps 1e-5 | applied before RoPE, outside this module |
| tokens | ~62.9K target rows at 1344×768×243f (paper quotes ~73.5K incl. conditioning) | the regime where attention dominates |

Two deliberate deviations:

1. **Permutations are shared across layers.** A permutation cannot change the
   exact attention result, so computing one per (head, step-window) and reusing
   it over all 50 layers is free mathematically and ~50× cheaper in compute and
   memory. `GroupingConfig.scope` keeps the per-layer option.
2. **`modality_aware` is off by default.** Using the H3 modality tag as the
   primary sort key measured **0.42 dB worse** than plain label sorting: it
   splits clusters that legitimately span modalities. Kept as an option.

---

## What is verified, and what is not

Verified here on CPU (torch 2.14, `tests/`, 73 assertions):

| Check | Result |
|---|---|
| E4M3 encoder matches PyTorch's native `float8_e4m3fn` conversion | byte-identical, 0 / 25005 differ |
| E4M3 relative error in the normal range | ≤ 2⁻⁴ (3 mantissa bits, as specified) |
| FP4 (E2M1) magnitude table and RNE ties | all 7 midpoints round to the even code |
| **ExpCast vs exp-then-cast byte agreement** | **0.7962** (paper: 79.6%) |
| **ExpCast total variation vs exact probabilities** | **1.36%** max (paper bound: 3.64%) |
| ExpCast row maximum lands on 2⁸ = 256 | yes |
| Hadamard transform is orthonormal and matches the explicit matrix | yes |
| Permutation changes nothing in exact arithmetic | max diff 1.4e-06 |
| Block demean + restore is exact | max diff 1.7e-06 |
| **V-Smooth vs per-channel FP8 baseline (V only)** | **+2.41 dB** (paper: +1.1 to +2.8 dB) |
| **V-Smooth, end-to-end low-bit attention PSNR** | **+0.62 dB** |
| **V-Smooth at 4 bit (NVFP4)** | **+2.20 dB** |
| Hook falls through to the original SDPA when it does not apply | bit-identical |
| Grouping schedule for the 8-step LoRA | groups on steps 0–1, recomputes at 0 and 4 |
| Kernel input layout (`prepare`) — shapes, fp8 dtype, padding, per-block scales, `mu × v_scale == true block means` | 12 assertions |

**Not verified — needs your GPU:**

- ~~The fused kernel body in `vc_attention/kernels/triton_attn.py`~~ → **verified on an
  RTX 4090 (Triton 3.6, torch 2.10/cu130), 2026-10-02.** It needed three fixes
  before it produced numbers, all now in the source:

  1. `LOG2E` / `P_SCALE` must be `tl.constexpr` instances — plain module
     globals fail to compile on Triton ≥ 3.2 (`NameError` at JIT time).
  2. The mean-restoration term was missing the `v_scale` factor:
     `MU` stores *mean / v_scale* (as `prepare` documents), so the kernel must
     add `r * mu * vs`, not `r * mu`. The old term drowned the output (~−40 dB).
  3. The PV `tl.dot` runs in **bf16**, not fp8: on sm_89 + Triton, an fp8 MMA
     whose A operand was computed in registers yields wrong values in every
     warp/stage configuration (the QK^T fp8 MMA with both operands from memory
     is fine, as is `torch._scaled_mm`). E4M3 → bf16 is exact, so the
     arithmetic is unchanged; the cost is half the PV MMA rate.

     Root-caused with minimal repros (`tests/_probe14/15/16`): **Triton's
     fp32→e4m3 `.to()` conversion itself produces wrong codes on sm_89** —
     still broken in Triton 3.8.0. The MMA is innocent: computing the e4m3
     codes with integer bit-math and `bitcast=True` gives a correct register-
     fp8 dot, bit-matching the oracle encoder (25.66 dB). That workaround is
     implemented and measured — and on the 4090 it is *slower* than the bf16
     PV (54.6 vs 48.6 ms at 16K tokens: the ~10 extra int ALU ops per element
     plus fp8 operand staging eat the 2× MMA gain), as is the one-FMA ExpCast
     variant (49.9 ms, −0.55 dB). **bf16 PV stays the default on Ada**; the
     fp8/bitcast path becomes interesting again on Hopper/Blackwell, where the
     cvt path differs and fp8 rates double again.

  Verified results (identity permutation + real k-means permutation, trap armed
  to catch silent fallback to the reference path):

  | Check | Result |
  |---|---|
  | Kernel compiles + runs (trap did not fire) | yes |
  | PSNR vs fp32 SDPA, 2048 i.i.d. tokens | 25.7 dB (= oracle 25.7 dB) |
  | PSNR, bench `h3_like` data, 8192 / 16384 tokens | 58.7 / 57.8 dB |
  | V-Smooth gain on bench structured data | +0.18 … +0.25 dB |
  | V-Smooth gain on i.i.d. data | +0.03 dB (≈ 0, as expected) |
  | CPU suite after the fixes | 73 / 73 assertions |

  **Speed (RTX 4090, 56 heads).** The fused kernel alone runs at **0.97×
  bf16 SDPA** (48.4 ms vs 46.9 ms at 16K tokens; best config
  `BLOCK_M=128, num_warps=8`). The first host-side `prepare()` implementation
  cost 201 ms per call — 4.3× the whole attention — which made end-to-end
  0.19×. The fast path (`_prepare_fast`, used automatically on CUDA) replaces
  the integer-op E4M3 oracle with the native `.to(float8_e4m3fn)` cast,
  the 7-stage Hadamard butterfly with one matmul, the expanded-index gather
  with a broadcast `take_along_dim`, and keeps the pipeline in bf16:
  **13.8 ms, 14.5× faster, −0.10 dB PSNR.** End-to-end after integration:

  | Tokens | bf16 SDPA | VC-Attention | ratio | PSNR | V-Smooth gain |
  |---|---|---|---|---|---|
  | 8192 | 11.6 ms | 18.7 ms | 0.62× | 58.1 dB | +0.27 dB |
  | 16384 | 46.2 ms | 61.9 ms | 0.75× | 56.4 dB | +0.16 dB |
  | 32768 | 188.5 ms | 224.5 ms | 0.84× | 57.8 dB | +0.11 dB |

  The ratio improves with size because prepare is linear while attention is
  quadratic — at H3's real ~50–70K-token sequences VC approaches parity, and
  would cross over with fp8 PV (blocked by the Triton bug above), ExpCast, or
  a tuned kernel. The k-means permutation is amortised (46 ms per step-window
  over 50 layers at 32K tokens). A silent fallback now prints to stderr.

- **Verified on an RTX 5090 (sm_120, Triton 3.8, torch 2.10/cu130), 2026-10-03.**
  The two things that blocked a speedup on Ada both clear on Blackwell:

  1. **The register-operand fp8 PV dot is correct on sm_120** (the sm_89
     fp32→e4m3 conversion bug is absent — `_probe19`, max|diff| = 0.0). Since
     both PV operands are already E4M3, this doubles the PV MMA rate at
     *identical* precision. Kernel: **51.7 ms → 25.0 ms (2.07×)** at 16K
     tokens; PSNR unchanged at 57.40 dB.
  2. **Launch config matters a lot**: `BLOCK_M=64, num_warps=4, num_stages=2`
     beats the old `128/8/3` by 1.27×. `BLOCK_M=256` with 4 warps spills
     catastrophically (966 ms). These are now the defaults.

  End-to-end on an RTX 5090 (auto backend → fp8), 56 heads, D=128. The
  `bf16 SDPA` baseline is the *same* FlashAttention backend the paper compares
  against — PyTorch's default SDPA dispatch measures 35.17 ms vs 35.23 ms for
  an explicit `SDPBackend.FLASH_ATTENTION` at 16K tokens (555.4 vs 555.3 ms at
  64K), i.e. it *is* FA, not a fallback:

  | Tokens | bf16 SDPA (=FA) | VC-Attention | ratio | PSNR | V-Smooth gain |
  |---|---|---|---|---|---|
  | 16384 | 35.1 ms | 32.7 ms | **1.07×** | 57.5 dB | +0.17 dB |
  | 32768 | 139.6 ms | 115.4 ms | **1.21×** | 56.0 dB | +0.11 dB |
  | 65536 | 557.1 ms | 431.9 ms | **1.29×** | 56.2 dB | +0.08 dB |

  The ratio climbs with length because `prepare` is linear while attention is
  quadratic; H3's real ~63K-token sequences sit at the 1.29× end. **The kernel
  alone is ~1.4×** (the rest is the 8.2 ms prepare at 16K, ~33 ms at 64K).

  **This does not reach the paper's RTX 5090 claim (kernel 2.3–3.6×, end-to-end
  1.36–1.70×), and the reason is specific and measured.** Both of our matmuls
  run at fp8 = 2× the bf16 MMA rate; the paper's number requires the *4-bit*
  path = 4×. That path is **marginal here, not impossible** — and the reason is
  worth stating precisely, because it is not what it first looks like:

  * **The hardware can do it.** sm_120 has a single-instruction converter,
    `cvt.rn.satfinite.e2m1x2.f32`, and Triton can reach it through
    `tl.inline_asm_elementwise`. Measured bit-exact against `quant.fp4_encode`
    (`tests/_probe28_fp4cvt.py`). PyTorch does not expose it, which is why the
    portable encoder needs 7 comparisons.
  * **But it still does not pay.** With the PTX converter the whole NVFP4 pack
    drops from 11.97 ms to 3.06 ms over a (16384×56, 128) tensor — a 3.9×
    improvement (`tests/_probe29_fp4pack.py`) — yet that is still **8× the fp8
    cast (0.37 ms)**. Q+K packing costs ~5.4 ms against the ~4.7 ms the fp4 QK
    kernel saves, so it is a wash before accuracy is even considered.
  * **PV is the structural one.** Its A operand is the softmax probability,
    which the kernel already produces as E4M3 — so fp8 PV is *free* (2× the
    bf16 rate for no extra work), while fp4 would need a genuine e2m1 encode of
    every tile. `tl.dot_scaled` also needs packed operands, so the encode
    cannot be skipped.
  * **And the shape is wrong for it.** QK^T reduces over D=128; fp4 needs a long
    GEMM to shine (3.82× at K=4096, but only 1.23× at K=128), and it costs ~10 dB
    (QK rel-err 3.6% → 13.4%).

  With the matmuls capped at 2× and roughly a third of kernel time being
  softmax/epilogue, ~1.4–1.5× kernel is the ceiling here — consistent with what
  we measure. Making 4-bit actually pay would mean moving the PTX converter
  *inside* the kernel (for P), which is real research work, not a flag.

- **Head-to-head against the alternatives already on this machine.** Same
  harness, same data (`tests/_probe23/26`), 56 heads, D=128, RTX 5090:

  | tokens | bf16 FlashAttention | Comfy Kitchen INT8 | VC-Attention fp8 |
  |---|---|---|---|
  | 16384 | 35.09 ms (1.00×) | **13.32 ms (2.63×)** | 32.49 ms (1.08×) |
  | 65536 | 555.2 ms (1.00×) | **203.0 ms (2.74×)** | 431.5 ms (1.29×) |

  and on fidelity at 16K: Kitchen **68.54 dB** vs VC-Attention **56.43 dB**.

  So on sm_120 this node is beaten on *both* axes by ComfyUI's own INT8 backend
  — which is already installed. Its kernel sustains ~600 TFLOP/s where ours
  reaches ~317, and it does not pay for V-Smooth's epilogue or the k-means
  permutation. **The fp8 path here uses no FP4 at all**, so "it works, it is
  just modest" is really "it is not competitive": the paper's case for
  workstation Blackwell is the 4-bit path, and that is the part that does not
  survive on this stack. Treat this port on sm_120 as a correctness reference,
  not a speedup — and use Kitchen or Sol-Attn for actual sampling.

- **Composing with other backends.** Monkey-patching `optimized_attention` put
  this node *outside* ComfyUI's dispatch, so any node registering an
  `optimized_attention_override` (Kitchen's backend node, Sol-Attn's generic
  node) silently shadowed it (`tests/_probe21_kitchen.py`). `install(model=...)`
  enters that chain and wraps the existing backend as its fallback, exactly as
  Sol-Attn does. Whether that helps depends on being the faster backend, and the
  answer changed when block sparsity was fused into the kernel
  (`tests/_probe35_phase3.py` + `tests/_probe26_kitchen_time.py`):

  | configuration | 16384 tokens | 65536 tokens |
  |---|---|---|
  | Kitchen alone | 13.45 ms | 202.1 ms |
  | VC-Attention in front, dense (fp8) | 32.9 ms (0.41×) | 432 ms (0.47×) |
  | VC-Attention in front, sparse τ=1.3 | 15.0 ms (0.90×) | **112.3 ms (1.80×)** |

  So `override_priority` now defaults to **`"front"`**: at H3's real (~63K
  token) length the fused kernel wins. At 16K it is ~11% behind Kitchen, so
  `"defer"` remains the right setting for short-sequence work. Note that this is
  a *single* fused kernel, not two nodes stacked — two attention nodes on one
  call site is pure overhead (0.98–0.99×, `tests/_probe27_stack_gain.py`).

- **NVFP4 for attention: works, but does not pay off on sm_120.** `tl.dot_scaled`
  with e2m1 + per-16 e4m3 microscales *is* native FP4 hardware on sm_120, not
  emulation — in a compute-bound GEMM (K=4096) it runs at **530 TFLOP/s vs 277
  for fp8 and 139 for bf16 (3.82×)**. But attention is the wrong shape for it:

  | | prepare | kernel | total | PSNR |
  |---|---|---|---|---|
  | fp8 QK + bf16 PV | 8.2 ms | 51.7 ms | 59.9 ms | 57.40 dB |
  | fp8 QK + fp8 PV | 8.2 ms | **25.0 ms** | **33.2 ms** | 57.40 dB |
  | fp4 QK + fp8 PV | **31.3 ms** | 20.2 ms | 51.5 ms | 47.19 dB |

  QK^T's reduction is only `D=128` (not a long GEMM), so FP4 buys just 1.23×
  there, while host-side NVFP4 quantization costs ~23 ms/layer — PyTorch 2.10
  has no `float → float4_e2m1fn_x2` cast, so the codes come from 7 elementwise
  comparisons (the e4m3 microscale does use the native cast). And the 2-bit
  mantissa costs ~10 dB: QK^T relative error 3.6% → 13.4%. PV *cannot* use FP4
  at all, because its A operand (the softmax probabilities) is computed in
  registers and `dot_scaled` needs packed operands from memory.

  So on sm_120 the default is **fp8**; `backend="nvfp4"` selects the FP4 QK path
  for anyone who wants the paper's 4-bit configuration or is on a shape where
  the tradeoff flips. This is what `tests/bench_pv.py` and `_probe18/19/20` measure.

- All speed figures besides the ones above. The paper's speedups additionally
  rely on ExpCast-FP8 (softmax-stage shortening), which is inactive on
  workstation cards.
- ExpCast-FP8 branch: compiles (shares the fixed PV call site) but was never
  exercised — it targets 8-bit datacenter cards.
- Real MiniMax-H3 weights. The PSNR figures above come from synthetic tensors
  shaped like H3, not from the model. In a real ComfyUI run the node was also
  **silently inert** until the router fix below (see Caveats).

```bash
python tests/bench_attention.py --tokens 32768 --heads 56 --backend auto
```

---

## Two findings worth knowing

**Scale granularity is the whole ballgame.** With a per-channel scale computed
over the whole tensor, grouping does nothing — measured 1.01×, because the
scale is set by the tensor's largest entry either way. The value scale must be
per *(value block × channel)*; then homogeneous blocks quantise tightly. The
8-bit path uses one E4M3 scale per block per channel; the 4-bit path uses
NVFP4's per-16-token microscales with the mean divided by the tensor-level
scale (the paper's "means are added unscaled").

**Grouping only helps on structured values.** On i.i.d. value tensors the
k-means objective is nearly flat and the resulting groups are arbitrary —
measured *worse* than sequence order. The gain appears when tokens cluster
(content regions, modalities) and those clusters are interleaved along the
sequence, which is what flattening a (t, h, w) latent grid produces.

Also: without the Q/K Hadamard rotation *and* per-token (not per-channel) key
quantisation, the QK term is 55% relative output error versus ~1.5% for V, and
no amount of V smoothing is visible end to end. Per-token key quantisation
brings it to 3.6%. Both are on by default.

---

## Fused block sparsity (Sol-Attn style)

**Two nodes cannot be stacked.** Attention overrides compete for the same call
site, so adding Sol-Attn next to this node measures 0.98–0.99× — pure overhead
(`tests/_probe27_stack_gain.py`). Sparsity and quantisation are *orthogonal*
axes, though: one skips work, the other makes work cheaper, and they compose
numerically (stacking costs ≤0.03 dB, `tests/_probe22_solstack.py`). So the only
combination that can pay is both inside one kernel, and that is what
`sparse=True` does.

**How it works** (`kernels/triton_attn.py::_vc_attn_fwd_sparse`). KV blocks are
walked a `group=16` at a time:

1. one tensor-core matmul `q @ kc^T` gives the per-row proxy score against every
   block's mean key in the group;
2. the group's row-mean proxy is compared with a host-side threshold
   `mean + tau·std` (computed from a pooled query centroid, `_prepare_fast`);
3. blocks below the threshold are **approximated, not dropped** — their proxy
   score is reused for the whole block against the block-mean value, so the
   softmax normaliser stays right. One small matmul folds the whole group in;
4. blocks above it run the exact quantised tile.

The approximation and the routing both run on tensor cores. That matters: the
first attempt computed the proxy and the approximation per tile with fp32
elementwise ops, which cost ~45% of the dense kernel and made skipping
pointless — measured *slower* than not skipping at all.

**Measured** (RTX 5090, sm_120, 56 heads, D=128, `h3_like` data,
`tests/_probe35_phase3.py` + `tests/_probe26_kitchen_time.py`):

| tokens | bf16 FlashAttention | Comfy Kitchen INT8 | VC dense (fp8) | **VC sparse τ=1.3** |
|---|---|---|---|---|
| 16384 | 35.4 ms (1.00×) | 13.45 ms (2.63×) | 32.9 ms (1.08×) | **15.0 ms (2.36×)** |
| 65536 | 559 ms (1.00×) | 202 ms (2.77×) | 432 ms (1.29×) | **112 ms (4.98×)** |

At H3's real sequence length (~63K tokens) the fused kernel is **1.80× faster
than Comfy Kitchen INT8**. At 16K it is ~11% behind Kitchen. Keep ratio at
τ=1.3 is ~12%; the τ sweep (`0.8 → 23%`, `1.0 → 18%`, `2.0 → 5%`) is in the
probe output.

**Verification.** `SPARSE=False` is a *separate* kernel, so the dense path is
untouched (G2). With every block forced exact (`tau=-1000`, `local=1e5`,
`sink=1e5`) the sparse kernel reproduces the dense kernel **bit for bit**
(rel-err `0.000e+00`, `tests/_probe33_sparse_isolate.py`). Against an independent
PyTorch simulation of the same algorithm on the same quantised operands, PSNR
agrees to 0.01 dB at every τ (`tests/_probe35_phase3.py`).

**Quality: read this before enabling it.** On the synthetic `h3_like`
benchmark the sparse path drops ~20 dB of PSNR against dense attention — at
*every* τ. That number is **not** a valid measure of the real quality loss: the
synthetic keys inside a 128-token block are near-orthogonal, so the block-mean
proxy is a poor stand-in and skipped blocks contribute almost nothing. It is
also not specific to this port — the independent fp32 Sol-Attn simulation of the
same rule scores *worse* (31.8 dB) than this kernel (35.5 dB) on the same data.
Real keys are correlated within a block, which is exactly the premise the
algorithm relies on. **End-to-end H3 image/video validation has not been run
here** and is the one gate this change has not passed. If you enable sparsity,
compare a dense and a sparse render first; raise `tau` or set
`enable_sparsity=false` if it degrades.

**Knobs** (`VC Attention (MiniMax-H3)` node):

| Parameter | Default | Meaning |
|---|---|---|
| `enable_sparsity` | `true` | off = pure quantisation path |
| `tau` | `1.3` | Sol-Attn's tuned value; larger = faster, lower quality |
| `sparsity_min_tokens` | `8192` | below this, sparsity is off |
| `sink_tokens` | `512` | leading tokens kept exact (H3 packs text/conditioning there) |
| `local_blocks` | `1` | ±N KV blocks around each query block kept exact |
| `override_priority` | `front` | take priority over another backend node |

`override_priority` defaulted to `defer` while this kernel was slower than
Comfy Kitchen INT8 (taking priority was then a 3× regression). Now that the
fused version wins at H3's length it defaults to `front` — that is what makes
the node take effect next to a Kitchen backend node. Set `defer` if you work at
short sequence lengths.

---

## Layout

```
vc_attention/
  device.py      GPU capability -> 8-bit or 4-bit path
  quant.py       E4M3 / NVFP4 / INT4 encoders (portable, bit-exact)
  expcast.py     ExpCast-FP8
  hadamard.py    orthonormal Walsh-Hadamard for Q and K
  grouping.py    online k-means + permutation
  h3.py          MiniMax-H3 facts, token estimator, defaults
  schedule.py    step tracking, grouping window, permutation cache
  patch.py       installs the attention hook and step counter
  kernels/
    reference.py portable oracle (defines the semantics)
    triton_attn.py fused fp8 kernel (fast path; fp8 PV default, opt-in NVFP4 QK)
nodes.py         ComfyUI node
tests/
  test_quant.py          32 assertions, CPU  (formats + ExpCast)
  test_vsmooth.py         8 assertions, CPU  (does grouping pay?)
  test_install.py        21 assertions, CPU  (hook, schedule, fallback, node.apply)
  test_prepare.py        12 assertions, CPU  (kernel input layout)
  bench_attention.py     speed + PSNR, needs CUDA
  bench_pv.py            prepare/kernel split, backend matrix, needs CUDA
  tune_attn.py           launch-config sweep, needs CUDA
  _probe14/15/16.py      sm_89 register-fp8 cvt bug repros
  _probe18/19_nvfp4.py   NVFP4 dot_scaled layouts + compute-bound GEMM bench
  _probe20_qkquant.py    fp8 vs NVFP4 Q/K fidelity
  _diag_decomp.py        error decomposition scratch script
```

## Hardware notes

**RTX 40-series (Ada, sm_89): the node works but does not speed anything up.**
This is measured, not guessed — on an RTX 4090 the fused kernel compiles,
runs the full dense sequence inside real ComfyUI sampling, produces correct
output (25+ dB PSNR at every sequence length), and still lands at **~0.97×
native SDPA** at the kernel level (48.6 ms vs 47.2 ms at 16K tokens) and
wall-clock parity in a real run. Three reasons, all hardware-bound: Ada has
no FP4 tensor cores; the fp8 PV path is blocked by Triton's fp32→e4m3
conversion bug (see above — still unfixed in Triton 3.8.0); and softmax ALU
doesn't scale with the tensor cores. On 40-series, use this node only to
exercise or verify the algorithm — for actual sampling speed on Ada, sparse
attention (e.g. block-sparse schedulers) wins by skipping work rather than
cheapening it. `auto` resolves to FP8 on Ada.

- **H200 / B200 / B300** (sm_90 / 10.x): the paper's 8-bit target — kernel
  1.46–1.59× vs BF16 FlashAttention-4, end-to-end 1.13–1.19× on long
  sequences. ExpCast-FP8 applies; the branch is wired but has not been
  exercised on these parts in this port.
- **RTX 5090 / RTX PRO 6000** (sm_120): the paper's 4-bit target — kernel
  2.3–3.6×, end-to-end 1.36–1.70× with NVFP4 (V-Smooth only; NVFP4 codes have
  no affine map from a log-domain score, and at 4 bits softmax is not the
  longest pipeline stage). NOTE: this port's kernel currently implements the
  FP8 path only — the NVFP4 branch is not written yet, so today the node runs
  FP8 there too.
- **Blackwell datacenter dropped INT4/INT8 MMA**, so `int4` resolves to FP8
  on 10.x; Ada and older keep INT8.

## Caveats

- The attention hook is **process-global**. Fine for a single-user sampling run;
  call `uninstall()` to put everything back.
- **ComfyUI + MiniMax-H3 needs the by-value-import rebind.** comfy's own H3
  port does `from comfy.ldm.modules.attention import optimized_attention` at
  module load, so replacing the attribute on the defining module is not enough —
  the model keeps calling the original forever and the node is silently inert
  (measured: identical step time to baseline, zero Triton compiles).
  `install()` now walks `sys.modules` and rebinds every comfy module that still
  holds the original object; the wrapper also handles `skip_reshape=True`
  4-D inputs, which H3 uses — and since comfy 0.38 those arrive wrapped in
  single-owner `AttentionTensorContainer`s on every H3 call; the hook peeks
  into them, defers to any registered `optimized_attention_override`, and
  falls back to the original on any internal error so it can never take a run
  down (first live run crashed exactly there before this guard existed).
  Validated end-to-end against the real comfy tree
  (`tests/_probe9_hook_rebind.py`): H3's own `optimized_attention` binding
  routes to the VC kernel bit-identically, and short calls defer to the
  original bit-identically. **Apply the node before the first sampling after a
  restart**, and note that ComfyUI's torch.compile ("Comfy model compiler")
  traces whatever function objects are installed at compile time — if the graph
  was compiled before the node ran, it must be re-traced (restart) or the
  compiler disabled (`--disable-comfy-compiler`) for the hook to be seen.
- Only non-causal attention is accelerated. Calls with an `attn_mask`, or
  causal, or below `min_tokens`, or with head_dim ∉ {64, 128} go straight to the
  original SDPA. H3's per-call sequence can fall below `min_tokens` on short
  segments — watch the stats, not the console banner.
- H3's padding rows (tag −1) need a masked backend; this node does not
  accelerate masked attention yet. The diffusers port runs unmasked.
