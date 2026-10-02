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
Defaults are already set for H3; `backend=auto` picks NVFP4 on RTX 50xx /
RTX PRO 6000 and FP8 elsewhere. Add **VC Attention Disable** after sampling to
restore the original attention.

Watch the console on install — it prints the detected GPU, the resolved backend
and whether the fused kernel is actually live.

## Use from Python

```python
from vc_attention.patch import install, VCAttentionConfig

install(VCAttentionConfig(
    backend="auto",        # nvfp4 on Blackwell workstation, fp8 elsewhere
    enable_vsmooth=True,
    enable_expcast=False,  # 8-bit datacenter only
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
  3. The PV `tl.dot` runs in **bf16**, not fp8: on sm_89 + Triton 3.6, an fp8
     MMA whose A operand was computed in registers yields NaNs in every
     warp/stage configuration (the QK^T fp8 MMA with both operands from memory
     is fine, as is `torch._scaled_mm`). E4M3 → bf16 is exact, so the
     arithmetic is unchanged; the cost is half the PV MMA rate.

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

- All speed figures besides the ones above. The paper's speedups additionally
  rely on fp8 PV (blocked here by the Triton bug above) and ExpCast-FP8
  (softmax-stage shortening), neither of which is active on this card.
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
    triton_attn.py fused fp8 kernel (fast path, unverified)
nodes.py         ComfyUI node
tests/
  test_quant.py          32 assertions, CPU  (formats + ExpCast)
  test_vsmooth.py         8 assertions, CPU  (does grouping pay?)
  test_install.py        21 assertions, CPU  (hook, schedule, fallback, node.apply)
  test_prepare.py        12 assertions, CPU  (kernel input layout)
  bench_attention.py     speed + PSNR, needs CUDA
  _diag_decomp.py        error decomposition scratch script
```

## Hardware notes

- **RTX 5090 / RTX PRO 6000** (sm_120): NVFP4, V-Smooth only. ExpCast-FP8 does
  not apply — NVFP4 codes have no affine map from a log-domain score, and at
  4 bits softmax is not the longest pipeline stage.
- **RTX 4090** (sm_89): no FP4 tensor cores. `auto` resolves to FP8.
- **B200 / B300 / H200**: 8-bit + ExpCast-FP8. Blackwell datacenter dropped
  INT4/INT8 MMA, so `int4` resolves to FP8 there.

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
