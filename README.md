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

### Nodes

**VC Attention (MiniMax-H3)** — takes `MODEL`, returns patched `MODEL`.
Everything defaults to the H3 distilled-LoRA setup (56 heads, head_dim 128,
8 steps), so in the common case you only drop it in and don't touch anything.

| Input | Default | Meaning |
|---|---|---|
| `backend` | `auto` | `nvfp4` on RTX 50xx / RTX PRO 6000, `fp8` on other fp8-capable cards, otherwise nothing runs. `reference` is the slow exact PyTorch path, for checking results. |
| `enable_vsmooth` | on | V-Smooth value grouping — the accuracy half. Leave on. |
| `enable_expcast` | off | ExpCast-FP8, the softmax-speed trick. Only effective when the resolved backend is `fp8` (datacenter cards). |
| `total_steps` | 8 | Denoising steps you plan to sample with; the grouping window is derived from it. |
| `block_rows` | 128 | Value rows per quantisation block. 128 matches H3's head_dim. |
| `min_tokens` | 8192 | Sequences shorter than this keep native SDPA. |
| `group_fraction` | 0.25 | Fraction of steps that run k-means grouping. |
| `reuse_every` | 4 | Steps one permutation is reused for. |
| `kmeans_iters` | 3 | Lloyd iterations on a cold start. |
| `modality_aware` | off | Sort by H3 modality tag first. Measured slightly worse; kept as an option. |

**VC Attention Disable** — takes and returns `MODEL`; call `uninstall()` and
restore the original attention. Put it on a branch you don't take if you want
the patch to last for the whole run.

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

- The fused kernel body in `vc_attention/kernels/triton_attn.py`. Written
  against the Triton FA2 shape; it has **never been compiled or run**. It is
  guarded so any failure falls back to the reference path. Its *inputs* are
  covered on CPU by `tests/test_prepare.py`, so a wrong result points at the
  kernel body rather than the host-side layout. Run `tests/bench_attention.py`
  before trusting its numbers.
- All speed figures. Nothing here has been timed on a GPU.
- Real MiniMax-H3 weights. The PSNR figures above come from synthetic tensors
  shaped like H3, not from the model.

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
- Only non-causal attention is accelerated. Calls with an `attn_mask`, or
  causal, or below `min_tokens`, or with head_dim ∉ {64, 128} go straight to the
  original SDPA.
- H3's padding rows (tag −1) need a masked backend; this node does not
  accelerate masked attention yet. The diffusers port runs unmasked.
