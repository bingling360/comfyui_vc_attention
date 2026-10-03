# VC-Attention for MiniMax-H3

ComfyUI node + PyTorch/Triton implementation of **VC-Attention** (Nunchux AI,
[arXiv:2609.15810](https://arxiv.org/abs/2609.15810)), adapted to
**MiniMax-H3**'s attention. Training-free; no checkpoint changes.

Two ideas from the paper:

- **V-Smooth** — group value tokens with a lightweight k-means so each
  quantisation block holds similar tokens, subtract the block mean, quantise the
  residual, and restore the mean from the row sum the online softmax already
  maintains.
- **ExpCast-FP8** — write the E4M3 probability *byte* with one FMA instead of an
  FP32 exponential plus a format conversion. (Off on workstation cards.)

This port adds a third, measured here: **fused block sparsity** — Sol-Attn's
block routing multiplied with the quantisation inside one kernel.

---

## Verdict (2026-10-04)

**On sm_120 with ComfyUI 0.38, use ComfyUI's own attention nodes — not this one.**

| backend | 16384 tokens | 65536 tokens | PSNR @16K / @64K |
|---|---|---|---|
| bf16 FlashAttention | 35.4 ms (1.00×) | 556 ms (1.00×) | 86.2 / 86.2 dB |
| Comfy Kitchen INT8 (dense) | 13.2 ms (2.68×) | 202 ms (2.76×) | 69.6 / 71.2 dB |
| **ComfyUI `Model Sparse Attention`** (`ck.sol_attn` τ=1.3) | **3.02 ms (11.7×)** | **30.7 ms (18.2×)** | 36.0 / 38.5 dB |
| this port, fused sparse τ=1.3 | 15.0 ms (2.4×) | 112 ms (5.0×) | 35.4 / 37.3 dB |

ComfyUI's own sparse kernel is **3.7–5.0× faster than this port's at
equal-or-better fidelity**, it avoids this port's host-side prepare entirely, and
it **re-installs its attention override on every prepare step**, so this node is
silently shadowed whenever it is present.

Recommended graph:

```
Model Attention Backend   = comfy kitchen attention
Model Sparse Attention    = sol-attn,  tau ≈ 1.3
```

Everything else in this document describes this port *on its own*, and remains
valid only when that node is not in the graph.

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
Defaults are set for H3; `backend=auto` picks FP8 on every card this port has
measured — on RTX 50xx that is deliberate, not a limitation: the FP4 path works
but loses end to end there. `backend="nvfp4"` opts into NVFP4 QK^T anyway. Add
**VC Attention Disable** after sampling to restore the original attention.

`enable_sparsity` is **on by default** (fused Sol-Attn-style block skipping).
It is the one setting with an unverified quality cost — read "Quality: the one
gate not passed" below before using it for real work, and set
`enable_sparsity=false` for the pure quantisation path.

Watch the console on install — it prints the detected GPU, the resolved backend,
the sparsity settings and whether the fused kernel is actually live.

---

## Measured results (2026-10-04)

All figures: RTX 5090 (sm_120), 56 heads, D=128, `h3_like` synthetic tensors,
Triton 3.8 / torch 2.10+cu130. Scripts are in `tests/`.

### 1. ComfyUI's own sparse backend already beats this one

`comfy_extras/nodes_sparse_attention.py` ("Model Sparse Attention",
experimental) runs on **comfy_kitchen's own sparse kernels** — `ck.sol_attn` /
`ck.sol_attn_chunked` — with Sol-Attn's adaptive τ threshold (plus SLA-style
top-k and FastVideo VSA). On MiniMax-H3 it patches the blocks directly and
projects QKV in 4096-token chunks straight into the kernel's int8 carriers, so
full Q/K/V are never materialised. Its dense fallback is whatever
`Model Attention Backend` selected.

So "Kitchen + official block-sparse" is **one sparse backend with a dense
fallback**, not two things to stack — and the third-party Sol-Attn node is a
third implementation of the same τ rule. All of them contend for the same
`optimized_attention_override` slot: two attention nodes in one graph measure
0.98–0.99× (`tests/_probe27_stack_gain.py`).

`tests/_probe37_official_sparse.py`:

| backend | 16384 tokens | 65536 tokens | PSNR @16K / @64K |
|---|---|---|---|
| bf16 FlashAttention | 35.4 ms | 556 ms | 86.2 / 86.2 dB |
| Comfy Kitchen INT8 (dense) | 13.2 ms | 202 ms | 69.6 / 71.2 dB |
| official `ck.sol_attn` τ=1.0 | 3.80 ms | 45.2 ms | 37.0 / 39.8 dB |
| **official `ck.sol_attn` τ=1.3** | **3.02 ms** | **30.7 ms** | 36.0 / 38.5 dB |
| official τ=1.3 + `extra_tokens=256` (node default) | 3.95 ms | 39.0 ms | 41.3 / 41.9 dB |
| official `ck.sol_attn` τ=2.0 | 1.96 ms | 12.5 ms | 35.0 / 36.1 dB |
| this port, fused sparse τ=1.3 | 15.0 ms | 112 ms | 35.4 / 37.3 dB |
| this port, dense fp8 | 32.6 ms | 432 ms | 57.0 / 56.8 dB |

Three things this table settles:

- The official kernel is **3.7–5.0× faster** than this port's fused sparse
  kernel, at slightly *better* PSNR.
- It is also **12 dB more accurate than this port's dense path** (int8 has a
  7-bit mantissa, e4m3 has 3). VC-Attention's case for fp8/4-bit is a speed
  argument, and on this stack int8 wins on both axes.
- This port's sparsity axis is the same algorithm as the official node's
  `sol-attn` mode, and its quantisation axis is the same idea as
  `comfy_kitchen` int8 — so there is no axis left where it adds anything.

### 2. Where a MiniMax-H3 block actually spends its time

`tests/_probe38_h3_block_budget.py`, real H3 shapes (hidden 5376, qkv 7168,
ffn 14336), bf16 GEMMs:

| | 16384 tokens | 65536 tokens |
|---|---|---|
| qkv_proj | 15.9 ms | 67.4 ms |
| out_proj | 5.6 ms | 22.4 ms |
| ffn (up + down) | 22.3 ms | 90.2 ms |
| **non-attention total** | **43.8 ms** | **179.9 ms** |
| + attention (Kitchen dense) | 57.0 ms — attention 23.2% | 381.9 ms — attention 52.9% |
| + attention (official `ck.sol_attn`) | 46.8 ms — attention **6.5%** | 210.6 ms — attention **14.6%** |

**With the official sparse kernel, attention is only 14.6% of a 64K block.**
Making it *completely free* would buy **1.17× end-to-end** — that is the ceiling
for any attention-side work, and the official kernel is already close to it.
(The deployed H3 weights are int8, so the real GEMMs run about 2× faster than
the bf16 figures above; attention's share would rise to ~25% and the ceiling to
~1.34×. Still small.)

The remaining time is the projections and the FFN (85.4% of a 64K block), where
`qkv_proj` alone costs 2.2× the attention. If you want to keep optimising,
that is where to look — not at attention.

### 3. This port's fused block sparsity

Sparsity (skip work) and quantisation (cheapen work) are orthogonal axes, and
they compose numerically — stacking costs ≤0.03 dB (`tests/_probe22_solstack.py`).
The only combination that can pay is both inside one kernel, which is what
`sparse=True` does (`kernels/triton_attn.py::_vc_attn_fwd_sparse`). KV blocks are
walked `group=16` at a time:

1. one tensor-core matmul `q @ kc^T` gives the per-row proxy score against every
   block's mean key in the group;
2. the group's row-mean proxy is compared with a host-side threshold
   `mean + tau·std` (computed from a pooled query centroid in `_prepare_fast`);
3. blocks below it are **approximated, not dropped** — the proxy score is reused
   for the whole block against the block-mean value, so the softmax normaliser
   stays right. One small matmul folds the whole group in;
4. blocks above it run the exact quantised tile.

`tests/_probe35_phase3.py` + `tests/_probe26_kitchen_time.py`:

| tokens | bf16 FlashAttention | Comfy Kitchen INT8 | VC dense (fp8) | VC sparse τ=1.3 |
|---|---|---|---|---|
| 16384 | 35.4 ms (1.00×) | 13.45 ms (2.63×) | 32.9 ms (1.08×) | 15.0 ms (2.36×) |
| 65536 | 559 ms (1.00×) | 202 ms (2.77×) | 432 ms (1.29×) | **112 ms (4.98×)** |

Keep ratio at τ=1.3 is ~12% (`0.8 → 23%`, `1.0 → 18%`, `2.0 → 5%`).

**The first attempt at this failed, and why is worth recording.** It computed
the proxy and the approximation per tile with fp32 elementwise ops. The proxy
materialised a (64,128) fp32 temporary per KV tile (+27% with everything kept),
and the approximation was a per-tile rank-1 fp32 update costing ~45% of the
dense kernel — so skipping saved nothing at all (keep 50% and keep 12% took the
same time, and the result did not respond to τ). Moving both onto tensor cores,
batched over a group, is what made it work. A normalisation bug went with it:
the approximation's numerator was missing the block-length factor its
denominator had, so skipped blocks were under-weighted; fixing it raised τ=1.3
PSNR from 31.6 to 34.9 dB.

**Verification.** `SPARSE=False` is a *separate* kernel, so the dense path is
untouched. With every block forced exact (`tau=-1000`, `local=1e5`, `sink=1e5`)
the sparse kernel reproduces the dense kernel **bit for bit** (rel-err
`0.000e+00`, `tests/_probe33_sparse_isolate.py`). Against an independent PyTorch
simulation of the same algorithm on the same quantised operands, PSNR agrees to
0.01 dB at every τ (`tests/_probe35_phase3.py`).

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
fused version wins against Kitchen at H3's length it defaults to `front`. It
does **not** help against ComfyUI's own sparse node, which re-asserts itself
every step — see the Verdict.

### Quality: the one gate not passed

On the synthetic `h3_like` benchmark the sparse path drops ~20 dB of PSNR
against dense attention — at *every* τ. **That number is not a valid measure of
the real quality loss.** Synthetic keys inside a 128-token block are
near-orthogonal, so the block-mean proxy is a poor stand-in and skipped blocks
contribute almost nothing. It is also not specific to this port: the independent
fp32 Sol-Attn simulation of the same rule scores *worse* (31.8 dB) than this
kernel (35.5 dB) on the same data. Real keys are correlated within a block,
which is the premise the algorithm relies on.

**End-to-end H3 image/video validation has not been run** and is the one gate
this port has not passed. The pod has the full H3 stack, so it is runnable; it
needs a working H3 graph and a ComfyUI restart to pick up node changes. If you
enable sparsity, compare a dense and a sparse render first, then raise `tau` or
set `enable_sparsity=false` if it degrades.

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
| tokens | ~62.9K target rows at 1344×768×243f (paper quotes ~73.5K incl. conditioning) | the regime where attention dominates the block |

Two deliberate deviations:

1. **Permutations are shared across layers.** A permutation cannot change the
   exact attention result, so computing one per (head, step-window) and reusing
   it over all 50 layers is free mathematically and ~50× cheaper in compute and
   memory. `GroupingConfig.scope` keeps the per-layer option.
2. **`modality_aware` is off by default.** Using the H3 modality tag as the
   primary sort key measured **0.42 dB worse** than plain label sorting: it
   splits clusters that legitimately span modalities. Kept as an option.

Note the k-means permutation is the one thing this port has that ComfyUI's
sparse node does not — but it measured only +0.08…+0.17 dB on structured data,
and it is structurally incompatible with the official H3 path, which tiles the
sequence into video cubes and keeps the conditioning prefix exact.

---

## What is verified, and what is not

Verified on CPU (73 assertions, `tests/`):

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

**RTX 4090 (Ada, sm_89), 2026-10-02.** The fused kernel compiles, runs inside
real ComfyUI sampling, and produces correct output — but lands at **~0.97×
native SDPA** at the kernel level, so on Ada this node is correctness-only. It
needed three fixes first, all now in the source:

1. `LOG2E` / `P_SCALE` must be `tl.constexpr` instances — plain module globals
   fail to compile on Triton ≥ 3.2.
2. The mean-restoration term was missing the `v_scale` factor: `MU` stores
   *mean / v_scale*, so the kernel must add `r * mu * vs`, not `r * mu`. The old
   term drowned the output (~−40 dB).
3. The PV `tl.dot` runs in **bf16**, not fp8: on sm_89 + Triton an fp8 MMA whose
   A operand was computed in registers yields wrong values in every warp/stage
   configuration. Root-caused with minimal repros (`tests/_probe14/15/16`):
   **Triton's fp32→e4m3 `.to()` conversion itself produces wrong codes on
   sm_89** — still broken in Triton 3.8.0. E4M3 → bf16 is exact, so the
   arithmetic is unchanged at half the PV rate.

Verified results there: PSNR vs fp32 SDPA 25.7 dB on i.i.d. tokens (= the
oracle), 58.7 / 57.8 dB on `h3_like` at 8192 / 16384 tokens, V-Smooth gain
+0.18…+0.25 dB on structured data and +0.03 dB on i.i.d. data (≈ 0, as
expected).

**RTX 5090 (sm_120), 2026-10-03.** The two things that blocked a speedup on Ada
both clear on Blackwell: the register-operand fp8 PV dot is correct (the sm_89
conversion bug is absent — max|diff| = 0.0), and `BLOCK_M=64, num_warps=4,
num_stages=2` beats the old `128/8/3` by 1.27×. Kernel 51.7 → 25.0 ms at 16K
tokens at unchanged 57.40 dB. End-to-end (auto → fp8):

| Tokens | bf16 SDPA (=FA) | VC-Attention | ratio | PSNR | V-Smooth gain |
|---|---|---|---|---|---|
| 16384 | 35.1 ms | 32.7 ms | **1.07×** | 57.5 dB | +0.17 dB |
| 32768 | 139.6 ms | 115.4 ms | **1.21×** | 56.0 dB | +0.11 dB |
| 65536 | 557.1 ms | 431.9 ms | **1.29×** | 56.2 dB | +0.08 dB |

The baseline *is* FlashAttention: PyTorch's default SDPA dispatch measures
35.17 ms vs 35.23 ms for an explicit `SDPBackend.FLASH_ATTENTION` at 16K (555.4
vs 555.3 ms at 64K). The ratio climbs with length because `prepare` is linear
while attention is quadratic; H3's real ~63K sequences sit at the 1.29× end.

**Not verified:**

- The paper's RTX 5090 claim (kernel 2.3–3.6×, end-to-end 1.36–1.70×). Both of
  this port's matmuls run at fp8 = 2× the bf16 MMA rate; the paper's number needs
  the *4-bit* path = 4×, and with roughly a third of kernel time being
  softmax/epilogue, ~1.4–1.5× kernel is the ceiling here. The 4-bit path is
  marginal rather than impossible, and precisely why is in "The FP4 question"
  below.
- ExpCast-FP8 branch: compiles but was never exercised — it targets 8-bit
  datacenter cards.
- Real MiniMax-H3 weights. Every PSNR figure in this document comes from
  synthetic tensors shaped like H3.
- End-to-end H3 quality with sparsity enabled (see above).

```bash
python tests/bench_attention.py --tokens 32768 --heads 56 --backend auto
```

---

## The FP4 question

`tl.dot_scaled` with e2m1 + per-16 e4m3 microscales *is* native FP4 hardware on
sm_120, not emulation — in a compute-bound GEMM (K=4096) it runs at **530
TFLOP/s vs 277 for fp8 and 139 for bf16 (3.82×)**. But attention is the wrong
shape for it:

| | prepare | kernel | total | PSNR |
|---|---|---|---|---|
| fp8 QK + bf16 PV | 8.2 ms | 51.7 ms | 59.9 ms | 57.40 dB |
| fp8 QK + fp8 PV | 8.2 ms | **25.0 ms** | **33.2 ms** | 57.40 dB |
| fp4 QK + fp8 PV | **31.3 ms** | 20.2 ms | 51.5 ms | 47.19 dB |

- **The hardware can do it.** sm_120 has a single-instruction converter,
  `cvt.rn.satfinite.e2m1x2.f32`, reachable through `tl.inline_asm_elementwise`,
  measured bit-exact against `quant.fp4_encode` (`tests/_probe28_fp4cvt.py`).
- **But it still does not pay.** With the PTX converter the whole NVFP4 pack
  drops from 11.97 ms to 3.06 ms over a (16384×56, 128) tensor — 3.9× — yet that
  is still **8× the fp8 cast (0.37 ms)**. Q+K packing costs ~5.4 ms against the
  ~4.7 ms the fp4 QK kernel saves: a wash before accuracy is considered.
- **PV is the structural one.** Its A operand is the softmax probability, which
  the kernel already produces as E4M3 — so fp8 PV is free (2× the bf16 rate for
  no extra work), while fp4 would need a genuine e2m1 encode of every tile, and
  `tl.dot_scaled` needs packed operands from memory.
- **And the shape is wrong for it.** QK^T reduces over D=128; fp4 needs a long
  GEMM to shine (3.82× at K=4096, only 1.23× at K=128), and it costs ~10 dB
  (QK rel-err 3.6% → 13.4%).

Making 4-bit actually pay would mean moving the PTX converter *inside* the
kernel (for P), which is real research work, not a flag.

---

## Two findings worth knowing

**Scale granularity is the whole ballgame.** With a per-channel scale computed
over the whole tensor, grouping does nothing — measured 1.01×, because the scale
is set by the tensor's largest entry either way. The value scale must be per
*(value block × channel)*; then homogeneous blocks quantise tightly.

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

## Hardware notes

- **RTX 40-series (Ada, sm_89):** works, correct, but does not speed anything up
  (~0.97× native SDPA at kernel level). Ada has no FP4 tensor cores; the fp8 PV
  path is blocked by Triton's fp32→e4m3 conversion bug; softmax ALU does not
  scale with the tensor cores. Use this node only to exercise or verify the
  algorithm. `auto` resolves to FP8.
- **H200 / B200 / B300** (sm_90 / 10.x): the paper's 8-bit target — kernel
  1.46–1.59× vs BF16 FlashAttention-4, end-to-end 1.13–1.19× on long sequences.
  ExpCast-FP8 applies; the branch is wired but has not been exercised here.
- **RTX 5090 / RTX PRO 6000** (sm_120): the paper's 4-bit target — kernel
  2.3–3.6×, end-to-end 1.36–1.70×. Not reached here; see "The FP4 question".
- **Blackwell datacenter dropped INT4/INT8 MMA**, so `int4` resolves to FP8 on
  10.x; Ada and older keep INT8.

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
    triton_attn.py fused fp8 kernel; dense + grouped-sparse variants
nodes.py         ComfyUI node
tests/
  test_quant.py          32 assertions, CPU  (formats + ExpCast)
  test_vsmooth.py         8 assertions, CPU  (does grouping pay?)
  test_install.py        21 assertions, CPU  (hook, schedule, fallback, node.apply)
  test_prepare.py        12 assertions, CPU  (kernel input layout)
  bench_attention.py     speed + PSNR, needs CUDA
  bench_pv.py            prepare/kernel split, backend matrix, needs CUDA
  tune_attn.py           launch-config sweep, needs CUDA
  _probe22_solstack.py   does Sol-Attn routing compose with the quantisation?
  _probe26/27_*.py       Kitchen timing; node-stacking gain
  _probe30_tilescale.py  kernel time vs number of KV tiles (the sparsity ceiling)
  _probe33_sparse_isolate.py   is the kept branch exact?
  _probe35_phase3.py     sparse gates G2/G3 + tau/speed/PSNR curve
  _probe37_official_sparse.py  official ck.sol_attn vs Kitchen vs this port
  _probe38_h3_block_budget.py  where an H3 block's time goes
  _probe14/15/16.py      sm_89 register-fp8 cvt bug repros
  _probe18/19/20/28/29_*.py    NVFP4 layouts, QK fidelity, PTX cvt/pack
```

## Caveats

- The attention hook is **process-global**. Fine for a single-user sampling run;
  call `uninstall()` to put everything back.
- **ComfyUI + MiniMax-H3 needs the by-value-import rebind.** comfy's own H3
  port does `from comfy.ldm.modules.attention import optimized_attention` at
  module load, so replacing the attribute on the defining module is not enough —
  the model keeps calling the original forever and the node is silently inert.
  `install()` walks `sys.modules` and rebinds every comfy module that still holds
  the original object; the wrapper also handles `skip_reshape=True` 4-D inputs,
  and since comfy 0.38 those arrive wrapped in single-owner
  `AttentionTensorContainer`s on every H3 call — the hook peeks into them,
  defers to any registered `optimized_attention_override`, and falls back to the
  original on any internal error so it can never take a run down. **Apply the
  node before the first sampling after a restart**, and note that ComfyUI's
  torch.compile traces whatever function objects are installed at compile time —
  if the graph was compiled before the node ran, re-trace (restart) or disable
  the compiler (`--disable-comfy-compiler`).
- Only non-causal attention is accelerated. Calls with an `attn_mask`, or
  causal, or below `min_tokens`, or with head_dim ∉ {64, 128} go straight to the
  original SDPA. H3's per-call sequence can fall below `min_tokens` on short
  segments — watch the stats, not the console banner.
- H3's padding rows (tag −1) need a masked backend; this node does not
  accelerate masked attention yet. The diffusers port runs unmasked.
