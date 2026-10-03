"""Installing VC-Attention into a running model.

ComfyUI has no per-model attention hook that covers every DiT, and MiniMax-H3
in particular arrives as a plain diffusers ``MiniMaxH3Transformer3DModel``. So
the installation is deliberately blunt:

1. Wrap the denoiser's ``forward`` to count denoising steps. Grouping is
   scheduled off the step index, and nothing else in the call stack knows it.
2. Replace :func:`torch.nn.functional.scaled_dot_product_attention` (and
   ComfyUI's ``optimized_attention`` when present) with a router that picks
   VC-Attention only for calls it can handle and forwards everything else to
   the original implementation unchanged.

Shape-based routing means unrelated attention calls in the same process (the
text encoder, for instance) are left alone unless they happen to match the
head-dimension and token-count conditions, which is the intended behaviour: the
conditions are exactly the ones under which low-bit attention pays off.

The hook is global. That is fine for a single-user sampling run, which is what
ComfyUI is; call :func:`uninstall` to put everything back.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import torch

from .device import detect_profile, resolve_backend
from .grouping import GroupingConfig, build_permutation
from .h3 import TAG
from .schedule import GroupSchedule, PermutationCache, StepTracker

__all__ = [
    "VCAttentionConfig", "VCAttentionRuntime", "get_runtime",
    "install", "uninstall", "is_installed",
]

_ORIGINALS: Dict[str, Callable] = {}
_NOTIFIED: set = set()


def _notify_once(msg: str) -> None:
    """Print once per unique message; the router must fail loudly, not silently."""
    if msg not in _NOTIFIED and len(_NOTIFIED) < 16:
        _NOTIFIED.add(msg)
        print(f"[VC-Attention] {msg}")

# comfy >= 0.38 wraps q/k/v in single-owner containers before calling
# optimized_attention (MiniMax-H3 does this on every call). Optional: the
# package must keep importing (and falling back to plain tensors) without it.
try:
    from comfy.ldm.modules.attention import AttentionTensorContainer as _ATTN_CONTAINER
except Exception:  # pragma: no cover
    _ATTN_CONTAINER = None


@dataclass
class VCAttentionConfig:
    enabled: bool = True
    backend: str = "auto"            # auto | fp8 | nvfp4 | int4 | reference
    enable_vsmooth: bool = True
    enable_expcast: bool = False     # 8-bit datacenter only; see expcast.py
    block_rows: int = 128            # == H3 attention_head_dim
    block_m: int = 64                # tuned on sm_120 (see TritonConfig)
    min_tokens: int = 8192
    hadamard: bool = True
    smooth_k: bool = True

    # grouping
    kmeans_iters: int = 3
    warm_iters: int = 2
    modality_aware: bool = False     # measured slightly worse; see grouping.py
    group_fraction: float = 0.25
    reuse_every: int = 4
    total_steps_hint: int = 8        # H3 distilled LoRA

    # Without a fused kernel the PyTorch reference path is a Python loop over
    # tiles and is *slower* than SDPA. So by default the node simply does
    # nothing when no kernel is available rather than making sampling slower.
    # Set this to True only to exercise the algorithm itself.
    allow_slow_fallback: bool = False

    supported_head_dims: tuple = (64, 128)


class VCAttentionRuntime:
    """Owns the schedule, the permutation cache and the patched callables."""

    def __init__(self, config: Optional[VCAttentionConfig] = None):
        self.config = config or VCAttentionConfig()
        self.profile = detect_profile()
        self.backend = resolve_backend(self.config.backend, self.profile)
        self.schedule = GroupSchedule(
            group_fraction=self.config.group_fraction,
            reuse_every=self.config.reuse_every,
            total_steps_hint=self.config.total_steps_hint,
            min_tokens=self.config.min_tokens,
        )
        self.tracker = StepTracker(total_steps_hint=self.config.total_steps_hint)
        self.cache = PermutationCache()
        self.tags: Optional[torch.Tensor] = None
        self.stats: Dict[str, int] = {"calls": 0, "used": 0, "skipped": 0, "grouped": 0}

    # -- schedule ----------------------------------------------------------
    def begin_step(self) -> int:
        return self.tracker.begin_step()

    def observe_total_steps(self, total: int) -> None:
        self.tracker.observe_total(total)

    def reset(self) -> None:
        self.tracker.reset()
        self.cache.clear()

    def set_tags(self, tags: Optional[torch.Tensor]) -> None:
        """Per-row modality tags of the packed sequence (H3: 0 video, 1 text, 2 audio)."""
        self.tags = tags

    # -- permutation -------------------------------------------------------
    def permutation(self, v: torch.Tensor):
        """Return a (G, N) permutation, or None when grouping is not active."""
        cfg = self.config
        if not cfg.enable_vsmooth or not self.schedule.should_group(
            self.tracker.current, self.tracker.total_steps
        ):
            return None

        b, h, n, d = v.shape
        G = b * h
        bucket = self.schedule.window_index(self.tracker.current)
        cached = self.cache.get(bucket, n, G)
        if cached is not None and not self.schedule.should_recompute(self.tracker.current):
            return cached[0]

        gcfg = GroupingConfig(
            block_rows=cfg.block_rows,
            iters=cfg.kmeans_iters if cached is None else cfg.warm_iters,
            warm_iters=cfg.warm_iters,
            modality_aware=cfg.modality_aware,
        )
        warm = None if cached is None else cached[1]
        res = build_permutation(v.reshape(G, n, d), gcfg, tags=self.tags, centroids=warm)
        self.cache.put(bucket, n, G, res.perm, res.centroids)
        return res.perm

    # -- attention ---------------------------------------------------------
    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
        attn_mask: Any = None,
        scale: Optional[float] = None,
        dropout_p: float = 0.0,
    ) -> Optional[torch.Tensor]:
        """Return the VC-Attention result, or None to defer to the original."""
        self.stats["calls"] += 1
        cfg = self.config
        if not cfg.enabled or is_causal or attn_mask is not None:
            self.stats["skipped"] += 1
            return None
        if q.shape != k.shape or q.shape != v.shape or q.dim() != 4:
            self.stats["skipped"] += 1
            return None
        b, h, n, d = q.shape
        if d not in cfg.supported_head_dims or n < cfg.min_tokens:
            self.stats["skipped"] += 1
            _notify_once(f"call n={n} (head_dim={d}) below min_tokens="
                         f"{cfg.min_tokens} -> native SDPA")
            return None
        if q.device.type != "cuda":
            self.stats["skipped"] += 1
            return None

        # A missing fused kernel must never make sampling slower than baseline.
        if self.backend != "reference" and not self.config.allow_slow_fallback \
                and not fast_path_available():
            self.stats["skipped"] += 1
            return None

        perm = self.permutation(v)
        if perm is not None:
            self.stats["grouped"] += 1
        self.stats["used"] += 1

        from .kernels.reference import RefConfig, vc_attention_reference
        from .kernels.triton_attn import TritonConfig, vc_attention_triton

        if self.backend == "reference":
            return vc_attention_reference(
                q, k, v,
                RefConfig(
                    backend="fp8",
                    block_rows=cfg.block_rows,
                    block_m=64,
                    enable_vsmooth=perm is not None,
                    enable_expcast=False,
                    hadamard=cfg.hadamard,
                    smooth_k=cfg.smooth_k,
                ),
                perm=perm,
                scale=scale,
            )

        tcfg = TritonConfig(
            block_m=cfg.block_m,
            block_n=cfg.block_rows,
            enable_expcast=bool(cfg.enable_expcast and self.backend == "fp8"),
            # "nvfp4" additionally moves QK^T onto the FP4 tensor cores. On
            # sm_120 that is a *net loss* for attention: the kernel gets 1.23x
            # faster but host-side NVFP4 quantization costs ~23 ms per layer
            # (no native float->fp4 cast exists) and QK fidelity drops ~10 dB.
            # Kept selectable because the paper specifies it and the tradeoff
            # depends on the sequence shape; see tests/bench_pv.py.
            pv_fp8=(True if self.backend == "nvfp4" else None),
            qk_fp4=(self.backend == "nvfp4"),
        )
        return vc_attention_triton(q, k, v, perm=perm, cfg=tcfg, scale=scale)


def fast_path_available() -> bool:
    """True when a fused kernel can actually run: CUDA *and* Triton.

    ComfyUI's Windows builds frequently ship without Triton (it is a separate
    ``triton-windows`` install). In that state the only implementation available
    is the portable PyTorch reference, which is slower than SDPA, so the router
    defers to the original attention instead of using it.
    """
    try:
        from .kernels.triton_attn import has_triton
    except Exception:
        return False
    return bool(has_triton()) and torch.cuda.is_available()


def kernel_status() -> str:
    """One-line status for the console: what will actually run."""
    try:
        from .kernels.triton_attn import has_triton
    except Exception:
        has_triton = lambda: False  # noqa: E731
    if not torch.cuda.is_available():
        return "no CUDA visible -> VC-Attention inactive, using native SDPA"
    if not has_triton():
        return ("CUDA present but Triton is missing -> VC-Attention inactive, "
                "using native SDPA (install triton to enable it)")
    return f"fused kernel available on {torch.cuda.get_device_name(0)}"


_RUNTIME = VCAttentionRuntime()


def get_runtime() -> VCAttentionRuntime:
    return _RUNTIME


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

def _make_hook(runtime: VCAttentionRuntime, orig: Callable) -> Callable:
    @functools.wraps(orig)
    def hooked(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
               scale=None, *args, **kwargs):
        if not runtime.config.enabled:
            return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                        is_causal=is_causal, scale=scale, *args, **kwargs)
        try:
            out = runtime.attention(
                query, key, value, is_causal=is_causal,
                attn_mask=attn_mask, scale=scale, dropout_p=dropout_p,
            )
        except Exception:
            out = None
        if out is None:
            return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                        is_causal=is_causal, scale=scale, *args, **kwargs)
        return out

    return hooked


def install(config: Optional[VCAttentionConfig] = None, model: Any = None) -> VCAttentionRuntime:
    """Patch attention globally and optionally count steps on ``model``."""
    global _RUNTIME
    if config is not None:
        _RUNTIME = VCAttentionRuntime(config)
    runtime = _RUNTIME

    if "sdpa" not in _ORIGINALS and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        orig = torch.nn.functional.scaled_dot_product_attention
        _ORIGINALS["sdpa"] = orig
        torch.nn.functional.scaled_dot_product_attention = _make_hook(runtime, orig)
        # ``torch.scaled_dot_product_attention`` exists in some builds only.
        if hasattr(torch, "scaled_dot_product_attention"):
            _ORIGINALS["sdpa_torch"] = torch.scaled_dot_product_attention
            torch.scaled_dot_product_attention = orig

    # ComfyUI routes many models through this helper instead of calling SDPA.
    try:
        from comfy.ldm.modules import attention as comfy_attention  # type: ignore

        if "comfy" not in _ORIGINALS and hasattr(comfy_attention, "optimized_attention"):
            orig = comfy_attention.optimized_attention
            _ORIGINALS["comfy"] = orig

            @functools.wraps(orig)
            def hooked(q, k, v, heads, *args, **kwargs):
                try:
                    t_opts = kwargs.get("transformer_options")
                    if isinstance(t_opts, dict) and \
                            t_opts.get("optimized_attention_override") is not None:
                        # Another attention override owns this call. That is how
                        # both the Comfy Kitchen backend node (ModelSampling* ->
                        # set_model_optimized_attention) and Sol-Attn's generic
                        # node register themselves, and Sol chains whatever was
                        # already there as its fallback. VC-Attention sits
                        # *outside* that chain, so it always loses -- say so,
                        # because otherwise the node looks enabled and does
                        # nothing. Measured: tests/_probe21_kitchen.py.
                        _notify_once(
                            "another attention backend is active "
                            "(`optimized_attention_override`: Comfy Kitchen or "
                            "Sol-Attn) -> VC-Attention is INACTIVE for those "
                            "calls. Remove that node, or disable VC-Attention."
                        )
                        return orig(q, k, v, heads, *args, **kwargs)

                    if _ATTN_CONTAINER is not None and isinstance(q, _ATTN_CONTAINER):
                        # comfy 0.38: MiniMax-H3 wraps every call's q/k/v in
                        # single-owner containers. peek() reads without consuming,
                        # so if we defer below, orig (which take()s them) still
                        # works — and orig also accepts plain tensors directly.
                        q, k, v = q.peek(), k.peek(), v.peek()

                    if not (isinstance(q, torch.Tensor) and isinstance(k, torch.Tensor)
                            and isinstance(v, torch.Tensor)):
                        return orig(q, k, v, heads, *args, **kwargs)

                    if q.dim() == 4:
                        # skip_reshape=True callers (MiniMax-H3 among them) pass
                        # (B, heads, S, D) already split.
                        b, h_, s_, d_ = q.shape
                        out = runtime.attention(q, k, v)
                        if out is None:
                            return orig(q, k, v, heads, *args, **kwargs)
                        return out.transpose(1, 2).reshape(b, s_, h_ * d_)

                    b, n, _ = q.shape
                    d = q.shape[-1] // heads
                    q4 = q.view(b, n, heads, d).transpose(1, 2)
                    k4 = k.view(k.shape[0], k.shape[1], heads, d).transpose(1, 2)
                    v4 = v.view(v.shape[0], v.shape[1], heads, d).transpose(1, 2)
                    out = runtime.attention(q4, k4, v4)
                    if out is None:
                        return orig(q, k, v, heads, *args, **kwargs)
                    return out.transpose(1, 2).reshape(b, n, heads * d)
                except Exception as e:
                    # The hook must never take sampling down with it. peek()
                    # consumed nothing, so orig is always callable here — but
                    # say why, or the node looks silently inert.
                    _notify_once(f"deferring to original attention after "
                                 f"{type(e).__name__}: {e}")
                    return orig(q, k, v, heads, *args, **kwargs)

            comfy_attention.optimized_attention = hooked
            _rebind_by_value_imports(orig, hooked)
    except Exception:
        pass

    if model is not None:
        _install_step_counter(model, runtime)
    return runtime


def _install_step_counter(model: Any, runtime: VCAttentionRuntime) -> None:
    """Wrap the denoiser forward so the runtime knows the step index.

    One call == one denoising step for the DiT-style models this targets: CFG is
    a batch duplication inside the call, not a second call.
    """
    denoiser = None
    for candidate in (
        getattr(model, "diffusion_model", None),
        getattr(getattr(model, "model", None), "diffusion_model", None),
    ):
        if candidate is not None and hasattr(candidate, "forward"):
            denoiser = candidate
            break
    if denoiser is None or getattr(denoiser, "_vc_step_wrapped", False):
        return

    orig_forward = denoiser.forward

    @functools.wraps(orig_forward)
    def wrapped(*args, **kwargs):
        runtime.begin_step()
        sigmas = None
        t_opts = kwargs.get("transformer_options") or {}
        if isinstance(t_opts, dict):
            sigmas = t_opts.get("sample_sigmas")
        if sigmas is not None:
            try:
                runtime.observe_total_steps(len(sigmas) - 1)
            except Exception:
                pass
        return orig_forward(*args, **kwargs)

    wrapped._vc_step_wrapped = True          # type: ignore[attr-defined]
    denoiser.forward = wrapped
    denoiser._vc_step_wrapped = True         # type: ignore[attr-defined]
    _ORIGINALS.setdefault("denoiser", orig_forward)


def _rebind_by_value_imports(orig: Callable, hooked: Callable) -> None:
    """Rebind ``optimized_attention`` in modules that imported it by value.

    ``from comfy.ldm.modules.attention import optimized_attention`` binds the
    function object into the importer's namespace at import time, so replacing
    the attribute on the defining module is not enough — comfy's own
    MiniMax-H3 port (``comfy.ldm.minimax.model``) imports exactly that way and
    would keep calling the original forever. Walk comfy's modules and swap any
    namespace that still holds the original object.
    """
    import sys

    for name, mod in list(sys.modules.items()):
        if not name.startswith("comfy"):
            continue
        mdict = getattr(mod, "__dict__", None)
        if mdict and mdict.get("optimized_attention", None) is orig:
            mdict["optimized_attention"] = hooked


def uninstall() -> None:
    """Restore every patched callable."""
    if "sdpa" in _ORIGINALS:
        torch.nn.functional.scaled_dot_product_attention = _ORIGINALS.pop("sdpa")
    if "sdpa_torch" in _ORIGINALS:
        torch.scaled_dot_product_attention = _ORIGINALS.pop("sdpa_torch")
    if "comfy" in _ORIGINALS:
        try:
            from comfy.ldm.modules import attention as comfy_attention  # type: ignore

            comfy_attention.optimized_attention = _ORIGINALS.pop("comfy")
        except Exception:
            _ORIGINALS.pop("comfy", None)
    _ORIGINALS.pop("denoiser", None)


def is_installed() -> bool:
    return "sdpa" in _ORIGINALS
