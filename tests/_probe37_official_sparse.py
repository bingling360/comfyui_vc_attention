"""Probe 37: the OFFICIAL ComfyUI block-sparse backend vs Kitchen dense INT8 vs
our fused VC sparse.

comfy_extras/nodes_sparse_attention.py ("Model Sparse Attention") runs on
comfy_kitchen's own sparse kernels -- ck.sol_attn / ck.sol_attn_chunked -- with
Sol-Attn's adaptive tau (plus SLA top-k and FastVideo VSA). So "Kitchen" and
"the official block-sparse backend" are the SAME kernel library, not two things
to stack. This probe measures the sparse kernel head-to-head with ours.

Ordering matters on this pod: Kitchen must be called before bf16 SDPA (the
libcublasLt shape-error bug), so every Kitchen call runs first and the fp32
reference is computed last.
"""
import inspect
import sys

import torch

sys.path.insert(0, "/root/ComfyUI")
sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention")
sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention/tests")

import comfy.ldm.modules.attention as A  # noqa: E402
import comfy_kitchen as ck  # noqa: E402
from bench_attention import h3_like, psnr, timeit  # noqa: E402
from vc_attention.grouping import GroupingConfig, build_permutation  # noqa: E402
from vc_attention.kernels.triton_attn import TritonConfig, vc_attention_triton  # noqa: E402

H, D, QB, BN = 56, 128, 64, 128

print("comfy_kitchen:", getattr(ck, "__version__", "?"), flush=True)
print("sol_attn_is_available:", ck.sol_attn_is_available(torch.device("cuda")), flush=True)
try:
    print("ck.sol_attn sig:", inspect.signature(ck.sol_attn), flush=True)
except Exception as e:  # noqa: BLE001
    print("sig introspect failed:", e, flush=True)
try:
    print("ck.sol_attn_chunked sig:", inspect.signature(ck.sol_attn_chunked), flush=True)
except Exception as e:  # noqa: BLE001
    print("chunked sig introspect failed:", e, flush=True)

for n in (16384, 65536):
    q, k, v = h3_like(n, H, D, "cuda")          # (1, H, N, D) bf16
    perm = build_permutation(v.reshape(H, n, D), GroupingConfig(block_rows=BN, iters=3)).perm

    # BTHD, which is what the official node hands to ck.sol_attn.
    qs = q.transpose(1, 2).contiguous()
    ks = k.transpose(1, 2).contiguous()
    vs = v.transpose(1, 2).contiguous()

    print(f"\n=== tokens={n}  heads={H}  head_dim={D} ===", flush=True)
    outs, times = {}, {}

    def bench(name, fn):
        try:
            o = fn()
            torch.cuda.synchronize()
            t = timeit(fn, warmup=2, repeat=5)
            outs[name] = o
            times[name] = t
            print(f"  {name:34}: {t:9.2f} ms", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {name:34}: FAIL {type(e).__name__}: {str(e)[:110]}", flush=True)

    # Kitchen-family first (ordering bug), then Triton, then the fp32 reference.
    bench("Kitchen dense INT8", lambda: A.attention_comfy_kitchen_int8(
        q, k, v, H, skip_reshape=True, skip_output_reshape=True))
    bench("official ck.sol_attn tau=1.3", lambda: ck.sol_attn(
        qs, ks, vs, tau=1.3, topk_ratio=0.0))
    # The node's real defaults: extra_tokens=256 (token_aug) pulls it toward dense.
    bench("official tau=1.3 +token_aug256", lambda: ck.sol_attn(
        qs, ks, vs, tau=1.3, topk_ratio=0.0, token_aug=256))
    bench("official ck.sol_attn tau=1.0", lambda: ck.sol_attn(
        qs, ks, vs, tau=1.0, topk_ratio=0.0))
    bench("official ck.sol_attn tau=2.0", lambda: ck.sol_attn(
        qs, ks, vs, tau=2.0, topk_ratio=0.0))
    for tau in (1.0, 1.3, 2.0):
        bench(f"VC fused sparse tau={tau}", lambda tau=tau: vc_attention_triton(
            q, k, v, perm=perm,
            cfg=TritonConfig(block_m=QB, block_n=BN, num_warps=4, num_stages=2,
                             sparse=True, tau=tau)))
    bench("VC dense (fp8)", lambda: vc_attention_triton(
        q, k, v, perm=perm,
        cfg=TritonConfig(block_m=QB, block_n=BN, num_warps=4, num_stages=2, sparse=False)))

    ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
    bench("bf16 FlashAttention", lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))

    print("  --- quality (PSNR vs fp32 dense) ---", flush=True)
    base = times.get("bf16 FlashAttention")
    for name, o in outs.items():
        if o is None:
            continue
        # Normalise to BHND: the official kernel returns BTHD, Kitchen returns
        # (B, N, H*D) unless skip_output_reshape was set.
        oo = o
        if oo.shape == (1, H, n, D):
            pass
        elif oo.shape == (1, n, H, D):
            oo = oo.transpose(1, 2)
        elif oo.shape == (1, n, H * D):
            oo = oo.view(1, n, H, D).transpose(1, 2)
        else:
            print(f"    {name:34}: shape {tuple(o.shape)} not comparable", flush=True)
            continue
        t = times[name]
        r = f"{base / t:5.2f}x" if base else "  -  "
        print(f"    {name:34}: PSNR {psnr(oo, ref):6.2f} dB   {r} vs SDPA", flush=True)

    del q, k, v, qs, ks, vs, ref, outs
    torch.cuda.empty_cache()
