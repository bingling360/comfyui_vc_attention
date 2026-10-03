"""Probe 38: where does a MiniMax-H3 block actually spend its time?

The official sparse kernel already took attention from 556 ms to 30.7 ms at 64K
tokens. Before designing anything to "assist" it, it is worth knowing how much
of a block is still attention -- MiniMax-H3's projections are wide (hidden 5376,
qkv 7168, ffn 14336), so the FFN and the QKV projection may dominate by now.

This times the GEMMs of the real H3 shapes at 16K and 64K tokens and prints the
per-layer breakdown next to the measured attention times from probe37. No model
weights needed -- shape and dtype are what matter.

MiniMax-H3 (transformer/config.json): hidden 5376, 56 heads x 128 = 7168 qkv,
ffn 14336, 50 blocks, 8-step distilled LoRA.
"""
import sys

import torch

sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention")
sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention/tests")

from bench_attention import timeit  # noqa: E402

HID, QKV, FFN, BLOCKS, STEPS = 5376, 7168, 14336, 50, 8

# Attention times measured in probe37 (RTX 5090, 56 heads, D=128).
ATTN = {
    16384: {"Kitchen dense INT8": 13.21, "official ck.sol_attn t=1.3": 3.02},
    65536: {"Kitchen dense INT8": 201.94, "official ck.sol_attn t=1.3": 30.66},
}


def gemm(tokens, din, dout, dtype, label):
    """One linear layer at this token count."""
    a = torch.randn(tokens, din, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(din, dout, device="cuda", dtype=torch.bfloat16)
    if dtype == torch.bfloat16:
        aa, ww = a, w
    elif dtype == torch.float8_e4m3fn:
        aa = (a / a.abs().amax()).to(dtype)
        ww = (w / w.abs().amax()).to(dtype)
    else:
        raise ValueError(dtype)
    try:
        t = timeit(lambda: torch.matmul(aa, ww), warmup=2, repeat=5)
    except Exception as e:  # noqa: BLE001
        print(f"    {label:26} {str(dtype):>22}: FAIL {type(e).__name__}: {str(e)[:60]}", flush=True)
        del a, w
        torch.cuda.empty_cache()
        return None
    del a, w, aa, ww
    torch.cuda.empty_cache()
    return t


for n in (16384, 65536):
    print(f"\n=== tokens={n}  (MiniMax-H3 block shapes) ===", flush=True)
    parts = {}
    for dtype, tag in ((torch.bfloat16, "bf16"), (torch.float8_e4m3fn, "fp8")):
        qkv = gemm(n, HID, 3 * QKV, dtype, "qkv_proj")
        oproj = gemm(n, QKV, HID, dtype, "out_proj")
        ffn1 = gemm(n, HID, FFN, dtype, "ffn_up")
        ffn2 = gemm(n, FFN, HID, dtype, "ffn_down")
        if None in (qkv, oproj, ffn1, ffn2):
            continue
        proj_total = qkv + oproj
        ffn_total = ffn1 + ffn2
        print(f"  [{tag}] qkv_proj {qkv:7.2f}  out_proj {oproj:7.2f}  "
              f"ffn {ffn_total:7.2f}  (ffn_up {ffn1:6.2f} / down {ffn2:6.2f})  "
              f"-> projections {proj_total:7.2f}, non-attention total "
              f"{proj_total + ffn_total:7.2f} ms", flush=True)
        parts[tag] = proj_total + ffn_total

    print("  --- per-block budget ---", flush=True)
    for tag, nonattn in parts.items():
        for aname, at in ATTN[n].items():
            tot = nonattn + at
            print(f"    {tag} GEMMs + {aname:26}: block {tot:7.2f} ms  "
                  f"(attention {at / tot * 100:4.1f}%)  "
                  f"x{BLOCKS} layers x{STEPS} steps = {tot * BLOCKS * STEPS / 1000:6.1f} s", flush=True)

    # Upper bound: what if attention became free?
    for tag, nonattn in parts.items():
        base = nonattn + ATTN[n]["Kitchen dense INT8"]
        best = nonattn + ATTN[n]["official ck.sol_attn t=1.3"]
        zero = nonattn
        print(f"    [{tag}] Kitchen->official {base / best:4.2f}x   "
              f"official->free (upper bound) {best / zero:4.2f}x   "
              f"so <= {(1 - zero / best) * 100:.1f}% is left in attention", flush=True)
    torch.cuda.empty_cache()
