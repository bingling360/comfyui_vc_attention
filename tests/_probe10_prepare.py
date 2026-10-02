"""Probe 10: decompose prepare() cost, validate an optimized fast path.

Stages of the current prepare (per call, G=56 N=16384 D=128):
  float upcasts, per-head gather (expanded int64 index), fwht x2 (7 cat stages),
  smooth_k mean, per-token E4M3 x2 (portable int64-heavy encoder), block demean,
  per-(block x channel) E4M3, scales.

Fast path replaces: fp32 pipeline -> bf16 pipeline; portable e4m3_encode ->
native `.to(float8_e4m3fn)` (README's own test says byte-identical); fwht ->
single matmul by the explicit orthonormal matrix; expanded-index gather ->
broadcast take_along_dim; repeat_interleave demean -> broadcast subtract.
"""
import math
import os
import sys
import time

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/tcache10")
import torch
import triton

sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention")
from vc_attention.kernels.triton_attn import Prepared, _vc_attn_fwd, prepare
from vc_attention.grouping import GroupingConfig, build_permutation
from vc_attention import quant as Q

torch.manual_seed(0)
T, H, D = 16384, 56, 128
BR = 128
dev = "cuda"
q = torch.randn(1, H, T, D, device=dev, dtype=torch.bfloat16)
k = torch.randn(1, H, T, D, device=dev, dtype=torch.bfloat16)
v = torch.randn(1, H, T, D, device=dev, dtype=torch.bfloat16)
ref_sdpa = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
perm = build_permutation(v.reshape(H, T, D), GroupingConfig(block_rows=BR, iters=3)).perm


def timeit(fn, warmup=2, repeat=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat * 1000


def psnr(out, ref):
    mse = ((out.float() - ref.float()) ** 2).mean()
    if not torch.isfinite(out.float()).all():
        return float("nan")
    return float(10 * torch.log10(ref.float().pow(2).mean() / mse.clamp(min=1e-30)))


# ---------------------------------------------------------------------------
# 1. stage-by-stage decomposition of the current prepare
# ---------------------------------------------------------------------------
def decompose():
    G = H
    n = T
    timings = {}

    def t(name, fn):
        timings[name] = timeit(fn, warmup=1, repeat=3)

    qf = q.reshape(G, n, D).float()
    kf = k.reshape(G, n, D).float()
    vf = v.reshape(G, n, D).float()
    p = perm.to(torch.int64).to(dev)
    idx = p.unsqueeze(-1).expand(G, n, D)

    t("float upcast x3", lambda: (q.reshape(G, n, D).float(), k.reshape(G, n, D).float(), v.reshape(G, n, D).float()))
    t("gather k (expand idx)", lambda: kf.gather(1, idx))
    t("gather k (take_along)", lambda: torch.take_along_dim(kf, p.unsqueeze(-1), dim=1))
    t("gather v (expand idx)", lambda: vf.gather(1, idx))
    t("fwht q (7 cat stages)", lambda: Q_fwht(qf))
    t("e4m3_encode q row", lambda: Q.e4m3_encode(qf / 0.01))
    t("native cast q row", lambda: (qf / 0.01).to(torch.float8_e4m3fn))
    vb = vf.reshape(G, n // BR, BR, D)
    mu = vb.mean(dim=2)
    t("block mean", lambda: vb.mean(dim=2))
    t("resid (repeat_interleave)", lambda: vf - mu.repeat_interleave(BR, dim=1))
    mub = mu.unsqueeze(2)
    t("resid (broadcast)", lambda: vb - mub)
    return timings


def Q_fwht(x):
    batch = x.shape[:-1]
    d = x.shape[-1]
    out = x
    h = 1
    while h < d:
        out = out.reshape(*batch, -1, 2, h)
        a = out[..., 0, :]
        b = out[..., 1, :]
        out = torch.cat((a + b, a - b), dim=-1)
        h *= 2
    return out.reshape(x.shape) / math.sqrt(d)


print("=== current prepare stage decomposition (ms) ===")
for name, ms in decompose().items():
    print(f"  {name:28s}: {ms:8.2f}")

# ---------------------------------------------------------------------------
# 2. fast prepare
# ---------------------------------------------------------------------------
_HM_CACHE = {}


def _hadamarian(d, device):
    key = (d, str(device))
    if key not in _HM_CACHE:
        h = torch.ones((1, 1), device=device, dtype=torch.bfloat16)
        while h.shape[0] < d:
            h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
        _HM_CACHE[key] = (h / math.sqrt(d)).contiguous()
    return _HM_CACHE[key]


def _native_e4m3(x, dim):
    """Per-row symmetric E4M3 via the hardware cast. x bf16, scale fp32 out."""
    amp = x.detach().abs().amax(dim=dim, keepdim=True).float().clamp(min=1e-30) / 448.0
    codes = (x / amp.to(torch.bfloat16)).to(torch.float8_e4m3fn)
    return codes, amp


def prepare_fast(q, k, v, perm=None, block_rows=128, hadamard=True, smooth_k=True):
    b, h, n, d = q.shape
    G = b * h
    dev = q.device
    n_pad = -(-n // block_rows) * block_rows
    nb = n_pad // block_rows

    qf = q.reshape(G, n, d)
    kf = k.reshape(G, n, d)
    vf = v.reshape(G, n, d)

    if perm is not None:
        p = perm.to(torch.int64).to(dev)
        if p.shape[0] == 1 and G > 1:
            p = p.expand(G, n)
        kf = torch.take_along_dim(kf, p.unsqueeze(-1), dim=1)
        vf = torch.take_along_dim(vf, p.unsqueeze(-1), dim=1)

    if hadamard:
        Hm = _hadamarian(d, dev)
        q_t = torch.matmul(qf, Hm)
        k_t = torch.matmul(kf, Hm)
    else:
        q_t, k_t = qf, kf
    if smooth_k:
        k_t = k_t - k_t.mean(dim=1, keepdim=True)

    q_codes, q_scale = _native_e4m3(q_t, dim=-1)
    k_codes, k_scale = _native_e4m3(k_t, dim=-1)
    q_scale = q_scale.reshape(G, n)
    k_scale = k_scale.reshape(G, n)

    pad = n_pad - n
    if pad:
        vf = torch.nn.functional.pad(vf, (0, 0, 0, pad))
        q_codes = torch.nn.functional.pad(q_codes, (0, 0, 0, pad))
        k_codes = torch.nn.functional.pad(k_codes, (0, 0, 0, pad))
        q_scale = torch.nn.functional.pad(q_scale, (0, pad))
        k_scale = torch.nn.functional.pad(k_scale, (0, pad))

    vb = vf.reshape(G, nb, block_rows, d)
    mu = vb.float().mean(dim=2)                       # fp32 (G,NB,D) small
    resid = vb - mu.to(torch.bfloat16).unsqueeze(2)   # bf16 broadcast, no repeat
    amp = resid.detach().abs().amax(dim=2).float().clamp(min=1e-30)  # (G,NB,D)
    v_scale = amp / 448.0
    v_codes = (resid / v_scale.to(torch.bfloat16).unsqueeze(2)).to(torch.float8_e4m3fn)
    mu_over_scale = mu / v_scale

    return Prepared(
        q=q_codes.reshape(b, h, n_pad, d).contiguous(),
        q_scale=q_scale.reshape(b, h, n_pad).contiguous().float(),
        k=k_codes.reshape(b, h, n_pad, d).contiguous(),
        k_scale=k_scale.reshape(b, h, n_pad).contiguous().float(),
        v=v_codes.reshape(b, h, n_pad, d).contiguous(),
        v_scale=v_scale.reshape(b, h, nb, d).contiguous().float(),
        mu=mu_over_scale.reshape(b, h, nb, d).contiguous().float(),
        n_pad=n_pad,
    )


print("\n=== timing @ 16384 tokens ===")
t_oracle = timeit(lambda: prepare(q, k, v, perm, block_rows=BR, hadamard=True), repeat=3)
p_o = prepare(q, k, v, perm, block_rows=BR, hadamard=True)
t_fast = timeit(lambda: prepare_fast(q, k, v, perm, block_rows=BR, hadamard=True), repeat=5)
p_f = prepare_fast(q, k, v, perm, block_rows=BR, hadamard=True)
t_sdpa = timeit(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))
print(f"  prepare (oracle) : {t_oracle:8.2f} ms")
print(f"  prepare (fast)   : {t_fast:8.2f} ms   ({t_oracle/t_fast:.1f}x faster)")
print(f"  bf16 SDPA        : {t_sdpa:8.2f} ms")


def run_kernel(p, bm=128, warps=8, stages=3):
    out = torch.empty_like(q)
    grid = (triton.cdiv(p.n_pad, bm), H)
    _vc_attn_fwd[grid](p.q, p.q_scale, p.k, p.k_scale, p.v, p.v_scale, p.mu, out,
                       D ** -0.5, T, p.n_pad, D, BLOCK_M=bm, BLOCK_N=BR,
                       EXPCAST=False, BETA=-0.35, num_warps=warps, num_stages=stages)
    return out


out_o = run_kernel(p_o)
out_f = run_kernel(p_f)
print(f"\n=== end-to-end correctness ===")
print(f"  kernel(oracle prepare) PSNR : {psnr(out_o, ref_sdpa):.2f} dB")
print(f"  kernel(fast prepare)   PSNR : {psnr(out_f, ref_sdpa):.2f} dB")
print(f"  outputs differ maximally by  : {float((out_o.float()-out_f.float()).abs().max()):.2e}")
t_kern = timeit(lambda: run_kernel(p_f), repeat=5)
print(f"  kernel alone (fast tensors)  : {t_kern:.2f} ms  -> end-to-end {t_kern + t_fast:.1f} ms "
      f"({t_sdpa / (t_kern + t_fast):.2f}x vs SDPA)")
