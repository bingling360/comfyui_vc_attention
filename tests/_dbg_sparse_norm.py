"""Debug the probe35 simulation: small shape, sim vs kernel vs dense."""
import sys

import torch

sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention")
sys.path.insert(0, "/root/ComfyUI/custom_nodes/comfyui_vc_attention/tests")

from bench_attention import h3_like, psnr  # noqa: E402
from vc_attention import quant as Q  # noqa: E402
from vc_attention.grouping import GroupingConfig, build_permutation  # noqa: E402
from vc_attention.kernels import triton_attn as T  # noqa: E402

H, D, QB, BN = 4, 128, 64, 128
N = 4096
q, k, v = h3_like(N, H, D, "cuda")
ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
perm = build_permutation(v.reshape(H, N, D), GroupingConfig(block_rows=BN, iters=3)).perm
scale = D ** -0.5


def dq_fp8(x):
    c, sc = Q.quantize_e4m3(x, dim=-1)
    return Q.dequantize_e4m3(c, sc)


def sim(tau, keep_all=False):
    Hh = H
    qf = q.reshape(Hh, N, D)
    kf = k.reshape(Hh, N, D)
    vf = v.reshape(Hh, N, D)
    Hm = T._hadamarian(D, q.device).float()
    nb, nqb = N // BN, N // QB
    ki = torch.arange(nb, device=q.device)
    outs, ratios = [], []
    for hh in range(Hh):
        q_t = qf[hh].float() @ Hm
        k_t = kf[hh].float() @ Hm
        k_t = k_t - k_t.mean(dim=0, keepdim=True)
        p = perm[hh].to(torch.int64)
        k_t = k_t[p]
        v_p = vf[hh].float()[p]
        q8 = dq_fp8(q_t)
        k8 = dq_fp8(k_t)
        vb = v_p.reshape(nb, BN, D)
        mu = vb.mean(dim=1)
        v8 = dq_fp8((vb - mu[:, None, :]).reshape(N, D)).reshape(nb, BN, D) + mu[:, None, :]
        v8 = v8.reshape(N, D)
        k_mean = k_t.reshape(nb, BN, D).mean(dim=1)
        q_cent = q_t.reshape(nqb, QB, D).mean(dim=1)
        proxy_h = (q_cent @ k_mean.T) * scale
        thr = proxy_h.mean(-1, keepdim=True) + tau * proxy_h.std(-1, unbiased=False, keepdim=True)
        proxy = (q8 @ k_mean.T) * scale
        pq = proxy.reshape(nqb, QB, nb).mean(dim=1)
        qi = torch.arange(nqb, device=q.device)
        keep = (pq > thr) | ((qi[:, None] // (BN // QB) - ki[None, :]).abs() <= 1)
        if keep_all:
            keep = torch.ones_like(keep)
        out = torch.empty(N, D, device=q.device, dtype=torch.float32)
        for j in range(nqb):
            rows = slice(j * QB, (j + 1) * QB)
            kb = keep[j]
            kb_e = kb.repeat_interleave(BN)[None, :]
            s = (q8[rows] @ k8.T) * scale
            s_mod = torch.where(kb_e, s, proxy[rows].repeat_interleave(BN, 1))
            m = s_mod.max(dim=-1, keepdim=True).values
            p_all = torch.exp(s_mod - m)
            num = (p_all * kb_e) @ v8
            p_blocks = p_all.reshape(QB, nb, BN).sum(dim=-1)
            num = num + ((p_blocks * (~kb)[None, :].float()) @ mu)
            out[rows] = num / p_all.sum(dim=-1, keepdim=True)
        outs.append(out)
        ratios.append(float(keep.float().mean()))
    return torch.stack(outs).reshape(1, Hh, N, D), sum(ratios) / Hh


print(f"ref abs mean {ref.abs().mean():.4f}", flush=True)
for tau in (1.3, 100.0):
    s, kr = sim(tau)
    cfg = T.TritonConfig(block_m=QB, block_n=BN, num_warps=4, num_stages=2, sparse=True, tau=tau)
    ker = T.vc_attention_triton(q, k, v, perm=perm, cfg=cfg)
    print(f"\ntau={tau} keep={kr:.3f}", flush=True)
    print(f"  sim    abs mean {s.abs().mean():.4f}  PSNR {psnr(s, ref):6.2f} dB", flush=True)
    print(f"  kernel abs mean {ker.abs().mean():.4f}  PSNR {psnr(ker, ref):6.2f} dB", flush=True)
    print(f"  kernel/sim abs-mean ratio {ker.abs().mean() / s.abs().mean():.4f}", flush=True)
    print(f"  sim vs kernel rel-err {float((ker.float()-s).norm()/s.norm()):.3e}", flush=True)

s, kr = sim(0.0, keep_all=True)
print(f"\nkeep-all sim PSNR {psnr(s, ref):6.2f} dB  (dense VC ~57 dB expected)", flush=True)
print(f"  keep ratio {kr:.3f}", flush=True)
