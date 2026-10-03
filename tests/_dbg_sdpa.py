"""Which backend is the 'bf16 SDPA' baseline, and how does it compare to FA?"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_attention import h3_like, timeit  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

print("flash:", torch.backends.cuda.flash_sdp_enabled(),
      "cudnn:", torch.backends.cuda.cudnn_sdp_enabled(),
      "mem_eff:", torch.backends.cuda.mem_efficient_sdp_enabled(),
      "math:", torch.backends.cuda.math_sdp_enabled(), flush=True)

for n in (16384, 65536):
    q, k, v = h3_like(n, 56, 128, "cuda")
    print(f"--- tokens={n} ---", flush=True)
    for name, be in (("default", None), ("flash", SDPBackend.FLASH_ATTENTION),
                     ("cudnn", SDPBackend.CUDNN_ATTENTION),
                     ("efficient", SDPBackend.EFFICIENT_ATTENTION)):
        try:
            if be is None:
                t = timeit(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v), repeat=5)
            else:
                with sdpa_kernel(be):
                    t = timeit(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v), repeat=5)
            print(f"  {name:10}: {t:8.2f} ms", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {name:10}: FAIL {type(e).__name__}: {str(e)[:70]}", flush=True)
    del q, k, v
    torch.cuda.empty_cache()
