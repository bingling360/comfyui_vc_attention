"""Kernel backends.

``reference``  -- portable PyTorch, defines the semantics, runs on CPU.
``triton_attn``-- fused fp8 kernel, the fast path on GPU.
"""
