"""Benchmark: matmul TFLOPS per dtype and memory bandwidth on the best device available
(CUDA/ROCm GPU, Apple Silicon MPS, or CPU). Prints one JSON line.

Run it directly and through the MCP server's run_shell to compare overhead."""
import json
import time

import torch

if torch.cuda.is_available():
    dev, name = "cuda", torch.cuda.get_device_name(0)
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    dev, name = "mps", "Apple Silicon (MPS)"
else:
    dev, name = "cpu", "CPU"
# problem sizes per device, so it finishes in seconds everywhere
N, ITERS, GB = {"cuda": (8192, 50, 4), "mps": (2048, 10, 0.5), "cpu": (512, 3, 0.25)}[dev]


def sync():
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


def bench_matmul(dtype, n=N, iters=ITERS):
    try:
        a = torch.randn(n, n, device=dev, dtype=dtype)
        b = torch.randn(n, n, device=dev, dtype=dtype)
        for _ in range(3):
            a @ b
    except (RuntimeError, TypeError):  # dtype not supported on this device
        return None
    sync()
    t = time.perf_counter()
    for _ in range(iters):
        a @ b
    sync()
    return round(2 * n**3 * iters / (time.perf_counter() - t) / 1e12, 3)


def bench_bandwidth(gb=GB, iters=10):
    x = torch.empty(int(gb * 1e9 / 4), device=dev, dtype=torch.float32)
    y = torch.empty_like(x)
    for _ in range(3):
        y.copy_(x)
    sync()
    t = time.perf_counter()
    for _ in range(iters):
        y.copy_(x)
    sync()
    return round(2 * x.numel() * 4 * iters / (time.perf_counter() - t) / 1e9, 1)  # read + write


res = {"device": dev, "name": name, "torch": torch.__version__}
t0 = time.perf_counter()
res["tflops_fp16"] = bench_matmul(torch.float16)
res["tflops_bf16"] = bench_matmul(torch.bfloat16)
res["tflops_fp32"] = bench_matmul(torch.float32)
if dev == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    res["tflops_tf32"] = bench_matmul(torch.float32)
res["mem_bandwidth_gb_s"] = bench_bandwidth()
if dev == "cuda":
    res["max_vram_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
res["bench_seconds"] = round(time.perf_counter() - t0, 2)
print(json.dumps(res))
