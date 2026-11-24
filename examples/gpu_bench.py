"""GPU benchmark: matmul TFLOPS (fp16/bf16/fp32), memory bandwidth, and timing.
Run directly on the instance or via the MCP run_shell tool to compare overhead."""
import json
import time

import torch

assert torch.cuda.is_available(), "CUDA not available"
dev = torch.device("cuda")
res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
       "cuda": torch.version.cuda}


def bench_matmul(dtype, n=8192, iters=50):
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    return round(2 * n**3 * iters / dt / 1e12, 1)


def bench_bandwidth(gb=4, iters=20):
    n = int(gb * 1e9 / 4)
    x = torch.empty(n, device=dev, dtype=torch.float32)
    y = torch.empty_like(x)
    for _ in range(3):
        y.copy_(x)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        y.copy_(x)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    return round(2 * n * 4 * iters / dt / 1e9, 1)  # read + write


t0 = time.perf_counter()
res["tflops_fp16"] = bench_matmul(torch.float16)
res["tflops_bf16"] = bench_matmul(torch.bfloat16)
res["tflops_fp32_tf32off"] = bench_matmul(torch.float32, iters=10)
torch.backends.cuda.matmul.allow_tf32 = True
res["tflops_tf32"] = bench_matmul(torch.float32, iters=20)
res["mem_bandwidth_gb_s"] = bench_bandwidth()
res["max_vram_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
res["bench_seconds"] = round(time.perf_counter() - t0, 2)
print(json.dumps(res))
