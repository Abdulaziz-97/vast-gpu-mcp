# Changelog

## 0.2.0

- Works on Linux, macOS and Windows. Commands use bash on Linux/macOS and PowerShell on Windows
  (configurable with `--shell`), with real exit codes and UTF-8 output.
- `gpu_status` detects NVIDIA, AMD (rocm-smi) and Apple Silicon GPUs and works without a GPU.
- `llm_generate` runs on CUDA/ROCm, Apple MPS or CPU.
- Background jobs use a small runner that is the same on every OS. On Windows they start through
  WMI when needed, so they survive the MCP client disconnecting.
- `stop_job` and `run_shell` timeouts stop the whole process tree on every OS.
- `list_jobs` returns `{"jobs": [...], "count": n}`.
- Examples run on any device; `train_cifar.py --limit` for quick checks.
- CI on Linux, macOS and Windows.

## 0.1.0

First release.

- 11 tools: `gpu_status`, `system_status`, `run_shell`, `start_job`, `job_status`, `list_jobs`,
  `stop_job`, `write_file`, `read_file`, `free_gpu_memory`, `llm_generate`.
- `stdio` transport (launched over SSH) and `http` transport (behind an SSH tunnel).
- Refuses to bind HTTP to a non-loopback address without `--allow-public`.
- Background jobs persist across server restarts and stop with their whole process group.
- Reconnecting SSH tunnel scripts for macOS/Linux and Windows.
