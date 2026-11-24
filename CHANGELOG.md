# Changelog

## 0.1.0

First release.

- 11 tools: `gpu_status`, `system_status`, `run_shell`, `start_job`, `job_status`, `list_jobs`,
  `stop_job`, `write_file`, `read_file`, `free_gpu_memory`, `llm_generate`.
- `stdio` transport (launched over SSH) and `http` transport (behind an SSH tunnel).
- Refuses to bind HTTP to a non-loopback address without `--allow-public`.
- Background jobs persist across server restarts and stop with their whole process group.
- Reconnecting SSH tunnel scripts for macOS/Linux and Windows.
