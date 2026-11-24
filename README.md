# vast-gpu-mcp

[![tests](https://github.com/Abdulaziz-97/vast-gpu-mcp/actions/workflows/test.yml/badge.svg)](https://github.com/Abdulaziz-97/vast-gpu-mcp/actions/workflows/test.yml)

An [MCP](https://modelcontextprotocol.io) server that turns a machine into a remote workstation
for AI agents such as Claude. The agent can run commands, start training jobs that keep running
in the background, watch the GPU, read logs, and run quick LLM inference, all on that machine
while you chat from your own computer.

It was built on [Vast.ai](https://vast.ai) GPU instances and works on any machine:

| | Linux | macOS | Windows |
| --- | --- | --- | --- |
| Shell for commands | bash | bash | PowerShell |
| GPU detection | NVIDIA, AMD | Apple Silicon | NVIDIA |
| `llm_generate` | CUDA / ROCm / CPU | MPS / CPU | CUDA / CPU |
| Background jobs survive restarts | yes | yes | yes |

Every tool works without a GPU too; work then runs on the CPU. The test suite runs on all three
systems on every push.

> **Unofficial.** This project is not affiliated with or endorsed by Vast.ai.

## Tools

| Tool | What it does |
| --- | --- |
| `gpu_status` | GPUs (NVIDIA, AMD, Apple) with memory, power, temperature, a `busy` flag, running jobs |
| `system_status` | OS, CPU, RAM, disk, uptime, which shell is used, Python environment |
| `run_shell` | Run a command and return exit code, stdout, stderr (short tasks) |
| `start_job` | Run a long command in the background (training, downloads, servers) |
| `job_status` | State, exit code, runtime and log tail of a job |
| `list_jobs` | All jobs, including ones started before a server restart |
| `stop_job` | Stop a job and all of its child processes |
| `write_file` / `read_file` | Put scripts and configs on the machine, read logs and results |
| `llm_generate` | Text generation with a Hugging Face model kept in VRAM (optional `[llm]` extra) |
| `free_gpu_memory` | Unload that model so training gets the whole card |

Commands run with the machine's Python environment active, so `python`, `pip`, `torchrun` and
`accelerate` work as-is. Background jobs are detached from the server: they keep running if the
server restarts or your connection drops, and their state is saved to disk. `stop_job` stops a
job together with every process it started.

## Security: read this first

This server **executes arbitrary commands** and has **no authentication**. That is the point of
it, and it is only safe because nobody but you can reach it:

- In `http` mode it binds to `127.0.0.1` and refuses any other address unless you pass
  `--allow-public`. Do not. Reach it through an SSH tunnel instead (below).
- In `stdio` mode it has no network port at all; your MCP client starts it over SSH.

Your SSH key is the only lock. Anyone who can talk to the server can do anything on the machine.

## Install (on the machine that does the work)

Python 3.10 or newer, on Linux, macOS or Windows:

```bash
pip install "git+https://github.com/Abdulaziz-97/vast-gpu-mcp"
```

For `llm_generate`, add the `[llm]` extra and [PyTorch](https://pytorch.org/get-started/locally/)
(GPU cloud images already ship PyTorch):

```bash
pip install "vast-gpu-mcp[llm] @ git+https://github.com/Abdulaziz-97/vast-gpu-mcp"
```

On Vast.ai PyTorch images, install into the image's venv: `/venv/main/bin/pip install "git+https://github.com/Abdulaziz-97/vast-gpu-mcp"`.

## Connect it to your MCP client

You can choose between two ways.

### Option A: stdio over SSH (simplest, nothing to keep running)

Your MCP client starts the server over SSH for each session. With Claude Code:

```bash
claude mcp add vast-gpu --scope user -- \
  ssh -o BatchMode=yes -p 40022 root@1.2.3.4 /venv/main/bin/vast-gpu-mcp --transport stdio
```

Use the SSH port and address your provider shows. Your SSH key must load without a prompt
(use `ssh-agent`). Jobs and their logs survive between sessions; a model loaded by
`llm_generate` does not.

To use the GPU in your own computer, skip SSH and let the client start the server directly:

```bash
claude mcp add local-gpu --scope user -- vast-gpu-mcp --transport stdio
```

### Option B: HTTP server plus SSH tunnel (server stays up)

On a Linux or macOS machine, install and start the server in the background:

```bash
curl -fsSLO https://raw.githubusercontent.com/Abdulaziz-97/vast-gpu-mcp/main/scripts/start_server.sh
bash start_server.sh
```

On any machine, including Windows, you can also run it yourself:
`vast-gpu-mcp --transport http` (listens on `127.0.0.1:8765`).

On your computer, keep a tunnel open. The script reconnects on its own and alternates between
the routes you give it (Vast.ai offers a direct route and a proxy route, and either can drop):

```bash
bash scripts/tunnel.sh root@1.2.3.4:40022 root@ssh5.vast.ai:27389      # macOS / Linux
```
```powershell
.\scripts\tunnel.ps1 root@1.2.3.4:40022 root@ssh5.vast.ai:27389         # Windows
```

Then register it:

```bash
claude mcp add --transport http --scope user vast-gpu http://localhost:8765/mcp
```

## Options

| Flag | Env var | Default |
| --- | --- | --- |
| `--transport stdio\|http` | `MCP_TRANSPORT` | `http` |
| `--host` | `MCP_HOST` | `127.0.0.1` |
| `--port` | `MCP_PORT` | `8765` |
| `--workdir` (commands, files, job logs) | `MCP_WORKDIR` | home directory |
| `--venv` (activated for commands; `''` for none) | `MCP_VENV` | the server's own venv, else `/venv/main` |
| `--shell` (for `run_shell` and `start_job`) | `MCP_SHELL` | bash (Linux/macOS), PowerShell (Windows) |
| `--model` (default for `llm_generate`) | `MODEL_ID` | `Qwen/Qwen2.5-7B-Instruct` |

Job metadata and logs go to `<workdir>/mcp_jobs/`. On machines with a `/workspace` disk,
`HF_HOME` defaults to `/workspace/.hf_home`.

## Things worth knowing

- **GPU utilization reads 0% on many cloud containers.** The container cannot see it. Power
  draw and memory used are reliable, and `gpu_status` sets `busy` from them.
- **On Windows, commands are PowerShell.** Windows PowerShell 5.1 has no `&&`; chain commands
  with `;`. Install [PowerShell 7](https://aka.ms/powershell) (`pwsh`) and it is used
  automatically, or pick another shell with `--shell`.
- **Windows jobs start through WMI** when the server runs inside a job object (MCP clients use
  them to clean up), so jobs are not killed when the client disconnects.
- **Apple Silicon** has no power or utilization readings without root; `gpu_status` reports the
  chip and its unified memory.
- **The overhead is small.** On an RTX 5090, a matmul benchmark run through the server reached
  the same speed as one run directly over SSH (215 vs 214 TFLOPS fp16), and an MCP round trip
  took about 0.2 s.
- **Gated models** (Llama, Mistral) need `hf auth login` on the machine first.
- **Stop the instance when you are done.** The server does not watch your bill.

## Examples

Both pick the best device available (CUDA/ROCm, Apple MPS or CPU).

- `examples/gpu_bench.py`: matmul TFLOPS per dtype and memory bandwidth.
- `examples/train_cifar.py`: ResNet-18 on CIFAR-10 with the data held on the device. Needs
  `pip install datasets torchvision`; `--limit 1024` trains on a small subset for a quick check.

## Development

```bash
pip install -e ".[dev]"
pytest          # Linux, macOS or Windows; no GPU needed (LLM tests run if torch is installed)
```

## License

MIT
