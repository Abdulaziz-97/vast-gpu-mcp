"""MCP tools that turn a rented GPU machine into a remote workstation for an AI agent.

The server runs on the GPU machine. All settings come from `configure()` (called by
the CLI) or from environment variables; see `vast_gpu_mcp.cli` for the options.
"""

from __future__ import annotations

import gc
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import psutil
from fastmcp import FastMCP

MAX_OUTPUT = 20_000  # characters returned to the client per call


def _default_venv() -> Path | None:
    """The venv that child commands should use: MCP_VENV, else the venv running
    this server, else the Vast.ai PyTorch image's /venv/main, else none."""
    if os.getenv("MCP_VENV"):
        return Path(os.environ["MCP_VENV"]).expanduser()
    if sys.prefix != sys.base_prefix:
        return Path(sys.prefix)
    if Path("/venv/main/bin").is_dir():
        return Path("/venv/main")
    return None


@dataclass
class Settings:
    workdir: Path = field(default_factory=lambda: Path(os.getenv("MCP_WORKDIR", "~")).expanduser())
    venv: Path | None = field(default_factory=_default_venv)
    # Ungated by default (no Hugging Face login needed).
    default_model: str = field(default_factory=lambda: os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct"))

    @property
    def jobs_dir(self) -> Path:
        return self.workdir / "mcp_jobs"


settings = Settings()
BASH = shutil.which("bash") or "/bin/bash"

mcp = FastMCP(
    "Vast GPU",
    instructions=(
        "Tools for a remote Linux GPU machine. Use run_shell for short commands and "
        "start_job for anything that takes more than a few minutes (training, downloads), "
        "then follow it with job_status. gpu_status shows memory and power; on many cloud "
        "containers utilization % reads 0, so power draw is the reliable busy signal."
    ),
)


def configure(workdir: str | None = None, venv: str | None = None, model: str | None = None) -> Settings:
    """Override settings before the server starts."""
    if workdir:
        settings.workdir = Path(workdir).expanduser().resolve()
    if venv is not None:
        settings.venv = Path(venv).expanduser() if venv else None
    if model:
        settings.default_model = model
    settings.workdir.mkdir(parents=True, exist_ok=True)
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    return settings


def _clip(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return f"...[{len(text) - limit} chars truncated]...\n" + text[-limit:]


def _env(extra: dict | None = None) -> dict:
    """Environment for child processes: venv active, unbuffered Python output."""
    env = dict(os.environ)
    venv = settings.venv
    if venv and (venv / "bin").is_dir():
        env["VIRTUAL_ENV"] = str(venv)
        env["PATH"] = f"{venv / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"  # job logs update immediately
    if "HF_HOME" not in env and Path("/workspace").is_dir():
        env["HF_HOME"] = "/workspace/.hf_home"  # Vast.ai convention: big disk
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    return env


def _resolve(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else settings.workdir / p


def _nvidia_smi(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=20)


# --------------------------------------------------------------------------
# Monitoring
# --------------------------------------------------------------------------

@mcp.tool()
def gpu_status() -> dict:
    """GPU memory, power draw, temperature, clocks and compute processes.
    Note: on many cloud containers the utilization % and process list read 0 / empty.
    Power draw and memory used are the reliable signs that the GPU is busy."""
    if not shutil.which("nvidia-smi"):
        return {"error": "nvidia-smi not found: no NVIDIA GPU/driver on this machine."}

    fields = ["index", "name", "utilization.gpu", "memory.used", "memory.total",
              "temperature.gpu", "power.draw", "power.limit", "clocks.sm",
              "clocks.max.sm", "pstate", "driver_version"]
    out = _nvidia_smi(f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits")
    if out.returncode != 0:
        return {"error": out.stderr.strip()}

    def num(x: str, default: float = 0.0) -> float:
        try:
            return float(x)
        except ValueError:  # "[N/A]" on some GPUs
            return default

    gpus = []
    for line in out.stdout.strip().splitlines():
        v = [x.strip() for x in line.split(",")]
        power, limit, mem = num(v[6]), num(v[7], 1.0) or 1.0, num(v[3])
        gpus.append({
            "index": int(v[0]), "name": v[1], "utilization_pct_reported": v[2],
            "memory_used_mib": int(mem), "memory_total_mib": int(num(v[4])),
            "temperature_c": v[5], "power_draw_w": power, "power_limit_w": limit,
            "power_pct": round(100 * power / limit, 1), "sm_clock_mhz": v[8],
            "sm_clock_max_mhz": v[9], "pstate": v[10], "driver": v[11],
            "busy": power / limit > 0.25 or mem > 1000,
        })

    procs = _nvidia_smi("--query-compute-apps=pid,process_name,used_memory",
                        "--format=csv,noheader,nounits")
    processes = [
        dict(zip(["pid", "process", "memory_mib"], [x.strip() for x in l.split(",")]))
        for l in procs.stdout.strip().splitlines() if l.strip()
    ]
    return {"gpus": gpus, "processes": processes,
            "running_jobs": [j["job_id"] for j in _all_jobs() if j["state"] == "running"]}


@mcp.tool()
def system_status() -> dict:
    """CPU, RAM, disk usage, uptime and Python environment of the machine."""
    vm = psutil.virtual_memory()
    wd = settings.workdir
    disk = psutil.disk_usage(str(wd) if wd.exists() else "/")
    py = settings.venv / "bin" / "python" if settings.venv else None
    return {
        "cpu_count": psutil.cpu_count(),
        "cpu_pct": psutil.cpu_percent(interval=0.5),
        "load_avg": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "ram_used_gb": round(vm.used / 1e9, 2),
        "ram_total_gb": round(vm.total / 1e9, 2),
        "disk_used_gb": round(disk.used / 1e9, 2),
        "disk_free_gb": round(disk.free / 1e9, 2),
        "disk_total_gb": round(disk.total / 1e9, 2),
        "uptime_hours": round((time.time() - psutil.boot_time()) / 3600, 2),
        "workdir": str(wd),
        "python": str(py) if py and py.exists() else shutil.which("python3"),
    }


# --------------------------------------------------------------------------
# Running work on the machine
# --------------------------------------------------------------------------

@mcp.tool()
def run_shell(command: str, timeout_seconds: int = 300, cwd: str = "") -> dict:
    """Run a bash command (venv active) and return exit code, stdout and stderr.
    Good for installs, quick scripts and inspecting files.
    For anything longer than a few minutes (training!) use start_job."""
    try:
        out = subprocess.run(
            command, shell=True, executable=BASH, capture_output=True, text=True,
            timeout=timeout_seconds, cwd=str(_resolve(cwd) if cwd else settings.workdir),
            env=_env(), stdin=subprocess.DEVNULL,
        )
        return {"exit_code": out.returncode,
                "stdout": _clip(out.stdout), "stderr": _clip(out.stderr)}
    except subprocess.TimeoutExpired as e:
        def s(x):
            return _clip(x.decode(errors="replace") if isinstance(x, bytes) else (x or ""))
        return {"error": f"timed out after {timeout_seconds}s; use start_job for long tasks",
                "stdout": s(e.stdout), "stderr": s(e.stderr)}


# Background jobs are persisted as <jobs_dir>/<id>.json so they survive a server
# restart. The exit code is written to <jobs_dir>/<id>.exit by a bash wrapper.
_procs: dict[str, subprocess.Popen] = {}


def _meta_path(job_id: str) -> Path:
    return settings.jobs_dir / f"{job_id}.json"


def _load_meta(job_id: str) -> dict | None:
    if "/" in job_id or job_id.startswith("."):
        return None
    p = _meta_path(job_id)
    return json.loads(p.read_text()) if p.exists() else None


def _job_state(meta: dict) -> dict:
    job_id = meta["job_id"]
    exit_file = settings.jobs_dir / f"{job_id}.exit"
    proc = _procs.get(job_id)
    if proc is not None:
        proc.poll()  # reap if finished
    if exit_file.exists():
        try:
            code = int(exit_file.read_text().strip() or -1)
        except ValueError:
            code = -1
        state = "stopped" if meta.get("stopped") else "finished"
        end = exit_file.stat().st_mtime
    else:
        alive = False
        try:
            p = psutil.Process(meta["pid"])
            alive = p.is_running() and p.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            pass
        if alive:
            state, code, end = "running", None, time.time()
        else:
            state = "stopped" if meta.get("stopped") else "lost"
            code, end = None, meta.get("stopped_at", time.time())
    return {**meta, "state": state, "exit_code": code,
            "runtime_s": round(end - meta["start_ts"], 1)}


def _all_jobs() -> list[dict]:
    if not settings.jobs_dir.exists():
        return []
    metas = []
    for p in settings.jobs_dir.glob("*.json"):
        try:
            metas.append(json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return [_job_state(m) for m in sorted(metas, key=lambda m: m["start_ts"])]


def _tail(path: Path, lines: int) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 200_000))
            data = f.read().decode(errors="replace")
        # progress bars use \r; keep only the latest state of each line
        out = [l.split("\r")[-1] for l in data.splitlines()]
        return _clip("\n".join(out[-lines:]))
    except FileNotFoundError:
        return ""


@mcp.tool()
def start_job(command: str, name: str = "", cwd: str = "", env: dict | None = None) -> dict:
    """Start a long-running bash command in the background (training, fine-tuning,
    benchmarks, dataset downloads, servers). The venv is active and Python output is
    unbuffered. stdout+stderr go to a log file; follow it with job_status.
    `env` adds environment variables, e.g. {"CUDA_VISIBLE_DEVICES": "0"}.
    Jobs keep running if the MCP server or the connection goes down."""
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    job_id = time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    log = settings.jobs_dir / f"{job_id}.log"
    exit_file = settings.jobs_dir / f"{job_id}.exit"
    wrapped = f"{command}\necho $? > {shlex.quote(str(exit_file))}"
    workdir = str(_resolve(cwd) if cwd else settings.workdir)
    with open(log, "w") as fh:
        proc = subprocess.Popen(
            [BASH, "-c", wrapped], stdout=fh, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, cwd=workdir, env=_env(env), start_new_session=True,
        )
    _procs[job_id] = proc
    meta = {"job_id": job_id, "name": name or command[:60], "command": command,
            "cwd": workdir, "pid": proc.pid, "log_file": str(log),
            "start_ts": time.time(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    _meta_path(job_id).write_text(json.dumps(meta))
    return _job_state(meta)


@mcp.tool()
def job_status(job_id: str, tail_lines: int = 50) -> dict:
    """State (running/finished/stopped/lost), exit code, runtime and the last
    lines of a background job's log."""
    meta = _load_meta(job_id)
    if not meta:
        return {"error": f"no job {job_id}", "known_jobs": [j["job_id"] for j in _all_jobs()]}
    st = _job_state(meta)
    st["log_tail"] = _tail(Path(meta["log_file"]), tail_lines)
    return st


@mcp.tool()
def list_jobs(only_running: bool = False) -> list[dict]:
    """All background jobs started through this server (also from before a restart)."""
    jobs = _all_jobs()
    return [j for j in jobs if j["state"] == "running"] if only_running else jobs


@mcp.tool()
def stop_job(job_id: str, grace_seconds: int = 10) -> dict:
    """Stop a background job and all its child processes (SIGTERM, then SIGKILL)."""
    meta = _load_meta(job_id)
    if not meta:
        return {"error": f"no job {job_id}"}
    if _job_state(meta)["state"] == "running":
        meta["stopped"] = True
        meta["stopped_at"] = time.time()
        _meta_path(job_id).write_text(json.dumps(meta))
        pgid = meta["pid"]  # start_new_session -> the job leads its own process group
        try:
            os.killpg(pgid, signal.SIGTERM)
            deadline = time.time() + grace_seconds
            while time.time() < deadline and psutil.pid_exists(meta["pid"]):
                if job_id in _procs and _procs[job_id].poll() is not None:
                    break
                time.sleep(0.25)
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if job_id in _procs:
            _procs[job_id].poll()
    st = _job_state(_load_meta(job_id))
    st["log_tail"] = _tail(Path(meta["log_file"]), 20)
    return st


@mcp.tool()
def write_file(path: str, content: str) -> dict:
    """Write a text file (relative paths are under the work dir).
    Use it to put training scripts, configs or data files on the GPU machine."""
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return {"path": str(p), "bytes": p.stat().st_size}


@mcp.tool()
def read_file(path: str, max_chars: int = MAX_OUTPUT, from_end: bool = False) -> dict:
    """Read a text file (relative paths are under the work dir).
    from_end=True returns the last max_chars characters (handy for big logs)."""
    p = _resolve(path)
    if not p.exists():
        return {"error": f"{p} does not exist"}
    text = p.read_text(errors="replace")
    content = text[-max_chars:] if from_end else text[:max_chars]
    return {"path": str(p), "size": len(text), "content": content,
            "truncated": len(text) > max_chars}


# --------------------------------------------------------------------------
# GPU inference (optional; needs the [llm] extra and PyTorch)
# --------------------------------------------------------------------------

_model_cache: dict = {}
_model_lock = threading.Lock()


def _unload() -> None:
    _model_cache.clear()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


@mcp.tool()
def free_gpu_memory() -> dict:
    """Unload the model cached by llm_generate and release its VRAM, so training
    or other GPU jobs get the whole card. Does not touch background jobs."""
    with _model_lock:
        had = _model_cache.get("id")
        _unload()
    used = None
    if shutil.which("nvidia-smi"):
        used = _nvidia_smi("--query-gpu=memory.used", "--format=csv,noheader,nounits").stdout.strip()
    return {"unloaded_model": had, "gpu_memory_used_mib_now": used}


def _load(model_id: str):
    with _model_lock:
        if _model_cache.get("id") != model_id:
            import torch  # imported lazily so the server starts fast
            from transformers import AutoModelForCausalLM, AutoTokenizer

            _unload()
            tok = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto",
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
            _model_cache.update(id=model_id, tok=tok, model=model)
        return _model_cache["tok"], _model_cache["model"]


@mcp.tool()
def llm_generate(prompt: str, max_new_tokens: int = 256, temperature: float = 0.0,
                 model_id: str = "") -> dict:
    """Generate text on the GPU with a Hugging Face chat model (default:
    Qwen2.5-7B-Instruct). The model stays loaded in VRAM (~15 GB for 7B) for fast
    follow-up calls; call free_gpu_memory before training to release it."""
    model_id = model_id or settings.default_model
    try:
        t0 = time.time()
        tok, model = _load(model_id)
    except ImportError as e:
        return {"error": f"{e}. Install the LLM extra: pip install 'vast-gpu-mcp[llm]' (and PyTorch)."}
    load_s = time.time() - t0

    messages = [{"role": "user", "content": prompt}]
    # return_dict=True gives the same shape on transformers 4.x and 5.x
    if getattr(tok, "chat_template", None):
        enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
    else:
        enc = tok(prompt, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    input_len = enc["input_ids"].shape[-1]

    gen_kwargs = {"max_new_tokens": max_new_tokens, "pad_token_id": tok.eos_token_id}
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    else:
        gen_kwargs.update(do_sample=False)

    t1 = time.time()
    out = model.generate(**enc, **gen_kwargs)
    new_tokens = out[0][input_len:]
    gen_s = time.time() - t1
    return {
        "model": model_id,
        "text": tok.decode(new_tokens, skip_special_tokens=True).strip(),
        "new_tokens": len(new_tokens),
        "tokens_per_s": round(len(new_tokens) / gen_s, 1) if gen_s else None,
        "load_s": round(load_s, 1),
    }
