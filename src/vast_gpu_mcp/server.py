"""MCP tools that turn any machine (Linux, macOS or Windows; NVIDIA, AMD, Apple Silicon or
CPU only) into a remote workstation for an AI agent.

Settings come from `configure()` (called by the CLI) or from environment variables;
see `vast_gpu_mcp.cli` for the options.
"""

from __future__ import annotations

import gc
import json
import os
import platform
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

from . import gpu

IS_WIN = os.name == "nt"
MAX_OUTPUT = 20_000  # characters returned to the client per call


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

def _default_venv() -> Path | None:
    """The Python env commands should use: MCP_VENV, else the venv running this
    server, else the Vast.ai PyTorch image's /venv/main, else none."""
    if os.getenv("MCP_VENV"):
        return Path(os.environ["MCP_VENV"]).expanduser()
    if sys.prefix != sys.base_prefix:
        return Path(sys.prefix)
    if Path("/venv/main/bin").is_dir():
        return Path("/venv/main")
    return None


def _default_shell() -> str:
    if os.getenv("MCP_SHELL"):
        return os.environ["MCP_SHELL"]
    if IS_WIN:
        return shutil.which("pwsh") or shutil.which("powershell") or "powershell.exe"
    return shutil.which("bash") or shutil.which("zsh") or "/bin/sh"


@dataclass
class Settings:
    workdir: Path = field(default_factory=lambda: Path(os.getenv("MCP_WORKDIR", "~")).expanduser())
    venv: Path | None = field(default_factory=_default_venv)
    shell: str = field(default_factory=_default_shell)
    # Ungated by default (no Hugging Face login needed).
    default_model: str = field(default_factory=lambda: os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct"))

    @property
    def jobs_dir(self) -> Path:
        return self.workdir / "mcp_jobs"

    @property
    def shell_kind(self) -> str:
        name = Path(self.shell).stem.lower()
        if name in ("powershell", "pwsh"):
            return "powershell"
        return "cmd" if name == "cmd" else "posix"


settings = Settings()

mcp = FastMCP(
    "Vast GPU",
    instructions=(
        "Tools for a remote machine with (usually) a GPU. Call system_status first to learn the "
        "OS and which shell run_shell uses (bash on Linux/macOS, PowerShell on Windows; in "
        "Windows PowerShell 5.1 chain commands with ';' because '&&' does not exist). Use "
        "run_shell for short commands and start_job for anything that takes more than a few "
        "minutes (training, downloads), then follow it with job_status. gpu_status reports "
        "NVIDIA, AMD and Apple Silicon GPUs; on many cloud containers NVIDIA utilization reads "
        "0, so power draw is the reliable busy signal."
    ),
)


def configure(workdir: str | None = None, venv: str | None = None, model: str | None = None,
              shell: str | None = None) -> Settings:
    """Override settings before the server starts."""
    if workdir:
        settings.workdir = Path(workdir).expanduser().resolve()
    if venv is not None:
        settings.venv = Path(venv).expanduser() if venv else None
    if model:
        settings.default_model = model
    if shell:
        settings.shell = shutil.which(shell) or shell
    settings.workdir.mkdir(parents=True, exist_ok=True)
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    return settings


# --------------------------------------------------------------------------
# Helpers: environment, shell, process trees
# --------------------------------------------------------------------------

def _clip(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return f"...[{len(text) - limit} chars truncated]...\n" + text[-limit:]


def _venv_bin(venv: Path | None) -> Path | None:
    """Directory holding the env's python: bin/ (POSIX venv), Scripts/ (Windows venv),
    or the prefix itself (a plain Windows Python install)."""
    if not venv:
        return None
    for d in (venv / "Scripts", venv / "bin", venv):
        if (d / ("python.exe" if IS_WIN else "python")).exists():
            return d
    return None


def _env(extra: dict | None = None) -> dict:
    """Environment for child processes: venv active, unbuffered UTF-8 Python output."""
    env = dict(os.environ)
    bindir = _venv_bin(settings.venv)
    if bindir:
        env["VIRTUAL_ENV"] = str(settings.venv)
        env["PATH"] = f"{bindir}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"   # job logs update immediately
    env["PYTHONIOENCODING"] = "utf-8"
    if "HF_HOME" not in env and Path("/workspace").is_dir():
        env["HF_HOME"] = "/workspace/.hf_home"  # Vast.ai convention: the big disk
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    return env


# PowerShell: UTF-8 output so non-English text survives the pipe, and the real exit code of
# the last program (plain -Command reports only 0 or 1).
_PS_PREFIX = ("try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch {}; "
              "$OutputEncoding = [Text.Encoding]::UTF8; $global:LASTEXITCODE = 0\n")
_PS_SUFFIX = ("\n$__ok = $?; $__code = $global:LASTEXITCODE\n"
              "if (-not $__ok) { if ($__code) { exit $__code } else { exit 1 } }\n"
              "exit 0")


def _shell_argv(command: str) -> list[str]:
    kind = settings.shell_kind
    if kind == "powershell":
        return [settings.shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                "-Command", _PS_PREFIX + command + _PS_SUFFIX]
    if kind == "cmd":
        return [settings.shell, "/d", "/s", "/c", command]
    return [settings.shell, "-c", command]


def _detach_kwargs() -> dict:
    """Start the child in its own process group / session so it can be stopped as a tree
    and does not die with the server."""
    if IS_WIN:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    return {"start_new_session": True}


def _kill_tree(pid: int, grace: float = 5.0) -> None:
    """Terminate a process and all of its descendants, then kill what is left."""
    try:
        root = psutil.Process(pid)
        procs = [root] + root.children(recursive=True)
    except psutil.NoSuchProcess:
        procs = []
    if not IS_WIN:
        try:
            os.killpg(pid, signal.SIGTERM)  # also catches children that re-parented
        except (ProcessLookupError, PermissionError):
            pass
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=grace)
    if not IS_WIN:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


def _resolve(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else settings.workdir / p


# --------------------------------------------------------------------------
# Monitoring
# --------------------------------------------------------------------------

@mcp.tool()
def gpu_status() -> dict:
    """GPUs (NVIDIA, AMD or Apple Silicon) with memory, power, temperature and a `busy`
    flag, plus running background jobs. On a machine without a GPU, `gpus` is empty and
    everything else still works on the CPU."""
    result = gpu.status()
    result["running_jobs"] = [j["job_id"] for j in _all_jobs() if j["state"] == "running"]
    return result


@mcp.tool()
def system_status() -> dict:
    """OS, CPU, RAM, disk, uptime, the shell run_shell uses, and the Python environment."""
    vm = psutil.virtual_memory()
    wd = settings.workdir
    disk = psutil.disk_usage(str(wd) if wd.exists() else str(Path.home()))
    bindir = _venv_bin(settings.venv)
    try:
        load = [round(x, 2) for x in psutil.getloadavg()]
    except (AttributeError, OSError):
        load = None
    return {
        "os": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "hostname": platform.node(),
        "shell": settings.shell,
        "shell_kind": settings.shell_kind,
        "cpu_count": psutil.cpu_count(),
        "cpu_pct": psutil.cpu_percent(interval=0.5),
        "load_avg": load,
        "ram_used_gb": round(vm.used / 1e9, 2),
        "ram_total_gb": round(vm.total / 1e9, 2),
        "disk_used_gb": round(disk.used / 1e9, 2),
        "disk_free_gb": round(disk.free / 1e9, 2),
        "disk_total_gb": round(disk.total / 1e9, 2),
        "uptime_hours": round((time.time() - psutil.boot_time()) / 3600, 2),
        "workdir": str(wd),
        "python": str(bindir / ("python.exe" if IS_WIN else "python")) if bindir
                  else shutil.which("python3") or shutil.which("python"),
    }


# --------------------------------------------------------------------------
# Running work on the machine
# --------------------------------------------------------------------------

@mcp.tool()
def run_shell(command: str, timeout_seconds: int = 300, cwd: str = "") -> dict:
    """Run a command and return exit code, stdout and stderr. The shell is bash on
    Linux/macOS and PowerShell on Windows (see system_status); the Python env is active.
    Good for installs, quick scripts and inspecting files.
    For anything longer than a few minutes (training!) use start_job."""
    proc = subprocess.Popen(
        _shell_argv(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL, cwd=str(_resolve(cwd) if cwd else settings.workdir),
        env=_env(), text=True, encoding="utf-8", errors="replace", **_detach_kwargs(),
    )
    try:
        out, err = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        _kill_tree(proc.pid, grace=2)  # kill grandchildren too, or the pipes never close
        out, err = proc.communicate()
        return {"error": f"timed out after {timeout_seconds}s; use start_job for long tasks",
                "stdout": _clip(out or ""), "stderr": _clip(err or "")}
    return {"exit_code": proc.returncode, "stdout": _clip(out), "stderr": _clip(err)}


# Background jobs are persisted as <jobs_dir>/<id>.json so they survive a server
# restart. A small runner script (same on every OS) starts the job from a spec file,
# sends its output to <id>.log and writes its exit code to <id>.exit.
_procs: dict[str, subprocess.Popen] = {}
_RUNNER = '''\
import json, os, subprocess, sys
spec_path = sys.argv[1]
with open(spec_path, encoding="utf-8") as f:
    spec = json.load(f)
os.remove(spec_path)  # it holds the job's environment, which may contain secrets
kw = {"creationflags": 0x08000000} if os.name == "nt" else {}  # CREATE_NO_WINDOW
code = 127
with open(spec["log"], "ab", buffering=0) as log:
    try:
        code = subprocess.call(spec["argv"], cwd=spec["cwd"], env=spec["env"], stdout=log,
                               stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **kw)
    except OSError as e:
        log.write(("vast-gpu-mcp: could not start %r: %s\\n" % (spec["argv"][0], e)).encode())
with open(spec["exit_file"] + ".tmp", "w") as f:
    f.write(str(code))
os.replace(spec["exit_file"] + ".tmp", spec["exit_file"])
'''


def _meta_path(job_id: str) -> Path:
    return settings.jobs_dir / f"{job_id}.json"


def _load_meta(job_id: str) -> dict | None:
    if not job_id or any(c in job_id for c in "/\\:") or job_id.startswith("."):
        return None
    p = _meta_path(job_id)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _save_meta(meta: dict) -> None:
    _meta_path(meta["job_id"]).write_text(json.dumps(meta), encoding="utf-8")


def _is_alive(meta: dict) -> bool:
    try:
        p = psutil.Process(meta["pid"])
        if meta.get("create_time") and abs(p.create_time() - meta["create_time"]) > 1:
            return False  # the pid was reused by another process
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


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
    elif _is_alive(meta):
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
            metas.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return [_job_state(m) for m in sorted(metas, key=lambda m: m["start_ts"])]


def _tail(path: Path, lines: int) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 200_000))
            data = f.read().decode("utf-8", errors="replace")
        # progress bars use \r; keep only the latest state of each line
        out = [l.rstrip("\r").split("\r")[-1] for l in data.split("\n")]
        if out and out[-1] == "":
            out.pop()
        return _clip("\n".join(out[-lines:]))
    except FileNotFoundError:
        return ""


def _runner_path() -> Path:
    p = settings.jobs_dir / "_runner.py"
    if not p.exists() or p.read_text(encoding="utf-8") != _RUNNER:
        p.write_text(_RUNNER, encoding="utf-8")
    return p


def _write_private(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)


def _wmi_start(cmdline: str, cwd: str) -> int | None:
    """Start a hidden process through WMI. It is created by the WMI service, so it is
    outside any job object the MCP client put the server in (those kill every member
    when the client disconnects, and usually forbid breakaway)."""
    ps = ("$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly "
          "-Property @{ShowWindow=[uint16]0}; "
          "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments "
          "@{CommandLine=$env:VGM_CMD; CurrentDirectory=$env:VGM_CWD; ProcessStartupInformation=$si}; "
          "if ($r.ReturnValue -ne 0) { exit 1 }; $r.ProcessId")
    env = dict(os.environ, VGM_CMD=cmdline, VGM_CWD=cwd)
    try:
        out = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps],
                             capture_output=True, text=True, timeout=60, env=env,
                             creationflags=subprocess.CREATE_NO_WINDOW)
        return int(out.stdout.strip()) if out.returncode == 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _launch_runner(argv: list[str], cwd: str) -> tuple[int, subprocess.Popen | None, str]:
    """Start the runner detached from the server. Returns (pid, popen or None, method)."""
    if not IS_WIN:
        p = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
        return p.pid, p, "session"
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    popen = dict(cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not _in_job(os.getpid()):
        p = subprocess.Popen(argv, creationflags=flags, **popen)
        return p.pid, p, "detached"
    # Inside a job object (MCP clients use them to kill the server's whole tree). Breakaway is
    # not reliable with nested jobs: CreateProcess can succeed while the child stays in the
    # client's job, which then kills it. WMI creates the process outside every job.
    pid = _wmi_start(subprocess.list2cmdline(argv), cwd)
    if pid:
        return pid, None, "wmi"
    try:
        p = subprocess.Popen(argv, creationflags=flags | 0x01000000, **popen)  # BREAKAWAY_FROM_JOB
        return p.pid, p, "breakaway"
    except OSError:
        pass
    p = subprocess.Popen(argv, creationflags=flags, **popen)
    return p.pid, p, "attached"  # may stop when the MCP client disconnects


def _in_job(pid: int) -> bool:
    """True if a Windows process is in a job object that kills it with the server."""
    if not IS_WIN:
        return False
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    try:
        res = wintypes.BOOL()
        return bool(k32.IsProcessInJob(h, None, ctypes.byref(res))) and bool(res.value)
    finally:
        k32.CloseHandle(h)


@mcp.tool()
def start_job(command: str, name: str = "", cwd: str = "", env: dict | None = None) -> dict:
    """Start a long-running command in the background (training, fine-tuning, benchmarks,
    dataset downloads, servers). Same shell as run_shell; the Python env is active and
    Python output is unbuffered. stdout+stderr go to a log file; follow it with job_status.
    `env` adds environment variables, e.g. {"CUDA_VISIBLE_DEVICES": "0"}.
    Jobs keep running if the MCP server or the connection goes down."""
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    job_id = time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    log = settings.jobs_dir / f"{job_id}.log"
    exit_file = settings.jobs_dir / f"{job_id}.exit"
    workdir = str(_resolve(cwd) if cwd else settings.workdir)
    log.write_bytes(b"")
    spec = settings.jobs_dir / f"{job_id}.spec.json"
    _write_private(spec, json.dumps({"argv": _shell_argv(command), "cwd": workdir, "env": _env(env),
                                     "log": str(log), "exit_file": str(exit_file)}))
    # -I: the runner ignores PYTHONPATH etc.; the job itself gets the env from the spec
    pid, proc, method = _launch_runner([sys.executable, "-I", str(_runner_path()), str(spec)], workdir)
    if proc is not None:
        _procs[job_id] = proc
    try:
        create_time = psutil.Process(pid).create_time()
    except psutil.NoSuchProcess:
        create_time = None
    meta = {"job_id": job_id, "name": name or command[:60], "command": command,
            "cwd": workdir, "pid": pid, "create_time": create_time, "log_file": str(log),
            "launch": method, "start_ts": time.time(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    _save_meta(meta)
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
def list_jobs(only_running: bool = False) -> dict:
    """All background jobs started through this server (also from before a restart)."""
    jobs = _all_jobs()
    if only_running:
        jobs = [j for j in jobs if j["state"] == "running"]
    return {"jobs": jobs, "count": len(jobs)}


@mcp.tool()
def stop_job(job_id: str, grace_seconds: int = 10) -> dict:
    """Stop a background job and all its child processes (terminate, then kill)."""
    meta = _load_meta(job_id)
    if not meta:
        return {"error": f"no job {job_id}"}
    if _job_state(meta)["state"] == "running":
        meta["stopped"] = True
        meta["stopped_at"] = time.time()
        _save_meta(meta)
        _kill_tree(meta["pid"], grace=grace_seconds)
        if job_id in _procs:
            _procs[job_id].poll()
    st = _job_state(_load_meta(job_id))
    st["log_tail"] = _tail(Path(meta["log_file"]), 20)
    return st


@mcp.tool()
def write_file(path: str, content: str) -> dict:
    """Write a UTF-8 text file (relative paths are under the work dir).
    Use it to put scripts, configs or data files on the machine."""
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:  # keep the line endings as given
        f.write(content)
    return {"path": str(p), "bytes": p.stat().st_size}


@mcp.tool()
def read_file(path: str, max_chars: int = MAX_OUTPUT, from_end: bool = False) -> dict:
    """Read a text file (relative paths are under the work dir).
    from_end=True returns the last max_chars characters (handy for big logs)."""
    p = _resolve(path)
    if not p.exists():
        return {"error": f"{p} does not exist"}
    text = p.read_text(encoding="utf-8", errors="replace")
    content = text[-max_chars:] if from_end else text[:max_chars]
    return {"path": str(p), "size": len(text), "content": content,
            "truncated": len(text) > max_chars}


# --------------------------------------------------------------------------
# LLM inference (optional; needs the [llm] extra and PyTorch). CUDA, ROCm, MPS or CPU.
# --------------------------------------------------------------------------

_model_cache: dict = {}
_model_lock = threading.Lock()


def _unload() -> None:
    _model_cache.clear()
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        torch.mps.empty_cache()


@mcp.tool()
def free_gpu_memory() -> dict:
    """Unload the model cached by llm_generate and release its memory, so training or
    other GPU jobs get the whole card. Does not touch background jobs."""
    with _model_lock:
        had = _model_cache.get("id")
        _unload()
    return {"unloaded_model": had, "gpu_memory_used_mib_now": gpu.memory_used_mib()}


def _pick_device():
    import torch
    if torch.cuda.is_available():  # NVIDIA, or AMD with a ROCm build of PyTorch
        bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        return "cuda", torch.bfloat16 if bf16 else torch.float16
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def _load(model_id: str):
    with _model_lock:
        if _model_cache.get("id") != model_id:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy: slow import

            _unload()
            device, dtype = _pick_device()
            tok = AutoTokenizer.from_pretrained(model_id)
            if device == "cuda":
                model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", dtype=dtype)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
            _model_cache.update(id=model_id, tok=tok, model=model, device=device)
        return _model_cache["tok"], _model_cache["model"]


@mcp.tool()
def llm_generate(prompt: str, max_new_tokens: int = 256, temperature: float = 0.0,
                 model_id: str = "") -> dict:
    """Generate text with a Hugging Face chat model (default: Qwen2.5-7B-Instruct) on the
    best available device: CUDA/ROCm GPU, Apple Silicon (MPS) or CPU. The model stays
    loaded (~15 GB for 7B) for fast follow-up calls; call free_gpu_memory to release it.
    On small machines pass a small model_id, e.g. Qwen/Qwen2.5-0.5B-Instruct."""
    model_id = model_id or settings.default_model
    try:
        t0 = time.time()
        tok, model = _load(model_id)
    except ImportError as e:
        return {"error": f"{e}. Install the LLM extra: pip install 'vast-gpu-mcp[llm]' plus PyTorch "
                         "(https://pytorch.org/get-started/locally/)."}
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
        "device": _model_cache.get("device"),
        "text": tok.decode(new_tokens, skip_special_tokens=True).strip(),
        "new_tokens": len(new_tokens),
        "tokens_per_s": round(len(new_tokens) / gen_s, 1) if gen_s else None,
        "load_s": round(load_s, 1),
    }
