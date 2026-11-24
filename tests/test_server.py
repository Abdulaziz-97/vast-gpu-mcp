"""End-to-end tests through a real MCP client (in-memory transport).

They run on Linux, macOS and Windows, with or without a GPU. Commands that differ between
bash and PowerShell are written as small Python scripts, which work the same in both.
"""

import asyncio
import json
import os
import sys
import time

import psutil
import pytest
from fastmcp import Client

from vast_gpu_mcp import server
from vast_gpu_mcp.cli import _is_loopback, main

IS_WIN = os.name == "nt"
ALL_TOOLS = {"gpu_status", "system_status", "run_shell", "start_job", "job_status",
             "list_jobs", "stop_job", "write_file", "read_file", "free_gpu_memory",
             "llm_generate"}


@pytest.fixture
def client(tmp_path):
    # the test interpreter's env, so `python` in commands is this Python
    server.configure(workdir=str(tmp_path), venv=sys.prefix)
    return Client(server.mcp)


async def call(client, name, args=None):
    async with client:
        r = await client.call_tool(name, args or {})
        return json.loads(r.content[0].text)


async def script(client, name, code):
    await call(client, "write_file", {"path": name, "content": code})
    return f"python {name}"


async def wait_done(client, job_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = await call(client, "job_status", {"job_id": job_id})
        if st["state"] != "running":
            return st
        await asyncio.sleep(0.25)
    raise AssertionError(f"job {job_id} still running after {timeout}s: {st}")


# ---------------------------------------------------------------- discovery

async def test_lists_all_tools(client):
    async with client:
        names = {t.name for t in await client.list_tools()}
    assert names == ALL_TOOLS


async def test_system_status(client, tmp_path):
    s = await call(client, "system_status")
    assert s["cpu_count"] >= 1 and s["ram_total_gb"] > 0 and s["disk_total_gb"] > 0
    assert s["workdir"] == str(tmp_path)
    assert s["shell_kind"] == ("powershell" if IS_WIN else "posix")
    assert s["python"] and os.path.exists(s["python"])


async def test_gpu_status_never_errors(client):
    g = await call(client, "gpu_status")
    assert "error" not in g
    assert isinstance(g["gpus"], list) and g["running_jobs"] == []
    for gpu in g["gpus"]:
        assert gpu["vendor"] in ("nvidia", "amd", "apple") and gpu["name"]
    if not g["gpus"]:
        assert "No GPU detected" in g["note"]


async def test_free_gpu_memory_without_model(client):
    r = await call(client, "free_gpu_memory")
    assert r["unloaded_model"] is None and isinstance(r["gpu_memory_used_mib_now"], list)


# ---------------------------------------------------------------- run_shell

async def test_run_shell_output_and_exit_code(client, tmp_path):
    cmd = await script(client, "a.py",
                       "import os, sys\nprint(os.getcwd())\nprint('err', file=sys.stderr)\nsys.exit(3)\n")
    r = await call(client, "run_shell", {"command": cmd})
    assert r["exit_code"] == 3, r
    assert os.path.samefile(r["stdout"].strip(), tmp_path)
    assert r["stderr"].strip().endswith("err")


async def test_run_shell_success_is_zero(client):
    r = await call(client, "run_shell", {"command": "echo hello"})
    assert r["exit_code"] == 0 and r["stdout"].strip() == "hello"


async def test_run_shell_explicit_exit(client):
    r = await call(client, "run_shell", {"command": "exit 5"})
    assert r["exit_code"] == 5


async def test_run_shell_unknown_command_fails(client):
    r = await call(client, "run_shell", {"command": "definitely-not-a-command-xyz"})
    assert r["exit_code"] != 0


async def test_run_shell_unicode(client):
    cmd = await script(client, "u.py", "print('مرحبا 你好 ✓')\n")
    r = await call(client, "run_shell", {"command": cmd})
    assert r["stdout"].strip() == "مرحبا 你好 ✓", r


async def test_run_shell_cwd(client, tmp_path):
    (tmp_path / "sub").mkdir()
    await call(client, "write_file", {"path": "sub/p.py", "content": "import os; print(os.getcwd())"})
    r = await call(client, "run_shell", {"command": "python p.py", "cwd": "sub"})
    assert os.path.samefile(r["stdout"].strip(), tmp_path / "sub")


async def test_run_shell_timeout_kills_children(client, tmp_path):
    cmd = await script(client, "spawn.py",
                       "import subprocess, sys, time\n"
                       "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
                       "open('child.pid', 'w').write(str(p.pid))\n"
                       "time.sleep(120)\n")
    t0 = time.time()
    r = await call(client, "run_shell", {"command": cmd, "timeout_seconds": 3})
    assert "timed out" in r["error"]
    assert time.time() - t0 < 30
    child = int((tmp_path / "child.pid").read_text())
    time.sleep(0.5)
    assert not psutil.pid_exists(child) or psutil.Process(child).status() == psutil.STATUS_ZOMBIE


# ---------------------------------------------------------------- files

async def test_write_and_read_file(client, tmp_path):
    w = await call(client, "write_file", {"path": "sub/a.txt", "content": "hello\nworld\n"})
    assert w["path"] == str(tmp_path / "sub" / "a.txt")
    assert (tmp_path / "sub" / "a.txt").read_bytes() == b"hello\nworld\n"  # no CRLF added
    r = await call(client, "read_file", {"path": "sub/a.txt", "max_chars": 6, "from_end": True})
    assert r["content"] == "world\n" and r["truncated"]
    r = await call(client, "read_file", {"path": str(tmp_path / "sub" / "a.txt")})
    assert r["content"] == "hello\nworld\n" and not r["truncated"]
    assert "error" in await call(client, "read_file", {"path": "missing.txt"})


async def test_files_unicode(client):
    await call(client, "write_file", {"path": "ar.txt", "content": "نص عربي"})
    assert (await call(client, "read_file", {"path": "ar.txt"}))["content"] == "نص عربي"


# ---------------------------------------------------------------- jobs

async def test_job_lifecycle(client):
    cmd = await script(client, "j.py", "import sys, time\nprint('start')\ntime.sleep(0.5)\n"
                                       "print('done')\nsys.exit(7)\n")
    job = await call(client, "start_job", {"command": cmd, "name": "t"})
    assert job["state"] == "running" and job["name"] == "t"
    st = await wait_done(client, job["job_id"])
    assert st["state"] == "finished" and st["exit_code"] == 7, st
    assert st["log_tail"].splitlines() == ["start", "done"]
    jobs = await call(client, "list_jobs")
    assert jobs["count"] == 1 and [j["job_id"] for j in jobs["jobs"]] == [job["job_id"]]
    assert await call(client, "list_jobs", {"only_running": True}) == {"jobs": [], "count": 0}


async def test_job_success_exit_zero(client):
    job = await call(client, "start_job", {"command": "echo ok"})
    st = await wait_done(client, job["job_id"])
    assert st["exit_code"] == 0 and st["log_tail"].strip() == "ok"


async def test_job_env_and_progress_bars(client):
    cmd = await script(client, "pb.py", "import os, sys\n"
                                        "sys.stdout.write('10%\\r50%\\r100%\\n')\n"
                                        "print(os.environ['FOO'])\n")
    job = await call(client, "start_job", {"command": cmd, "env": {"FOO": "bar"}})
    st = await wait_done(client, job["job_id"])
    assert st["log_tail"].splitlines() == ["100%", "bar"], st


async def test_job_shows_in_gpu_status_while_running(client):
    job = await call(client, "start_job", {"command": await script(
        client, "s.py", "import time; time.sleep(30)")})
    g = await call(client, "gpu_status")
    assert g["running_jobs"] == [job["job_id"]]
    await call(client, "stop_job", {"job_id": job["job_id"], "grace_seconds": 2})


async def test_stop_job_kills_whole_tree(client, tmp_path):
    cmd = await script(client, "tree.py",
                       "import subprocess, sys, time\n"
                       "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
                       "open('child.pid', 'w').write(str(p.pid))\n"
                       "print('spawned', flush=True)\n"
                       "time.sleep(300)\n")
    job = await call(client, "start_job", {"command": cmd})
    for _ in range(60):
        if (tmp_path / "child.pid").exists():
            break
        await asyncio.sleep(0.25)
    child = int((tmp_path / "child.pid").read_text())
    assert psutil.pid_exists(child)
    st = await call(client, "stop_job", {"job_id": job["job_id"], "grace_seconds": 3})
    assert st["state"] == "stopped", st
    time.sleep(0.5)
    assert not psutil.pid_exists(child) or psutil.Process(child).status() == psutil.STATUS_ZOMBIE
    # stopping again is harmless
    assert (await call(client, "stop_job", {"job_id": job["job_id"]}))["state"] == "stopped"


async def test_jobs_survive_server_restart(client):
    """A new server process sees jobs from the old one: state lives on disk."""
    job = await call(client, "start_job", {"command": await script(
        client, "r.py", "import time; time.sleep(1); print('after')")})
    server._procs.clear()  # forget the Popen handles, as a restarted server would
    assert (await call(client, "job_status", {"job_id": job["job_id"]}))["state"] == "running"
    st = await wait_done(client, job["job_id"])
    assert st["state"] == "finished" and st["exit_code"] == 0 and "after" in st["log_tail"]


async def test_lost_job_is_reported(client):
    job = await call(client, "start_job", {"command": await script(
        client, "l.py", "import time; time.sleep(60)")})
    server._procs.clear()
    psutil.Process(job["pid"]).kill()  # the wrapper dies without writing an exit code
    await asyncio.sleep(1)
    st = await call(client, "job_status", {"job_id": job["job_id"]})
    assert st["state"] == "lost", st
    for p in psutil.process_iter(["cmdline"]):  # clean up the orphaned shell
        if p.info["cmdline"] and any("l.py" in c for c in p.info["cmdline"]):
            p.kill()


async def test_unknown_job_and_path_traversal(client):
    for bad in ("nope", "../etc/passwd", "..\\x", "C:x", ""):
        assert "error" in await call(client, "job_status", {"job_id": bad})
        assert "error" in await call(client, "stop_job", {"job_id": bad})


# ---------------------------------------------------------------- llm

async def test_llm_generate_without_transformers_gives_clear_error(client, monkeypatch):
    def boom(model_id):
        raise ImportError("No module named 'transformers'")
    monkeypatch.setattr(server, "_load", boom)
    r = await call(client, "llm_generate", {"prompt": "hi"})
    assert "pip install" in r["error"]


@pytest.mark.parametrize("model", ["trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",  # chat template
                                   "hf-internal-testing/tiny-random-gpt2"])          # plain tokenizer
async def test_llm_generate_and_free(client, model):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    r = await call(client, "llm_generate", {"prompt": "Hello", "max_new_tokens": 5, "model_id": model})
    assert "error" not in r, r
    assert r["model"] == model and r["device"] in ("cuda", "mps", "cpu") and r["new_tokens"] >= 1
    r2 = await call(client, "llm_generate", {"prompt": "Hi", "max_new_tokens": 3, "model_id": model,
                                             "temperature": 0.7})
    assert r2["load_s"] < r["load_s"] + 1  # cached
    f = await call(client, "free_gpu_memory")
    assert f["unloaded_model"] == model


# ---------------------------------------------------------------- cli

def test_loopback_guard():
    assert _is_loopback("127.0.0.1") and _is_loopback("::1") and _is_loopback("localhost")
    assert not _is_loopback("0.0.0.0") and not _is_loopback("example.com")
    with pytest.raises(SystemExit):
        main(["--host", "0.0.0.0"])


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0 and "vast-gpu-mcp" in capsys.readouterr().out
