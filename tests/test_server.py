"""End-to-end tests through a real MCP client (in-memory transport). No GPU needed."""

import asyncio
import json
import time

import pytest
from fastmcp import Client

from vast_gpu_mcp import server
from vast_gpu_mcp.cli import _is_loopback, main


@pytest.fixture
def client(tmp_path):
    server.configure(workdir=str(tmp_path), venv="")
    return Client(server.mcp)


async def call(client, name, args=None):
    async with client:
        r = await client.call_tool(name, args or {})
        return json.loads(r.content[0].text)


async def test_lists_all_tools(client):
    async with client:
        names = {t.name for t in await client.list_tools()}
    assert names == {"gpu_status", "system_status", "run_shell", "start_job", "job_status",
                     "list_jobs", "stop_job", "write_file", "read_file", "free_gpu_memory",
                     "llm_generate"}


async def test_run_shell(client, tmp_path):
    r = await call(client, "run_shell", {"command": "pwd; echo err >&2; exit 3"})
    assert r["exit_code"] == 3
    assert r["stdout"].strip() == str(tmp_path)
    assert r["stderr"].strip() == "err"


async def test_run_shell_timeout(client):
    r = await call(client, "run_shell", {"command": "sleep 5", "timeout_seconds": 1})
    assert "timed out" in r["error"]


async def test_write_and_read_file(client, tmp_path):
    w = await call(client, "write_file", {"path": "sub/a.txt", "content": "hello\nworld\n"})
    assert w["path"] == str(tmp_path / "sub" / "a.txt")
    r = await call(client, "read_file", {"path": "sub/a.txt", "max_chars": 6, "from_end": True})
    assert r["content"] == "world\n" and r["truncated"]
    assert "error" in await call(client, "read_file", {"path": "missing.txt"})


async def test_job_lifecycle(client):
    job = await call(client, "start_job", {"command": "echo start; sleep 0.5; echo done; exit 7",
                                           "name": "t", "env": {"FOO": "bar"}})
    assert job["state"] == "running"
    for _ in range(40):
        st = await call(client, "job_status", {"job_id": job["job_id"]})
        if st["state"] != "running":
            break
        await asyncio.sleep(0.25)
    assert st["state"] == "finished" and st["exit_code"] == 7
    assert st["log_tail"].splitlines() == ["start", "done"]
    jobs = await call(client, "list_jobs")
    assert [j["job_id"] for j in jobs] == [job["job_id"]]


async def test_job_env_and_progress_bars(client):
    job = await call(client, "start_job", {"command": "printf '10%%\\r50%%\\r100%%\\n'; echo $FOO",
                                           "env": {"FOO": "bar"}})
    time.sleep(1)
    st = await call(client, "job_status", {"job_id": job["job_id"]})
    assert st["log_tail"].splitlines() == ["100%", "bar"]


async def test_stop_job_kills_children(client):
    job = await call(client, "start_job", {"command": "sleep 300 & sleep 300; wait"})
    time.sleep(0.5)
    st = await call(client, "stop_job", {"job_id": job["job_id"], "grace_seconds": 2})
    assert st["state"] == "stopped"
    r = await call(client, "run_shell", {"command": "pgrep -f '[s]leep 300' || true"})
    assert r["stdout"].strip() == ""


async def test_unknown_job_and_path_traversal(client):
    assert "error" in await call(client, "job_status", {"job_id": "nope"})
    assert "error" in await call(client, "job_status", {"job_id": "../etc/passwd"})


async def test_status_tools(client):
    s = await call(client, "system_status")
    assert s["cpu_count"] >= 1 and s["disk_total_gb"] > 0
    g = await call(client, "gpu_status")
    assert "gpus" in g or "error" in g


def test_loopback_guard():
    assert _is_loopback("127.0.0.1") and _is_loopback("::1") and _is_loopback("localhost")
    assert not _is_loopback("0.0.0.0")
    with pytest.raises(SystemExit):
        main(["--host", "0.0.0.0"])
