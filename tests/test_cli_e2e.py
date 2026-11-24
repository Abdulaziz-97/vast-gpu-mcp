"""Runs the installed `vast-gpu-mcp` command as a real separate process, over stdio and HTTP,
and checks that a job outlives the server process."""

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

EXE = shutil.which("vast-gpu-mcp", path=os.path.dirname(sys.executable)) or shutil.which("vast-gpu-mcp")
pytestmark = pytest.mark.skipif(EXE is None, reason="package not installed (pip install -e .)")


def text(r):
    return json.loads(r.content[0].text)


def stdio_client(workdir):
    return Client(StdioTransport(command=EXE, args=["--transport", "stdio", "--workdir", str(workdir),
                                                    "--venv", sys.prefix]))


async def test_stdio_job_outlives_server(tmp_path):
    (tmp_path / "slow.py").write_text("import time; time.sleep(4); print('survived')")
    async with stdio_client(tmp_path) as c:
        assert len(await c.list_tools()) == 11
        job = text(await c.call_tool("start_job", {"command": "python slow.py"}))
        assert job["state"] == "running"
    # that server process has exited; a brand-new one must still see the job running
    await asyncio.sleep(1)
    async with stdio_client(tmp_path) as c:
        st = text(await c.call_tool("job_status", {"job_id": job["job_id"]}))
        assert st["state"] == "running", st
    await asyncio.sleep(5)
    async with stdio_client(tmp_path) as c:
        st = text(await c.call_tool("job_status", {"job_id": job["job_id"]}))
        assert st["state"] == "finished" and st["exit_code"] == 0 and "survived" in st["log_tail"], st


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_http_server(tmp_path):
    port = _free_port()
    proc = subprocess.Popen([EXE, "--transport", "http", "--port", str(port), "--workdir", str(tmp_path)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        url = f"http://127.0.0.1:{port}/mcp"
        for _ in range(60):
            try:
                async with Client(url) as c:
                    tools = await c.list_tools()
                break
            except Exception:
                await asyncio.sleep(0.5)
        else:
            raise AssertionError(proc.stdout.read().decode(errors="replace") if proc.poll() is not None
                                 else "server did not come up")
        assert len(tools) == 11
        async with Client(url) as c:
            r = text(await c.call_tool("run_shell", {"command": "echo via-http"}))
            assert r["exit_code"] == 0 and r["stdout"].strip() == "via-http"
            s = text(await c.call_tool("system_status"))
            assert os.path.samefile(s["workdir"], tmp_path)
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_refuses_public_bind():
    r = subprocess.run([EXE, "--transport", "http", "--host", "0.0.0.0"], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode != 0 and "refusing" in r.stderr
