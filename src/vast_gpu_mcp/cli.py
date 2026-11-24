"""Command line entry point: `vast-gpu-mcp`."""

from __future__ import annotations

import argparse
import ipaddress
import os
import sys

from . import __version__


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="vast-gpu-mcp",
        description="MCP server that lets an AI agent use a remote GPU machine "
                    "(run commands, background training jobs, monitoring, LLM inference).",
    )
    ap.add_argument("--transport", choices=["stdio", "http"],
                    default=os.getenv("MCP_TRANSPORT", "http"),
                    help="stdio: launched per session over SSH, no port or tunnel needed. "
                         "http: long-running server on --host:--port, reached through an SSH tunnel. "
                         "(default: http)")
    ap.add_argument("--host", default=os.getenv("MCP_HOST", "127.0.0.1"),
                    help="http bind address (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=int(os.getenv("MCP_PORT", "8765")),
                    help="http port (default 8765)")
    ap.add_argument("--workdir", default=os.getenv("MCP_WORKDIR"),
                    help="default directory for commands, files and job logs (default: home)")
    ap.add_argument("--venv", default=None,
                    help="Python venv activated for commands (default: MCP_VENV, the server's own "
                         "venv, or /venv/main). Pass '' to disable.")
    ap.add_argument("--model", default=None,
                    help="default Hugging Face model for llm_generate (default Qwen/Qwen2.5-7B-Instruct)")
    ap.add_argument("--allow-public", action="store_true",
                    help="allow binding http to a non-loopback address. DANGEROUS: anyone who can "
                         "reach the port can run any command on the machine. There is no auth.")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = ap.parse_args(argv)

    if sys.platform == "win32":
        ap.error("the server runs on the Linux GPU machine, not on Windows.")
    if args.transport == "http" and not _is_loopback(args.host) and not args.allow_public:
        ap.error(f"refusing to bind to {args.host}: this server executes arbitrary commands and "
                 "has no authentication. Keep it on 127.0.0.1 and use an SSH tunnel, "
                 "or pass --allow-public if you really mean it.")

    from .server import configure, mcp  # imported after arg checks: fastmcp is slow to import

    s = configure(workdir=args.workdir, venv=args.venv, model=args.model)
    # stdout belongs to the MCP protocol in stdio mode, so log to stderr
    print(f"vast-gpu-mcp {__version__}: transport={args.transport} workdir={s.workdir} "
          f"venv={s.venv or 'none'}", file=sys.stderr, flush=True)
    if args.transport == "stdio":
        mcp.run(transport="stdio", show_banner=False)
    else:
        print(f"listening on http://{args.host}:{args.port}/mcp", file=sys.stderr, flush=True)
        mcp.run(transport="streamable-http", host=args.host, port=args.port, show_banner=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
