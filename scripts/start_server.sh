#!/usr/bin/env bash
# Run ON the GPU machine. Installs vast-gpu-mcp into the image's venv (if it has one)
# and starts the HTTP server in the background on 127.0.0.1:8765.
#
#   bash start_server.sh                 # install from GitHub
#   bash start_server.sh ./vast_gpu_mcp-0.1.0-py3-none-any.whl   # or a local wheel / path / git URL
set -euo pipefail

SRC="${1:-git+https://github.com/Abdulaziz-97/vast-gpu-mcp}"
PORT="${MCP_PORT:-8765}"
LOG="${MCP_LOG:-$HOME/vast-gpu-mcp.log}"

if [ -x /venv/main/bin/python ]; then      # Vast.ai PyTorch images
  PY=/venv/main/bin/python
else
  PY="$(command -v python3)"
fi
echo "python: $PY"

if command -v uv >/dev/null 2>&1; then
  uv pip install -q --python "$PY" "$SRC"
else
  "$PY" -m pip install -q "$SRC"
fi

BIN="$(dirname "$PY")/vast-gpu-mcp"
pkill -f "vast-gpu-mcp --transport http" 2>/dev/null || true
sleep 1
nohup "$BIN" --transport http --port "$PORT" > "$LOG" 2>&1 &
sleep 3
cat "$LOG"
if ! kill -0 $! 2>/dev/null; then
  echo "server failed to start (log above)" >&2
  exit 1
fi
echo "running (pid $!). Log: $LOG"
