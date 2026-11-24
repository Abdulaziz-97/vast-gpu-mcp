#!/usr/bin/env bash
# Run on YOUR computer. Keeps localhost:8765 forwarded to the server on the GPU machine,
# reconnecting automatically. Give one or more SSH routes; it alternates between them on failure
# (Vast.ai offers a direct route and a proxy route, either of which can drop).
#
#   bash tunnel.sh root@1.2.3.4:40022 root@ssh5.vast.ai:27389
set -u
[ $# -ge 1 ] || { echo "usage: $0 user@host:port [user@host:port ...]" >&2; exit 2; }
LOCAL_PORT="${MCP_PORT:-8765}"
i=0
while true; do
  route="${@:$(( i % $# + 1 )):1}"
  target="${route%:*}"; port="${route##*:}"
  echo "$(date '+%F %T') connecting via $target port $port"
  ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
      -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=accept-new \
      -N -p "$port" -L "$LOCAL_PORT:127.0.0.1:8765" "$target"
  echo "$(date '+%F %T') tunnel closed; retrying in 5s"
  i=$((i + 1)); sleep 5
done
