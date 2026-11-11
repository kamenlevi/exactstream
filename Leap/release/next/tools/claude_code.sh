#!/bin/bash
# Claude Code on this Mac's model, in one command:
#
#   tools/claude_code.sh                  # the 32 GB chat config; then Claude Code in the current folder
#   tools/claude_code.sh --mem 45         # or 24; anything after it goes to claude (e.g. -p "fix the failing test")
#
# Starts serve.py unless one already answers on the port (EXACTSTREAM_PORT, default 8080), waits until the engine is
# loaded (about a minute), then runs Claude Code with the settings in tools/claude_local.sh, for this process only
# (~/.claude is not changed), and stops the server it started when Claude Code exits. The server runs in its own process
# group, so Ctrl-C in Claude Code (which stops a reply) does not reach it. Its log goes to out/claude_code_serve_<time>.log.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MEM=32
if [ "$1" = "--mem" ]; then MEM=$2; shift 2; fi
case $MEM in 45|32|24) ;; *) echo "claude_code.sh: --mem takes 45, 32 or 24" >&2; exit 2;; esac
# The context window Claude Code is told (EXACTSTREAM_CONTEXT overrides it): 64K at 45 and 32 GB, 128K at 24 GB.
# A cold 128K prefill at 32 GB with the layer-major prefill was forecast at ~61 GB, and 45 GB at 128K at ~59 GB, above
# what a 64 GB Mac can spare; at 24 GB it peaked at 51.6 GB (data/long_ctx_1111.md).
case $MEM in 24) CTX_DEFAULT=131072;; *) CTX_DEFAULT=65536;; esac
export EXACTSTREAM_CONTEXT=${EXACTSTREAM_CONTEXT:-$CTX_DEFAULT}
PORT=${EXACTSTREAM_PORT:-8080}
health() { curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null; }
STARTED=""
if ! health | grep -q ready; then
  if [ -n "$(health)" ]; then
    echo "claude_code.sh: a server on port $PORT is still loading; waiting for it"
  else
    mkdir -p "$ROOT/out"
    LOG="$ROOT/out/claude_code_serve_$(date +%m%d_%H%M%S).log"
    set -m                                                 # its own process group: Ctrl-C in Claude Code stays there
    ( cd "$ROOT" && exec .venv/bin/python serve.py --mem "$MEM" --port "$PORT" ${EXACTSTREAM_SERVE_ARGS:-} ) > "$LOG" 2>&1 &
    STARTED=$!
    set +m
    echo "claude_code.sh: loading the engine (--mem $MEM, about a minute); log: $LOG"
  fi
  until health | grep -q ready; do
    if [ -n "$STARTED" ] && ! kill -0 "$STARTED" 2>/dev/null; then
      echo "claude_code.sh: serve.py stopped while loading; see $LOG" >&2
      exit 1
    fi
    sleep 2
  done
fi
stop_server() {
  [ -z "$STARTED" ] && return
  kill -TERM "$STARTED" 2>/dev/null
  for _ in $(seq 1 60); do kill -0 "$STARTED" 2>/dev/null || return; sleep 1; done
  kill -KILL "$STARTED" 2>/dev/null
}
trap stop_server EXIT
# The status line (tools/statusline.sh: thinking time + mark, live tok/s, the last answer), for this session only,
# re-run every second so it also ticks while the prompt is being read.
SL="{\"statusLine\":{\"type\":\"command\",\"command\":\"'$ROOT/tools/statusline.sh' $PORT\",\"padding\":0,\"refreshInterval\":1}}"
EXACTSTREAM_PORT=$PORT "$ROOT/tools/claude_local.sh" --settings "$SL" "$@"
