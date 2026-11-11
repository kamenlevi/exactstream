#!/bin/bash
# Codex on this Mac's model, in one command:
#
#   tools/codex_code.sh                   # the 32 GB chat config; then Codex in the current folder
#   tools/codex_code.sh --mem 45          # or 24; anything after it goes to codex (e.g. exec "fix the failing test")
#
# Starts serve.py unless one already answers on the port (EXACTSTREAM_PORT, default 8080), waits until the engine is
# loaded (about a minute), then runs Codex with the settings in tools/codex_local.sh (its own CODEX_HOME)
# (~/.codex is not changed), and stops the server it started when Codex exits. The server runs in its own process
# group, so Ctrl-C in Codex does not reach it. Its log goes to out/codex_code_serve_<time>.log.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MEM=32
if [ "$1" = "--mem" ]; then MEM=$2; shift 2; fi
case $MEM in 45|32|24) ;; *) echo "codex_code.sh: --mem takes 45, 32 or 24" >&2; exit 2;; esac
PORT=${EXACTSTREAM_PORT:-8080}
health() { curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null; }
STARTED=""
if ! health | grep -q ready; then
  if [ -n "$(health)" ]; then
    echo "codex_code.sh: a server on port $PORT is still loading; waiting for it"
  else
    mkdir -p "$ROOT/out"
    LOG="$ROOT/out/codex_code_serve_$(date +%m%d_%H%M%S).log"
    set -m                                                 # its own process group: Ctrl-C in Codex stays there
    ( cd "$ROOT" && exec .venv/bin/python serve.py --mem "$MEM" --port "$PORT" ${EXACTSTREAM_SERVE_ARGS:-} ) > "$LOG" 2>&1 &
    STARTED=$!
    set +m
    echo "codex_code.sh: loading the engine (--mem $MEM, about a minute); log: $LOG"
  fi
  until health | grep -q ready; do
    if [ -n "$STARTED" ] && ! kill -0 "$STARTED" 2>/dev/null; then
      echo "codex_code.sh: serve.py stopped while loading; see $LOG" >&2
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
EXACTSTREAM_PORT=$PORT "$ROOT/tools/codex_local.sh" "$@"
