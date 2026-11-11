#!/bin/bash
# T1 of the conversation cache (docs/CONV_CACHE_DESIGN.md sections 7 + 13; the orchestrator Tue 23:3x: "Yes, run T1 tonight
# ... once that runner has ended and released its lock ... latest start 06:30"). Leap's serve.py --mem 32 (the chat config,
# PF-A on) from Leap's cache working copy (never the clone Kamen chats in) with CONV_CACHE_CHECK=1: every cache hit runs a
# cold prefill first in the same process and compares last logits, every state array, the taps tail, the widened rows and
# the U3C footprint. conv/t1_replay.py posts Leap's 45 recorded requests in their order (answers capped at 64 tokens).
# Gate (polled each minute): no long_ctx.sh (Leap's runner), no chat.py, no live quiet lock of another session, the hook in
# serve.py, NO-SWAP >= 38 GB; latest start T1_DEADLINE (epoch from a full date: HHMM wraps at midnight). Swap guard: the
# server is stopped if swap grows > 2 GB. The server is exec'd (so $! is it), stopped with SIGINT + wait + a port check.
# Out: data/conv/t1/ (run.log, convcache.log, serve.log, t1_replay.json). Exit 0 = ran (verdict in t1_replay.json).
# usage: bash conv/t1_run.sh     (T1_WC=<working copy>  T1_PORT=8091  T1_DEADLINE="2026-11-11 06:30:00"  T1_DRY=1)
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
WC="${T1_WC:-$HOME/Leap/release/cleanclone/cache/clone}"
PORT="${T1_PORT:-8091}"
OUT="$HOME/BigMoE/data/conv/t1"
mkdir -p "$OUT"
LOG="$OUT/run.log"
CLOG="$OUT/convcache.log"
DL=$(date -j -f "%Y-%m-%d %H:%M:%S" "${T1_DEADLINE:-2026-11-11 06:30:00}" +%s) || exit 2
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; [ -z "$T1_DRY" ] || echo "$*"; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap_gb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); printf "%.2f", $6 / 1024}'; }
quiet_held() { /usr/bin/python3 -I -S -c '
import glob, json, os, sys, time
me = sys.argv[1]
for p in glob.glob(os.path.expanduser("~/.claude-peers/locks/*.json")):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    if d.get("mode") == "quiet" and d.get("until", 0) > time.time() and os.path.basename(p) != me + ".json":
        print(d.get("why", "?")[:80]); sys.exit(0)
sys.exit(1)' "${CLAUDE_PID:-none}"; }
gate() {
  local b=""
  pgrep -f 'agent/long/long_ct[x]\.sh' > /dev/null && b="$b long_ctx.sh"
  pgrep -f 'chat\.p[y] ' > /dev/null && b="$b chat.py"
  q=$(quiet_held) && b="$b quiet-lock($q)"
  grep -q "convcache" "$WC/serve.py" || b="$b no-hook-in-serve.py"
  [ -x "$WC/.venv/bin/python" ] || b="$b no-venv"
  if [ -z "$b" ]; then
    local a; a=$(noswap)
    [ "$a" -ge 38 ] || b=" NO-SWAP $a GB < 38"
  fi
  echo "$b"
}
if [ -n "$T1_DRY" ]; then
  say "dry: WC $WC, port $PORT, deadline $(date -r "$DL" '+%m-%d %H:%M:%S'), gate now:$(gate), NO-SWAP $(noswap) GB, swap $(swap_gb) GB"
  exit 0
fi
say "T1 armed (pid $$): WC $WC, port $PORT; gate = no long_ctx.sh / chat.py / quiet lock, the hook, NO-SWAP >= 38 GB; latest start $(date -r "$DL" '+%m-%d %H:%M')"
last=""
open_n=0                                                   # the gate must stay open 3 checks in a row (3 min): Leap re-arms
while :; do                                                # its runner (new pids), and a check can fall in the gap
  if [ "$(date +%s)" -gt "$DL" ]; then say "T1 not started: the latest start passed (T1 moves to Wednesday morning)"; exit 3; fi
  b=$(gate)
  if [ -z "$b" ]; then
    open_n=$((open_n + 1))
    [ "$open_n" -ge 3 ] && break
    b=" (gate open $open_n/3)"
  else
    open_n=0
  fi
  [ "$b" != "$last" ] && say "waiting:$b"
  last="$b"
  sleep 60
done
SW0=$(swap_gb)
say "gate open: NO-SWAP $(noswap) GB, swap $SW0 GB; peer share lock + server start"
peer lock share 60m --cpu 400 --gpu 90 "BigMoE T1: conversation cache on the real engine (serve.py --mem 32, CONV_CACHE_CHECK=1, ~32 GB, ~20 min)" >> "$LOG" 2>&1
: > "$CLOG"
( cd "$WC" && exec env CONV_CACHE_CHECK=1 CONV_CACHE_LOG="$CLOG" HF_HUB_OFFLINE=1 \
    "$WC/.venv/bin/python" serve.py --mem 32 --port "$PORT" ) > "$OUT/serve.log" 2>&1 &
SPID=$!
say "server pid $SPID"
stop_server() {
  kill -INT "$SPID" 2>/dev/null
  for _ in $(seq 1 60); do kill -0 "$SPID" 2>/dev/null || break; sleep 2; done
  kill -0 "$SPID" 2>/dev/null && { kill -TERM "$SPID" 2>/dev/null; sleep 10; }
  kill -0 "$SPID" 2>/dev/null && { kill -KILL "$SPID" 2>/dev/null; say "server needed SIGKILL"; }
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN > /dev/null 2>&1 && say "port $PORT still listening after stop" || say "server stopped, port $PORT free"
}
guard() {                                                  # swap guard while the server runs
  while kill -0 "$SPID" 2>/dev/null; do
    sw=$(swap_gb)
    if awk -v a="$sw" -v b="$SW0" 'BEGIN {exit !(a - b > 2.0)}'; then
      say "SWAP GUARD: swap $SW0 -> $sw GB (> +2 GB): stopping the server"
      touch "$OUT/swap_guard.hit"
      kill -INT "$SPID" 2>/dev/null
      break
    fi
    sleep 20
  done
}
rm -f "$OUT/swap_guard.hit"
guard &
GPID=$!
ok=""
for _ in $(seq 1 120); do                                  # up to 10 min to load
  kill -0 "$SPID" 2>/dev/null || break
  curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ready"' && { ok=1; break; }
  sleep 5
done
if [ -z "$ok" ]; then
  say "server not ready (log tail: $(tail -n 3 "$OUT/serve.log" | tr '\n' ' ' | cut -c1-300))"
  stop_server; kill "$GPID" 2>/dev/null; peer unlock > /dev/null 2>&1; exit 4
fi
say "server ready; replay starts"
/usr/bin/python3 -I conv/t1_replay.py --port "$PORT" --log "$CLOG" --cap 64 --out "$OUT/t1_replay.json" >> "$LOG" 2>&1
rc=$?
say "replay rc=$rc: $(tail -n 1 "$LOG" | cut -c1-300)"
stop_server
kill "$GPID" 2>/dev/null
peer unlock > /dev/null 2>&1
say "T1 done (swap $SW0 -> $(swap_gb) GB$( [ -f "$OUT/swap_guard.hit" ] && echo ', SWAP GUARD hit'))"
exit 0
