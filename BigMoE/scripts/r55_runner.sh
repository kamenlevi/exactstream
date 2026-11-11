#!/bin/bash
# R-55 runner (NewIdeas 22:50 GO; Kamen 21:1x: 45 GB first): scripts/r55_screen.py at C 192 (BIGMOE_MAXK=160, ext/build_next) under one quiet lock, with a
# swap guard (stops the job's process group if swap grows > 2 GB; C 176 loads tripped Leap's guard at 13 GB compressor).
# usage: HAVE_LOCK=1 nohup bash scripts/r55_runner.sh > data/r55/runner.out 2>&1 &
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
set -m   # each run is its own process group: time-up or a lost lock stops the whole tree (RULES, orchestrator 20:5x)
cd "$HOME/BigMoE" || exit 1
OUT="$HOME/BigMoE/data/r55"; mkdir -p "$OUT"
LOG="$OUT/r55.log"
PY="$HOME/BigMoE/.venv/bin/python"
REF="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12"
LOCKF="$HOME/.claude-peers/locks/${CLAUDE_PID}.json"
keepalive() { "$PY" -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF"; }
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
source "$HOME/BigMoE/scripts/ssd_gate.sh"
run() {  # run NAME TIMEOUT OUTFILE CMD...  (memory gate at most 30 s: the lock is only 15 min)
  local name=$1 to=$2 of=$3; shift 3
  keepalive || { say "lost the lock before $name: stopping the batch"; exit 1; }
  while [ -e "$HOME/.local/share/quiet/PAUSE" ]; do sleep 5; keepalive || { say "lock ended during a PAUSE"; exit 1; }; done
  ssd_wait 66
  local i=0
  while { [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -lt 60 ] || [ "$(ps -axo rss=,pid= | awk -v me=$$ '$1 > 8000000 && $2 != me' | wc -l)" -gt 0 ]; } && [ $i -lt 6 ]; do sleep 5; i=$((i+1)); done
  [ $i -ge 6 ] && say "memory busy at the start of $name (ran anyway after 30 s)"
  say "run $name"
  perl -e 'alarm shift; exec @ARGV' "$to" "$@" >> "$OUT/$of" 2>&1 &
  local jp=$!
  while kill -0 $jp 2>/dev/null; do
    for k in $(seq 36); do kill -0 $jp 2>/dev/null || break; sleep 5; done
    kill -0 $jp 2>/dev/null || break
    keepalive || { say "lost the lock during $name: stopping"; kill -- -$jp; sleep 5; kill -9 -- -$jp 2>/dev/null; exit 1; }
  done
  wait $jp; local rc=$?; kill -9 -- -$jp 2>/dev/null; say "done $name rc=$rc"   # the job's whole group (RULES)
  ssd_check || say "SSD hot after $name"
  return $rc
}
say "r55 start (CLAUDE_PID=$CLAUDE_PID)"
if [ -z "$HAVE_LOCK" ]; then
  PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 12m "BigMoE R-55: sync-free 16-row gate on the exact REF2 check path at C 192, ~6 min" >> "$LOG" 2>&1 || { say "no lock"; exit 1; }
fi
swap_gb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}'; }
S0=$(swap_gb); say "swap at start: $S0 GB"
( seen=0; while sleep 5; do
    S=$(swap_gb)
    if awk -v s="$S" -v s0="$S0" 'BEGIN { exit !(s - s0 > 2) }'; then
      say "SWAP GUARD: swap $S GB (start $S0): stopping the screen"
      P=$(pgrep -P $$ -f "scripts/r55_screen.py" | head -1)     # the runner's own job, by PID (RULES: no pkill)
      [ -n "$P" ] && kill -TERM -- -"$P"; break
    fi
    if pgrep -P $$ -f "scripts/r55_screen.py" >/dev/null; then seen=1; elif [ $seen = 1 ]; then break; fi
  done ) &
GUARD=$!
run screen 600 r55_run.txt env BIGMOE_MAXK=160 BIGMOE_EXT_BUILD=build_next $PY scripts/r55_screen.py 192 14 || say "screen failed"
kill $GUARD 2>/dev/null
[ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1
say "lock released"
tail -4 "$OUT/r55_screen.txt" >> "$LOG" 2>/dev/null
say "r55 end"
