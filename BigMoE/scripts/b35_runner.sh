#!/bin/bash
# b35 (S10c all-layer confirmation, NewIdeas GO Tue ~20:50 after b34 BETWEEN; QUEUE S10b row): b33's capture again with x
# kept at EVERY source layer L = 0..45 (BIGMOE_S10B_LAYERS), so TrainLab can fit M1a for all 46 pairs L -> L + 2 (judge
# models on prompts 0-3, as for the 8 pairs). 20 prompts x 400 tokens, res.bin on, ~2.2 GB of x. One 10-min quiet lock.
# The confirmation A/B (b36, after TrainLab's fits): PASS whole-process SSD bytes / token <= -5 % at tok/s no worse than
# -1 %; KILL > -2 %. This capture is deleted after TrainLab's fits.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
set -m   # each run is its own process group: time-up or a lost lock stops the whole tree (RULES, orchestrator 20:5x)
cd "$HOME/BigMoE" || exit 1
OUT="$HOME/BigMoE/data/b35"; mkdir -p "$OUT"
LOG="$OUT/b35.log"
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
run() {  # run NAME TIMEOUT OUTFILE CMD...  (memory gate at most 60 s: the lock is only 10 min)
  local name=$1 to=$2 of=$3; shift 3
  keepalive || { say "lost the lock before $name: stopping the batch"; exit 1; }
  while [ -e "$HOME/.local/share/quiet/PAUSE" ]; do sleep 5; keepalive || { say "lock ended during a PAUSE"; exit 1; }; done
  ssd_wait 66
  local i=0
  while { [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -lt 60 ] || [ "$(ps -axo rss=,pid= | awk -v me=$$ '$1 > 8000000 && $2 != me' | wc -l)" -gt 0 ]; } && [ $i -lt 12 ]; do sleep 5; i=$((i+1)); done
  [ $i -ge 12 ] && say "memory busy at the start of $name (ran anyway after 60 s)"
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
}
say "b35 start (CLAUDE_PID=$CLAUDE_PID)"
free_gb() { df -k / | awk 'NR==2 {printf "%.1f", $4 / 1048576}'; }
if awk -v f="$(free_gb)" 'BEGIN { exit !(f < 110.0) }'; then   # ~2.2 GB of x: stay well above the 100 GB floor
  say "disk $(free_gb) GB free < 110: not capturing"; exit 1; fi
if [ -z "$HAVE_LOCK" ]; then
  PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 10m "BigMoE b35 (S10c all-layer capture): S10b trace with x at all 46 source layers (24 GB v3 decode, 20 prompts x 400 tok, ~7 min)" >> "$LOG" 2>&1 || { say "no lock"; exit 1; }
fi
X="$REF BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_FRSPEC=1 BIGMOE_SCOUT_BUNDLE=1 MTPLX_QWEN4_VERIFY_GLUE=1 BIGMOE_MTP_STREAM=1"
V3="BIGMOE_TIER=4433333333 BIGMOE_TIER_GLU=3c BIGMOE_TIER_COMBINE=1 BIGMOE_OOO=1"
G="BIGMOE_GDN_EXACT=1"
PG="BIGMOE_TIER_PF_GROUP=64 BIGMOE_PF_NOLOGITS=1"
FULL="$G $V3 $PG BIGMOE_EVICT_LFU=0.99 BIGMOE_VERIFY_ROWS_SDPA=1"
J="BIGMOE_JOINT=1"
LAY=$(seq -s, 0 45 | sed 's/,*$//')   # BSD seq ends the list with a separator
if lsof 2>/dev/null | grep -q "BigMoE/ext/build/bigmoe_ext"; then export BIGMOE_EXT_BUILD=build_next; B=ext/build_next; else B=ext/build; fi
(cd $B && make 2>&1 | tail -3) >> "$LOG" 2>&1
if [ ! "$B/bigmoe_ext.cpython-312-darwin.so" -nt ext/bigmoe_ext.cpp ]; then   # never run a stale ext
  say "BUILD FAILED in $B: stopping"; [ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1; exit 1; fi
say "ext built in $B"
T="$OUT/s10b"; mkdir -p "$T"
run s10b 570 trace.txt env $X $FULL $J BIGMOE_S10B=1 BIGMOE_S10B_LAYERS=$LAY BIGMOE_S10B_MISS=$T/miss.bin BIGMOE_S10B_RES=$T/res.bin $PY scripts/s10b_trace.py 0.25 400 $T
if ! lsof 2>/dev/null | grep -q "BigMoE/ext/build/bigmoe_ext"; then (cd ext/build && make 2>&1 | tail -1) >> "$LOG" 2>&1; say "ext/build updated"; fi
[ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1
say "trace: $(du -sh "$T" | cut -f1), disk $(free_gb) GB free"
peer wait --timeout 5h >> "$LOG" 2>&1   # CPU check only when nobody measures (RULES: CPU-only jobs <= 4 cores)
env OMP_NUM_THREADS=4 taskpolicy -b $PY scripts/s10b_labels.py "$T" > "$OUT/labels.txt" 2>&1; say "labels rc=$?"
say "b35 end"
