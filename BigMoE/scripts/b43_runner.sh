#!/bin/bash
# b43 = R-43 DFL24 pricing screen (NewIdeas Wed ~16:25; no build): what would DFlash2 16-row checks cost in the 24 GB
# engine? The gate row carries <= 92 expert ids (MAXK 92, one extra id line), so a gated 16-row check needs a build;
# this sweeps the verify width on copy-heavy prompts (edit = code text, copy_math = GSM solutions; long n-gram drafts,
# fixed widths) at 4 / 8 / 9 rows + the shipped settings, traced (gate wait, GPU, reads per pass; spec125 timers), and
# scripts/r43_price.py extrapolates per-pass costs to 16 rows with D-02b's kept-per-check (9.57 code, 7.00 math) and
# 12.5 / 7.9 ms drafting. Lines (NewIdeas): PASS (build) >= +25 % on code and math and >= 0 on prose; KILL < +10 % on code.
set -m   # each run is its own process group: time-up or a lost lock stops the whole tree (RULES)
cd "$HOME/BigMoE" || exit 1
OUT="$HOME/BigMoE/data/b43"; mkdir -p "$OUT"
LOG="$OUT/b43.log"
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
run() {  # run NAME TIMEOUT OUTFILE CMD...  (memory gate at most 30 s: the lock is only 10 min)
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
say "b43 start (CLAUDE_PID=$CLAUDE_PID)"
if [ -z "$HAVE_LOCK" ]; then
  PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 10m "BigMoE b43 (R-43 DFL24 pricing): verify-width sweep 4 / 8 / 9 rows + shipped reference, traced (24 GB v3, 4 runs x 2 prompts x 256 tok, ~6 min)" >> "$LOG" 2>&1 || { say "no lock"; exit 1; }
fi
X="$REF BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_FRSPEC=1 BIGMOE_SCOUT_BUNDLE=1 MTPLX_QWEN4_VERIFY_GLUE=1 BIGMOE_MTP_STREAM=1"
V3="BIGMOE_TIER=4433333333 BIGMOE_TIER_GLU=3c BIGMOE_TIER_COMBINE=1 BIGMOE_OOO=1"
G="BIGMOE_GDN_EXACT=1"
PG="BIGMOE_TIER_PF_GROUP=64 BIGMOE_PF_NOLOGITS=1"
FULL="$G $V3 $PG BIGMOE_EVICT_LFU=0.99 BIGMOE_VERIFY_ROWS_SDPA=1"
J="BIGMOE_JOINT=1"
if lsof 2>/dev/null | grep -q "BigMoE/ext/build/bigmoe_ext"; then export BIGMOE_EXT_BUILD=build_next; B=ext/build_next; else B=ext/build; fi
(cd $B && make 2>&1 | tail -3) >> "$LOG" 2>&1
if [ ! "$B/bigmoe_ext.cpython-312-darwin.so" -nt ext/bigmoe_ext.cpp ]; then   # never run a stale ext
  say "BUILD FAILED in $B: stopping"; [ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1; exit 1; fi
say "ext built in $B"
run A 200 A.txt env $X $FULL $J BIGMOE_MTP_PF_GROUP=48 BIGMOE_TRACE=1 $PY scripts/s10_anatomy.py 0.25 256 "$OUT/A.json" edit,copy_math
run W4 200 W4.txt env $X $FULL $J BIGMOE_MTP_PF_GROUP=48 BIGMOE_TRACE=1 S10_FIXED=1 S10_MAX_DRAFT=3 S10_LOOKUP=2 $PY scripts/s10_anatomy.py 0.25 256 "$OUT/W4.json" edit,copy_math
run W8 200 W8.txt env $X $FULL $J BIGMOE_MTP_PF_GROUP=48 BIGMOE_TRACE=1 S10_FIXED=1 S10_MAX_DRAFT=7 S10_LOOKUP=2 $PY scripts/s10_anatomy.py 0.25 256 "$OUT/W8.json" edit,copy_math
run W9 200 W9.txt env $X $FULL $J BIGMOE_MTP_PF_GROUP=48 BIGMOE_TRACE=1 S10_FIXED=1 S10_MAX_DRAFT=8 S10_LOOKUP=2 BIGMOE_MAXK=92 $PY scripts/s10_anatomy.py 0.25 256 "$OUT/W9.json" edit,copy_math
if ! lsof 2>/dev/null | grep -q "BigMoE/ext/build/bigmoe_ext"; then (cd ext/build && make 2>&1 | tail -1) >> "$LOG" 2>&1; say "ext/build updated"; fi
[ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1
say "runs done"
say "b43 end"
