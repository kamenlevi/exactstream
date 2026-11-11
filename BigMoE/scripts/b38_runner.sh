#!/bin/bash
# b38 = S10c all-layer CONFIRMATION A/B (NewIdeas GO Tue ~20:50 on b34 BETWEEN; K = 12 only): 4 runs A1 S1 S2 A2 in one
# 10-min quiet lock, prompt 4 of each domain, 400 tokens, b33 settings (S10b miss labels + BIGMOE_TRACE on, no x / res).
# A = the shipped decode; S = TrainLab's M1a lead-2 lists at ALL 46 pairs L -> L + 2 (~/TrainLab/data/m1a_all.npz, top-12),
# replacing every covered lead-1 router list. All runs PRED_LINES=3 + PRED_CAP=61. Scored by r27_ab.py (generic A vs S):
# PASS whole-process SSD bytes per token (rusage) <= -5 % at tok/s >= -1 %; KILL > -2 %; INVALID if any sha differs.
# M1a adds ~145 MB of f16 weights outside the pools (reported, not charged).
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
set -m   # each run is its own process group: time-up or a lost lock stops the whole tree (RULES, orchestrator 20:5x)
cd "$HOME/BigMoE" || exit 1
OUT="$HOME/BigMoE/data/b38"; mkdir -p "$OUT"
LOG="$OUT/b38.log"
PY="$HOME/BigMoE/.venv/bin/python"
M1A="${M1A:-$HOME/TrainLab/data/m1a_all.npz}"
REF="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12"
LOCKF="$HOME/.claude-peers/locks/${CLAUDE_PID}.json"
keepalive() { "$PY" -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF"; }
left_s() { "$PY" -I -S -c 'import json, sys, time; print(int(json.load(open(sys.argv[1])).get("until", 0) - time.time()))' "$LOCKF" 2>/dev/null || echo 0; }
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
say "b38 start (CLAUDE_PID=$CLAUDE_PID, M1A=$M1A)"
[ -s "$M1A" ] || { say "no M1a export at $M1A: not taking the lock"; exit 1; }
if [ -z "$HAVE_LOCK" ]; then
  PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 10m "BigMoE b38 (S10c all-layer confirmation A/B): M1a lead-2 at all 46 layers vs the shipped forecast (24 GB v3 decode, 4 runs x 4 prompts x 400 tok, ~7 min)" >> "$LOG" 2>&1 || { say "no lock"; exit 1; }
fi
X="$REF BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_FRSPEC=1 BIGMOE_SCOUT_BUNDLE=1 MTPLX_QWEN4_VERIFY_GLUE=1 BIGMOE_MTP_STREAM=1"
V3="BIGMOE_TIER=4433333333 BIGMOE_TIER_GLU=3c BIGMOE_TIER_COMBINE=1 BIGMOE_OOO=1"
G="BIGMOE_GDN_EXACT=1"
PG="BIGMOE_TIER_PF_GROUP=64 BIGMOE_PF_NOLOGITS=1"
FULL="$G $V3 $PG BIGMOE_EVICT_LFU=0.99 BIGMOE_VERIFY_ROWS_SDPA=1"
J="BIGMOE_JOINT=1"
AB="BIGMOE_PRED_LINES=3 BIGMOE_PRED_CAP=61 BIGMOE_TRACE=1 S10B_IDX=4 S10B_X_IDX=9"
if lsof 2>/dev/null | grep -q "BigMoE/ext/build/bigmoe_ext"; then export BIGMOE_EXT_BUILD=build_next; B=ext/build_next; else B=ext/build; fi
(cd $B && make 2>&1 | tail -3) >> "$LOG" 2>&1
if [ ! "$B/bigmoe_ext.cpython-312-darwin.so" -nt ext/bigmoe_ext.cpp ]; then   # never run a stale ext
  say "BUILD FAILED in $B: stopping"; [ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1; exit 1; fi
say "ext built in $B"
ok=1
for R in A1 S1 S2 A2; do
  T="$OUT/$R"; mkdir -p "$T"
  ARM=""; [ "${R:0:1}" = "S" ] && ARM="BIGMOE_M1A=$M1A BIGMOE_M1A_K=12"
  run "$R" 200 "$R.txt" env $X $FULL $J $AB $ARM BIGMOE_S10B=1 BIGMOE_S10B_MISS=$T/miss.bin $PY scripts/s10b_trace.py 0.25 400 $T || { say "$R failed: stopping the batch"; ok=0; break; }
done
[ $ok = 1 ] || say "batch incomplete"
if ! lsof 2>/dev/null | grep -q "BigMoE/ext/build/bigmoe_ext"; then (cd ext/build && make 2>&1 | tail -1) >> "$LOG" 2>&1; say "ext/build updated"; fi
[ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1
say "traces: $(du -sh "$OUT" | cut -f1)"
peer wait --timeout 5h >> "$LOG" 2>&1   # CPU scoring only when nobody measures
env OMP_NUM_THREADS=4 taskpolicy -b $PY scripts/r27_ab.py "$OUT" > "$OUT/ab.txt" 2>&1; say "ab rc=$?"
say "b38 end"
