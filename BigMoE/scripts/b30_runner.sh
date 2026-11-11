#!/bin/bash
# b30 (S10 follow-ups, NewIdeas 15:0x, pre-registered: adopt an arm at >= +3 % tok/s on code AND prose, edit not below -2 %,
# shas identical, CLEAN prompts only, paired rounds). All on the JOINT v3 stack, adaptive width:
#   j = reference; t3 = precision gate BIGMOE_PF_THETA=0.3; rt3 = ranked unique forecast lists (24) + THETA 0.3;
#   p2t3 = L+2 forecasts (PRED_LINES=3, PREFETCH2=8) + THETA 0.3; p2t3f = p2t3 + per-distance queue (BIGMOE_PF_QD_FAR=16:
#   next-layer reads bypass QD, deeper ones at QD 16); 2 rounds, order reversed in round 2. Then one p23f run (+ L+3
#   forecasts, PREFETCH3=6) for the raw recall at lead 1 / 2 / 3. The ext logs per forecast distance: raw recall of the
#   misses, unique list size, reads issued, precision, recall of issued (s10_anatomy "forecasts:" line). -> data/b30/.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
OUT="$HOME/BigMoE/data/b30"; mkdir -p "$OUT"
LOG="$OUT/b30.log"
PY="$HOME/BigMoE/.venv/bin/python"
REF="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12"
LOCKF="$HOME/.claude-peers/locks/${CLAUDE_PID}.json"
# keepalive (MoonShots' fix): peer counts a lock as active from max(taken, active_at); stamp active_at in our lock file
# (atomic replace), as peer's own hook does for a session's commands. Exit 1 = lock gone / not quiet / expired.
keepalive() { "$PY" -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF"; }
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
# ONE quiet lock for the whole batch (RULES.md "One lock per batch, not per job"); keepalive every 3 min; memory gate per job.
run() {  # run NAME TIMEOUT OUTFILE CMD...
  local name=$1 to=$2 of=$3; shift 3
  keepalive || { say "lost the lock before $name: stopping the batch"; exit 1; }
  while [ -e "$HOME/.local/share/quiet/PAUSE" ]; do sleep 5; keepalive; done   # quiet mode: never start a job into a pause
  local i=0
  while { [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -lt 60 ] || [ "$(ps -axo rss=,pid= | awk -v me=$$ '$1 > 8000000 && $2 != me' | wc -l)" -gt 0 ]; } && [ $i -lt 120 ]; do sleep 5; i=$((i+1)); [ $((i % 36)) -eq 0 ] && keepalive; done
  [ $i -ge 120 ] && { say "skip $name (memory busy)"; return; }
  say "run $name"
  perl -e 'alarm shift; exec @ARGV' "$to" "$@" >> "$OUT/$of" 2>&1 &
  local jp=$!
  while kill -0 $jp 2>/dev/null; do
    for k in $(seq 36); do kill -0 $jp 2>/dev/null || break; sleep 5; done
    kill -0 $jp 2>/dev/null || break
    keepalive || { say "lost the lock during $name: stopping"; kill $jp; sleep 5; kill -9 $jp 2>/dev/null; exit 1; }
  done
  wait $jp; say "done $name rc=$?"
}
say "b30 start"
if [ -z "$HAVE_LOCK" ]; then
  peer lock quiet --queue 120m "BigMoE b30: S10 forecast follow-ups (THETA, ranked, L+2, per-distance QD) at 24 GB (~9 min)" >> "$LOG" 2>&1 || { say "no lock"; exit 1; }
fi
X="$REF BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_FRSPEC=1 BIGMOE_SCOUT_BUNDLE=1 MTPLX_QWEN4_VERIFY_GLUE=1 BIGMOE_MTP_STREAM=1"
V3="BIGMOE_TIER=4433333333 BIGMOE_TIER_GLU=3c BIGMOE_TIER_COMBINE=1 BIGMOE_OOO=1"
SPEC="$HOME/SpecDecode/scripts/test_spec125.py"
CF="3:5:0,3:7:1,a:2.0"
G="BIGMOE_GDN_EXACT=1"
PG="BIGMOE_TIER_PF_GROUP=64 BIGMOE_PF_NOLOGITS=1"
NG="BIGMOE_LOOP_GUARD=0 MTPLX_LOOP_GUARD=0"
E3="BIGMOE_RECORDS=3"
FULL="$G $V3 $PG BIGMOE_EVICT_LFU=0.99 BIGMOE_VERIFY_ROWS_SDPA=1"
if lsof 2>/dev/null | grep -q "ext/build/bigmoe_ext"; then export BIGMOE_EXT_BUILD=build_next; B=ext/build_next; else B=ext/build; fi
(cd $B && make 2>&1 | tail -3) >> "$LOG" 2>&1
if [ ! "$B/bigmoe_ext.cpython-312-darwin.so" -nt ext/bigmoe_ext.cpp ]; then   # never time a stale ext
  say "BUILD FAILED in $B: stopping"; [ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1; exit 1; fi
say "ext built in $B"
A="$PY scripts/s10_anatomy.py 0.25 256"
J="BIGMOE_JOINT=1"; SB="BIGMOE_SPIN_G2=1.05 BIGMOE_SPIN_PAIRS=256"
T3="BIGMOE_PF_THETA=0.3"; P2="BIGMOE_PRED_LINES=3 BIGMOE_PREFETCH2=8"
arm() { case $1 in j) run "j$2" 900 ab.txt env $X $FULL $J $A $OUT/ab_j$2.json;;
                 t3) run "t3_$2" 900 ab.txt env $X $FULL $J $T3 $A $OUT/ab_t3_$2.json;;
                 rt3) run "rt3_$2" 900 ab.txt env $X $FULL $J BIGMOE_PF_RANKED=1 BIGMOE_PREFETCH=24 $T3 $A $OUT/ab_rt3_$2.json;;
                 p2t3) run "p2t3_$2" 900 ab.txt env $X $FULL $J $P2 $T3 $A $OUT/ab_p2t3_$2.json;;
                 p2t3f) run "p2t3f$2" 900 ab.txt env $X $FULL $J $P2 $T3 BIGMOE_PF_QD_FAR=16 $A $OUT/ab_p2t3f$2.json;; esac; }
arm j 1; arm t3 1; arm rt3 1; arm p2t3 1; arm p2t3f 1
arm p2t3f 2; arm p2t3 2; arm rt3 2; arm t3 2; arm j 2
run p23f 900 ab.txt env $X $FULL $J $P2 BIGMOE_PREFETCH3=6 $T3 BIGMOE_PF_QD_FAR=16 $A $OUT/ab_p23f1.json
if ! lsof 2>/dev/null | grep -q "ext/build/bigmoe_ext"; then (cd ext/build && make 2>&1 | tail -1) >> "$LOG" 2>&1; say "ext/build updated"; fi
[ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1
say "b30 end"
