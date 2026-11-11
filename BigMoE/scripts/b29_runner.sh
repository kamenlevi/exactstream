#!/bin/bash
# b29 (S10 / B5s step 2, confirmation): b28's round 2 was frozen by a 10-min quiet window (13:46-13:56) and its soft-end
# ramp. 3 clean rounds of base / joint / joint+sb1, order rotated per round; s10_anatomy now flags prompts a PAUSE or a
# freeze touched (gated.PAUSED), scripts/ab_summary.py leaves them out. Shas must match. Then the S10 ceiling: an oracle
# replay of the true ids 1 / 3 layers ahead (fixed widths). -> data/b29/.
# (b28 (S10 / B5s step 2): JOINT (one read burst per layer: base + sidecar, port of Fit12 B1 v2) and SB1 (spin back-off
# G2 1.05, PAIRS 256) on the shipped 24 GB v3 stack: ABAB rounds base / joint / sb1 / joint+sb1, shas must match; one
# traced joint run for the anatomy. -> data/b28/.)
# (b26: b25's loaded half (b25's loaded runs were frozen by the fan-quiet window). b25: gate stamps (MoonShots torn-handoff fix, ported): quiet shas (exact v1, v3b) + the same under 50 % GPU load
# (MoonShots b8mr/gpuload.py): shas must hold, stamp re-reads show caught tears. -> data/b25/. (b24: b23's skipped arms (memory gate at 85 % free blocked them; 24 GB runs need ~40 %: gate 60 % here). b23: opt-in fast mode = per-pass expert budget b16 + budget-aware JIT (TrainLab, NON-EXACT) on the full v3 stack at 24 GB
# (+ GDN, PF_GROUP, LFU, VERIFY_ROWS): natural vs b16+JIT, 256 tokens, MTP 3:3:0 and adaptive. -> data/b23/.
# b9: OoO (MoEPrefetch) adopted: ext post_ids partial-hit row (all modes) + BIGMOE_OOO; v1/v3b shas, A/B, GSM chunk 2.
# b8: tier prefill = the reference call shape (compact scratch); exact via 3c (4444444444) recheck, v3 shas, v3b GSM8K chunk.
# v3 validation (b7): GSM8K 0-471 with the 3-bit engine (3c + combine), one CHUNK per lock, resumable (data/v3/). (2026-10-20 evening): records/ -> records3/ conversion (MoonShots make_records3.py --delete-src),
# then guard_sim, exact mode on records3 (BIGMOE_RECORDS=3, v1 shas), loop retest, first BIGMOE_TIER=4433333333 runs.
# One quiet lock for the batch. Results -> data/b4/.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
OUT="$HOME/BigMoE/data/b29"; mkdir -p "$OUT"
LOG="$OUT/b29.log"
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
say "b29 start"
if [ -z "$HAVE_LOCK" ]; then
  peer lock quiet --queue 120m "BigMoE b29: JOINT confirmation A/B + S10 oracle ceiling at 24 GB (~12 min)" >> "$LOG" 2>&1 || { say "no lock"; exit 1; }
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
arm() { case $1 in base) run "base$2" 900 ab.txt env $X $FULL $A $OUT/ab_base$2.json;;
                 joint) run "joint$2" 900 ab.txt env $X $FULL $J $A $OUT/ab_joint$2.json;;
                 jsb) run "jsb$2" 900 ab.txt env $X $FULL $J $SB $A $OUT/ab_jsb$2.json;; esac; }
arm base 1; arm joint 1; arm jsb 1
arm jsb 2; arm base 2; arm joint 2
arm joint 3; arm jsb 3; arm base 3
# S10 ceiling (oracle bound, MoonShots' ORACLE machinery, per-family look-ahead): fixed widths (S10_FIXED, so the replay
# sees the recording's passes); fref1 records the true ids of every (pass, row), fref2 = the same without recording;
# fd1 / fd3 reserve the true ids 1 / 3 layers ahead through the JIT queue (QD 4), fd3q at QD 16. All JOINT; shas match.
F="$J S10_FIXED=1"; ORF=$OUT/oracle.bin
run fref1 900 oracle.txt env $X $FULL $F BIGMOE_ORACLE_REC=$ORF $A $OUT/ab_fref1.json
run fref2 900 oracle.txt env $X $FULL $F $A $OUT/ab_fref2.json
run fd1 900 oracle.txt env $X $FULL $F BIGMOE_ORACLE_PLAY=$ORF BIGMOE_ORACLE_DIST=1 $A $OUT/ab_fd1_1.json
run fd3 900 oracle.txt env $X $FULL $F BIGMOE_ORACLE_PLAY=$ORF BIGMOE_ORACLE_DIST=3 $A $OUT/ab_fd3_1.json
run fd3q 900 oracle.txt env $X $FULL $F BIGMOE_ORACLE_PLAY=$ORF BIGMOE_ORACLE_DIST=3 BIGMOE_PF_QD=16 $A $OUT/ab_fd3q1.json
if ! lsof 2>/dev/null | grep -q "ext/build/bigmoe_ext"; then (cd ext/build && make 2>&1 | tail -1) >> "$LOG" 2>&1; say "ext/build updated"; fi
[ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1
say "b29 end"
