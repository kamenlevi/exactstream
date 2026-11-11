#!/bin/bash
# 3-bit build batch 4: exact mode on records3 after the serial-rebuild fix (b5: spin-gate timeouts in spec) (2026-10-20 evening): records/ -> records3/ conversion (MoonShots make_records3.py --delete-src),
# then guard_sim, exact mode on records3 (BIGMOE_RECORDS=3, v1 shas), loop retest, first BIGMOE_TIER=4433333333 runs.
# One quiet lock for the batch. Results -> data/b4/.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
OUT="$HOME/BigMoE/data/b6"; mkdir -p "$OUT"
LOG="$OUT/b6.log"
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
  local i=0
  while { [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -lt 85 ] || [ "$(ps -axo rss=,pid= | awk -v me=$$ '$1 > 8000000 && $2 != me' | wc -l)" -gt 0 ]; } && [ $i -lt 120 ]; do sleep 5; i=$((i+1)); [ $((i % 36)) -eq 0 ] && keepalive; done
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
say "b6 start"
if [ -z "$HAVE_LOCK" ]; then   # HAVE_LOCK=1: the session already holds this PID's quiet lock (taken in the queue)
  peer lock quiet --queue 120m "BigMoE b6: records3 exact-mode spec fix (~12 min)" >> "$LOG" 2>&1 || { say "no lock"; exit 1; }
fi
export BIGMOE_EXT_BUILD=build_next     # ext/build may be loaded by another session's runner
mkdir -p ext/build_next
(cd ext/build_next && { [ -f Makefile ] || cmake .. -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$PY" > cmake.log 2>&1; } && make 2>&1 | tail -1) >> "$LOG" 2>&1
run test_records3 300 tests.txt $PY scripts/test_records3.py 9 128
grep -q '^PASS' "$OUT/tests.txt" || { say "test_records3 failed: stop"; peer unlock >> "$LOG" 2>&1; exit 1; }
X="$REF BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_FRSPEC=1 BIGMOE_SCOUT_BUNDLE=1 MTPLX_QWEN4_VERIFY_GLUE=1 BIGMOE_MTP_STREAM=1"
SPEC="$HOME/SpecDecode/scripts/test_spec125.py"
for P in prose code edit; do     # exact mode on records3 after the serial-rebuild fix: v1 shas, no spin-gate timeouts
  run "r3_$P" 600 r3.txt /usr/bin/time -l env $X BIGMOE_RECORDS=3 PROMPT=$P $PY $SPEC 0.25 96 3:5:0,a:2.0
done
run tier3cc_code 900 tier3cc.txt env $X BIGMOE_TIER=4433333333 BIGMOE_TIER_GLU=3c BIGMOE_TIER_COMBINE=1 PROMPT=code $PY $SPEC 0.25 96 a:2.0
if ! lsof 2>/dev/null | grep -q "ext/build/bigmoe_ext"; then (cd ext/build && make 2>&1 | tail -1) >> "$LOG" 2>&1; say "ext/build updated"; fi
peer unlock >> "$LOG" 2>&1
say "b6 end"
