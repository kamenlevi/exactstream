#!/bin/bash
# 3-bit build batch 1 (2026-10-20 evening): ext rebuild, tier unit tests, v1 shas with the repetition guard on (24 GB
# mode, whole-process footprint via /usr/bin/time -l), and the ifeval/1219 loop prompt (guard arms; spec == ar2).
# One quiet lock for the batch (segtest_runner pattern). Results -> data/b3/.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
OUT="$HOME/BigMoE/data/b3"; mkdir -p "$OUT"
LOG="$OUT/b3.log"
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
say "b3 start"
peer lock quiet --queue 90m "BigMoE: 3-bit build tests + v1 sha recheck (~20 min)" >> "$LOG" 2>&1 || { say "no lock"; exit 1; }
if lsof 2>/dev/null | grep -q bigmoe_ext.cpython; then say "ext in use: not rebuilding, stop"; peer unlock >> "$LOG" 2>&1; exit 1; fi
(cd ext/build && make 2>&1 | tail -2) >> "$LOG" 2>&1
run test_records3 300 tests.txt $PY scripts/test_records3.py 11 16 "$OUT"
run test_tier 300 tests.txt $PY scripts/test_tier.py 3
X="$REF BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_FRSPEC=1 BIGMOE_SCOUT_BUNDLE=1 MTPLX_QWEN4_VERIFY_GLUE=1 BIGMOE_MTP_STREAM=1 BIGMOE_RECORDS=1"
SPEC="$HOME/SpecDecode/scripts/test_spec125.py"
for P in prose code edit; do     # expect prose efab73b6f6, code 8c56402b7f, edit cd4289b64d (guard disarmed at 96 tokens)
  run "v1_$P" 600 v1guard.txt /usr/bin/time -l env $X PROMPT=$P $PY $SPEC 0.25 96 3:5:0,a:2.0
done
run loop1219 900 loop.txt env $X PROMPT=loop $PY $SPEC 0.25 1024 a:2.0
run loop1219_off 900 loop.txt env $X PROMPT=loop BIGMOE_LOOP_GUARD=0 MTPLX_LOOP_GUARD=0 $PY $SPEC 0.25 1024 a:2.0
peer unlock >> "$LOG" 2>&1
say "b3 end"
