#!/bin/bash
# BigMoE night runner (2026-10-19, Plan B): detached GPU jobs back to back while no Claude session runs (03:15-07:15).
# Each step: skip if past the deadline, take a queued quiet lock, wait for free memory, run under a hard timeout,
# release the lock. Results -> ~/BigMoE/data/night/. Exits by ~07:05. Needs nobody.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
OUT="$HOME/BigMoE/data/night2"; mkdir -p "$OUT"
LOG="$OUT/batch.log"
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
say "batch start: queue one lock"
peer lock quiet --queue 60m "BigMoE batch: 14 skipped night jobs (~45 min)" >> "$LOG" 2>&1 || { say "no lock"; exit 1; }
X="$REF BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_FRSPEC=1 BIGMOE_SCOUT_BUNDLE=1 MTPLX_QWEN4_VERIFY_GLUE=1"
M24="$X BIGMOE_PF_GROUP=64 BIGMOE_MTP_STREAM=1 BIGMOE_RECORDS=1"
SPEC="$HOME/SpecDecode/scripts/test_spec125.py"
for round in 2 3; do
  for P in code edit prose; do
    run m24_${P}_r$round 900 final_24gb.txt env $M24 PROMPT=$P $PY $SPEC 0.25 96 3:5:0,3:7:1
    run s50_${P}_r$round 900 final_50.txt env $X BIGMOE_RECORDS=1 PROMPT=$P $PY $SPEC 0.50 96 3:5:0,3:7:1
  done
done
run m24_mem 600 final_24gb_mem.txt env $M24 $PY scripts/mem_probe.py 0.25 64
run long 2400 long.txt env $X BIGMOE_RECORDS=1 $PY scripts/longctx.py 0.50 128 16384,32768
peer unlock >> "$LOG" 2>&1
say "batch end"
