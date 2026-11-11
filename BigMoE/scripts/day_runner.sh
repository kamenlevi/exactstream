#!/bin/bash
# BigMoE day runner (pacing sleeps): detached GPU jobs back to back while no Claude session runs (03:15-07:15).
# Each step: skip if past the deadline, take a queued quiet lock, wait for free memory, run under a hard timeout,
# release the lock. Results -> ~/BigMoE/data/night/. Exits by ~07:05. Needs nobody.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
OUT="$HOME/BigMoE/data/day"; mkdir -p "$OUT"
LOG="$OUT/day_runner.log"
PY="$HOME/BigMoE/.venv/bin/python"
DEADLINE=$(( $(date +%s) + 3*3600 ))
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
step() {  # step NAME TIMEOUT_S OUTFILE CMD...
  local name=$1 to=$2 of=$3; shift 3
  local now; now=$(date +%s)
  if [ "$now" -ge "$DEADLINE" ]; then say "skip $name (past 06:40)"; return; fi
  local left=$(( DEADLINE + 1200 - now )); [ "$to" -gt "$left" ] && to=$left        # never run past ~07:00
  say "queue $name"
  peer lock quiet --queue 45m "night_runner: BigMoE $name" >> "$LOG" 2>&1 || { say "no lock for $name"; return; }
  # memory guard: >= 85 % free AND no other process with > 8 GB resident (another model loaded), else skip
  local i=0
  while { [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -lt 85 ] || [ "$(ps -axo rss=,pid= | awk -v me=$$ '$1 > 8000000 && $2 != me' | wc -l)" -gt 0 ]; } && [ $i -lt 120 ]; do sleep 5; i=$((i+1)); [ $((i % 48)) -eq 0 ] && keepalive; done
  if [ $i -ge 120 ]; then say "skip $name (memory busy: another big process or < 85 % free)"; peer unlock >> "$LOG" 2>&1; return; fi
  say "run $name (timeout ${to}s)"
  perl -e 'alarm shift; exec @ARGV' "$to" "$@" >> "$OUT/$of" 2>&1 &
  local jp=$!
  # keepalive: peer attributes lock activity to Claude sessions, not to this detached runner, so re-take the lock every
  # 4 min while the job runs; if that is refused (someone is waiting), stop the job rather than overlap two models
  while kill -0 $jp 2>/dev/null; do
    for k in $(seq 36); do kill -0 $jp 2>/dev/null || break; sleep 5; done
    kill -0 $jp 2>/dev/null || break
    if ! keepalive; then
      say "lost the quiet lock during $name: stopping it"; kill $jp 2>/dev/null; sleep 5; kill -9 $jp 2>/dev/null; break
    fi
  done
  wait $jp
  say "done $name rc=$?"
  peer unlock >> "$LOG" 2>&1
}
OF_DIR="$OUT/"
say "day runner start"
V1="$REF BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_FRSPEC=1"
# 1: forced sorted expert gather at verify widths: spec == plain == c09a8163f9 means it is exact (not v2)
step sort_verify 1200 sort_verify.txt env $V1 BIGMOE_SORT_VERIFY=1 $PY "$HOME/SpecDecode/scripts/test_spec125.py" 0.50 64 3:5:0,3:5:0,3:7:1,5:5:0
# 2: stacked v1 build (FR-Spec) vs today's baseline, code / edit / prose, 0.50/96
for P in code edit prose; do
  step base_$P 1200 stack_$P.txt env $REF BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 PROMPT=$P $PY "$HOME/SpecDecode/scripts/test_spec125.py" 0.50 96 3:5:0,3:5:0,3:7:1
  step stack_$P 1200 stack_$P.txt env $V1 PROMPT=$P $PY "$HOME/SpecDecode/scripts/test_spec125.py" 0.50 96 3:5:0,3:5:0,3:7:1
done
say "day runner end"
