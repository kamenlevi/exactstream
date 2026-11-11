#!/bin/bash
# BigMoE night runner (2026-10-19, Plan B): detached GPU jobs back to back while no Claude session runs (03:15-07:15).
# Each step: skip if past the deadline, take a queued quiet lock, wait for free memory, run under a hard timeout,
# release the lock. Results -> ~/BigMoE/data/night/. Exits by ~07:05. Needs nobody.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
OUT="$HOME/BigMoE/data/night"; mkdir -p "$OUT"
LOG="$OUT/runner.log"
PY="$HOME/BigMoE/.venv/bin/python"
DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "$(date +%Y-%m-%d) 06:40" +%s)
[ "$DEADLINE" -le "$(date +%s)" ] && DEADLINE=$((DEADLINE + 86400))   # next 06:40
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
say "night runner start"
# 1-2: LC1 bisect (which QSA state leaks: indexer cores or lru caches). Fresh sha a67851027e.
step lc1_reset_core 1300 lc1_bisect.txt env $REF $PY scripts/lc1_reset.py reset-core 128
step lc1_reset_lru 1300 lc1_bisect.txt env $REF $PY scripts/lc1_reset.py reset-lru 128
# 3: v2a GSM8K-150 arm (reference + exact-or-not candidate flags: GDN CONVNORM/VERIFY, fused GDN in_proj), 5 x 30 problems
V2A="W125_UPSTREAM=1 W125_TAG=v2a W125_CHUNK=30 BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 MTPLX_FUSED_GDN_CONVNORM=1 MTPLX_FUSED_CONVNORM_VERIFY=1 MTPLX_FUSED_GDN_INPROJ=1"
for c in 1 2 3 4 5; do
  step gsm_v2a_$c 1600 gsm_v2a.txt env $V2A $PY gsmroot/scripts/waittop125.py 0.50 exact gsm150
done
# 4: stacked re-measure, best exact stack (code prompt, 50 %), 2 processes
BEST="$REF BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80"
for r in 1 2; do
  step total_r$r 900 total.txt env $BEST $PY "$HOME/SpecDecode/scripts/test_spec125.py" 0.50 64 3:5:0,3:5:0,3:7:1,3:7:1
done
# 5: 32K prefill with layer-major + full scratch (exact), and PF3 (v2)
step prefill_32k 1800 prefill32k.txt env $REF BIGMOE_PF_PREFILL=1 BIGMOE_PF_PROF=1 BIGMOE_PF_FULLSCR=1 LMP_MODES=lmp,pf3 $PY "$HOME/MoonShots/patches/layer_major_prefill/bench_lmp125.py" 0.5 32768 8
say "night runner end"
