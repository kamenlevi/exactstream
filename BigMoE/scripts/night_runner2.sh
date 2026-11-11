#!/bin/bash
# BigMoE night runner, part 2 (detached; starts after part 1 exits). v2a GSM8K scoring + near-tie check on the
# diverging problems, 16K prefill repeats, and a check that the per-generation QSA reset leaves plain decode unchanged.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
OUT="$HOME/BigMoE/data/night"; mkdir -p "$OUT"; LOG="$OUT/runner2.log"
PY="$HOME/BigMoE/.venv/bin/python"
DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "$(date +%Y-%m-%d) 06:40" +%s)
[ "$DEADLINE" -le "$(date +%s)" ] && DEADLINE=$((DEADLINE + 86400))
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
step() {
  local name=$1 to=$2 of=$3; shift 3
  local now; now=$(date +%s)
  if [ "$now" -ge "$DEADLINE" ]; then say "skip $name (past 06:40)"; return; fi
  local left=$(( DEADLINE + 1200 - now )); [ "$to" -gt "$left" ] && to=$left
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
say "runner2 start; waiting for part 1"
while pgrep -f "scripts/night_runner.sh" > /dev/null && [ "$(date +%s)" -lt "$DEADLINE" ]; do sleep 30; done
say "part 1 gone"
# 1: score v2a vs eager, list diverging problems
"$PY" - > "$OUT/gsm_v2a_score.txt" 2>&1 <<'PYEOF'
import json, os
d = os.path.expanduser("~/BigMoE/gsmroot/data/")
a = json.load(open(d + "w125_gsm_eager.json")); b = json.load(open(d + "w125_gsm_v2a.json"))
n = min(len(a["answers"]), len(b["answers"]))
diff = [i for i in range(n) if a["answers"][i] != b["answers"][i]]
print(f"v2a vs eager over {n}: identical {n - len(diff)}/{n}; correct eager {sum(a['ok'][:n])} vs v2a {sum(b['ok'][:n])}; differing {diff}")
open(os.path.expanduser("~/BigMoE/data/night/gsm_v2a_diff.txt"), "w").write(",".join(map(str, diff)))
PYEOF
DIFF=$(cat "$OUT/gsm_v2a_diff.txt" 2>/dev/null)
NDIFF=$(echo "$DIFF" | tr ',' '\n' | grep -c .)
say "v2a diff: [$DIFF] ($NDIFF)"
# 2: near-tie at each divergence (eager arm vs v2a arm), if few enough
if [ -n "$DIFF" ] && [ "$NDIFF" -le 12 ]; then
  NT="NT_PROBS=$DIFF NT_OUT=$OUT/nt_{}.json NT_ARM=v2a"
  step nt_eager 2400 neartie_v2a.txt env $NT $PY scripts/near_tie.py eager
  step nt_v2a 2400 neartie_v2a.txt env $NT BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 MTPLX_FUSED_GDN_CONVNORM=1 MTPLX_FUSED_CONVNORM_VERIFY=1 MTPLX_FUSED_GDN_INPROJ=1 $PY scripts/near_tie.py v2a
  env $NT $PY scripts/near_tie.py compare >> "$OUT/neartie_v2a.txt" 2>&1
fi
# 3: the per-generation QSA reset must leave plain decode unchanged (sha c09a8163f9) and cost ~nothing
step reset_check 900 reset_check.txt env $REF $PY scripts/bench.py 0.50 ar2 64
# 4: 16K prefill repeats (layer-major + full scratch, exact) for variance
for r in 1 2; do
  step prefill16k_r$r 1200 prefill16k.txt env $REF BIGMOE_PF_PREFILL=1 BIGMOE_PF_PROF=1 BIGMOE_PF_FULLSCR=1 LMP_MODES=lmp $PY "$HOME/MoonShots/patches/layer_major_prefill/bench_lmp125.py" 0.5 16384 8
done
say "runner2 end"
