# Breakthrough's per-item quiet-lock functions for detached runners (orchestrator 10-27 19:34: a run over 10 min takes
# the quiet lock PER ITEM and releases it between items, so queued <= 10-min screens slot in). Waits while PAUSE exists,
# the health monitor says ACT, soft_end.sh's resume ramp runs (it SIGSTOPs new children: Fit12 10-27) or it is outside
# 08:01-23:50. Each step stamps the lock's active_at, is not timed while a quiet window holds it stopped, and stops on a
# lost lock, swap growth > 2 GB or health ACT (Leap's run_locked.sh guards).
# Source it with OUT (output dir) and LOG (summary file) set, then: item NAME LIMIT_S WHY CMD...
PY=$HOME/BigMoE/.venv/bin/python; Q=$HOME/.local/share/quiet
LOCKF=$HOME/.claude-peers/locks/$CLAUDE_PID.json
export PYTHONDONTWRITEBYTECODE=1
log() { echo "$(date '+%a %H:%M:%S') $*" >> "$LOG"; }
in_hours() { [ "${NIGHT_OK:-0}" = 1 ] && return 0; h=$(date +%H%M); [ "$h" -ge 0801 ] && [ "$h" -le 2350 ]; }   # NIGHT_OK=1: Kamen lifted the silent night
act() { grep -q '^\*\*Verdict: ACT' $Q/HEALTH.md 2>/dev/null; }
ramp() { pgrep -f '^/bin/zsh [^ ]*soft_end\.sh' > /dev/null 2>&1; }
ok() { in_hours && [ ! -e $Q/PAUSE ] && ! act && ! ramp; }
keepalive() { $PY -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF"; }
swap_mb() { sysctl -n vm.swapusage | awk '{for (i = 1; i <= NF; i++) if ($i == "used") {v = $(i + 2); sub("M", "", v); print int(v)}}'; }
tree() { echo "$1"; for c in $(pgrep -P "$1"); do tree "$c"; done; }
paused() { for p in $(tree $P); do case "$(ps -o stat= -p $p 2>/dev/null)" in *T*) return 0;; esac; done; return 1; }
stop() { log "$1: stopping $name"; T=$(tree $P); kill $T 2>/dev/null; sleep 5; kill -9 $T 2>/dev/null; }
step() {   # step NAME SECONDS_LIMIT CMD... (the lock must be held)
  name=$1; lim=$2; shift 2
  keepalive || { log "lock lost before $name"; return 125; }
  log "start $name"
  SW0=$(swap_mb)
  ( "$@" > "$OUT/$name.txt" 2>&1 ) & P=$!
  i=0; rc=0
  while kill -0 $P 2>/dev/null && [ $i -lt "$lim" ]; do
    sleep 5
    if paused; then continue; fi
    i=$((i + 5))
    if [ $((i % 120)) -eq 0 ]; then keepalive || { stop "lost the lock"; rc=125; break; }; fi
    SW=$(swap_mb); if [ $((SW - SW0)) -gt 2048 ]; then stop "swap grew ${SW0} -> ${SW} MB"; rc=126; break; fi
    if act; then stop "health monitor at ACT"; rc=127; break; fi
  done
  kill -0 $P 2>/dev/null && { stop "TIMEOUT"; rc=124; }
  wait $P; w=$?; [ $rc -eq 0 ] && rc=$w
  log "done $name (rc $rc): $(grep -E 'RESULT|phase (prefill|decode)|decode [0-9]+:|Error|Traceback|B12_PFR skipped' "$OUT/$name.txt" | tail -6 | cut -c1-300 | tr '\n' ' ')"
  return $rc
}
item() {   # item NAME SECONDS_LIMIT WHY CMD...: wait until ok, queue a quiet lock (LOCK_MIN, default 10 min), run the step, unlock
  name=$1; lim=$2; why=$3; shift 3
  if [ "${HELD:-0}" = 1 ] && keepalive; then        # HELD=1: the session took this lock and launched the runner on it
    HELD=0; log "$name: on the lock the session already holds"
  else
    until ok; do sleep 30; done
    log "$name: queued"
    if ! PEER_QUEUE_TIMEOUT=7200 peer lock quiet --queue ${LOCK_MIN:-10}m "Breakthrough $why"; then log "$name: no lock"; return 125; fi
  fi
  if ! ok; then log "$name: lock came with PAUSE / ACT / ramp / out of hours, released"; peer unlock; return 125; fi
  step "$name" "$lim" "$@"; rc=$?
  peer unlock
  return $rc
}
