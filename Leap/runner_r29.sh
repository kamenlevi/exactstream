#!/bin/sh
# Leap's twenty-ninth detached screen runner (Thu 2026-10-29 22:05; RULES.md 'Rapid screening'): waits for the pid
# given as $1 (0: none), then runs R-40 STEP 3t (where REF2's live loop leaves its AR: start-state digests +
# per-layer row records) under its own quiet lock, with the 30 s memory settle. If this session already holds a
# quiet lock when a screen starts (Leap took it with a background `peer lock quiet --queue` and launched this runner on
# the grant), the screen uses it instead of queueing behind the sessions that wait for that lock. run_locked.sh
# releases the lock after each screen, so the later screens queue as usual. No time-of-day gate; Kamen's stops come as
# PAUSE, which every gate here honours. No retry. Lines are pre-registered in TABLE_350.md. Log: data/runner_r.log
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
Q=$HOME/.local/share/quiet
export PEER_QUEUE_TIMEOUT=21600
log() { echo "[runner29] $* $(date '+%H:%M:%S'); swap $(sysctl -n vm.swapusage | cut -d'(' -f1 | sed 's/ *$//'), disk $(df -g / | tail -1 | awk '{print $4}') GB free"; }
lid_closed() { ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes; }
ok_now() {   # also waits while soft_end.sh ramps held jobs back (it SIGSTOPs their new children too: Fit12, 10-27)
  [ ! -e $Q/PAUSE ] && ! lid_closed && ! grep -q "Verdict: ACT" $Q/HEALTH.md 2>/dev/null \
    && ! pgrep -f '^/bin/zsh [^ ]*soft_end\.sh' >/dev/null \
    && [ "$(memory_pressure 2>/dev/null | awk '/free percentage/ {sub("%", "", $NF); print int($NF)}')" -ge 58 ]   # ~37 GB free: room for the full streamed 125B at C = 176 (~35 GB peak)
}
held() {     # this session holds an unexpired quiet lock
  $PY -I -S -c 'import json, sys, time
d = json.load(open(sys.argv[1]))
sys.exit(0 if d.get("mode") == "quiet" and d.get("until", 0) > time.time() else 1)' "$HOME/.claude-peers/locks/$CLAUDE_PID.json" 2>/dev/null
}
screen() {   # screen <id> <why> <command...>
  id=$1; why=$2; shift 2
  while :; do
    w=0; while ! ok_now; do [ $w -eq 0 ] && log "$id: waiting (PAUSE / ACT / lid / memory)"; w=1; sleep 120; done
    if held; then log "$id: lock already held"; else log "$id: queued"; fi
    if held || peer lock quiet --queue 10m "Leap $id: $why"; then
      if ! ok_now; then peer unlock; log "$id: lock came under PAUSE / ACT / closed lid / low memory: released"; continue; fi
      sleep 30                                       # let the previous holder's memory drain
      if ! ok_now; then peer unlock; log "$id: memory not back 30 s after the grant: released"; sleep 60; continue; fi
      log "$id: lock"
      ./run_locked.sh 10 "$@"; rc=$?
      log "$id: rc $rc"
      return $rc
    fi
    log "$id: no lock, queueing again"; sleep 60
  done
}
{
  log "then: R-40 STEP 3t (start states + row records, C 128)"
  while [ "$1" != 0 ] && kill -0 "$1" 2>/dev/null; do sleep 30; done
  screen R40S3T "R-40 step 3t: where the live loop leaves REF2's AR (state digests + row records), full streamed 125B (~2 min)" \
    $PY scripts/r40s3t_screen.py 128
  cat data/r40s3t_screen.txt
  log "done"
} >> data/runner_r.log 2>&1
