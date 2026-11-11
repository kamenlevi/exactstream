#!/bin/sh
# Leap's twentieth detached screen runner (Thu 2026-10-29 17:40; RULES.md 'Rapid screening'): runs R-40 STEP 3y, R-21
# GDNB step 1 and R-40 STEP 3z in that order, each under its own quiet lock (supersedes runners 17 / 19 / 18, which the
# peers hook kept from launching while other sessions held or queued for the Mac). If this session already holds a
# quiet lock when a screen starts (Leap took it with a background `peer lock quiet --queue` and launched this runner on
# the grant), the screen uses it instead of queueing behind the sessions that wait for that lock. run_locked.sh
# releases the lock after each screen, so the later screens queue as usual. No time-of-day gate; Kamen's stops come as
# PAUSE, which every gate here honours. No retry. Lines are pre-registered in TABLE_350.md. Log: data/runner_r.log
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
Q=$HOME/.local/share/quiet
export PEER_QUEUE_TIMEOUT=21600
log() { echo "[runner20] $* $(date '+%H:%M:%S'); swap $(sysctl -n vm.swapusage | cut -d'(' -f1 | sed 's/ *$//'), disk $(df -g / | tail -1 | awk '{print $4}') GB free"; }
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
      log "$id: lock"
      ./run_locked.sh 10 "$@"; rc=$?
      log "$id: rc $rc"
      return $rc
    fi
    log "$id: no lock, queueing again"; sleep 60
  done
}
{
  log "then: R-40 STEP 3y, R-21 GDNB step 1, R-40 STEP 3z (C=176)"
  screen R40S3Y "R-40 step 3y: prefill state-leak bisection, full streamed 125B (~1-2 min)" \
    $PY scripts/r40s3y_screen.py 176
  cat data/r40s3y_screen.txt
  screen R21S1 "R-21 GDNB step 1: batched GDN replay bound, synthetic GDN shapes (~1-2 min)" \
    $PY scripts/r21_screen.py
  cat data/r21_screen.txt
  screen R40S3Z "R-40 step 3z: STEP 3 re-run, REF2 + SB vs today, full streamed 125B (~7-9 min)" \
    $PY scripts/r40s3z_screen.py 176
  cat data/r40s3z_screen.txt
  log "done"
} >> data/runner_r.log 2>&1
