#!/bin/sh
# Leap's sixteenth detached screen runner (Thu 2026-10-29 02:12; RULES.md 'Rapid screening'): waits for the pid given
> (a quote from the project owner, kept private)
# for now" (STATE.md); his stop comes as PAUSE, which every gate here honours.
# Pass / kill lines are pre-registered in TABLE_350.md. Log: data/runner_r.log (shared with runner_r.sh)
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
Q=$HOME/.local/share/quiet
export PEER_QUEUE_TIMEOUT=21600
log() { echo "[runner16] $* $(date '+%H:%M:%S'); swap $(sysctl -n vm.swapusage | cut -d'(' -f1 | sed 's/ *$//'), disk $(df -g / | tail -1 | awk '{print $4}') GB free"; }
lid_closed() { ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes; }
ok_now() {   # also waits while soft_end.sh ramps held jobs back (it SIGSTOPs their new children too: Fit12, 10-27)
  [ ! -e $Q/PAUSE ] && ! lid_closed && ! grep -q "Verdict: ACT" $Q/HEALTH.md 2>/dev/null \
    && ! pgrep -f '^/bin/zsh [^ ]*soft_end\.sh' >/dev/null \
    && [ "$(memory_pressure 2>/dev/null | awk '/free percentage/ {sub("%", "", $NF); print int($NF)}')" -ge 58 ]   # ~37 GB free: room for the full streamed 125B at C = 176 (~35 GB peak) + the drafter
}
screen() {   # screen <id> <why> <command...>
  id=$1; why=$2; shift 2; tries=0
  while :; do
    w=0; while ! ok_now; do [ $w -eq 0 ] && log "$id: waiting (PAUSE / ACT / lid / memory)"; w=1; sleep 120; done
    log "$id: queued"
    if peer lock quiet --queue 10m "Leap $id: $why"; then
      if ! ok_now; then peer unlock; log "$id: lock came under PAUSE / ACT / closed lid / low memory: released"; continue; fi
      log "$id: lock"
      ./run_locked.sh 10 "$@"; rc=$?
      log "$id: rc $rc"; tries=$((tries + 1))
      return $rc
    fi
    log "$id: no lock, queueing again"; sleep 60
  done
}
{
  log "then: R-40 STEP 3x (localize the exactness break; warm 16-row check prices; C=176)"
  while [ "$1" != 0 ] && kill -0 "$1" 2>/dev/null; do sleep 30; done
  screen R40S3X "R-40 step 3x: REF2 exactness variants + warm 16-row check prices, full streamed 125B (~3 min)" \
    $PY scripts/r40s3x_screen.py 176
  cat data/r40s3x_screen.txt
  log "done"
} >> data/runner_r.log 2>&1
