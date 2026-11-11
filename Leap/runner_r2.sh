#!/bin/sh
# Leap's second detached screen runner (Tue 2026-10-27; RULES.md 'Rapid screening'): waits for runner_r.sh (pid given
# as $1) to finish, so the two never hold or queue a lock at once, then runs its screens exactly as runner_r.sh does.
# Pass / kill lines are pre-registered in TABLE_350.md. Log: data/runner_r.log (shared with runner_r.sh)
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
Q=$HOME/.local/share/quiet
export PEER_QUEUE_TIMEOUT=21600
log() { echo "[runner2] $* $(date '+%H:%M:%S'); swap $(sysctl -n vm.swapusage | cut -d'(' -f1 | sed 's/ *$//'), disk $(df -g / | tail -1 | awk '{print $4}') GB free"; }
lid_closed() { ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes; }
ok_now() {   # also waits while soft_end.sh ramps held jobs back (it SIGSTOPs their new children too: Fit12, 10-27)
  h=$(date +%H%M)
  [ ! -e $Q/PAUSE ] && [ "$h" -ge 0802 ] && [ "$h" -lt 2348 ] && ! lid_closed && ! grep -q "Verdict: ACT" $Q/HEALTH.md 2>/dev/null \
    && ! pgrep -f '^/bin/zsh [^ ]*soft_end\.sh' >/dev/null
}
screen() {   # screen <id> <why> <command...>
  id=$1; why=$2; shift 2; tries=0
  while :; do
    w=0; while ! ok_now; do [ $w -eq 0 ] && log "$id: waiting (PAUSE / ACT / lid / night)"; w=1; sleep 120; done
    log "$id: queued"
    if peer lock quiet --queue 10m "Leap $id: $why"; then
      if ! ok_now; then peer unlock; log "$id: lock came under PAUSE / ACT / closed lid / night: released"; continue; fi
      log "$id: lock"
      ./run_locked.sh 10 "$@"; rc=$?
      log "$id: rc $rc"; tries=$((tries + 1))
      [ $rc -eq 143 ] && [ $tries -lt 2 ] && continue
      return $rc
    fi
    log "$id: no lock, queueing again"; sleep 60
  done
}
{
  log "waiting for runner_r.sh (pid $1) to finish; then: R-11, R-09b build, R-15 step 1"
  while kill -0 "$1" 2>/dev/null; do sleep 30; done
  screen R-11 "GPU idle per draft / check / accept / commit cycle: 4 arms on CA's 12-layer slice + drafter (~2 min)" \
    $PY scripts/r11_screen.py
  grep -E "R-11|per-cycle|paired|components|PLE staging|idle|48 layers" data/r11_screen.txt
  screen R-09b "MLX's NAX split-K order with loads prefetched: bitwise + 96-read chain timing (~1 min)" \
    $PY scripts/r09c_screen.py
  grep -E "verdict|bitwise|per 96|saved" data/r09c_screen.txt
  screen R-15 "drafter kernel-time breakdown by class (knockouts), b16, bf16 + 4-bit (~2 min)" \
    $PY scripts/r15_screen.py
  grep -E "drafter|class time|top two|R-15" data/r15_screen.txt
  log "done"
} >> data/runner_r.log 2>&1
