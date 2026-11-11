#!/bin/sh
# Leap's detached screen runner (Mon 2026-10-26; RULES.md 'Rapid screening'): each screen waits until it may run (no
# PAUSE, no health ACT, lid open, 08:02-23:47 so a 10-min screen never reaches the 00:01 silent night), then takes a
# 10-min quiet lock and runs under run_locked.sh (keepalive, swap + health guards, unlock). A lock that arrives outside
# those conditions is released and the screen waits again; a screen stopped by a guard (rc 143) gets one more try.
# Pass / kill lines are pre-registered in TABLE_350.md. Log: data/runner_r.log
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
Q=$HOME/.local/share/quiet
export PEER_QUEUE_TIMEOUT=21600
log() { echo "[runner] $* $(date '+%H:%M:%S'); swap $(sysctl -n vm.swapusage | cut -d'(' -f1 | sed 's/ *$//'), disk $(df -g / | tail -1 | awk '{print $4}') GB free"; }
lid_closed() { ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes; }
ok_now() {
  h=$(date +%H%M)
  [ ! -e $Q/PAUSE ] && [ "$h" -ge 0802 ] && [ "$h" -lt 2348 ] && ! lid_closed && ! grep -q "Verdict: ACT" $Q/HEALTH.md 2>/dev/null
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
  log "queue: R-09b step 1, R-10, R-12 step 1, R-05"
  screen R-09b "where the exact 16-row HC read's time goes + which GEMM MLX runs at 16 rows (~1 min)" \
    $PY scripts/r09b_screen.py
  grep -E "bitwise equal|in-chain|R-09b step 1|E .* X5" data/r09b_screen.txt
  screen R-10 "drafter compiled (per-layer blocks + head + selector), 24 HumanEval replays (~4 min)" \
    $PY scripts/r10_screen.py
  grep -E "verdict|tokens per check|GB of its own" data/r10_screen.txt
  screen R-12 "commit anatomy: GDN replays vs PLE layer replay vs trims by keep length, 12-layer slice (~2 min)" \
    $PY scripts/ca_anat.py commit
  grep -E "R-12|keep|keeping" data/ca_commit.txt
  screen R-05 "W vs U drafter loss, 21 steps each from L1-20 + replays at block 32 / 16 (~9 min, RAM disk)" \
    ./r05_batch.sh
  cat data/r05_report.txt
  log "done"
} >> data/runner_r.log 2>&1
