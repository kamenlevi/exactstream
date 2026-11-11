#!/bin/sh
# Leap's fifty-sixth detached screen runner (Sun 2026-11-01 06:25; FAST INTEGRATION: a proof of concept on a SHARED GPU
# slot, rough numbers): runs K-60-T (scripts/k60t_tier_bw.py: the 3c tier kernel's GB/s vs the read ceiling, synthetic, < 1 GB) under
# `peer lock share`, night mode. Waits for PAUSE / a closed lid / ACT; a share lock is
# refused during quiet locks (it retries every 2 min). NO-SWAP memory >= 9 GB (the ~1 GB peak + 8). One run. Log:
# data/runner_r.log
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
Q=$HOME/.local/share/quiet
log() { echo "[runner56] $* $(date '+%H:%M:%S'); swap $(sysctl -n vm.swapusage | cut -d'(' -f1 | sed 's/ *$//')"; }
noswap_gb() { vm_stat | awk '/page size of/ {ps = $8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v = $NF; sub("[.]", "", v); s += v} END {printf "%d", s * ps / 1e9}'; }
lid_closed() { ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes; }
ok_pre() { [ ! -e $Q/PAUSE ] && ! lid_closed && ! grep -q "Verdict: ACT" $Q/HEALTH.md 2>/dev/null && [ "$(noswap_gb)" -ge 9 ]; }
{
  log "then: K-60-T (3c tier kernel GB/s vs the read ceiling, shared GPU, rough)"
  while :; do
    while ! ok_pre; do sleep 120; done
    if peer lock share 10m --cpu 100 --gpu 30 "Leap K-60-T: 3c tier kernel GB/s vs the read ceiling (synthetic, <1 GB, ~1 min, shared GPU rough)" >/dev/null 2>&1; then
      log "K60T: share lock"
      $PY scripts/k60t_tier_bw.py >> logs/job.out 2>&1; rc=$?
      peer unlock >/dev/null 2>&1
      log "K60T: rc $rc"
      cat data/k60t_tier_bw.txt
      break
    fi
    sleep 120
  done
  log "done"
} >> data/runner_r.log 2>&1
