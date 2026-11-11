#!/bin/sh
# Leap's fifty-second detached screen runner (Sat 2026-10-31 20:05; FAST INTEGRATION: a proof of concept on a SHARED GPU
# slot, rough numbers): runs R-62 ATT16 (scripts/r62_att16.py: one SDPA call per QSA layer vs 12 head chunks, bitwise + ms, synthetic, < 1 GB) under
# `peer lock share`, to finish before the 20:45 end of Kamen's hour. Waits for PAUSE / a closed lid / ACT; a share lock is
# refused during quiet locks (it retries every 2 min). NO-SWAP memory >= 9 GB (the ~1 GB peak + 8). One run. Log:
# data/runner_r.log
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
Q=$HOME/.local/share/quiet
log() { echo "[runner52] $* $(date '+%H:%M:%S'); swap $(sysctl -n vm.swapusage | cut -d'(' -f1 | sed 's/ *$//')"; }
noswap_gb() { vm_stat | awk '/page size of/ {ps = $8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v = $NF; sub("[.]", "", v); s += v} END {printf "%d", s * ps / 1e9}'; }
lid_closed() { ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes; }
ok_pre() { [ ! -e $Q/PAUSE ] && ! lid_closed && ! grep -q "Verdict: ACT" $Q/HEALTH.md 2>/dev/null && [ "$(noswap_gb)" -ge 9 ]; }
{
  log "then: R-62 ATT16 (one SDPA call vs 12 head chunks, shared GPU, rough)"
  while :; do
    while ! ok_pre; do sleep 120; done
    if peer lock share 10m --cpu 100 --gpu 30 "Leap R-62 ATT16: one SDPA call vs 12 head chunks (synthetic, <1 GB, ~1 min, shared GPU rough)" >/dev/null 2>&1; then
      log "R62: share lock"
      $PY scripts/r62_att16.py >> logs/job.out 2>&1; rc=$?
      peer unlock >/dev/null 2>&1
      log "R62: rc $rc"
      cat data/r62_att16.txt
      break
    fi
    sleep 120
  done
  log "done"
} >> data/runner_r.log 2>&1
