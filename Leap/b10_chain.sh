#!/bin/sh
# B10(b) unattended (Leap, Sun 2026-10-25): wait for a quiet window, run b10_batch.sh under it, release.
# Log: data/b10_chain.log (swap + disk before and after).
cd "$HOME/Leap" || exit 1
{
  date '+[chain] queued %H:%M:%S'
  if peer lock quiet --queue 30m "Leap B10(b): trellis-decode expert GLU kernels vs 3c on one real layer (~4 GB, ~6 min: test, sweep, bench)"; then
    date '+[chain] lock %H:%M:%S'; sysctl vm.swapusage; df -h / | tail -1
    ./b10_batch.sh; rc=$?
    sysctl vm.swapusage
    peer unlock
    date "+[chain] batch rc $rc, unlocked %H:%M:%S"
  else
    date '+[chain] no lock %H:%M:%S'
  fi
} >> data/b10_chain.log 2>&1
