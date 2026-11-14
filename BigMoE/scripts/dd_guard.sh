#!/bin/bash
# Sourced by BigMoE runners (Sat 14 Nov 2026, after a yielded share lock held Leap's quiet ticket for 17 min).
#   guarded LABEL MEM_GB CMD...   runs CMD under the session's share lock and yields to other sessions' quiet locks:
#     - waits while another session's quiet lock is held, lapsed or queued;
#     - takes `peer lock share` (MEM_GB in the reason) right before each start;
#     - polls every 10 s; on another session's quiet lock: SIGTERM to CMD's whole process group, `peer unlock`, wait,
#       rerun from the start;
#     - `peer unlock` when CMD ends; returns CMD's exit code.
# Needs `say` defined by the caller. CMD runs in its own process group (set -m), so a stop reaches its children.
set -m
_oq() { peer ls 2>/dev/null | grep "^- " | grep -oE "· (benchmarking|quiet lock lapsed|waiting for a quiet window)" | grep -c .; }
guarded() {
  local lab=$1 mem=$2 p rc; shift 2
  while :; do
    while [ "$(_oq)" -gt 0 ]; do sleep 30; done
    peer lock share 120m --cpu 300 --gpu 80 "BigMoE: $lab (MEM ~$mem GB; yields to quiet locks)" >/dev/null 2>&1
    "$@" & p=$!
    while kill -0 $p 2>/dev/null; do
      sleep 10
      if [ "$(_oq)" -gt 0 ]; then
        say "$lab: a quiet lock of another session -> SIGTERM to group $p, unlock, rerun after"
        kill -TERM -- -$p 2>/dev/null; wait $p; p=""
        peer unlock >/dev/null 2>&1
        break
      fi
    done
    if [ -n "$p" ]; then wait $p; rc=$?; peer unlock >/dev/null 2>&1; return $rc; fi
  done
}
