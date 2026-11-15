#!/bin/sh
# turn45's AFTER for natfri2_turn.sh (Leap, Fri 2026-11-06 01:5x; the orchestrator 01:4x: MTP Head's L1-20 first).
# Queue once MTP Head (peer ojox-4a) holds or waits for a quiet lock, so this ticket lines up behind theirs; or from
# 02:40 if they never show.
peer ls 2>/dev/null | grep -E '^- ojox-4a ' | grep -qE 'benchmarking: quiet lock until|waiting for a quiet window' && exit 0
[ "$(date +%Y%m%d%H%M)" -ge 202611060240 ] && exit 0
exit 1
