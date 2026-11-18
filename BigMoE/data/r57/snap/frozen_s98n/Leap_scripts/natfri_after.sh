#!/bin/sh
# NATIVE-FRI (Leap, Thu 2026-11-05 19:0x): turn45's AFTER condition. The quiet ticket is queued only after BigMoE's Friday
# ticket: its data/fri/fri.log has a Fri 11-06 "Fri run end" line, or (from Fri 03:40, if that ticket never ran) BigMoE
# neither holds nor waits for a quiet lock. Exit 0 = go.
F=$HOME/BigMoE/data/fri/fri.log
grep -qE '^11-06 [0-9:]{8} Fri run end' "$F" 2>/dev/null && exit 0
[ "$(date +%Y%m%d%H%M)" -ge 202611060340 ] || exit 1
out=$(peer ls 2>/dev/null)
[ -n "$out" ] || exit 1
printf '%s\n' "$out" | grep -F 'BigMoE' | grep -qE 'benchmarking: quiet lock|waiting for a quiet window' && exit 1
exit 0
