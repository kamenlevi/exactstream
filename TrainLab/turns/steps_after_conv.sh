#!/bin/bash
# MTP Head, Thu 11-05 09:2x: the PORT steps check's fallback slot. The orchestrator 21:5x: beside runner 42 only if
# NO-SWAP stays >= 20 GB; at 09:13 runner 42 ran as a shared run (MEM 36 GB) with only 15 GB available beside it
# (swap 10.7 GB), so not beside it. Runner 42 now ends ~10:15, and BigMoE converts records3 right after it (quiet
# lock, ~5 min; the orchestrator's heads-up 08:54). This wrapper keeps the steps check from starting in the gap between
# 42's end and the conversion's ticket: it waits for a BigMoE quiet lock or ticket seen after 09:45 to appear and then
# end (or until $NOCONV, default 10:50, if none appears), then runs turns/port_steps_thu.sh with its rules unchanged
# (no quiet lock / ticket and >= 32 GB available for 3 clear minutes; GIVE_UP 11:30 here; then the walk re-measure
# under the same lock). Log: data/port/steps_after_conv.log (port_steps_thu.sh's lines follow in the same log).
cd ~/TrainLab || exit 1
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
others() { peer ls 2>/dev/null | grep "^- "; }
locks() { grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"; }
bigmoe_quiet() { others | grep "~/BigMoE" | locks | grep -cE "^· (benchmarking|waiting for a quiet window)"; }
NOCONV=${NOCONV:-1050}
seen=0
log "waiting for BigMoE's records3 conversion (a quiet lock / ticket after 09:45) to start and end; none by $NOCONV -> go on"
while :; do
  now=$(date +%H%M); n=$(bigmoe_quiet)
  if [ "$now" -ge 945 ]; then
    if [ "$n" -gt 0 ] && [ "$seen" -eq 0 ]; then seen=1; log "BigMoE quiet lock / ticket seen: $(others | grep "~/BigMoE" | locks | head -1 | cut -c1-140)"; fi
    if [ "$seen" -eq 1 ] && [ "$n" -eq 0 ]; then log "BigMoE's quiet lock ended"; break; fi
    if [ "$seen" -eq 0 ] && [ "$now" -ge "$NOCONV" ]; then log "no BigMoE quiet lock by $NOCONV: go on"; break; fi
  fi
  sleep 20
done
GIVE_UP=${GIVE_UP:-1130} ./turns/port_steps_thu.sh
