#!/bin/sh
# Breakthrough's detached runner (Mon 2026-10-26; QUEUE 'RAPID SCREENS': open work goes into ONE detached runner before
# a pacing sleep; Leap's runner_r.sh pattern) for QUEUE L2-06t-dx run 3 (scripts/day_kvdx.sh: pools emptied before every
# item). Starts only in 08:01-22:45 (the job takes ~45 min and must end before the 00:01 silent night), without PAUSE and
# without a health ACT (20:45: disk 97 GB < the 100 GB floor stopped the first try in its build step); waits through
# all three, up to 26 h. Once clear it waits 5 more min, so 10-min screens queued at the same moment go first (dx is
# lower priority), then queues a 60m quiet lock, runs day_kvdx.sh once, unlocks. Log: data/kv8/runner.log
cd "$HOME/Breakthrough" || exit 1
Q=$HOME/.local/share/quiet
LOG=$HOME/Breakthrough/data/kv8/runner.log
log() { echo "[runner] $* $(date '+%a %H:%M:%S'); swap $(sysctl -n vm.swapusage | cut -d'(' -f1 | sed 's/ *$//'), disk $(df -g / | tail -1 | awk '{print $4}') GB free" >> "$LOG"; }
in_hours() { h=$(date +%H%M); [ "$h" -ge 0801 ] && [ "$h" -le 2245 ]; }
act() { grep -q '^\*\*Verdict: ACT' $Q/HEALTH.md 2>/dev/null; }
ok() { in_hours && [ ! -e $Q/PAUSE ] && ! act; }
END=$(( $(date +%s) + 93600 ))
log "dx: waiting for 08:01-22:45 without PAUSE or ACT (pid $$; health now: $(grep -m1 '^\*\*Verdict' $Q/HEALTH.md | cut -c1-50))"
while [ "$(date +%s)" -lt $END ]; do
  until ok || [ "$(date +%s)" -ge $END ]; do sleep 60; done
  ok || break
  sleep 300
  ok || continue
  log "dx: queued"
  if PEER_QUEUE_TIMEOUT=7200 peer lock quiet --queue 60m "Breakthrough L2-06t-dx run 3: 2K self-test, then 32K + 60K bf16 vs kv:64 (~45 min)"; then
    if ! ok; then log "dx: lock came outside the hours or with PAUSE / ACT, released"; peer unlock; continue; fi
    log "dx: lock"
    caffeinate -i -w $$ &
    sh scripts/day_kvdx.sh; rc=$?
    log "dx: rc $rc"
    peer unlock
    log "dx: unlocked"
    exit $rc
  fi
  log "dx: no lock (queue timeout)"
done
log "dx: gave up after 26 h of waiting"
