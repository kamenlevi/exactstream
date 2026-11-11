#!/bin/sh
# Fit12 (Tue 27 Oct 2026): stop runner $1 by pid between R-14 and R-01, before R-01's first lock. Orchestrator 19:4x: a run
# over 10 min takes the quiet lock per item; scr_conf's 15m / 20m pair locks get split per arm, and R-01 is relaunched alone.
# Exits (waking Fit12) after that cut, at the runner's end or death, or at 23:59. usage: sh cut_before_conf.sh RUNNER_PID
R=$1; F=/Users/kamen/Fit12; L=$F/logs/summary.txt; n0=$(wc -l < $L)
log() { echo "$(date +%H:%M:%S) SCR $*" >> $L; }
while :; do
  if tail -n +$((n0 + 1)) $L | grep -q -E "hcx/h2_3 done|screen hcx stopped|conf/|R-01 CONF"; then
    kids=$(pgrep -P $R); kill -KILL $R 2>/dev/null; for c in $kids; do kill -KILL $c 2>/dev/null; done
    CLAUDE_PID=23445 peer unlock >/dev/null 2>&1
    log "runner $R stopped by Fit12 by pid after R-14 (runs over 10 min lock per item: R-01 relaunched with one lock per arm)"
    break
  fi
  tail -n +$((n0 + 1)) $L | grep -q "runner end" && break
  kill -0 $R 2>/dev/null || break
  [ "$(date +%H%M)" -ge 2359 ] && break
  sleep 2
done
date '+%a %H:%M:%S'; ps -o pid=,stat= -p $R; tail -n +$((n0 + 1)) $L | cut -c1-260 | tail -40
