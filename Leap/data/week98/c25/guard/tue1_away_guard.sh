#!/bin/bash
# tue1_away_guard.sh (Leap, Tue 17 Nov 2026 06:3x): Kamen's AWAY rule (11-16 21:0x: a job over ~6 GB starts only while he
# is away >= 900 s, and yields mid-run when he is back) for slot #1 tue1, whose frozen runner (pid 13160, armed Mon 14:08,
# before the rule) has no away gate. From 09:29:30 until the runner exits: every 10 s read away_secs.sh; on away < 900,
# TERM the runner once (its trap kills the loop, releases the ticket and writes C25_tue1_DONE "stopped by a signal"),
# log it and exit. Before 09:29:30 it only waits (the runner holds nothing before 09:30). Never -9.
# Usage: tue1_away_guard.sh <runner pid> ; DRY=1 AWAY_CMD='echo N' FROM_EPOCH=<s> for the head test.
RP=${1:?runner pid}; LOG=/Users/kamen/Leap/data/week98/c25/guard/tue1_away_guard.log
AWAY_CMD=${AWAY_CMD:-/Users/kamen/.local/share/quiet/away_secs.sh}
FROM=${FROM_EPOCH:-$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-17 09:29:30" +%s)}
say() { echo "$(date '+%m-%d %H:%M:%S') [tue1_guard$([ "${DRY:-0}" = 1 ] && echo _dry)] $*" >> $LOG; }
say "armed (pid $$): runner $RP, checks from $(date -r $FROM '+%m-%d %H:%M:%S'), rule away >= 900"
while kill -0 $RP 2>/dev/null && [ $(date +%s) -lt $FROM ]; do sleep 20; done
kill -0 $RP 2>/dev/null || { say "runner $RP gone before the checks: exit"; exit 0; }
say "checks on (away now $($AWAY_CMD) s)"
while kill -0 $RP 2>/dev/null; do
  a=$($AWAY_CMD 2>/dev/null); a=${a%%.*}
  if [ -n "$a" ] && [ "$a" -lt 900 ] 2>/dev/null; then
    say "Kamen at the Mac (away $a s < 900): TERM runner $RP (the away rule; loop running: $(pgrep -f 'c25_loo[p].py' | tr '\n' ' '))"
    kill -TERM $RP
    exit 0
  fi
  sleep 10
done
say "runner $RP ended; guard done (away $($AWAY_CMD) s)"
