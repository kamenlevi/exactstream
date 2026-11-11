#!/bin/bash
# W24f's turn (the orchestrator 09:0x: Breakthrough's c47, then Leap's --mem 24 package check, then W24f; one > 5 GB job at a
# time; W24f ends by 14:50): exit 0 once Leap's package check has ended = a ~/Leap/data/pkg24_check.done newer than Tue 09:00
# (asked of Leap 09:0x). turn45 re-reads this file every loop, so it can be pointed at another done-file without a restart.
F=$HOME/Leap/data/pkg24_check.done
T0=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-10 09:00:00" +%s)
[ -f "$F" ] && [ "$(stat -f %m "$F")" -gt "$T0" ]
