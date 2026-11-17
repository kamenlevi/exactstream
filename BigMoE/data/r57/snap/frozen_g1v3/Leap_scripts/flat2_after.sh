#!/bin/sh
# FLAT-2 (zero's fixed-cost measurement; Leap runs it, the orchestrator Thu 18:5x): turn45's AFTER condition. Go once
# NATIVE-FRI's phase C has ended on Fri 11-06 (~/Leap/data/natfri/natfri.log), or from Fri 06:30 if it never ran.
# Thu 23:2x: never while NATIVE-FRI's turn45 still waits, queues or runs. Both turns hold THIS session's lock, so a second
# ticket would share or drop the first one's (peer unlock drops the session's queue), and natfri_c.sh writes its end line
# a moment before its turn45 releases the lock.
pgrep -f 'turn45[.]sh [0-9]+ [0-9]+ Leap: NATIVE-FRI' >/dev/null && exit 1
grep -qE '^11-06 [0-9:]{8} NATIVE-FRI phase C end' "$HOME/Leap/data/natfri/natfri.log" 2>/dev/null && exit 0
[ "$(date +%Y%m%d%H%M)" -ge 202611060630 ] && exit 0
exit 1
