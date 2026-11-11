#!/bin/sh
# (Tue 22:2x, after the move to main: the night starts NOW) the night ticket queues after Breakthrough's map prefix: its log's
# last line, written within 2 h, says "prefix done" or "everything stays for Wed 15:00"; or, from 23:15, no prefix process
# runs and its log has not moved for 10 min (it never started, or ended without those lines); or past 01:20 in the night
L="$HOME/Breakthrough/data/night_prefix.log"
age=$(( $(date +%s) - $(stat -f %m "$L" 2>/dev/null || echo 0) ))
[ -f "$L" ] && [ $age -lt 7200 ] && tail -1 "$L" | grep -qE 'prefix done|everything stays for Wed 15:00' && exit 0
t=$(date +%H%M)
if [ "$t" -ge 2315 ] || [ "$t" -lt 1200 ]; then
  pgrep -f night_prefix > /dev/null || { [ $age -ge 600 ] && exit 0; }
fi
[ "$t" -ge 0120 ] && [ "$t" -lt 1200 ] && exit 0
exit 1
