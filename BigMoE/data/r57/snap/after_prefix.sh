#!/bin/sh
# the wall-1 sprint turn queues only after Breakthrough's Wed 00:30 map prefix (share lock MEM 17 GB, ~25 min, hard stop
# 01:15; Breakthrough 11-03 19:18): its log's last line, written within the last 3 h, says "prefix done" or that it did not
# start ("everything stays for Wed 15:00"); or it is past 01:20 (its hard stop + 5 min) in the night
L="$HOME/Breakthrough/data/night_prefix.log"
if [ -f "$L" ] && [ $(( $(date +%s) - $(stat -f %m "$L") )) -lt 10800 ] && tail -1 "$L" | grep -qE 'prefix done|everything stays for Wed 15:00'; then exit 0; fi
t=$(date +%H%M)
[ "$t" -ge 0120 ] && [ "$t" -lt 1200 ] && exit 0
exit 1
