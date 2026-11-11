#!/bin/sh
# Fit12 (Tue 27 Oct 2026): keep the Mac from idle sleep only while runner PID $1 is busy (it has a child other than sleep /
# caffeinate: a lock wait or an arm), is not held (stat T) and PAUSE is off. Holds nothing while the runner waits in its gate
# or a quiet window holds it. Replaces the runner's own `caffeinate -i -w`, which kept the Mac awake through the quiet window
# of 26-28 Oct. Overlapping 90 s caffeinates every 60 s, so there is no gap while busy. usage: nohup sh awake.sh RUNNER_PID &
R=$1; P=$HOME/.local/share/quiet/PAUSE
while kill -0 "$R" 2>/dev/null; do
  st=$(ps -o stat= -p "$R" | cut -c1)
  kids=$(pgrep -l -P "$R" | awk -v me=$$ '$1 != me && $2 != "sleep" && $2 != "caffeinate"' | wc -l)
  [ ! -e "$P" ] && [ "$st" != T ] && [ "$kids" -gt 0 ] && caffeinate -i -t 90 &
  sleep 60
done
