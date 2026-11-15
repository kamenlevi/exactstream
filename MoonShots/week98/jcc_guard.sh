#!/bin/sh
# WEEK98 J(CC) guard (MoonShots, Sun 15 Nov 2026): run one command; every 5 s read its phys_footprint (footprint -p) and
# the swap in use; kill it if the footprint passes MAXGB or swap grows more than SWAPGB over the start (NO-SWAP rule).
# Prints the measured peak footprint at the end.
# usage: MAXGB=3 SWAPGB=1 LOG=<log> sh jcc_guard.sh <command ...>
MAXGB=${MAXGB:-3}; SWAPGB=${SWAPGB:-1}; LOG=${LOG:-/dev/null}
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/'; }
fp_mb() {
  footprint -p "$1" 2>/dev/null | awk '/Footprint:/ { for (i = 1; i <= NF; i++) if ($i == "Footprint:") { v = $(i+1); u = $(i+2);
    if (u == "KB") v /= 1024; else if (u == "GB") v *= 1024; else if (u == "B") v /= 1048576; printf "%d\n", v; exit } }'
}
S0=$(swap_mb)
"$@" >> "$LOG" 2>&1 &
PID=$!
PEAK=0
echo "[guard] pid $PID, MAXGB $MAXGB, swap at start ${S0} MB (kill at +${SWAPGB} GB)" >> "$LOG"
while kill -0 "$PID" 2>/dev/null; do
  FP=$(fp_mb "$PID"); FP=${FP:-0}
  [ "$FP" -gt "$PEAK" ] && PEAK=$FP
  SW=$(swap_mb)
  if [ "$FP" -gt $((MAXGB * 1024)) ]; then
    echo "[guard] KILL: footprint ${FP} MB > ${MAXGB} GB" >> "$LOG"; kill "$PID"; sleep 2; kill -9 "$PID" 2>/dev/null; break
  fi
  if awk -v a="$SW" -v b="$S0" -v g="$SWAPGB" 'BEGIN { exit !(a - b > g * 1024) }'; then
    echo "[guard] KILL: swap ${SW} MB, +$(awk -v a="$SW" -v b="$S0" 'BEGIN { printf "%d", a - b }') MB over the start" >> "$LOG"
    kill "$PID"; sleep 2; kill -9 "$PID" 2>/dev/null; break
  fi
  sleep 5
done
wait "$PID"; RC=$?
echo "[guard] exit $RC, peak footprint ${PEAK} MB" >> "$LOG"
exit $RC
