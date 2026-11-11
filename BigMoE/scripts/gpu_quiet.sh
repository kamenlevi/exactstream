#!/bin/sh
# Wait until the GPU is idle (ioreg "Device Utilization %" < 10 for 3 consecutive seconds), max 120 s.
# Prints the utilisation seen; exit 1 if it never went quiet (the caller should skip or mark the run).
ok=0; i=0
while [ $i -lt 120 ]; do
  u=$(ioreg -r -d 1 -c IOAccelerator 2>/dev/null | grep -oE '"Device Utilization %"=[0-9]+' | head -1 | cut -d= -f2)
  if [ "${u:-100}" -lt 10 ]; then ok=$((ok+1)); else ok=0; fi
  [ $ok -ge 3 ] && { echo "gpu quiet (${u}%)"; exit 0; }
  sleep 1; i=$((i+1))
done
echo "gpu NOT quiet after 120 s (last ${u}%)"; exit 1
