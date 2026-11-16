#!/bin/sh
# Fit12: run res1c_check.py (GPU bit tests, MEM ~4 GB) under a share lock once memory allows. Gate: NO-SWAP (free +
# purgeable + file-backed) >= 12 GB on two reads 30 s apart, no quiet (benchmarking) lock held by others, no PAUSE / HOLD.
# Gives up at 17:40. Stops: swap +1 GB, 10 min.
F=/Users/kamen/Fit12; PY=/Users/kamen/BigMoE/.venv/bin/python; O=$F/data/res1c; Q=/Users/kamen/.local/share/quiet
mkdir -p $O; rm -f $O/CHECK_DONE
nosw() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swapmb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
TE=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-16 17:40" +%s)
while :; do
  [ "$(date +%s)" -ge "$TE" ] && { echo "$(date +%H:%M:%S) no slot by 17:40: NOT RUN"; echo "rc 5 not run" > $O/CHECK_DONE; exit 5; }
  if [ "$(nosw)" -ge 12 ] && ! peer ls 2>/dev/null | grep -q "· benchmarking" && [ ! -e $Q/PAUSE ] && [ ! -e $Q/HOLD ]; then
    sleep 30
    [ "$(nosw)" -ge 12 ] && ! peer ls 2>/dev/null | grep -q "· benchmarking" && break
  fi
  sleep 30
done
peer lock share 15m --cpu 200 --gpu 50 "MEM 4 GB Fit12 res1c bit tests (installer GPU check, ~5 min)" >/dev/null 2>&1
SW0=$(swapmb); T0=$(date +%s)
echo "$(date +%H:%M:%S) start (NO-SWAP $(nosw) GB, swap $SW0 MB)"
cd $F/res1c && OMP_NUM_THREADS=2 $PY -I res1c_check.py > $O/res1c_check_run.txt 2>&1 &
P=$!; rc=0
while kill -0 $P 2>/dev/null; do
  sleep 5
  if [ $(( $(swapmb) - SW0 )) -gt 1024 ]; then rc=127; echo "swap +1 GB: stopping"; kill -TERM $P; break; fi
  if [ $(( $(date +%s) - T0 )) -ge 600 ]; then rc=124; echo "10 min: stopping"; kill -TERM $P; break; fi
done
wait $P; w=$?; [ $rc -eq 0 ] && rc=$w
peer unlock >/dev/null 2>&1
echo "$(date +%H:%M:%S) end rc $rc"; echo "rc $rc $(date +%H:%M:%S)" > $O/CHECK_DONE
tail -12 $O/res1c_check_run.txt
exit $rc
