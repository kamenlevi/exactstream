#!/bin/sh
> (a quote from the project owner, kept private)
# ahead of M-16 / r36 / runner 62): D-23c v2 (turn62, <= ~30 min, its own gate + v2b fallback), then the DLOG capture (turn64)
# with the ticket's time left (TL_LIMIT from run_locked), ending at the target (30K answer tokens per language).
# A short memory grant at the start retries D-23c's gate for up to ~10 min. A D-23c crash does not stop DLOG.
cd $HOME/TrainLab
T0=$(date +%s); LIM=${TL_LIMIT:-5910}
trap 'pkill -f d23c_mtptrunc.py; pkill -f dlog_capture.py; pkill -P $$; exit 143' TERM
for i in 1 2 3 4 5 6 7 8 9 10; do
  sh turns/turn62.sh > logs/turn65_d23c.tmp 2>&1; cat logs/turn65_d23c.tmp
  grep -q "not starting" logs/turn65_d23c.tmp || break
  echo "== D-23c gate short ($i): retry in 30 s"; sleep 30
done
LEFT=$(( LIM - ($(date +%s) - T0) ))
echo "== D-23c part done $(date +%H:%M:%S); $LEFT s left in the ticket for DLOG"
[ $LEFT -ge 900 ] || { echo "== DLOG: under 15 min left: skipped"; exit 0; }
DLOG_BUDGET=$((LEFT - 300)) DLOG_HARD=$((LEFT - 200)) sh turns/turn64.sh
