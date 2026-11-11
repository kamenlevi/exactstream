#!/bin/sh
# D-23c' at C 192 inside a 45-min ticket: turn70_d23c192.sh with its gate retried up to 6 times (a slow release of the holder).
cd $HOME/TrainLab
trap 'pkill -f d23c_mtptrunc.py; pkill -P $$; exit 143' TERM
for i in 1 2 3 4 5 6; do
  sh turns/turn70_d23c192.sh > logs/turn70.tmp 2>&1; cat logs/turn70.tmp
  grep -q "not starting" logs/turn70.tmp || exit 0
  echo "== D-23c' gate short ($i): retry in 30 s"; sleep 30
done
exit 1
