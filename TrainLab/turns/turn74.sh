#!/bin/sh
# D-23e inside a 40-min ticket: turn74_d23e.sh with its gate retried up to 6 times (a slow release of the holder).
cd $HOME/TrainLab
trap 'pkill -f d23c_mtptrunc.py; pkill -P $$; exit 143' TERM
for i in 1 2 3 4 5 6; do
  sh turns/turn74_d23e.sh > logs/turn74.tmp 2>&1; cat logs/turn74.tmp
  grep -q "not starting" logs/turn74.tmp || exit 0
  echo "== D-23e gate short ($i): retry in 30 s"; sleep 30
done
exit 1
