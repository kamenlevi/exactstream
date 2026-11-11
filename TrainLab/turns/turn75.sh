#!/bin/sh
# D-23x inside a 25-min ticket: turn75_d23x.sh with its gate retried up to 6 times (a slow release of the holder).
cd $HOME/TrainLab
trap 'pkill -f d23c_mtptrunc.py; pkill -P $$; exit 143' TERM
for i in 1 2 3 4 5 6; do
  sh turns/turn75_d23x.sh > logs/turn75.tmp 2>&1; cat logs/turn75.tmp
  grep -q "not starting" logs/turn75.tmp || exit 0
  echo "== D-23x gate short ($i): retry in 30 s"; sleep 30
done
exit 1
