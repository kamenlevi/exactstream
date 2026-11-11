#!/bin/sh
# D-23d inside a 40-min ticket: turn72_d23d.sh with its gate retried up to 6 times (a slow release of the holder).
cd $HOME/TrainLab
trap 'pkill -f d23c_mtptrunc.py; pkill -P $$; exit 143' TERM
for i in 1 2 3 4 5 6; do
  sh turns/turn72_d23d.sh > logs/turn72.tmp 2>&1; cat logs/turn72.tmp
  grep -q "not starting" logs/turn72.tmp || exit 0
  echo "== D-23d gate short ($i): retry in 30 s"; sleep 30
done
exit 1
