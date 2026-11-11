#!/bin/sh
# D-23c v2 alone (MTP Head, Sun 11-01 night; orchestrator 23:3x: the idle NO-SWAP measured at the DLOG grant, 45 GB, fits
# D-23c's 44 GB gate at C 224): turn62 (dfl vs mtp15e, head b5, D-23's 18 prompts, C 224, v2b preset with the NEWDEF=0
# fallback), its gate retried up to 6 times (a slow release of the previous holder), inside a 40-min quiet ticket.
cd $HOME/TrainLab
trap 'pkill -f d23c_mtptrunc.py; pkill -P $$; exit 143' TERM
for i in 1 2 3 4 5 6; do
  sh turns/turn62.sh > logs/turn68_d23c.tmp 2>&1; cat logs/turn68_d23c.tmp
  grep -q "not starting" logs/turn68_d23c.tmp || exit 0
  echo "== D-23c gate short ($i): retry in 30 s"; sleep 30
done
exit 1
