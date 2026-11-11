#!/bin/sh
# D-22 chain (MTP Head, Fri 10-30 18:1x): finish training base + its read-out (turn41), then the real-runtime decode
# with the trained head (turn44, resumable). Each part keeps its own guards.
cd $HOME/TrainLab
trap 'pkill -P $$; exit 143' TERM
turns/turn41.sh & wait $!
grep -q '"status": "done"' ckpt/d22_base.json 2>/dev/null || exit 0
turns/turn44.sh & wait $!
