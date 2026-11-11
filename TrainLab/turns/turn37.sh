#!/bin/sh
# D-16 chain (MTP Head, Fri 10-30 00:5x): finish part 1 (turn35, resumable), then part 2 (turn36: H arm, skipped if done;
# walks; d16_read) in the same quiet lock if the capture completed. Each part keeps its own guards.
cd $HOME/TrainLab
trap 'pkill -P $$; exit 143' TERM
turns/turn35.sh & wait $!
[ "$(ls data/d16w/*.npz 2>/dev/null | wc -l)" -ge 205 ] || exit 0
sleep 5
turns/turn36.sh & wait $!
