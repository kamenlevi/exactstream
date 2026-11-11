#!/bin/sh
# D-17 chain (MTP Head, Fri 10-30 02:0x): finish part 1 (turn33, resumable: multi), then part 2 (turn34: capture,
# resumable; every-position walks one set at a time, resumable; router read-out) in the same quiet lock.
cd $HOME/TrainLab
trap 'pkill -P $$; exit 143' TERM
[ "$(ls data/d17/*.npz 2>/dev/null | wc -l)" -ge 32 ] || { turns/turn33.sh & wait $!; }
[ "$(ls data/d17/*.npz 2>/dev/null | wc -l)" -ge 32 ] || exit 0
sleep 5
turns/turn34.sh & wait $!
