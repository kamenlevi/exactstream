#!/bin/sh
# MTP Head shared chain (Fri 10-30 19:5x): the base76 real-runtime read-out (turn48), then run b2 + its read-out (turn49).
cd $HOME/TrainLab
trap 'pkill -P $$; exit 143' TERM
turns/turn48.sh & wait $!
turns/turn49.sh & wait $!
