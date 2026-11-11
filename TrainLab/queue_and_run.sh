#!/bin/sh
# TrainLab: queue a quiet lock (waits for the turn), then run one job under it via run_locked.sh (keepalive, unlock at
# the end). For pace sleeps: the GPU job runs while the session sleeps. usage: queue_and_run.sh <minutes> <job> "<why>"
PEER_QUEUE_TIMEOUT=${PEER_QUEUE_TIMEOUT:-21600} peer lock quiet --queue ${1}m "$3" > /dev/null 2>&1 || { echo "$(date '+%m-%d %H:%M:%S') queue_and_run: no lock for $2" >> $HOME/TrainLab/logs/turns.log; exit 1; }
exec $HOME/TrainLab/run_locked.sh "$1" "$2"
