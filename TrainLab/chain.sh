#!/bin/sh
# Keep TrainLab in the GPU line: after the running run_turn.sh (if any) exits, queue turns one after another until
# turns/stop exists or MAX turns ran. Each turn runs turn_job.sh at grant time (turns/turnN.sh or turns/default).
T=$HOME/TrainLab; MAX=${1:-12}; n=0
while pgrep -f "run_turn.sh" > /dev/null; do sleep 20; done
while [ ! -f $T/turns/stop ] && [ $n -lt $MAX ]; do
  $T/run_turn.sh 45 "TrainLab: 35B LoRA healing of expert dataflows (~45 min, ~30 GB)"
  n=$((n + 1)); sleep 30
done
echo "$(date '+%m-%d %H:%M:%S') chain ended after $n turns" >> $T/logs/turns.log
