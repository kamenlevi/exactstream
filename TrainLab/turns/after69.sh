#!/bin/sh
# Waits for the E1 / E5 ticket (run_locked pid $1, turn69) to end, then queues D-23c' at C 192 (one 45-min quiet FIFO ticket,
# behind the decision map; REPORT "D-23c' at C 192: pre-registration") and runs it. Queued only after that release: a queue
# entry made while run_locked held the lock would be cancelled by its unlock.
cd $HOME/TrainLab
LOG=logs/turns.log
say() { echo "$(date '+%m-%d %H:%M:%S') after69: $*" >> $LOG; }
while kill -0 $1 2>/dev/null; do sleep 20; done
say "E1 / E5 ticket ended; queueing D-23c' at C 192 (45 min)"
PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 45m "MTP Head D-23c' at C 192: MTP K15 + the default's confidence cut vs the 45 GB default (both arms at C 192), 18 prompts (turn70)" >> $LOG 2>&1 || { say "queueing failed"; exit 1; }
N=$(wc -l < logs/turn_job.out)
./run_locked.sh 45 turns/turn70.sh
say "D-23c' ticket done; read-out in data/d23c_mtptrunc.txt (C 192 block)"
