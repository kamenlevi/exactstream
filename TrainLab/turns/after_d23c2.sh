#!/bin/sh
# HPD-1's slot, v2 (Mon 11-02 ~06:0x; replaces after_d23c.sh): the "Leap fixed-constant capture" in NewIdeas' night order
# has no script in ~/Leap and was never queued (Leap's session idle), so v1 would have waited for it until 12:00. v2 waits
# for the D-23c' re-run ticket (run_locked, pid $1) to end, then for the quiet line to drain: 25 minutes in a row without
# another session's quiet lock or ticket (Leap's runner 62 waits 20, so it and any other night item still go first). From
# 09:00 (day: tonight's follow-ups first) it queues at once. Then one quiet FIFO ticket of 40 min for turns/turn72.sh (D-23d,
# the timing run, first: it decides the chat switch) and, after its release, one of 90 min for turns/hpd1.sh; queue timeouts
# 18:00 (Kamen: no day hold Monday). Each queued only after the last release: two lock requests of one session interfere.
cd $HOME/TrainLab
LOG=logs/turns.log
say() { echo "$(date '+%m-%d %H:%M:%S') after_d23c2: $*" >> $LOG; }
others_quiet() { peer ls 2>/dev/null | grep -v '^You:' | grep -c -E 'waiting for a quiet window|quiet lock until'; }
hm() { date +%H%M | sed 's/^0*//'; }
STOPAT=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-02 18:00" +%s)
while kill -0 $1 2>/dev/null; do sleep 20; done
say "D-23c' re-run ticket ended; waiting for the quiet line to drain (25 min in a row; at once from 09:00)"
quiet=0
while [ $quiet -lt 25 ]; do
  t=$(hm); if [ -n "$t" ] && [ "$t" -ge 900 ]; then say "09:00 (day: tonight's follow-ups first): queueing now"; break; fi
  if [ "$(others_quiet)" -eq 0 ]; then quiet=$((quiet + 1)); else quiet=0; fi
  sleep 60
done
LEFT=$(( STOPAT - $(date +%s) ))
say "queueing D-23d (40 min; queue timeout ${LEFT} s = 18:00)"
if PEER_QUEUE_TIMEOUT=$LEFT peer lock quiet --queue 40m "MTP Head D-23d: MTP head vs DFlash2 on 20 fresh chat / agent prompts (timing, C 192, ~38 GB), ~35 min" >> $LOG 2>&1; then
  ./run_locked.sh 40 turns/turn72.sh
  say "D-23d ticket done: $(grep 'PRIMARY chat' data/d23d_read.txt 2>/dev/null | tail -1 | cut -c1-200)"
else
  say "D-23d not granted by 18:00"; exit 0
fi
LEFT=$(( STOPAT - $(date +%s) ))
say "queueing HPD-1 (90 min; queue timeout ${LEFT} s = 18:00)"
PEER_QUEUE_TIMEOUT=$LEFT peer lock quiet --queue 90m "MTP Head HPD-1: drafter training screen (small model, ~15 GB, no timing; last in NewIdeas' night order)" >> $LOG 2>&1 || { say "not granted by 18:00: HPD-1 not run"; exit 0; }
./run_locked.sh 90 turns/hpd1.sh
say "HPD-1 ticket done: $(grep 'hpd1:' $LOG | tail -1 | cut -c1-200)"
