#!/bin/sh
# MTP Head chain (Sun 11-01), launched beside the first D-23 ticket's run_locked.sh (the peers hook only lets a session launch
# while it holds its lock): wait for that ticket, then the remaining D-23 quiet tickets (turn59) until its 36 clean runs are
# saved (up to 5 more; a short grant waits 15 min; a crash stops the chain), then b5 (turn60) under a SHARED lock, resumable,
# up to 3 shared slots of 30 min. One lock request at a time: this script is the session's only lock user while it runs.
cd $HOME/TrainLab
LOG=logs/turns.log
say() { echo "$(date '+%m-%d %H:%M:%S') after_d23: $*" >> $LOG; }
done_n() { ~/BigMoE/.venv/bin/python -c "
import json, os
f = 'data/d23_runs.jsonl'
R = [json.loads(l) for l in open(f)] if os.path.exists(f) else []
print(len({(r['key'], r['arm']) for r in R if r['head'] == 'b3' and r['C'] == 224 and not (r['cut'] or r['paused'] or r['invalid'])}))" 2>/dev/null || echo 0; }
while pgrep -f "run_locked.sh 15 turns/turn59.sh" > /dev/null; do sleep 15; done
tail -40 logs/turn_job.out | grep -q "Traceback" && { say "D-23 ticket 1 crashed (see logs/turn_job.out): stopping"; exit 1; }
for t in 2 3 4 5 6; do
  [ "$(done_n)" -ge 36 ] && break
  say "D-23: $(done_n) / 36 runs; ticket $t"
  PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 15m "MTP Head D-23: the drafter switch on the 45 GB default (RANKED #5; turn59)" >> $LOG 2>&1 || exit 1
  N=$(wc -l < logs/turn_job.out)
  ./run_locked.sh 15 turns/turn59.sh
  tail -n +$((N + 1)) logs/turn_job.out | grep -q "Traceback" && { say "D-23 crashed (see logs/turn_job.out): stopping"; exit 1; }
  tail -n +$((N + 1)) logs/turn_job.out | grep -q "not starting" && { say "memory short; next ticket in 15 min"; sleep 900; }
done
say "D-23 done ($(done_n) / 36); b5 next"
for t in 1 2 3; do
  [ -s data/d22_eval_b5.txt ] && break
  ./run_shared.sh 30 300 60 turns/turn60.sh "D-22 b5: b4's recipe + Spec-AUF loss support, training + read-out"
done
say "done (b5 read-out: $([ -s data/d22_eval_b5.txt ] && echo yes || echo no))"
