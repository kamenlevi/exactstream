#!/bin/sh
# D-23 queue (MTP Head, Sun 11-01): quiet tickets for turns/turn59.sh until its 36 runs (18 prompts x dfl / mtp7, head b3,
# C 224) are saved, up to 6 tickets; a grant short on memory waits 15 min. The counter uses the harness's own resume key
# (key, arm, head, C): night3_q's counter ignored C and re-ran a whole set. Run in the background from ~/TrainLab.
cd $HOME/TrainLab
LOG=logs/turns.log
say() { echo "$(date '+%m-%d %H:%M:%S') d23_q: $*" >> $LOG; }
done_n() { ~/BigMoE/.venv/bin/python -c "
import json, os
f = 'data/d23_runs.jsonl'
R = [json.loads(l) for l in open(f)] if os.path.exists(f) else []
print(len({(r['key'], r['arm']) for r in R if r['head'] == 'b3' and r['C'] == 224 and not (r['cut'] or r['paused'] or r['invalid'])}))" 2>/dev/null || echo 0; }
for t in 1 2 3 4 5 6; do
  [ "$(done_n)" -ge 36 ] && break
  say "D-23: $(done_n) / 36 runs; ticket $t"
  PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 15m "MTP Head D-23: the drafter switch on the 45 GB default (RANKED #5; turn59)" >> $LOG 2>&1 || exit 1
  N=$(wc -l < logs/turn_job.out)
  ./run_locked.sh 15 turns/turn59.sh
  tail -n +$((N + 1)) logs/turn_job.out | grep -q "Traceback" && { say "the harness crashed (see logs/turn_job.out): stopping"; exit 1; }
  tail -n +$((N + 1)) logs/turn_job.out | grep -q "not starting" && { say "memory short; next ticket in 15 min"; sleep 900; }
done
say "done ($(done_n) / 36)"
