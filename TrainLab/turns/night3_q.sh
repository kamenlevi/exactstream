#!/bin/sh
# MTP Head night queue 3 (Sun 11-01): after night2_q (the shipped-line A-B-A redo) is done, D-20b = the switch confirmed
# on 18 new prompts at C 224 (turn58: dfl / mtp7, head b3), quiet tickets until its 36 runs are saved (up to 6 tickets;
# a grant short on memory waits 15 min). Run in the background from ~/TrainLab.
cd $HOME/TrainLab
LOG=logs/turns.log
say() { echo "$(date '+%m-%d %H:%M:%S') night3_q: $*" >> $LOG; }
while pgrep -f "turns/night2_q.sh" > /dev/null; do sleep 30; done
KEYS=$(grep -o 'KEYS="[^"]*"' turns/turn58.sh | cut -d'"' -f2)
done_n() { ~/BigMoE/.venv/bin/python -c "
import json, sys
ks = set(sys.argv[1].split(','))
R = [json.loads(l) for l in open('data/d20_runs.jsonl')]
print(len({(r['key'], r['arm']) for r in R if r['head'] == 'b3' and r['key'] in ks and r['arm'] in ('dfl', 'mtp7')}))" "$KEYS" 2>/dev/null || echo 0; }
for t in 1 2 3 4 5 6; do
  [ "$(done_n)" -ge 36 ] && break
  say "D-20b: $(done_n) / 36 runs; ticket $t"
  PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 15m "MTP Head D-20b: the drafter switch on 18 new prompts at C 224 (turn58)" >> $LOG 2>&1 || exit 1
  N=$(wc -l < logs/turn_job.out)
  ./run_locked.sh 15 turns/turn58.sh
  tail -n +$((N + 1)) logs/turn_job.out | grep -q "not starting" && { say "memory short; next ticket in 15 min"; sleep 900; }
done
say "done ($(done_n) / 36)"
