#!/bin/sh
# MTP Head night chain (Sun 11-01 -> Mon): D-23c v2 (turn62: dfl vs mtp15e, head b5, D-23's 18 prompts, C 224) in ONE 30-min quiet
# ticket from 23:00 (NewIdeas' night order: BigMoE C45 -> MTP Head D-23c -> MoonShots M-16 -> EvalSuite r36 -> Leap K-80); a
# short grant (NO-SWAP < 44) waits 15 min and re-queues; up to 3 tickets; a crash stops the chain. The session's only lock user.
cd $HOME/TrainLab
LOG=logs/turns.log
say() { echo "$(date '+%m-%d %H:%M:%S') night5_chain: $*" >> $LOG; }
NIGHT=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-01 23:00" +%s)
done_n() { ~/BigMoE/.venv/bin/python -c "
import json, os
f = 'data/d23c_runs.jsonl'
R = [json.loads(l) for l in open(f)] if os.path.exists(f) else []
print(len({(r['key'], r['arm']) for r in R if r['head'] == 'b5' and r['C'] == 224 and not (r['cut'] or r['paused'] or r['invalid'])}))" 2>/dev/null || echo 0; }
say "start; D-23c v2 from 23:00"
while [ $(date +%s) -lt $NIGHT ]; do sleep 60; done
for t in 1 2 3; do
  [ "$(done_n)" -ge 36 ] && break
  say "D-23c v2: $(done_n) / 36 runs; ticket $t"
  PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 30m "MTP Head D-23c: MTP K15 + the default's confidence cut vs the 45 GB default, 18 prompts (turn62)" >> $LOG 2>&1 || exit 1
  N=$(wc -l < logs/turn_job.out)
  ./run_locked.sh 30 turns/turn62.sh
  tail -n +$((N + 1)) logs/turn_job.out | grep -q "Traceback" && { say "D-23c crashed (see logs/turn_job.out): stopping"; exit 1; }
  tail -n +$((N + 1)) logs/turn_job.out | grep -q "not starting" && { say "memory short; next ticket in 15 min"; sleep 900; }
done
say "done (D-23c v2: $(done_n) / 36 runs)"
