#!/bin/sh
# MTP Head chain (Sun 11-01 day -> night). The session's only lock user while it runs (one lock request at a time).
# Day (orchestrator 11:3x: small tests, shared locks with peak + 8 GB free): the b4 / b5 read-out (turn61) under a shared lock
# whenever NO-SWAP >= 20 GB, until 22:50. Night (big tests from 23:00): D-23c (turn62) quiet tickets until its 20 clean runs
# (10 prompts x 2 arms) are saved, up to 6; a short grant waits 15 min; a crash stops the chain.
cd $HOME/TrainLab
LOG=logs/turns.log
say() { echo "$(date '+%m-%d %H:%M:%S') day4_chain: $*" >> $LOG; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
ro_done() { grep -q "^  mgsm" data/d22_eval_b5.txt 2>/dev/null && grep -q "^  mgsm" data/d22_eval_b4.txt 2>/dev/null; }
DAY_END=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-01 22:50" +%s); NIGHT=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-01 23:00" +%s)
say "start; read-out done: $(ro_done && echo yes || echo no)"
n=0
while ! ro_done && [ $(date +%s) -lt $DAY_END ] && [ $n -lt 4 ]; do
  if [ "$(noswap)" -ge 21 ]; then
    n=$((n + 1)); say "b4 / b5 read-out, shared slot $n ($(noswap) GB NO-SWAP)"
    N=$(wc -l < logs/turn_job.out)
    ./run_shared.sh 25 300 40 turns/turn61.sh "D-22 read-out b4 vs b5 (Spec-AUF), tokens per pass"
    tail -n +$((N + 1)) logs/turn_job.out | grep -q "Traceback" && { say "read-out crashed (see logs/turn_job.out)"; break; }
    tail -n +$((N + 1)) logs/turn_job.out | grep -q "not starting" && sleep 600
  else
    sleep 120
  fi
done
say "day part over (read-out done: $(ro_done && echo yes || echo no)); D-23c from 23:00"
while [ $(date +%s) -lt $NIGHT ]; do sleep 60; done
done_n() { ~/BigMoE/.venv/bin/python -c "
import json, os
f = 'data/d23c_runs.jsonl'
R = [json.loads(l) for l in open(f)] if os.path.exists(f) else []
print(len({(r['key'], r['arm']) for r in R if r['head'] == 'b3' and r['C'] == 224 and not (r['cut'] or r['paused'] or r['invalid'])}))" 2>/dev/null || echo 0; }
for t in 1 2 3 4 5 6; do
  [ "$(done_n)" -ge 20 ] && break
  say "D-23c: $(done_n) / 20 runs; ticket $t"
  PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 15m "MTP Head D-23c: MTP chain + the default's confidence cut on the 45 GB default (turn62)" >> $LOG 2>&1 || exit 1
  N=$(wc -l < logs/turn_job.out)
  ./run_locked.sh 15 turns/turn62.sh
  tail -n +$((N + 1)) logs/turn_job.out | grep -q "Traceback" && { say "D-23c crashed (see logs/turn_job.out): stopping"; exit 1; }
  tail -n +$((N + 1)) logs/turn_job.out | grep -q "not starting" && { say "memory short; next ticket in 15 min"; sleep 900; }
done
say "done (D-23c: $(done_n) / 20 runs)"
