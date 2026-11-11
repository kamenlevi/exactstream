#!/bin/sh
# MTP Head night chain 2 (Sun 11-01 -> Mon): the DLOG capture (turn64) AFTER night5_chain (D-23c) and Leap's runner 62 have exited
# (the session has one lock user at a time; NewIdeas' order). Quiet FIFO tickets of 30 min (behind tonight's queue: M-16, EvalSuite r36, Leap K-80); up to 4;
# stops when every language has >= 30K answer tokens, at a crash / DLOG-ONPATH-FAIL, under 114 GB free disk, or when a ticket
# would start after 06:00. A short grant (NO-SWAP < 44) waits 15 min and re-queues.
cd $HOME/TrainLab
LOG=logs/turns.log
say() { echo "$(date '+%m-%d %H:%M:%S') night6_capture: $*" >> $LOG; }
tot() { ~/BigMoE/.venv/bin/python -c "
import json, os
d = 'data/dlog/n1101'; f = os.path.join(d, 'index.jsonl'); t = {'de': 0, 'ru': 0, 'ja': 0}
for l in (open(f) if os.path.exists(f) else []):
    r = json.loads(l)
    if os.path.exists(os.path.join(d, r['key'].replace('/', '__') + '.npz')):
        t[r['lang']] = t.get(r['lang'], 0) + r['n']
print(min(t.values()), sum(t.values()))" 2>/dev/null || echo "0 0"; }
say "start; waits for night5_chain (D-23c) to exit"
while pgrep -f night5_chain.sh > /dev/null; do sleep 60; done
# NewIdeas 20:1x: first ticket only after Leap's runner 62 (pid 44084, sh runner_r62.sh: K-80 + the k83 exact-stack check) has
# exited: it waits 20 min between steps holding no ticket, so back-to-back FIFO tickets would starve an adoption verdict.
while ps -p 44084 -o command= 2>/dev/null | grep -q runner_r62; do sleep 60; done
say "night5_chain and Leap's runner 62 done; DLOG capture tickets"
END=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-02 06:00" +%s)
for t in 1 2 3 4; do
  set -- $(tot)
  [ "$1" -ge 30000 ] && { say "every language >= 30K answer tokens ($2 in all): done"; break; }
  [ $(date +%s) -ge $END ] && { say "06:00: no new ticket"; break; }
  [ $(df -g $HOME | awk 'NR==2 {print $4}') -ge 114 ] || { say "free disk under 114 GB: stopping"; break; }
  say "DLOG: $2 answer tokens so far (min language $1); ticket $t"
> (a quote from the project owner, kept private)
  [ $(date +%s) -ge $((END + 1800)) ] && { peer unlock > /dev/null 2>&1; say "granted after 06:30: released unused"; break; }
  N=$(wc -l < logs/turn_job.out)
  ./run_locked.sh 30 turns/turn64.sh
  tail -n +$((N + 1)) logs/turn_job.out | grep -q "Traceback\|DLOG-ONPATH-FAIL" && { say "DLOG capture failed (see logs/turn_job.out): stopping"; exit 1; }
  tail -n +$((N + 1)) logs/turn_job.out | grep -q "not starting" && { say "memory short; next ticket in 15 min"; sleep 900; }
done
say "done (DLOG: $(tot | awk '{print $2}') answer tokens)"
