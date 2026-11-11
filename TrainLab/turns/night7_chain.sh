#!/bin/sh
> (a quote from the project owner, kept private)
# third, right after D-23c, ahead of M-16 / r36 / runner 62). The session's only lock user.
# 1. From 23:00: ONE quiet FIFO ticket of 100 min (behind BigMoE's C45): turns/turn65.sh = D-23c v2 then the DLOG capture
#    with the time left; released early when DLOG reaches 30K answer tokens per language.
# 2. Top-up, only if DLOG is short of 30K per language, FOURTH after the night queue (NewIdeas 20:4x: BigMoE C 300/320 ->
#    BigMoE per-row split -> Leap's fixed-constant capture -> DLOG): after Leap's runner 62 (pid 44084) exits, wait until a quiet
#    ticket of Leap's capture (Leap's session pid 63330, 'capture' / 'fixed-constant' in its why) has been seen and is gone;
#    none seen within 6 h -> stop, the session queues the top-up by hand. Then up to 4 quiet tickets of 30 min until the
#    target (Kamen 20:3x: runs may continue past 06:00 and through Monday).
# Safety stops (orchestrator 20:2x / 20:3x): 8 GB written, under 114 GB free disk, no timing, no bit changes.
cd $HOME/TrainLab
LOG=logs/turns.log
say() { echo "$(date '+%m-%d %H:%M:%S') night7_chain: $*" >> $LOG; }
tot() { ~/BigMoE/.venv/bin/python -c "
import json, os
d = 'data/dlog/n1101'; f = os.path.join(d, 'index.jsonl'); t = {'de': 0, 'ru': 0, 'ja': 0}
for l in (open(f) if os.path.exists(f) else []):
    r = json.loads(l)
    if os.path.exists(os.path.join(d, r['key'].replace('/', '__') + '.npz')):
        t[r['lang']] = t.get(r['lang'], 0) + r['n']
print(min(t.values()), sum(t.values()))" 2>/dev/null || echo "0 0"; }
NIGHT=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-01 23:00" +%s)
say "start; one 100-min ticket from 23:00 (D-23c then DLOG), top-up after Leap's runner 62"
other() { ~/BigMoE/.venv/bin/python -c "
import glob, json, os, time
now, me, n = time.time(), int(os.environ.get('CLAUDE_PID', '0')), 0
for p in glob.glob(os.path.expanduser('~/.claude-peers/queue/*.json')) + glob.glob(os.path.expanduser('~/.claude-peers/locks/*.json')):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    if not isinstance(d, dict) or d.get('pid', int(os.path.basename(p).split('.')[0].split('-')[-1])) == me:
        continue
    if '/queue/' in p or (d.get('mode') == 'quiet' and d.get('until', 0) > now):
        n += 1
print(n)" 2>/dev/null || echo 0; }
while [ $(date +%s) -lt $NIGHT ]; do sleep 20; done
# NewIdeas' order keeps BigMoE's C45 first: queue as soon as another session's quiet ticket is in line or held (C45 queues at
# 23:00), at the latest 23:00:45, ahead of M-16 (23:01) and r36 (23:05)
while [ $(date +%s) -lt $((NIGHT + 45)) ] && [ "$(other)" -eq 0 ]; do sleep 2; done
say "queueing at $(date +%H:%M:%S) ($(other) other quiet tickets in line or held)"
> (a quote from the project owner, kept private)
say "granted; D-23c then DLOG (turn65)"
./run_locked.sh 100 turns/turn65.sh
set -- $(tot); say "ticket 1 done: DLOG $2 answer tokens (min language $1)"
[ "$1" -ge 30000 ] && { say "DLOG target reached: done"; exit 0; }
while ps -p 44084 -o command= 2>/dev/null | grep -q runner_r62; do sleep 60; done
leapcap() { ~/BigMoE/.venv/bin/python -c "
import glob, json, os, re, time
now, n = time.time(), 0
for p in glob.glob(os.path.expanduser('~/.claude-peers/queue/*.json')) + glob.glob(os.path.expanduser('~/.claude-peers/locks/*.json')):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    pid = d.get('pid', int(os.path.basename(p).split('.')[0].split('-')[-1])) if isinstance(d, dict) else 0
    if pid == 63330 and re.search(r'captur|fixed.?const', d.get('why') or '', re.I) and ('/queue/' in p or (d.get('mode') == 'quiet' and d.get('until', 0) > now)):
        n += 1
print(n)" 2>/dev/null || echo 0; }
say "runner 62 done; waiting for Leap's fixed-constant capture to run and exit (NewIdeas 20:4x)"
SEEN=0; T1=$(date +%s)
while :; do
  if [ "$(leapcap)" -gt 0 ]; then SEEN=1; elif [ $SEEN = 1 ]; then break; fi
  [ $SEEN = 0 ] && [ $(( $(date +%s) - T1 )) -gt 21600 ] && { say "Leap's capture not seen within 6 h: top-up left to the session"; exit 0; }
  sleep 30
done
say "Leap's capture done: DLOG top-up"
for t in 2 3 4 5; do
  set -- $(tot)
  [ "$1" -ge 30000 ] && { say "DLOG target reached ($2 answer tokens): done"; break; }
  [ $(df -g $HOME | awk 'NR==2 {print $4}') -ge 114 ] || { say "free disk under 114 GB: stopping"; break; }
  say "DLOG top-up: $2 answer tokens so far (min language $1); ticket $t"
> (a quote from the project owner, kept private)
  N=$(wc -l < logs/turn_job.out)
  ./run_locked.sh 30 turns/turn64.sh
  tail -n +$((N + 1)) logs/turn_job.out | grep -q "Traceback\|DLOG-ONPATH-FAIL" && { say "DLOG failed (see logs/turn_job.out): stopping"; exit 1; }
  tail -n +$((N + 1)) logs/turn_job.out | grep -q "not starting" && { say "memory short; next ticket in 15 min"; sleep 900; }
done
say "done (DLOG: $(tot | awk '{print $2}') answer tokens)"
