#!/bin/sh
# MTP Head night chain 3 (Sun 11-01 23:2x -> Mon), replaces night7_chain, whose 23:01 ticket ran nothing (the memory gates of
# D-23c and DLOG at C 224 were short: 36 GB NO-SWAP, needs 44; orchestrator 23:1x). The session's only lock user, in order:
# 1. DLOG at C 160 / 128 (turns/turn66.sh) until every language has 20K answer tokens (E1's set): quiet FIFO tickets of 75 min,
#    up to 3; a short grant waits 10 min and re-queues. D-23c is NOT run tonight: it is a timing test pre-registered at the
#    default's C 224, which needs ~46 GB; it waits for a night when C 224 fits.
# 2. E1 + E5: turns/night8_train.sh in the foreground (beside r36 if its gate passes; else ONE 3-h quiet ticket first after
#    runner 62, NewIdeas 21:5x); its exit creates data/E15_SLOT_DONE (BigMoE's C50 queues after that).
# 3. DLOG top-ups to 30K per language after Leap's capture (NewIdeas' order: BigMoE C50 -> per-row split -> Leap's capture ->
#    DLOG): 30-min tickets of turn66 (DLOG_TARGET 30000), up to 4.
# Caps everywhere: 8 GB written tonight, stop under 114 GB free disk, no timing, no bit changes to anyone's run.
cd $HOME/TrainLab
LOG=logs/turns.log
PY=$HOME/BigMoE/.venv/bin/python
say() { echo "$(date '+%m-%d %H:%M:%S') night9_chain: $*" >> $LOG; }
tot() { $PY -c "
import json, os
d = 'data/dlog/n1101'; f = os.path.join(d, 'index.jsonl'); t = {'de': 0, 'ru': 0, 'ja': 0}
for l in (open(f) if os.path.exists(f) else []):
    r = json.loads(l)
    if os.path.exists(os.path.join(d, r['key'].replace('/', '__') + '.npz')):
        t[r['lang']] = t.get(r['lang'], 0) + r['n']
print(min(t.values()), sum(t.values()))" 2>/dev/null || echo "0 0"; }
leapcap() { $PY -c "
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
mkdir data/night9.lock 2>/dev/null || exit 0          # one instance (turn67 may launch it too)
trap 'rmdir $HOME/TrainLab/data/night9.lock 2>/dev/null' EXIT
rm -f data/E15_SLOT_DONE
if [ "$NIGHT9_AFTER" = 1 ]; then                  # started by turn67 at the end of a DLOG ticket: wait for run_locked's release
  while $PY -c "
import json, os, sys, time
try: d = json.load(open(os.path.expanduser('~/.claude-peers/locks/%s.json' % os.environ['CLAUDE_PID'])))
except Exception: sys.exit(1)
sys.exit(0 if d.get('mode') == 'quiet' and d.get('until', 0) > time.time() else 1)" 2>/dev/null; do sleep 10; done
fi
say "start: D-23c at C 224, DLOG at C 160 / 128 to 20K per language, then E1 + E5 (night8_train), then DLOG top-ups"
# 0. ONE 4-h quiet ticket (orchestrator 01:0x: E1 then E5 right after D-23c; runner 62 can go last): turn69 = D-23c v2 at C 224
#    (turn68, <= 40 min) then E1 -> E5 (night8_train in-lock mode) on the DLOG set (>= 15K answer tokens per language, ~17.4K
#    tonight). Queued only now, after the DLOG ticket's release (a queue entry made while run_locked held the DLOG lock would be
#    cancelled by its unlock).
if [ ! -e data/E15_DONE ]; then
  set -- $(tot)
  if [ "$1" -ge 15000 ]; then
    $PY sticky/e15_set.py >> logs/night8_set.out 2>&1; say "$(tail -1 logs/night8_set.out)"
    say "D-23c + E1 / E5: queueing a 4-h ticket (turn69)"
    if PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 4h "MTP Head: D-23c (MTP K15 + cut vs the 45 GB default, C 224, ~30 min) then E1 + E5 drafter training (~2.5 h, small model)" >> $LOG 2>&1; then
      N=$(wc -l < logs/turn_job.out)
      ./run_locked.sh 240 turns/turn69.sh
      tail -n +$((N + 1)) logs/turn_job.out | grep -q "Traceback" && say "a traceback in the 4-h ticket (see logs/turn_job.out)"
      say "4-h ticket done: D-23c $([ -e data/D23C_DONE ] && echo ran || echo 'not run'); E1/E5 $([ -e data/E15_DONE ] && echo done || echo 'not done')"
    else
      say "queueing the 4-h ticket failed"
    fi
  fi
fi
for t in 1 2 3; do                                # E1's set: 20K per language; >= 15K goes ahead (00:3x: an SSD stall left ~16K;
  set -- $(tot); [ "$1" -ge 15000 ] && break      # a 75-min re-ticket before runner 62 would delay the E1 / E5 slot past 09:30)
  [ $(df -g $HOME | awk 'NR==2 {print $4}') -ge 114 ] || { say "free disk under 114 GB: no DLOG"; break; }
  say "DLOG: $2 answer tokens so far (min language $1); ticket $t"
> (a quote from the project owner, kept private)
  N=$(wc -l < logs/turn_job.out)
  ./run_locked.sh 75 turns/turn66.sh
  tail -n +$((N + 1)) logs/turn_job.out | grep -q "Traceback\|DLOG-ONPATH-FAIL" && { say "DLOG failed (see logs/turn_job.out): stopping"; exit 1; }
  tail -n +$((N + 1)) logs/turn_job.out | grep -q "not starting" && { say "memory short; next ticket in 10 min"; sleep 600; }
done
set -- $(tot); say "DLOG phase 1 done: $2 answer tokens (min language $1)"
[ "$1" -gt 0 ] || { say "no DLOG data: E1 / E5 cannot run; stopping"; touch data/E15_SLOT_DONE; exit 1; }
[ -e data/E15_DONE ] || sh turns/night8_train.sh >> logs/night8.out 2>&1    # only what the 4-h ticket left (its slot after runner 62)
touch data/E15_SLOT_DONE
say "E1 / E5 phase over (E15_SLOT_DONE); top-ups wait for Leap's capture"
SEEN=0; T1=$(date +%s)
while :; do
  if [ "$(leapcap)" -gt 0 ]; then SEEN=1; elif [ $SEEN = 1 ]; then break; fi
  [ $SEEN = 0 ] && [ $(( $(date +%s) - T1 )) -gt 21600 ] && { say "Leap's capture not seen within 6 h: top-up left to the session"; exit 0; }
  sleep 30
done
for t in 1 2 3 4; do
  set -- $(tot); [ "$1" -ge 30000 ] && { say "DLOG target 30K reached ($2): done"; break; }
  [ $(df -g $HOME | awk 'NR==2 {print $4}') -ge 114 ] || { say "free disk under 114 GB: stopping"; break; }
  say "DLOG top-up: $2 answer tokens so far (min language $1); ticket $t"
  PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 30m "MTP Head DLOG top-up: drafter training data capture at C 160, ~32 GB, no timing (turn66)" >> $LOG 2>&1 || exit 1
  N=$(wc -l < logs/turn_job.out)
  DLOG_TARGET=30000 DLOG_BUDGET=1450 DLOG_HARD=1550 ./run_locked.sh 30 turns/turn66.sh
  tail -n +$((N + 1)) logs/turn_job.out | grep -q "Traceback\|DLOG-ONPATH-FAIL" && { say "DLOG failed: stopping"; exit 1; }
  tail -n +$((N + 1)) logs/turn_job.out | grep -q "not starting" && { say "memory short; next ticket in 15 min"; sleep 900; }
done
say "done (DLOG: $(tot | awk '{print $2}') answer tokens)"
