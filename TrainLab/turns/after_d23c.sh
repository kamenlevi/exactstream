#!/bin/sh
# HPD-1's slot (Mon 11-02; REPORT "HPD-1: pre-registration"): waits for the D-23c' ticket (after69, pid $1) to end, then for
# Leap's fixed-constant capture (NewIdeas' night order: ... -> per-row split -> Leap's capture -> my slot, where the DLOG
# top-ups were) to be seen and gone; then ONE quiet FIFO ticket of 90 min for turns/hpd1.sh. If 12:00 comes first (day
# rules: small tests under shared locks), it stops and the session runs HPD-1 under a shared lock by hand.
# Queued only after after69's release: a queue entry made while run_locked holds or awaits a lock is cancelled by its unlock.
cd $HOME/TrainLab
LOG=logs/turns.log
say() { echo "$(date '+%m-%d %H:%M:%S') after_d23c: $*" >> $LOG; }
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
STOPAT=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-02 12:00" +%s)
while kill -0 $1 2>/dev/null; do sleep 20; done
say "D-23c' ticket ended; waiting for Leap's capture to be seen and gone (HPD-1 goes last in the night queue)"
SEEN=0
while :; do
  if [ "$(leapcap)" -gt 0 ]; then SEEN=1; elif [ $SEEN = 1 ]; then break; fi
  [ $(date +%s) -ge $STOPAT ] && { say "12:00 before Leap's capture ended (seen $SEEN): HPD-1 left for a shared lock by day"; exit 0; }
  sleep 30
done
LEFT=$(( STOPAT - $(date +%s) ))
[ $LEFT -lt 600 ] && { say "Leap's capture done but < 10 min to 12:00: HPD-1 left for a shared lock by day"; exit 0; }
say "Leap's capture done: queueing HPD-1 (90 min; queue timeout ${LEFT} s = 12:00)"
PEER_QUEUE_TIMEOUT=$LEFT peer lock quiet --queue 90m "MTP Head HPD-1: drafter training screen (small model, ~15 GB, no timing; last in NewIdeas' night order)" >> $LOG 2>&1 || { say "not granted by 12:00: HPD-1 left for a shared lock by day"; exit 0; }
./run_locked.sh 90 turns/hpd1.sh
say "HPD-1 ticket done: $(grep 'hpd1:' $LOG | tail -1 | cut -c1-200)"
