#!/bin/sh
# E2 / D2 chain v10 = v9 from EARLYTAP on (the night). v9 = v8 + COPYGATE (Fable's copygate session's gated MATCH walk, ~10 min) before EARLYTAP. v8 = v7 + EARLYTAP (Fable mission 18; the orchestrator's queue) after x4, before the night items (starts
# only before 21:30 so its slot ends before BigMoE's ticket). v7 = v6 + the MATCH walk (Fable 2.1 / 2.5; the orchestrator: tonight after x4's walk, shared, ahead of the
# night items). v6 (night order, the orchestrator ~13:2x: BigMoE's C 300 / 320 ticket first once the apps close; quiet
# runs done by ~07:00; shared x4 may run to 08:30): x2 is done when this starts. x4 slots start only before 22:00 or
# after the night's quiet runs; D-23x / D-23e quiet tickets only after BigMoE's lock was seen and gone (or from 04:00),
# D-23x not after 06:30, D-23e not after 06:00 and not if data/D23E_DROP exists. v5 = v4 with the deadline Tue 08:30.
# v4 (MTP Head, Mon 11-02: v3 with the K-HEAD first, since it fits the memory E2's capture is waiting for
# (12:0x, NO-SWAP 17 GB under EvalSuite's 20 GB decode); E2 then as v3. v3 (Mon 11-02 11:24: v2 + the K-HEAD walk and heads after x2's verdict, Kamen's 'Yes, queue
# both'; it first waits for a running slot of v2 to end). v2 (10:5x): v1 gave the capture only 4 tries 10 min apart while the day's other tests held the memory; ROUTE_350 condition 8, Kamen's day order via the orchestrator). One step at a
# time, each under its own lock taken only after the previous release (a session's lock requests interfere):
# 1. MBPP capture under a SHARE lock (turns/e2_cap.sh, 110-min slots) until x2's data exists (>= 16,171 code tokens);
# 2. x2: train + walk + read under a SHARE lock (turns/e2_train_x2.sh, 120 min; resumed in a second slot if needed);
# 3. capture on to x4's data (>= 48,513), then x4 (turns/e2_train_x4.sh, 180 min; resumed if needed);
# 4. D-23e (quiet 40 min at NO-SWAP >= 47) and D-23x (quiet 25 min at >= 44) when the memory allows, until the deadline (v5: Tue 08:30).
# Result lines go to NewIdeas by peer say; the steps to logs/turns.log. Replaces after_hpd2.sh / fwd_results.sh.
cd $HOME/TrainLab
LOG=logs/turns.log
PY=$HOME/BigMoE/.venv/bin/python
say() { echo "$(date '+%m-%d %H:%M:%S') e2_chain: $*" >> $LOG; }
fwd() { peer say --to ojox-bd "$1" > /dev/null 2>&1; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
ntok() { [ -f data/dlog/d1102mbpp/index.jsonl ] && grep -o '"n": [0-9]*' data/dlog/d1102mbpp/index.jsonl | awk '{s += $2} END {print s + 0}' || echo 0; }
STOPAT=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-03 08:30" +%s)   # v5: Monday night allowed (Kamen: no day hold Monday)
late() { [ $(date +%s) -ge $STOPAT ]; }
waitmem() {     # $1 = GB: wait (no lock held) until NO-SWAP >= $1 or the deadline
  w=0
  while [ "$(noswap)" -lt $1 ]; do
    late && return 1
    [ $((w % 15)) -eq 0 ] && say "waiting for NO-SWAP >= $1 GB (now $(noswap)) before taking the lock"
    w=$((w + 1)); sleep 120
  done
}
capture_to() {   # $1 = code tokens needed; a lock only when the memory is there; 3 empty slots in a row = stop
  i=0; bad=0
  while [ "$(ntok)" -lt $1 ] && [ $bad -lt 3 ] && ! late; do
    waitmem 25 || break
    i=$((i + 1)); n0=$(ntok)
    say "capture slot $i: $n0 MBPP code tokens so far, need $1 ($(noswap) GB NO-SWAP)"
    ./run_shared.sh 110 400 70 turns/e2_cap.sh "E2 / D2: the 125B's MBPP answers (capture, C by memory, no timing)"
    if [ "$(ntok)" -le "$n0" ]; then bad=$((bad + 1)); say "capture slot $i added nothing ($bad in a row): retry in 5 min"; sleep 300; else bad=0; fi
  done
  [ "$(ntok)" -ge $1 ]
}
train() {      # $1 = x2 / x4, $2 = minutes per slot
  for j in 1 2; do
    late && return 1
    waitmem 23 || return 1
    say "train $1, slot $j ($(noswap) GB NO-SWAP)"
    ./run_shared.sh $2 300 60 turns/e2_train_$1.sh "E2 / D2: train $1 + walk (D-09's protocol, ~15 GB, no timing)"
    [ -f data/d02b_walk_e2$1.txt ] && return 0
  done
  return 1
}
hm() { date +%H%M | sed 's/^0*//;s/^$/0/'; }
day_ok() { t=$(hm); [ "$t" -ge 1300 ] && [ "$t" -lt 2200 ]; }      # x4 slots by day: start before 22:00 only
x4_day() {     # capture to x4's data and train it while the day window lasts
  while day_ok && [ "$(ntok)" -lt 48513 ]; do
    [ "$(noswap)" -ge 25 ] || { sleep 120; continue; }
    n0=$(ntok); say "x4 capture slot (day): $n0 MBPP code tokens, need 48513 ($(noswap) GB NO-SWAP)"
    ./run_shared.sh 110 400 70 turns/e2_cap.sh "E2 / D2: the 125B's MBPP answers to x4's data (capture, no timing)"
    [ "$(ntok)" -le "$n0" ] && sleep 300
  done
  [ "$(ntok)" -ge 48513 ] && $PY sticky/e2_sets.py >> $LOG 2>&1 || return 1
  while day_ok && [ ! -f data/d02b_walk_e2x4.txt ]; do
    [ "$(noswap)" -ge 23 ] || { sleep 120; continue; }
    say "train x4 (day slot, $(noswap) GB NO-SWAP)"
    ./run_shared.sh 180 300 60 turns/e2_train_x4.sh "E2 / D2: train x4 + walk (D-09's protocol, ~15 GB, no timing)"
  done
  [ -f data/d02b_walk_e2x4.txt ]
}
report_x4() {
  L=$(grep 'LINE' data/e2_read.txt | tail -1 | sed 's/^ *//')
  say "x4 read: $L"; fwd "MTP Head (auto) E2/D2 condition 8, x2 + x4: $L (data/e2_read.txt)"
}
while pgrep -f "turns/(e2_cap|e2_train_x|kh_walk|match_walk|copygate_job|et_job|turn74|turn75)" > /dev/null; do sleep 20; done   # a running slot of mine ends first
say "v10 start: x2 walked: $([ -f data/d02b_walk_e2x2.txt ] && echo yes || echo NO); $(ntok) MBPP code tokens"
# BigMoE's A6 ran 15:18-15:37 (v6 waited for it); x4 goes on
# v10: x4, MATCH and COPYGATE are done (19:21); EARLYTAP (its first try failed on an import) and the night follow
# EARLYTAP (REPORT 'EARLYTAP'): the evening, shared, before the night items; each step resumable
if [ ! -f data/et_read.txt ] && ! late && [ "$(hm)" -ge 1500 ] && [ "$(hm)" -lt 2130 ] && waitmem 27; then
  say "EARLYTAP ($(noswap) GB NO-SWAP)"
  ./run_shared.sh 200 300 60 turns/et_job.sh "EARLYTAP: tap-subset DFlash2 trainings + walks (Fable mission 18; ~19-25 GB, no timing)"
  L=$(grep 'LINE' data/et_read.txt 2>/dev/null | tail -1 | sed 's/^ *//')
  say "EARLYTAP: ${L:-not finished (resumable)}"; fwd "MTP Head (auto) EARLYTAP (Fable mission 18): ${L:-not finished tonight, resumes Tuesday}"
fi
# the night: BigMoE's ticket first (its lock seen and gone, or 04:00), then D-23x / D-23e by 07:00, then x4 shared to 08:30
say "night: waiting for BigMoE's C 300 / 320 ticket to run first (or 04:00)"
seen=0
while ! late; do
  t=$(hm)
  if [ -f $HOME/.claude-peers/locks/65439.json ] && grep -q -E '"mode": "quiet"' $HOME/.claude-peers/locks/65439.json \
     && grep -q -E 'C 3[02]0|300 */ *320|C3[02]0' $HOME/.claude-peers/locks/65439.json; then
    [ $seen = 0 ] && say "BigMoE's ticket seen"; seen=1
  elif [ $seen = 1 ]; then
    say "BigMoE's ticket gone"; break
  fi
  [ "$t" -ge 400 ] && [ "$t" -lt 1200 ] && { say "04:00 without BigMoE's ticket done (seen $seen): going on"; break; }
  sleep 60
done
qrun2() {   # $1 = GB, $2 = minutes, $3 = turn, $4 = why, $5 = latest start (HHMM)
  while [ "$(noswap)" -lt $1 ]; do
    [ "$(hm)" -ge $5 ] && [ "$(hm)" -lt 1200 ] && { say "$3 not started by $5 (no $1 GB NO-SWAP)"; return 1; }; sleep 60
  done
  [ "$(hm)" -ge $5 ] && [ "$(hm)" -lt 1200 ] && { say "$3 not started: past $5"; return 1; }
  LEFT=$(( $(date -j -f "%Y-%m-%d %H%M" "2026-11-03 $(printf %04d $5)" +%s) - $(date +%s) ))
  [ $LEFT -gt 60 ] || { say "$3 not started: past $5"; return 1; }
  say "queueing $3 ($(noswap) GB NO-SWAP; $2 min; queue timeout ${LEFT} s, latest start $5)"
  PEER_QUEUE_TIMEOUT=$LEFT peer lock quiet --queue ${2}m "$4" >> $LOG 2>&1 || { say "$3 not granted by $5"; return 1; }
  ./run_locked.sh $2 $3
}
if qrun2 44 25 turns/turn75.sh "MTP Head D-23x: first-check read trace (C 192, ~38 GB), ~15 min" 630; then
  say "D-23x done (data/d23x_read.txt)"; fwd "MTP Head (auto) D-23x first-check read trace done: data/d23x_read.txt"
fi
if [ -f data/D23E_DROP ]; then
  say "D-23e dropped (data/D23E_DROP: BigMoE measured the wired switch at C 256)"
elif qrun2 47 40 turns/turn74.sh "MTP Head D-23e: the chat switch at C 224 (timing, ~43 GB), ~25 min" 600; then
  L=$(grep 'PRIMARY chat' data/d23d_read.txt | tail -1 | sed -E 's/.*time-weighted G ([+-][0-9.]+) %, geo-mean ratio ([0-9.]+), mtp15e faster on ([0-9]+).*\): ([A-Z]+)(.*)$/chat G \1 %, geo \2, faster on \3 -> \4\5/')
  A=$(grep 'SECONDARY agent' data/d23d_read.txt | tail -1 | sed -E 's/.*G ([+-][0-9.]+) %, geo-mean ([0-9.]+) -> agent (joins|stays).*/agent G \1 %, geo \2, \3/')
  say "D-23e: $L; $A"; fwd "MTP Head (auto) D-23e, chat switch at C 224: $L; $A (data/d23d_read.txt)"
fi
# x4 (shared) to the deadline if it is not done
if [ ! -f data/d02b_walk_e2x4.txt ]; then
  capture_to 48513 && $PY sticky/e2_sets.py >> $LOG 2>&1 && train x4 180 && report_x4
fi
say "chain end"
