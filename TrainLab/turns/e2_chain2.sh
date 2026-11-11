#!/bin/sh
# E2 / D2 chain v2 (MTP Head, Mon 11-02 10:5x; v1 gave the capture only 4 tries 10 min apart while the day's other tests held the memory; ROUTE_350 condition 8, Kamen's day order via the orchestrator). One step at a
# time, each under its own lock taken only after the previous release (a session's lock requests interfere):
# 1. MBPP capture under a SHARE lock (turns/e2_cap.sh, 110-min slots) until x2's data exists (>= 16,171 code tokens);
# 2. x2: train + walk + read under a SHARE lock (turns/e2_train_x2.sh, 120 min; resumed in a second slot if needed);
# 3. capture on to x4's data (>= 48,513), then x4 (turns/e2_train_x4.sh, 180 min; resumed if needed);
# 4. D-23e (quiet 40 min at NO-SWAP >= 47) and D-23x (quiet 25 min at >= 44) when the memory allows, until 23:30.
# Result lines go to NewIdeas by peer say; the steps to logs/turns.log. Replaces after_hpd2.sh / fwd_results.sh.
cd $HOME/TrainLab
LOG=logs/turns.log
PY=$HOME/BigMoE/.venv/bin/python
say() { echo "$(date '+%m-%d %H:%M:%S') e2_chain: $*" >> $LOG; }
fwd() { peer say --to ojox-bd "$1" > /dev/null 2>&1; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
ntok() { [ -f data/dlog/d1102mbpp/index.jsonl ] && grep -o '"n": [0-9]*' data/dlog/d1102mbpp/index.jsonl | awk '{s += $2} END {print s + 0}' || echo 0; }
STOPAT=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-02 23:30" +%s)
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
if capture_to 16171 && $PY sticky/e2_sets.py >> $LOG 2>&1; [ -d data/e2_x2 ] && train x2 120; then
  L=$(grep 'LINE' data/e2_read.txt | tail -1 | sed 's/^ *//')
  say "x2 read: $L"; fwd "MTP Head (auto) E2/D2 condition 8, x2 point: $L (data/e2_read.txt)"
else
  say "x2 not finished: $(ntok) code tokens captured"; fwd "MTP Head (auto) E2/D2: x2 not finished today ($(ntok) MBPP code tokens captured)"
fi
if [ -f data/d02b_walk_e2x2.txt ] && capture_to 48513 && $PY sticky/e2_sets.py >> $LOG 2>&1 && train x4 180; then
  L=$(grep 'LINE' data/e2_read.txt | tail -1 | sed 's/^ *//')
  say "x4 read: $L"; fwd "MTP Head (auto) E2/D2 condition 8, x2 + x4: $L (data/e2_read.txt)"
fi
# D-23e / D-23x (queued since this morning; they need the memory Kamen's apps hold by day)
qrun() {   # $1 = GB, $2 = minutes, $3 = turn, $4 = why
  while [ "$(noswap)" -lt $1 ]; do late && { say "23:30 without $1 GB NO-SWAP: $3 not run"; return 1; }; sleep 120; done
  LEFT=$(( STOPAT - $(date +%s) ))
  say "queueing $3 ($(noswap) GB NO-SWAP; $2 min; queue timeout ${LEFT} s)"
  PEER_QUEUE_TIMEOUT=$LEFT peer lock quiet --queue ${2}m "$4" >> $LOG 2>&1 || { say "$3 not granted"; return 1; }
  ./run_locked.sh $2 $3
}
if qrun 47 40 turns/turn74.sh "MTP Head D-23e: the chat switch at C 224 (timing, ~43 GB), ~25 min"; then
  L=$(grep 'PRIMARY chat' data/d23d_read.txt | tail -1 | sed -E 's/.*time-weighted G ([+-][0-9.]+) %, geo-mean ratio ([0-9.]+), mtp15e faster on ([0-9]+).*\): ([A-Z]+)(.*)$/chat G \1 %, geo \2, faster on \3 -> \4\5/')
  A=$(grep 'SECONDARY agent' data/d23d_read.txt | tail -1 | sed -E 's/.*G ([+-][0-9.]+) %, geo-mean ([0-9.]+) -> agent (joins|stays).*/agent G \1 %, geo \2, \3/')
  say "D-23e: $L; $A"; fwd "MTP Head (auto) D-23e, chat switch at C 224: $L; $A (data/d23d_read.txt)"
fi
qrun 44 25 turns/turn75.sh "MTP Head D-23x: first-check read trace (C 192, ~38 GB), ~15 min" && say "D-23x done (data/d23x_read.txt)" \
  && fwd "MTP Head (auto) D-23x first-check read trace done: data/d23x_read.txt"
say "chain end"
