#!/bin/sh
# MTP Head night / Tuesday-morning chain v11 (Mon 11-02 evening; Kamen's laptop schedule, RULES.md top: Mon-Fri big tests
# 00:30 -> 15:00, his time until 00:00 = small tests only). Replaces e2_chain10.sh (EARLYTAP paused at 19:5x, step 6).
# 1. nothing big before 00:30; 2. BigMoE's C 300 / 320 night ticket first (armed 01:00-05:30): its lock seen and gone, or
# 06:00 if it never shows; 3. D-23x (quiet, 25 min; NewIdeas' first-check trace); 4. EARLYTAP resumed (shared; each step
# resumable); 5. DUALDRAFT (shared; REPORT 'DUALDRAFT'). Every slot ends by 15:00 (minutes capped); deadline 15:00.
# Result lines go to NewIdeas by peer say (the orchestrator gets them from me when I wake).
cd $HOME/TrainLab
LOG=logs/turns.log
say() { echo "$(date '+%m-%d %H:%M:%S') night_1102: $*" >> $LOG; }
fwd() { peer say --to ojox-bd "$1" > /dev/null 2>&1; echo "$1" | peer send 7333 - > /dev/null 2>&1; }   # NewIdeas + the orchestrator (pid 7333)
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
hm() { date +%H%M | sed 's/^0*//;s/^$/0/'; }
STOPAT=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-03 15:00" +%s)
left_min() { echo $(( (STOPAT - $(date +%s)) / 60 )); }
late() { [ $(date +%s) -ge $STOPAT ]; }
while pgrep -f "turns/(e2_cap|e2_train_x|kh_walk|match_walk|copygate_job|et_job|dd_job|turn74|turn75)" > /dev/null; do sleep 20; done
> (a quote from the project owner, kept private)
# 1. the big-test window opens at 00:30 (Tuesday)
while :; do t=$(hm); [ "$t" -ge 30 ] && [ "$t" -lt 1500 ] && [ "$(date +%d)" = "03" ] && break; sleep 60; done
# 2. BigMoE first
seen=0
while ! late; do
  f=$HOME/.claude-peers/locks/65439.json
  if [ -f $f ] && python3 -I -c "import json,sys,time; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('mode') == 'quiet' and d.get('until', 0) > time.time() else 1)" $f 2>/dev/null; then
    [ $seen = 0 ] && say "BigMoE's quiet ticket seen"; seen=1
  elif [ $seen = 1 ]; then
    say "BigMoE's ticket gone"; break
  fi
  [ "$(hm)" -ge 600 ] && { say "06:00 without BigMoE's ticket done (seen $seen): going on"; break; }
  sleep 60
done
# 3. D-23x (quiet)
qrun() {   # $1 = GB, $2 = minutes, $3 = turn, $4 = why, $5 = latest start HHMM
  while [ "$(noswap)" -lt $1 ]; do [ "$(hm)" -ge $5 ] && { say "$3 not started by $5 (no $1 GB NO-SWAP)"; return 1; }; sleep 60; done
  [ "$(hm)" -ge $5 ] && { say "$3 not started: past $5"; return 1; }
  LEFT=$(( $(date -j -f "%Y-%m-%d %H%M" "2026-11-03 $(printf %04d $5)" +%s) - $(date +%s) ))
  say "queueing $3 ($(noswap) GB NO-SWAP; $2 min; queue timeout ${LEFT} s)"
  PEER_QUEUE_TIMEOUT=$LEFT peer lock quiet --queue ${2}m "$4" >> $LOG 2>&1 || { say "$3 not granted by $5"; return 1; }
  ./run_locked.sh $2 $3
}
if [ ! -f data/d23x_read.txt ] && qrun 44 25 turns/turn75.sh "MTP Head D-23x: first-check read trace (C 192, ~38 GB), ~15 min" 1330; then
  say "D-23x done (data/d23x_read.txt)"; fwd "MTP Head (auto) D-23x first-check read trace done: data/d23x_read.txt"
fi
# 4. EARLYTAP resumed (shared)
srun() {   # $1 = GB, $2 = max minutes, $3 = job, $4 = why, $5 = latest start HHMM
  while [ "$(noswap)" -lt $1 ]; do [ "$(hm)" -ge $5 ] && { say "$3 not started by $5 (no $1 GB NO-SWAP)"; return 1; }; sleep 120; done
  [ "$(hm)" -ge $5 ] && { say "$3 not started: past $5"; return 1; }
  m=$2; l=$(left_min); [ $l -lt $m ] && m=$l
  say "$3 ($(noswap) GB NO-SWAP; slot $m min)"
  ./run_shared.sh $m 300 60 $3 "$4"
}
if [ ! -f data/et_read.txt ]; then
  srun 27 200 turns/et_job.sh "EARLYTAP: tap-subset DFlash2 trainings + walks (Fable mission 18; ~19-30 GB, no timing)" 1300
  L=$(grep 'LINE' data/et_read.txt 2>/dev/null | tail -1 | sed 's/^ *//')
  say "EARLYTAP: ${L:-not finished (resumable)}"; fwd "MTP Head (auto) EARLYTAP (Fable mission 18): ${L:-not finished}"
fi
# 5. DUALDRAFT (shared)
if [ ! -f data/dd_read.txt ]; then
  srun 14 40 turns/dd_job.sh "DUALDRAFT: L1-20 + E2's x2 joint-chain walk (Fable DRAFTER_98 3.2; ~6 GB, no timing)" 1415
  L=$(grep 'LINE' data/dd_read.txt 2>/dev/null | tail -1 | sed 's/^ *//')
  say "DUALDRAFT: ${L:-not finished}"; fwd "MTP Head (auto) DUALDRAFT: ${L:-not finished}"
fi
say "v11 end"
