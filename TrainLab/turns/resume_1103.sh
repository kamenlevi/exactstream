#!/bin/sh
# MTP Head, Tue 11-03 (replaces after_night / after_night2 / after_night3, stopped 09:29 for the orchestrator's heat order:
# one GPU job at a time; Breakthrough's map, then EvalSuite's RELAX smoke, then my chain from ~11:45). From 11:45, before
# each job: no other session's active lock. Order: Breakthrough's two CF walks (~2 GB, ~10 min; it needs them for Wed),
# KWALL (a) (~2 GB walk), DIVERSE2 (conf-out + 3 trainings + walks, ~2 h; resumable), all shared slots ending by 15:00.
# DEPTH32 waits for Wednesday's window. Results to NewIdeas (ojox-3f) + the orchestrator (address at send time).
cd $HOME/TrainLab
LOG=logs/turns.log
say() { echo "$(date '+%m-%d %H:%M:%S') resume_1103: $*" >> $LOG; }
orch() { $HOME/.local/share/pacer/orch_addr.sh 2>/dev/null | sed 's|.*/||; s|\.sock$||'; }
fwd() { echo "$1" | peer send ojox-3f - > /dev/null 2>&1; O=$(orch); [ -n "$O" ] && echo "$1" | peer send "$O" - > /dev/null 2>&1; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
hm() { date +%H%M | sed 's/^0*//;s/^$/0/'; }
STOPAT=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-03 15:00" +%s)
left_min() { echo $(( (STOPAT - $(date +%s)) / 60 )); }
others_active() {   # another session's lock that has not expired
  for f in $HOME/.claude-peers/locks/*.json; do
    [ "$(basename $f .json)" = "$CLAUDE_PID" ] && continue
    python3 -I -c "import json,sys,time; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('until', 0) > time.time() and d.get('mode') in ('quiet', 'share') else 1)" $f 2>/dev/null && return 0
  done
  return 1
}
slot() {   # $1 = GB, $2 = max minutes, $3 = job, $4 = why, $5 = latest start HHMM
  while :; do
    [ "$(hm)" -ge $5 ] && { say "$3 not started: past $5"; return 1; }
    if ! others_active && [ "$(noswap)" -ge $1 ]; then break; fi
    sleep 60
  done
  m=$2; l=$(left_min); [ $l -lt $m ] && m=$l
  say "$3 ($(noswap) GB NO-SWAP; slot $m min)"
  ./run_shared.sh $m 300 60 $3 "$4"
}
say "start: waiting for 11:45 (the orchestrator's heat order), then one job at a time"
while [ "$(hm)" -lt 1145 ]; do sleep 60; done
if [ ! -f data/d02b_walk_hpd0_mbpp.npz ]; then
  slot 14 25 turns/bt_walks.sh "Breakthrough's CF walks: agent + MBPP (L1-20, MLX ~2 GB, no timing)" 1440
  if [ -f data/d02b_walk_hpd0_agent.npz ] && [ -f data/d02b_walk_hpd0_mbpp.npz ]; then
    echo "MTP Head (auto): your three HPD-0 CF walks are ready in ~/TrainLab/data: d02b_walk_hpd0_math.npz (it already existed: the same 60 tf_gsm windows, all fields), d02b_walk_hpd0_agent.npz (12 agent windows), d02b_walk_hpd0_mbpp.npz (100 MBPP answers of d1102mbpp, h through identity fc / hidden_norm). Read-only for you, please." | peer send 97689 - > /dev/null 2>&1
    say "Breakthrough's walks done and sent"
  fi
fi
if [ ! -f data/kw_read.txt ]; then
  slot 14 25 turns/kw_job.sh "KWALL: DFlash2 top-16 walk at HPD-0's anchors (mission 24; MLX ~2 GB, no timing)" 1440
  L=$(grep 'LINE' data/kw_read.txt 2>/dev/null | tail -1 | sed 's/^ *//')
  say "KWALL (a): ${L:-not finished}"; fwd "MTP Head (auto) KWALL (a), mission 24: ${L:-not finished}"
fi
if [ ! -f data/dv_read.txt ]; then
  slot 23 200 turns/dv_job.sh "DIVERSE2: conf-out + three member-2 trainings + walks (~15 GB, no timing; resumable)" 1300
  L=$(grep 'LINE' data/dv_read.txt 2>/dev/null | tail -1 | sed 's/^ *//')
  say "DIVERSE2: ${L:-not finished (resumable)}"; fwd "MTP Head (auto) DIVERSE2: ${L:-not finished, resumable in the next window}"
fi
say "end"
