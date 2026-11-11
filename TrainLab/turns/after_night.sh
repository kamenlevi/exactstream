#!/bin/sh
# After the night chain (MTP Head, Mon 11-02 evening; Kamen's Ultra week, the orchestrator's queue). While night_1102.sh runs:
# forward its result lines (D-23x, EARLYTAP, DUALDRAFT) to the orchestrator, whose address is looked up at send time (the
# chain's own `peer send 7333` targets the pre-restart pid). When it has ended, in Tuesday's big window only (00:30-15:00):
# KWALL (a) (mission 24's top-16 walk, ~5 GB) and DIVERSE2 (three trainings + walks, ~15 GB), shared slots ending by 15:00.
cd $HOME/TrainLab
LOG=logs/turns.log
say() { echo "$(date '+%m-%d %H:%M:%S') after_night: $*" >> $LOG; }
orch() { $HOME/.local/share/pacer/orch_addr.sh 2>/dev/null | sed 's|.*/||; s|\.sock$||'; }
fwd() { peer say --to ojox-bd "$1" > /dev/null 2>&1; O=$(orch); [ -n "$O" ] && echo "$1" | peer send "$O" - > /dev/null 2>&1; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
hm() { date +%H%M | sed 's/^0*//;s/^$/0/'; }
STOPAT=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-03 15:00" +%s)
left_min() { echo $(( (STOPAT - $(date +%s)) / 60 )); }
N0=$(wc -l < $LOG)
say "start: forwarding night_1102's results to the orchestrator; then KWALL + DIVERSE2 in Tuesday's big window"
while pgrep -f "turns/night_1102.sh" > /dev/null; do
  N1=$(wc -l < $LOG)
  if [ $N1 -gt $N0 ]; then
    tail -n +$((N0 + 1)) $LOG | head -n $((N1 - N0)) | grep -E "night_1102: (D-23x done|EARLYTAP: |DUALDRAFT: )" | while read -r l; do
      fwd "MTP Head (auto, night chain) ${l#* night_1102: }"
    done
    N0=$N1
  fi
  sleep 60
done
while pgrep -f "turns/(e2_cap|e2_train_x|kh_walk|match_walk|copygate_job|et_job|dd_job|dv_conf|turn74|turn75)" > /dev/null; do sleep 20; done
t=$(hm)
if ! { [ "$(date +%d)" = "03" ] && [ "$t" -ge 30 ] && [ "$t" -lt 1500 ]; }; then say "outside Tuesday's big window: nothing run"; exit 0; fi
srun() {   # $1 = GB, $2 = max minutes, $3 = job, $4 = why, $5 = latest start HHMM
  while [ "$(noswap)" -lt $1 ]; do [ "$(hm)" -ge $5 ] && { say "$3 not started by $5 (no $1 GB NO-SWAP)"; return 1; }; sleep 120; done
  [ "$(hm)" -ge $5 ] && { say "$3 not started: past $5"; return 1; }
  m=$2; l=$(left_min); [ $l -lt $m ] && m=$l
  say "$3 ($(noswap) GB NO-SWAP; slot $m min)"
  ./run_shared.sh $m 300 60 $3 "$4"
}
if [ ! -f data/kw_read.txt ]; then
  srun 14 30 turns/kw_job.sh "KWALL: DFlash2 top-16 walk at HPD-0's anchors (mission 24; ~5 GB, no timing)" 1430
  L=$(grep 'LINE' data/kw_read.txt 2>/dev/null | tail -1 | sed 's/^ *//')
  say "KWALL (a): ${L:-not finished}"; fwd "MTP Head (auto) KWALL (a), mission 24: ${L:-not finished}"
fi
if [ ! -f data/dv_read.txt ]; then
  srun 23 240 turns/dv_job.sh "DIVERSE2: three member-2 trainings + walks (~15 GB, no timing)" 1230
  L=$(grep 'LINE' data/dv_read.txt 2>/dev/null | tail -1 | sed 's/^ *//')
  say "DIVERSE2: ${L:-not finished (resumable)}"; fwd "MTP Head (auto) DIVERSE2: ${L:-not finished, resumable}"
fi
say "end"
