#!/bin/bash
# Night Thu 12 -> Fri 13 Nov (the orchestrator 19:1x). After the 1.2.0 release gate (Leap's rel120_check, then EvalSuite's
# r120, done by about 02:00), BigMoE's three GPU steps in order, each behind the usual gates and locks:
#  1. the 45 GB P3 + P2 A/B (scripts/p3.sh) via turn45, window 02:15-04:00. p3.sh itself needs NO-SWAP >= 47.2 GB, else it
#     logs "not run". Fans are medium until 07:30; the no-heat-freeze validity rule is applied when its report is read.
#  2. DEEPDRAFT TTT30B (docs/PREREG_DEEPDRAFT_TTT30B_2026-11-12.md): one pass over night 1 (4,206 steps) under a share lock
#     with MEM 18 GB. It yields to other sessions' quiet locks, a closed lid and a HEALTH ACT: SIGTERM makes dd_ttt.py save
#     the weights, Adam's moments and the step, and exit 3; it resumes once the gates clear and NO-SWAP >= 24 GB. Then the
#     eval of b5, t30a, t30b_s600, t30b_s1500, t30b_s3000 and t30b.
#  3. the DEEPDRAFT capture continuation (scripts/dd_capture_night.sh, DD_NIGHT=n1112, which skips prompts it already has)
#     via turn45 PEAK 24, last start 12:30, hard stop 13:30.
# usage (a harness background task, not nohup): bash scripts/dd_night_n1113.sh      DRY=1: print the plan, run nothing
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
D=data/deepdraft/ttt30b; mkdir -p $D
LOG="$PWD/$D/night.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; [ -n "$DRY" ] && echo "$*"; }
PY=$HOME/BigMoE/.venv/bin/python
Q="$HOME/.local/share/quiet"
if [ "$(date +%H)" -lt 12 ]; then DAY=$(date +%Y-%m-%d); else DAY=$(date -v+1d +%Y-%m-%d); fi   # the morning this night ends
ep() { date -j -f "%Y-%m-%d %H:%M:%S" "$DAY $1:00" +%s; }
CAP_LAST=$(ep 12:30); CAP_END=$(ep 13:30)
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
others_quiet() { peer ls 2>/dev/null | grep -E "benchmarking: quiet lock|waiting for a quiet" | grep -v "BigMoE" | grep -c .; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
health_act() { grep -q "Verdict: ACT" "$Q/HEALTH.md" 2>/dev/null; }
gates_clear() { [ "$(others_quiet)" -eq 0 ] && ! lid_closed && ! health_act; }
gate_state() { echo "quiet locks of others $(others_quiet), lid $(lid_closed && echo closed || echo open), health $(health_act && echo ACT || echo ok), NO-SWAP $(avail_gb) GB"; }
say "night n1113 chain start (pid $$; capture last start $(date -r $CAP_LAST '+%a %H:%M'), hard stop $(date -r $CAP_END '+%a %H:%M'))"

# 1. the 45 GB P3 + P2 A/B
say "step 1: P3 + P2 45 GB A/B via turn45 (window 02:15-04:00, PEAK 34, ticket 45 min)"
if [ -z "$DRY" ]; then
  NOTBEFORE=02:15 NOTAFTER=04:00 bash scripts/turn45.sh 34 45 "P3 first-token A/B" /bin/bash scripts/p3.sh
  say "step 1 ended rc $?"
fi

# 2. TTT30B: one pass, yielding to quiet locks
ARGS="--run t30b --manifest $D/split.json --pass-seed 1113 --save-at 600,1500,3000"
say "step 2: TTT30B train $ARGS"
trained=0; fails=0
while [ -z "$DRY" ] && [ $trained -eq 0 ]; do
  n=0
  until gates_clear && [ "$(avail_gb)" -ge 24 ]; do
    [ $((n % 30)) -eq 0 ] && say "train waits: $(gate_state)"
    n=$((n + 1)); sleep 60
  done
  peer lock share 480m --cpu 300 --gpu 60 "BigMoE DEEPDRAFT TTT30B: one pass over night 1 (4,206 steps), yields to quiet locks, MEM 18 GB" >/dev/null 2>&1
  say "train (re)start: $(gate_state)"
  taskpolicy -b $PY scripts/dd_ttt.py train $ARGS >> $D/run.log 2>&1 &
  TP=$!; y=0
  while kill -0 $TP 2>/dev/null; do
    sleep 20
    if ! gates_clear; then
      say "yield: $(gate_state) -> SIGTERM $TP"; kill -TERM $TP; y=1; break
    fi
  done
  wait $TP; rc=$?
  peer unlock >/dev/null 2>&1
  say "train run ended rc $rc (yield $y): $(tail -1 $D/run.log | cut -c1-160)"
  if [ $rc -eq 0 ]; then trained=1
  elif [ $rc -eq 3 ] && [ $y -eq 1 ]; then sleep 120
  else
    fails=$((fails + 1))
    [ $fails -ge 2 ] && { say "train stopped twice without a yield (rc $rc): no more training tonight"; break; }
    sleep 120
  fi
done

if [ $trained -eq 1 ]; then
  n=0
  until gates_clear && [ "$(avail_gb)" -ge 20 ]; do
    [ $((n % 30)) -eq 0 ] && say "eval waits: $(gate_state)"
    n=$((n + 1)); sleep 60
  done
  peer lock share 60m --cpu 300 --gpu 60 "BigMoE DEEPDRAFT TTT30B eval (6 heads, held-out HumanEval+ and MBPP), MEM 12 GB" >/dev/null 2>&1
  say "eval start"
  taskpolicy -b $PY scripts/dd_ttt.py eval --runs b5,t30a,t30b_s600,t30b_s1500,t30b_s3000,t30b --out $D >> $D/run.log 2>&1
  say "eval ended rc $?: $(tail -1 $D/run.log | cut -c1-200)"
  peer unlock >/dev/null 2>&1
fi

# 3. the capture continuation
NOW=$(date +%s)
if [ -n "$DRY" ]; then
  say "step 3: DD_NIGHT=n1112 NOTAFTER=12:30 turn45.sh 24 $(( (CAP_END - NOW) / 60 + 5 )) ... dd_capture_night.sh"
elif [ $NOW -lt $CAP_LAST ]; then
  MIN=$(( (CAP_END - NOW) / 60 + 5 ))
  say "step 3: the capture continuation via turn45 (PEAK 24, ticket $MIN min, last start 12:30, hard stop 13:30)"
  DD_NIGHT=n1112 NOTAFTER=12:30 bash scripts/turn45.sh 24 $MIN "DEEPDRAFT capture night 2" /bin/bash scripts/dd_capture_night.sh
  say "step 3 ended rc $?"
else
  say "step 3 skipped: past 12:30"
fi
say "night n1113 chain end"
