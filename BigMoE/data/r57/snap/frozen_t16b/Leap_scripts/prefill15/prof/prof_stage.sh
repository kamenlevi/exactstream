#!/bin/bash
# Prefill 1.5K, step 2: stage split of PF-B (R57_PFLM=3) at 32 GB on ~10K and ~30K-token prompts, through the cache clone's
# run.py with the instrumented pflm copy (scripts/prefill15/prof: sitecustomize maps `pflm` to pflm_prof.py; the clone is
# untouched). Four runs, one process each: 10K and 30K, R57_PFLM_PROF_SYNC=0 (timestamps only: the real overlapped run)
# and =1 (a barrier after each sub-step: GPU time per stage). --max-tokens 1. Gates: no engine / chat / server running, no
# PAUSE / HOLD, NO-SWAP >= 44 GB before each run; a share lock; swap +2 GB stops; last start 13:30 (the 14:00 push stays
# free). Usage: prof_stage.sh <work dir>
W=$1; P=/Users/kamen/Leap/scripts/prefill15/prof; CL=/Users/kamen/Leap/release/cleanclone/cache/clone; LOG=$W/prof.log
LAST=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-11 13:30:00" +%s)
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
procs() { ps -axo pid=,args= | awk '/[Pp]ython/ && /r57_screen_|[ \/](chat|serve|run)[.]py( |$)/ && !/ awk / {printf "%s ", $1}'; }
say() { echo "$(date '+%H:%M:%S') $*" | tee -a $LOG; }
RP=""; GUARD=""; LOCKED=0
cleanup() { [ -n "$GUARD" ] && kill $GUARD 2>/dev/null; [ -n "$RP" ] && kill -0 $RP 2>/dev/null && kill -TERM $RP
            [ $LOCKED = 1 ] && peer unlock >/dev/null 2>&1; say "exit (NO-SWAP $(noswap) GB, swap $(swap) MB)"; }
trap cleanup EXIT; trap 'say "stopped by a signal"; exit 1' INT TERM
peer lock share 75m --cpu 400 --gpu 100 "MEM 44 GB: Leap prefill stage split (PF-B, 32 GB, 10K / 30K), ~15 min" > /dev/null 2>&1 && LOCKED=1
S0=$(swap); ( while :; do [ $(( $(swap) - S0 )) -gt 2048 ] && { echo "$(date +%T) SWAP GUARD: $(swap) MB" >> $LOG; kill -TERM $$; exit; }; sleep 5; done ) &
GUARD=$!
for spec in "10k 0" "10k 1" "30k 0" "30k 1"; do
  set -- $spec; N=$1; SY=$2
  [ $(date +%s) -ge $LAST ] && { say "SKIP $N sync=$SY: past the 13:30 last start"; continue; }
  [ -n "$(procs)" ] && { say "SKIP $N sync=$SY: an engine runs ($(procs))"; continue; }
  [ -e ~/.local/share/quiet/PAUSE ] || [ -e ~/.local/share/quiet/HOLD ] && { say "SKIP $N sync=$SY: PAUSE / HOLD"; continue; }
  [ "$(noswap)" -lt 44 ] && { say "SKIP $N sync=$SY: NO-SWAP $(noswap) < 44"; continue; }
  J=$W/prof_${N}_sync$SY.jsonl; rm -f $J
  say "run $N sync=$SY (NO-SWAP $(noswap) GB)"
  ( cd $CL && exec env PYTHONPATH=$P R57_PFLM_PROF=$J R57_PFLM_PROF_SYNC=$SY .venv/bin/python run.py --config chat --mem 32 \
      --prompt-file $W/prompt_$N.txt --max-tokens 1 --env R57_PFLM=3 ) > $W/run_${N}_sync$SY.out 2>&1 &
  RP=$!; wait $RP; rc=$?; RP=""
  say "  rc $rc; $(grep -c '"L"' $J 2>/dev/null) layer lines; total $(grep -o '"total": \[[^]]*\]' $J 2>/dev/null) tokens $(grep -o '"tokens": [0-9]*' $J 2>/dev/null | cut -d' ' -f2); $(grep -o 'first token [0-9.]* s' $W/run_${N}_sync$SY.out | head -1); $(grep -o 'lifetime peak [0-9.]* GB' $W/run_${N}_sync$SY.out | tail -1)"
done
