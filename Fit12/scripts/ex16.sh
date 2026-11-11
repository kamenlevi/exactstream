#!/bin/sh
# Fit12 (Fri 6 Nov 21:4x; the orchestrator 21:3x, NewIdeas QUEUE f2243e8 '12GB-CLEAN'): the 12 GB VERIFIED-GREEDY loop ALONE.
# /Users/kamen/Fit12/scripts/fit_t3.py arm ex only: DFlash2 L1-20 (4-bit) drafts 15 tokens per check and the 12 GB model
# verifies every token in <= 4-row calls (verify rows wait for all ten ranks; ranks 0-1 at 4 bits, 2-9 at 3). NOT exact:
# 2-4-row verify calls round differently from one-token decoding. All 16 of T3R2's HumanEval prompts, greedy, 256 tokens,
# full outputs, C_m 47 / C_l 32, in its own process (no other arm, no ports: F12_KPORTS unset), so the process's lifetime
# peak phys_footprint (RUSAGE_INFO_V4) is the loop's own. F12_RERUN=1: a prompt with a PAUSE inside is re-run once at the
# end (paused twice: excluded). Report: /Users/kamen/Fit12/scripts/ex16_read.py -> $O/ex16_report.txt.
# Slot (NewIdeas): Saturday morning, first after tonight's chain, just before CERT's reference decodes. START GATE: from
# NOT_BEFORE (arg 1, default "2026-11-07 06:00") the first CLEAR minutes in a row (arg 2, default 10; 1 once $O/GO exists,
# which the chain's owner may touch at its end) with no other session's quiet lock or ticket, no other lock showing
# MEM > 5 GB, no PAUSE / HOLD / ACT verdict / resume ramp / closed lid, and >= 16 GB free (NO-SWAP: peak 11.4 + 4 under a
# quiet lock). Then a 25-min quiet lock (queue wait <= 60 min, else back to the gate) and an idle GPU. Starts by DEADLINE (arg 3,
# default "2026-11-07 18:00") or not at all. Stops on swap +1 GB, 30 min, or DEADLINE + 30 min. Writes $O/DONE at the end
# (rc, times), for CERT's runner to wait on. Run ATTACHED (Bash run_in_background). Usage: sh ex16.sh [NB] [CLEAR] [DL]
# Fri 23:3x RE-ARMED with NB = now (the orchestrator 23:3x: Kamen opened the window; it may run before 06:00 in a GPU gap
# NewIdeas places it in). $O/WAIT (NewIdeas may touch it to keep the slot for later) blocks the gate like HOLD does.
F=/Users/kamen/Fit12; PY=/Users/kamen/BigMoE/.venv/bin/python; LOG=$F/logs/summary.txt; O=$F/data/scr/ex16
K=humaneval/HumanEval; Q=/Users/kamen/.local/share/quiet; PAUSEF=$Q/PAUSE; HOLDF=$Q/HOLD; HEALTH=$Q/HEALTH.md
NB=$(date -j -f "%Y-%m-%d %H:%M" "${1:-2026-11-07 06:00}" +%s) || exit 9
CLR=${2:-10}
DL=$(date -j -f "%Y-%m-%d %H:%M" "${3:-2026-11-07 18:00}" +%s) || exit 9
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
swapmb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
freegb() { vm_stat | awk '/Pages free|Pages inactive|Pages speculative|Pages purgeable/ {gsub(/\./,"",$NF); s+=$NF} END {printf "%d", s*16384/1e9}'; }
others() { peer ls 2>/dev/null | grep "^- "; }
locks() { grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"; }
nquiet() { others | locks | grep -cE "^· (benchmarking|waiting for a quiet window)"; }
nbig() { others | locks | grep -oE "MEM [0-9]+ GB" | awk '$2 > 5' | wc -l | tr -d " "; }
gpu() { peer top 2>/dev/null | head -1 | grep -o 'GPU [0-9]*%' | grep -o '[0-9]*'; }
act() { grep -q '^\*\*Verdict: ACT' "$HEALTH" 2>/dev/null; }
ramp() { pgrep -f '^/bin/zsh [^ ]*soft_end\.sh' >/dev/null; }
lid_closed() { ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes; }
blocked() { [ -e "$PAUSEF" ] || [ -e "$HOLDF" ] || [ -e "$O/WAIT" ] || act || ramp || lid_closed; }
cd $F || exit 9
mkdir -p $O
rm -f $O/DONE
log "ex16 armed: not before $(date -r $NB '+%a %H:%M'), $CLR clear minutes (1 with $O/GO; held while $O/WAIT exists), start by $(date -r $DL '+%a %H:%M')"
while :; do
  clear=0
  while [ "$(date +%s)" -lt "$DL" ]; do
    if [ "$(date +%s)" -ge "$NB" ]; then
      q=$(nquiet); b=$(nbig); a=$(freegb)
      if [ "$q" -eq 0 ] && [ "$b" -eq 0 ] && ! blocked && [ "$a" -ge 16 ]; then clear=$((clear + 1)); else clear=0; fi
      need=$CLR; [ -e $O/GO ] && need=1
      [ $clear -ge $((need * 2)) ] && break
    fi
    sleep 30
  done
  if [ "$(date +%s)" -ge "$DL" ]; then
    log "ex16: no clear slot by the deadline (quiet $q, big $b, $a GB free, PAUSE $([ -e $PAUSEF ] && echo on || echo off)): NOT STARTED"
    echo "$(date +%H:%M:%S) SCR ex16 NOT STARTED: no clear slot by the deadline" >> $LOG
    echo "rc 5 not started $(date '+%a %H:%M:%S')" > $O/DONE
    exit 5
  fi
  log "ex16: clear for $((clear / 2)) min ($a GB free): asking for the quiet lock"
  if ! PEER_QUEUE_TIMEOUT=3600 peer lock quiet --queue 25m "Fit12 12GB-CLEAN: the 12 GB verified-greedy loop alone, 16 HumanEval, MEM 12 GB, ~8 min (NewIdeas f2243e8)" >/dev/null 2>&1; then
    log "ex16: no quiet lock within 60 min: back to the gate"; continue
  fi
  sleep 3; g=$(gpu); gp=100; gn=1                       # the previous holder's exit: two samples <= 10 % in a row, <= 2 min
  while { [ "${g:-0}" -gt 10 ] || [ "$gp" -gt 10 ]; } && [ $gn -lt 24 ]; do gp=${g:-0}; sleep 5; g=$(gpu); gn=$((gn + 1)); done
  if blocked || [ "$(freegb)" -lt 16 ]; then
    log "ex16: granted, but blocked or < 16 GB free ($(freegb) GB, GPU ${g}%): lock given back"; peer unlock >/dev/null 2>&1; continue
  fi
  break
done
SW0=$(swapmb); T0=$(date +%s)
echo "$(date +%H:%M:%S) SCR ex16 start: fit_t3.py ex16 arm ex ALONE (12GB-CLEAN, verified greedy), 16 HumanEval prompts, quiet lock ($(freegb) GB free, swap $SW0 MB, GPU ${g}%)" >> $LOG
{ echo "ex16 manifest $(date '+%a %d %b %H:%M:%S')"; shasum -a 256 $F/scripts/fit_t3.py $F/scripts/t3_twopass.py $F/scripts/t2_twopass.py \
    $F/pinned/8db7/bt_b12.py /Users/kamen/Leap/scripts/d7b_live.py; sed -n '/^ENV = /,/^for k in/p' $F/scripts/fit_t3.py; } > $O/MANIFEST.txt 2>&1
env F12_OUT=$O F12_CAP=256 F12_PHLOG=1 F12_RERUN=1 F12_T3ARMS=ex F12_CMCL=47,32 \
  F12_KEYS=$K/156,$K/42,$K/68,$K/91,$K/16,$K/6,$K/79,$K/19,$K/161,$K/17,$K/56,$K/141,$K/101,$K/63,$K/114,$K/77 \
  $PY -B $F/scripts/fit_t3.py ex16 > $O/ex16.txt 2>&1 &
P=$!; j=0; rc=0
while kill -0 $P 2>/dev/null; do
  sleep 5; j=$((j + 5))
  [ $(( $(swapmb) - SW0 )) -gt 1024 ] && { rc=127; log "ex16: swap +1 GB: stopping"; }
  [ $j -ge 1800 ] && { rc=124; log "ex16: 30 min: stopping"; }
  [ "$(date +%s)" -ge $((DL + 1800)) ] && { rc=125; log "ex16: deadline + 30 min: stopping"; }
  [ $rc -ne 0 ] && { kill -TERM $P; sleep 5; kill -KILL $P 2>/dev/null; break; }
done
wait $P; w=$?; [ $rc -eq 0 ] && rc=$w
peer unlock >/dev/null 2>&1
echo "$(date +%H:%M:%S) SCR ex16 end (rc $rc, $(( $(date +%s) - T0 )) s, swap $(swapmb) MB): $(ls $O/ex16_*_ex_toks.json 2>/dev/null | wc -l | tr -d ' ') prompt outputs" >> $LOG
$PY -I -B $F/scripts/ex16_read.py $O/ex16.txt ex16 > $O/ex16_report.txt 2>&1
echo "rc $rc started $(date -r $T0 '+%a %H:%M:%S') ended $(date '+%a %H:%M:%S')" > $O/DONE
grep -E 'Error|Traceback' $O/ex16.txt | head -3
cat $O/ex16_report.txt
exit $rc
