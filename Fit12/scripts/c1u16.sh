#!/bin/sh
# Fit12 (Sat 7 Nov 08:2x; the orchestrator 08:1x): the MEASURED t80ck1u run, the 12 GB FAST candidate. Breakthrough's map
# step 0 (be2c175): P1-K1 theta 0.8 KU 4 (T3R2's form) passes the code line, 45.8 tok/s [P] from Fit12's replay.
# /Users/kamen/Fit12/scripts/fit_t3.py arm t80ck1u ALONE in its own process, on ex16's footing: theta 0.8, cf 0.9, pass 1 =
# P1-K1 (the rows' rank-0 union, KU 4, 4 bits; top-2 tiers), the exact check for the rows pass 1 does not certify; all 16 of
# T3R2's HumanEval prompts, greedy, cap 256, full outputs, C_m 47 / C_l 32, F12_RERUN=1 (a paused prompt is re-run once),
# F12_PHLOG=1 (per-phase marks: the full-period figure). FAST: not exact and not verified greedy.
# GATE: the first quiet slot (the orchestrator: Leap may need the GPU next, keep it short): no PAUSE / HOLD / ACT verdict /
# resume ramp / closed lid and >= 16 GB free, then a 15-min quiet lock (queue wait <= 60 min, else back to the gate) and an
# idle GPU. Starts by DL (arg 1, default "2026-11-07 14:00") or not at all. Stops on swap +1 GB or 20 min.
# Report: /Users/kamen/Fit12/scripts/c1u16_read.py (run after). Out: /Users/kamen/Fit12/data/scr/c1u16 (c1u16.txt, per-prompt
# json, MANIFEST.txt, DONE). Run ATTACHED (Bash run_in_background). Usage: sh c1u16.sh [DL]
F=/Users/kamen/Fit12; PY=/Users/kamen/BigMoE/.venv/bin/python; LOG=$F/logs/summary.txt; O=$F/data/scr/c1u16
K=humaneval/HumanEval; Q=/Users/kamen/.local/share/quiet; PAUSEF=$Q/PAUSE; HOLDF=$Q/HOLD; HEALTH=$Q/HEALTH.md
DL=$(date -j -f "%Y-%m-%d %H:%M" "${1:-2026-11-07 14:00}" +%s) || exit 9
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
swapmb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
freegb() { vm_stat | awk '/Pages free|Pages inactive|Pages speculative|Pages purgeable/ {gsub(/\./,"",$NF); s+=$NF} END {printf "%d", s*16384/1e9}'; }
gpu() { peer top 2>/dev/null | head -1 | grep -o 'GPU [0-9]*%' | grep -o '[0-9]*'; }
act() { grep -q '^\*\*Verdict: ACT' "$HEALTH" 2>/dev/null; }
ramp() { pgrep -f '^/bin/zsh [^ ]*soft_end\.sh' >/dev/null; }
lid_closed() { ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes; }
blocked() { [ -e "$PAUSEF" ] || [ -e "$HOLDF" ] || act || ramp || lid_closed; }
cd $F || exit 9
mkdir -p $O
rm -f $O/DONE
log "c1u16 armed: the first quiet slot, start by $(date -r $DL '+%a %H:%M')"
while :; do
  while { blocked || [ "$(freegb)" -lt 16 ]; } && [ "$(date +%s)" -lt "$DL" ]; do sleep 30; done
  if [ "$(date +%s)" -ge "$DL" ]; then
    log "c1u16: no slot by the deadline ($(freegb) GB free, PAUSE $([ -e $PAUSEF ] && echo on || echo off)): NOT STARTED"
    echo "$(date +%H:%M:%S) SCR c1u16 NOT STARTED: no slot by the deadline" >> $LOG
    echo "rc 5 not started $(date '+%a %H:%M:%S')" > $O/DONE
    exit 5
  fi
  log "c1u16: asking for the quiet lock ($(freegb) GB free)"
  if ! PEER_QUEUE_TIMEOUT=3600 peer lock quiet --queue 15m "Fit12 12GB-FAST: t80ck1u measured (the 12 GB FAST candidate) alone, 16 HumanEval, MEM 12 GB, ~5 min (orchestrator 08:1x)" >/dev/null 2>&1; then
    log "c1u16: no quiet lock within 60 min: back to the gate"; continue
  fi
  sleep 3; g=$(gpu); gp=100; gn=1                       # the previous holder's exit: two samples <= 10 % in a row, <= 2 min
  while { [ "${g:-0}" -gt 10 ] || [ "$gp" -gt 10 ]; } && [ $gn -lt 24 ]; do gp=${g:-0}; sleep 5; g=$(gpu); gn=$((gn + 1)); done
  if blocked || [ "$(freegb)" -lt 16 ]; then
    log "c1u16: granted, but blocked or < 16 GB free ($(freegb) GB, GPU ${g}%): lock given back"; peer unlock >/dev/null 2>&1; continue
  fi
  break
done
SW0=$(swapmb); T0=$(date +%s)
log "c1u16: start ($(freegb) GB free, swap $SW0 MB, GPU ${g}%)"
echo "$(date +%H:%M:%S) SCR c1u16 start: fit_t3.py c1u16 arm t80ck1u ALONE (12 GB FAST candidate, measured), 16 HumanEval prompts, quiet lock ($(freegb) GB free, swap $SW0 MB, GPU ${g}%)" >> $LOG
{ echo "c1u16 manifest $(date '+%a %d %b %H:%M:%S')"; shasum -a 256 $F/scripts/fit_t3.py $F/scripts/t3_twopass.py $F/scripts/t2_twopass.py \
    $F/pinned/8db7/bt_b12.py /Users/kamen/Leap/scripts/d7b_live.py; sed -n '/^ENV = /,/^for k in/p' $F/scripts/fit_t3.py; } > $O/MANIFEST.txt 2>&1
env F12_OUT=$O F12_CAP=256 F12_PHLOG=1 F12_RERUN=1 F12_T3ARMS=t80ck1u F12_CMCL=47,32 \
  F12_KEYS=$K/156,$K/42,$K/68,$K/91,$K/16,$K/6,$K/79,$K/19,$K/161,$K/17,$K/56,$K/141,$K/101,$K/63,$K/114,$K/77 \
  $PY -B $F/scripts/fit_t3.py c1u16 > $O/c1u16.txt 2>&1 &
P=$!; j=0; rc=0
while kill -0 $P 2>/dev/null; do
  sleep 5; j=$((j + 5))
  [ $(( $(swapmb) - SW0 )) -gt 1024 ] && { rc=127; log "c1u16: swap +1 GB: stopping"; }
  [ $j -ge 1200 ] && { rc=124; log "c1u16: 20 min: stopping"; }
  [ $rc -ne 0 ] && { kill -TERM $P; sleep 5; kill -KILL $P 2>/dev/null; break; }
done
wait $P; w=$?; [ $rc -eq 0 ] && rc=$w
peer unlock >/dev/null 2>&1
echo "$(date +%H:%M:%S) SCR c1u16 end (rc $rc, $(( $(date +%s) - T0 )) s, swap $(swapmb) MB): $(ls $O/c1u16_*_t80ck1u_toks.json 2>/dev/null | wc -l | tr -d ' ') prompt outputs" >> $LOG
echo "rc $rc started $(date -r $T0 '+%a %H:%M:%S') ended $(date '+%a %H:%M:%S')" > $O/DONE
log "c1u16: end rc $rc"
grep -E 'Error|Traceback' $O/c1u16.txt | head -3
grep -E '^\[c1u16\] (TOTAL|done; lifetime|PAUSE summary)' $O/c1u16.txt | cut -c1-300
exit $rc
