#!/bin/sh
# Fit12 (Wed 4 Nov 19:4x; the orchestrator 19:3x, branch (b): "Finish layer 41 Thu early on CPU (e.g. ~06:00-06:30, beside
# RELAX's replays; CPU only, pausing for quiet tickets). Then Breakthrough runs s7_trl in a ~10-min gap right after BigMoE's
# Thu ticket (~08:55) ... All of it before the ~10:55 conversion, and your encoder must not run across the conversion."):
# waits until START; then, only if the TRL store is NOT whole, runs the same runner (/Users/kamen/Fit12/scripts/
# trl_chunks2.sh: resumable, finished chunks are skipped without starting Python; peer wait + a share lock per chunk; normal
# priority, 4 threads) on the layers that are not whole, with its last chunk start at STOP_AT (0830: the store is whole
# well before s7_trl's ~08:55 gap, and the encoder is off records3 ~2.5 h before the ~10:55 conversion).
# Does nothing (logged in summary.txt) if: the store is whole; records3's index.json already has a "layout" key (the
# conversion ran; fit_q.read would refuse anyway); it is past STOP_AT; a STOP file exists. Waits while the lid is closed
# (Fit12 rule: lid closed = pause). Run ATTACHED (Bash run_in_background, CLAUDE_PID set).
# Usage: sh trl_thu.sh ["YYYY-mm-dd HH:MM:SS" START, default 2026-11-05 06:00:00] [STOP_AT HHMM, default 0830]
START=${1:-"2026-11-05 06:00:00"}; STOP_AT=${2:-0830}
F=/Users/kamen/Fit12; OUT=$F/data/trl; LOG=$F/logs/summary.txt; IDX=/Users/kamen/BigMoE/models/bare/records3/index.json
t0=$(date -j -f "%Y-%m-%d %H:%M:%S" "$START" +%s) || { echo "bad START $START"; exit 2; }
nfiles() { find $OUT/L$1 -name 'e*_k*.safetensors' ! -name '*.tmp.*' 2>/dev/null | wc -l | tr -d ' '; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
echo "$(date +%H:%M:%S) SCR trl thu armed: at $START, only if the store is not whole, the same runner with last chunk start $STOP_AT (the orchestrator 19:3x, branch b) (pid $$)" >> $LOG
while [ $(date +%s) -lt $t0 ]; do sleep 60; done
while lid_closed && [ "$(date +%H%M)" -lt "$STOP_AT" ]; do sleep 60; done
todo=""; n=0
for L in 18 30 7 41; do c=$(nfiles $L); n=$((n + c)); [ "$c" -lt 1536 ] && todo="$todo${todo:+,}$L"; done
if [ -z "$todo" ]; then
  why="the store is whole ($n files)"
elif grep -q '"layout"' $IDX; then
  why="records3 index.json already has a \"layout\" key (converted); not run ($n of 6144 files, layers $todo not whole)"
elif [ "$(date +%H%M)" -ge "$STOP_AT" ]; then
  why="it is $(date +%H:%M), past the last chunk start $STOP_AT ($n of 6144 files, layers $todo not whole)"
elif lid_closed; then
  why="lid closed until $STOP_AT ($n of 6144 files)"
elif [ -e $OUT/STOP ]; then
  why="STOP file present ($n of 6144 files)"
else
  echo "$(date +%H:%M:%S) SCR trl thu: $n of 6144 files; finishing layers $todo with the same runner, last chunk start $STOP_AT (the orchestrator 19:3x, branch b)" >> $LOG
  cd $F && exec sh $F/scripts/trl_chunks2.sh $todo 0 511 mix 4 $STOP_AT normal
fi
echo "$(date +%H:%M:%S) SCR trl thu: not run: $why" >> $LOG
echo "thu: not run: $why"
