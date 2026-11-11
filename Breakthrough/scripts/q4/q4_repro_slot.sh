#!/bin/bash
# Qwen 4 day-0 "today's capture reproduced" + the 24 GB chat map gates (Breakthrough, Sun 2026-11-08; the orchestrator's
# 11:3x slot, NewIdeas' board 66766b1). Pre-registered in SCOREBOARD.md (Sun 04:2x / 04:4x). One share lock, MEM 18 GB.
#   1. capture  (36 windows, DM_C 51) with DM_OUT=<scratch>/cap   -> compared with out/baseline.json, out/capture.json, out/routing
#   2. xcapture (3 windows, --max 1) with DM_OUT=<scratch>/xcap   -> compared with out/xcap (x / w / ids)
#   3. map arms (default out/, 36 windows): $Q4_ARMS (default qom_0_live_c147 qom_0_live_ball_c147)
# Guards: swap +2 GB over the start kills the running map process; a wall cap per step; the log's last line is the verdict.
# Usage: q4_repro_slot.sh <scratch dir> [steps, default "cap xcap arms"]
SP=${1:?usage: q4_repro_slot.sh scratch_dir [steps]}; STEPS=${2:-cap xcap arms}
ARMS=${Q4_ARMS:-qom_0_live_c147 qom_0_live_ball_c147}
DM=~/FableGod/decision_map; PY=~/BigMoE/.venv/bin/python3
cd ~/Breakthrough || exit 1
mkdir -p "$SP"
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
S0=$(swap_mb); echo "[q4] start $(date +%H:%M:%S) swap ${S0} MB; steps: $STEPS; arms: $ARMS"
peer lock share 75m --cpu 200 --gpu 80 "Breakthrough: Qwen 4 map capture repro + 24 GB chat map gates (decision map, C 51), MEM 18 GB" || { echo "[q4] lock FAILED"; exit 1; }
trap 'peer unlock >/dev/null 2>&1' EXIT
run() {   # run <label> <wall cap s> <env...> -- <run_map args...>
  local label=$1 cap=$2 want=$3; shift 3; local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  echo "[q4] $label: start $(date +%H:%M:%S)"
  env PYTHONDONTWRITEBYTECODE=1 "${envs[@]}" nice -n 5 $PY -B $DM/run_map.py "$@" > "$SP/$label.log" 2>&1 &
  local pid=$! t0=$(date +%s) why=""
  while kill -0 $pid 2>/dev/null; do
    sleep 20
    local s=$(swap_mb)
    if [ "$s" -gt $((S0 + 2048)) ]; then why="swap ${s} MB > start ${S0} + 2048"; kill -TERM $pid; break; fi
    if [ $(( $(date +%s) - t0 )) -gt "$cap" ]; then why="wall cap ${cap} s"; kill -TERM $pid; break; fi
  done
  wait $pid; local rc=$?
  # success = every requested name logged "[dm] <name> done in" and none FAILED: run_map's closing C.report() reads
  # <OUT>/baseline.json, absent from a scratch DM_OUT, so its rc is 1 after a complete capture (Sun 11:39)
  local nd=$(grep -c '^\[dm\] .* done in ' "$SP/$label.log") nf=$(grep -c '^\[dm\] .* FAILED' "$SP/$label.log")
  echo "[q4] $label: end $(date +%H:%M:%S) rc $rc${why:+ STOPPED: $why}; done $nd of $want, failed $nf; $(grep -c ' positions ' "$SP/$label.log") windows logged; $(grep -m1 'loaded in' "$SP/$label.log")"
  [ -z "$why" ] && [ "$nd" -eq "$want" ] && [ "$nf" -eq 0 ]
}
ok=1
for st in $STEPS; do
  case $st in
    cap)  run cap 2400 1 DM_OUT="$SP/cap" DM_C=51 -- capture --max 12 || ok=0 ;;
    xcap) run xcap 900 1 DM_OUT="$SP/xcap" DM_C=51 -- xcapture --max 1 || ok=0 ;;
    arms) run arms 3000 $(echo $ARMS | wc -w) DM_C=51 -- $ARMS --max 12 --skip-done || ok=0 ;;
  esac
  [ $ok -eq 1 ] || break
done
peer unlock >/dev/null 2>&1; trap - EXIT
echo "[q4] RUNNER DONE $(date +%H:%M:%S) ok=$ok swap $(swap_mb) MB (start $S0)"
