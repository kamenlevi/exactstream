#!/bin/bash
# docbench runner, 2026-10-25 (detached, from a live evalsuite session; see PREREG.md). Quality runs under a share
# lock: first the configs already on disk (Kamen's Qwen3.8-27B as Q4 and 6-bit, Qwen3.6-35B-A3B), then the four 9B
# configs once the 9B BF16 download has finished. Speed runs (batch 1) last, each under a short quiet lock, when the
# download no longer loads the Mac. A memory gate holds each run until free memory covers its peak, and the runner
# stops if swap grows by more than 2 GB. Then the pre-registered verdict -> results/docbench/verdict.txt.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/docbench/runner.log
PY=$HOME/BigMoE/.venv324/bin/python
NINE=$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-bf16
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
free_gb() {   # free + inactive + speculative + purgeable pages
  vm_stat | awk '/page size of/ {ps=$8} /Pages free|Pages inactive|Pages speculative|Pages purgeable/ {gsub("\\.","",$NF); s+=$NF} END {printf "%d", s*ps/1e9}'
}
SWAP0=$(swap_mb)
gate() {      # gate NEED_GB: wait (up to 30 min) for the memory, and stop on swap growth
  local need=$1 i
  for i in $(seq 1 60); do
    if [ $(( $(swap_mb) - SWAP0 )) -gt 2048 ]; then say "swap grew by more than 2 GB since start: runner stops"; exit 4; fi
    [ "$(free_gb)" -ge "$need" ] && return 0
    [ "$i" = 1 ] && say "waiting for $need GB free (now $(free_gb) GB)"
    sleep 30
  done
  say "memory gate: $need GB not free after 30 min"; return 1
}
share() {
  local i
  for i in $(seq 1 20); do
    peer lock share 45m --cpu 200 --gpu 100 "docbench quality: $1" >> "$LOG" 2>&1 && return 0
    say "share lock refused (someone needs the Mac quiet): peer wait"
    peer wait --timeout 30m >> "$LOG" 2>&1
  done
  return 1
}
quality() {   # quality CONFIG BATCH NEED_GB
  [ -s "results/docbench/$1/outputs.jsonl" ] && { say "$1: outputs exist, skipping quality"; return 0; }
  gate "$3" || return 1
  share "$1" || { say "$1: no share lock"; return 1; }
  say "$1: quality start (batch $2, swap $(swap_mb) MB, free $(free_gb) GB)"
  $PY docbench/run.py "$1" --batch "$2" --pbatch 2 --pstep 512 >> "$LOG" 2>&1
  local rc=$?
  peer unlock >> "$LOG" 2>&1
  say "$1: quality rc=$rc (swap $(swap_mb) MB)"
  return $rc
}
speed() {     # speed CONFIG NEED_GB
  [ -s "results/docbench/$1/speed.json" ] && return 0
  gate "$2" || return 1
  PEER_QUEUE_TIMEOUT=5400 peer lock quiet --queue 15m "docbench speed: $1 (batch-1 prefill / decode tok/s)" >> "$LOG" 2>&1 \
    || { say "$1: no quiet lock"; return 1; }
  say "$1: speed start"
  $PY docbench/run.py "$1" --speed >> "$LOG" 2>&1
  local rc=$?
  peer unlock >> "$LOG" 2>&1
  say "$1: speed rc=$rc"
}
nine_ready() { ! pgrep -f "[h]f download mlx-community/Qwen3.5-9B" > /dev/null && [ -d "$NINE/snapshots" ] \
  && [ -z "$(ls "$NINE/blobs" | grep incomplete)" ] && [ "$(ls "$NINE"/snapshots/*/model-0000*-of-00004.safetensors 2>/dev/null | wc -l | tr -d ' ')" = 4 ]; }

say "runner start (pid $$, swap $SWAP0 MB, free $(free_gb) GB)"
quality 38-27b-q4 4 22
quality 38-27b-q6 4 28
quality 36-35b-a3b-q4 8 26
for i in $(seq 1 120); do nine_ready && break; [ "$i" = 1 ] && say "waiting for the 9B download"; sleep 60; done
if nine_ready; then
  say "9B download complete ($(du -sh "$NINE" | cut -f1))"
  for c in 9b-bf16 9b-q8 9b-q4 9b-q4mixed; do quality $c 8 27; done
else
  say "9B download not complete after 2 h: 9B configs skipped"
fi
for c in 38-27b-q4:20 38-27b-q6:26 36-35b-a3b-q4:24 9b-bf16:24 9b-q8:14 9b-q4:10 9b-q4mixed:11; do
  [ -s "results/docbench/${c%%:*}/outputs.jsonl" ] && speed "${c%%:*}" "${c##*:}"
done
$PY docbench/score.py --verdict > results/docbench/verdict.txt 2>&1
say "verdict written: $(head -1 results/docbench/verdict.txt)"
say "runner end"
