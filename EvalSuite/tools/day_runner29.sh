#!/bin/bash
# EvalSuite runner 29, 2026-10-30 (detached, from a live evalsuite session): R-56 LIVE16 (docs/PREREG_r56_2026-10-30.md).
# The arms run in the order given (tools/day_runner29.sh ref2 stock): each decodes the 36 prompts of results/r56/keys.txt
# in the 16-row live drafting loop (evalsuite/live16.py), in processes that start no new prompt after 9 min of running,
# each under one queued quiet lock (cli.gpu: memory gate 34 + 4 GB free at C 192, swap guard, keepalive, an 18-min stop).
# A process whose reference check fails stops the runner (exit 5: VOID), as does a harness error (exit 4). Then
# tools/r56.py -> results/r56.txt and the marker results/R56_DONE. Code pin: Leap's ref2.py and the kernel modules it
# imports, the drafter code and weights (size, mtime), ~/BigMoE/bigmoe, EvalSuite's evalsuite/, MLX / mlx_lm / mtplx; a
# change stops the runner (exit 7). No process while soft_end.sh ramps, PAUSE exists, the lid is closed with no external
# display, or < 34 GB is free. Deadline Fri 2026-10-30 07:10. Stop it by pid only.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day29_runner.log
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-30 07:10" +%s)
export PEER_QUEUE_TIMEOUT=14400
export BIGMOE_SCRATCH_KEEP=1                     # BigMoE's default since Thu 22:1x, set explicitly (live16 asserts it)
unset MTPLX_FUSED_GATE_UP BIGMOE_TIER BIGMOE_VERIFY_MIDLOOP
PY=.venv/bin/python
KEYS_SHA=cedc284e5f66                            # results/r56/keys.txt as pre-registered (sha256[:12])
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
left() { echo $(( EVALSUITE_DEADLINE - $(date +%s) )); }
nrows() { if [ -f "results/r56/$1.jsonl" ]; then wc -l < "results/r56/$1.jsonl" | tr -d ' '; else echo 0; fi; }
free_gb() { memory_pressure 2>/dev/null | tail -1 | sed 's/.*: *//; s/%//' | awk '{print int($1 * 64 / 100)}'; }
lid_shut() {   # lid closed and no external display (lid_guard.sh's test): research holds
  ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes || return 1
  (( $(system_profiler SPDisplaysDataType -detailLevel mini 2>/dev/null | grep -c "Resolution:") > 1 )) && return 1
  return 0
}
LASTWAIT=0
waitlog() { [ $(( $(date +%s) - LASTWAIT )) -ge 600 ] && { say "$*"; LASTWAIT=$(date +%s); }; }
calm() {
  while pgrep -f '^/bin/zsh [^ ]*soft_end\.sh' >/dev/null || [ -e "$HOME/.local/share/quiet/PAUSE" ] || lid_shut \
        || [ "$(free_gb)" -lt 34 ]; do
    waitlog "waiting: soft_end.sh, PAUSE, a closed lid or < 34 GB free ($(free_gb) GB)"
    sleep 30
  done
}
pin() {
  { (cd "$HOME/Leap" && shasum scripts/ref2.py scripts/leapk.py scripts/qmm16.py scripts/qmmg25.py scripts/gmr24.py \
       third_party/dflash_model_mlx.py && stat -f '%z %m' data/l120_ft.safetensors)
    (cd "$HOME/BigMoE" && find bigmoe -type f \( -name '*.py' -o -name '*.so' -o -name '*.metal' \) ! -path '*__pycache__*' \
       | sort | xargs shasum)
    find evalsuite -name '*.py' ! -path '*__pycache__*' | sort | xargs shasum
    ls "$HOME/BigMoE/.venv/lib/python3.12/site-packages" | grep -E '^(mlx|mlx_lm|mlx_metal|mtplx)-.*dist-info$'; } 2>&1 \
    | shasum | cut -c1-12
}
PIN0=$(pin)
checkpin() {
  local p; p=$(pin)
  [ "$p" = "$PIN0" ] || { say "code changed (Leap's ref2 / kernels / drafter, BigMoE's bigmoe, evalsuite, MLX / mtplx): pin $PIN0 -> $p: runner stops"; return 1; }
}
chunk() {   # chunk ARM: one process, one quiet lock
  local v=$1 rc b; b=$(nrows "$v")
  calm
  checkpin || exit 7
  [ "$(left)" -gt 1500 ] || { say "deadline near: $v not started: runner stops"; exit 6; }
  ./bin/evalsuite gpu live16 ref --arm "$v" --keys results/r56/keys.txt --chunk-minutes 9 --C 192 --minutes 18 >> "$LOG" 2>&1; rc=$?
  say "$v: chunk rc=$rc ($(nrows "$v")/36 rows)"
  if tail -4 "$LOG" | grep -q "can't tell which Claude session\|left the queue"; then
    say "the session that owns this runner's locks is gone: runner stops (relaunch it from a live session)"; exit 3
  fi
  if tail -4 "$LOG" | grep -q "swap guard"; then say "swap guard: runner stops"; exit 3; fi
  if [ -f results/r56/chunks.jsonl ] && tail -1 results/r56/chunks.jsonl | grep -q '"refcheck": {[^}]*"equal": false'; then
    say "$v: the reference row did not reproduce (VOID): runner stops"; exit 5
  fi
  if [ "$rc" != 0 ] && [ "$(nrows "$v")" -le "$b" ] && tail -40 results/gpu_stage.log | grep -q "Traceback"; then
    say "$v: the stage failed (results/gpu_stage.log): runner stops"; exit 4
  fi
  return 0
}
NOPROG=0
progress() {   # progress BEFORE AFTER: three chunks in a row without a new row stop the runner
  if [ "$2" -le "$1" ]; then
    NOPROG=$((NOPROG + 1)); [ "$NOPROG" -ge 3 ] && { say "three chunks without a new row: runner stops"; exit 3; }
    sleep 60
  else
    NOPROG=0
  fi
}

say "runner 29 start (pid $$); arms: $*; code pin $PIN0"
[ $# -ge 1 ] || { say "no arm given: runner stops"; exit 2; }
[ "$(shasum -a 256 results/r56/keys.txt | cut -c1-12)" = "$KEYS_SHA" ] || { say "results/r56/keys.txt is not the pre-registered list: runner stops"; exit 2; }
for v in "$@"; do
  while [ "$(nrows "$v")" -lt 36 ]; do
    b=$(nrows "$v")
    chunk "$v"
    progress "$b" "$(nrows "$v")"
  done
done
$PY tools/r56.py "$@" > results/r56.txt 2>&1
for v in "$@"; do say "$(grep -m1 "^$v: " results/r56.txt)"; done
touch results/R56_DONE
say "runner 29 end"
