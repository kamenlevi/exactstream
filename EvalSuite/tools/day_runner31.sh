#!/bin/bash
# EvalSuite runner 31, 2026-10-30 (detached, from a live evalsuite session): R-56 on REF2b, two arms
# (docs/PREREG_r56b_2026-10-30.md). Waits for results/REF2B_DONE (runner 30: the REF2b reference rows + decision 4's
# test); stops if runner 30 ends without it. Then arm ref2b on the 36 keys of results/r56/keys.txt (per prompt REF2b's
# AR to max_tokens, saved as results/r56b/ar/*.json, then the live loop: every token and every kept row bitwise), then
# arm ref2b_gate16 (BIGMOE_MAXK=160, BIGMOE_EXT_BUILD=build_next; the loop only, against the saved AR). Processes start
# no new prompt after 9 min of running, each under one queued quiet lock (cli.gpu: memory gate 36 + 4 GB, swap guard,
# 18-min stop); a harness error stops the runner (exit 4). Then tools/r56b.py -> results/r56b.txt and results/R56B_DONE.
# BigMoE from the frozen snap/bigmoe-6c64148. Code pin: the two snapshots, evalsuite/, Leap's dflash_model_mlx.py and
# l120_ft, MLX / mlx_lm / mtplx; a change stops the runner (exit 7). No process while soft_end.sh ramps, PAUSE exists,
# the lid is closed with no external display, or < 34 GB is free. Deadline Sat 2026-10-31 08:00. Stop it by pid only,
# and its cli child too.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day31_runner.log
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-31 08:00" +%s)
export PEER_QUEUE_TIMEOUT=14400
unset MTPLX_FUSED_GATE_UP BIGMOE_TIER BIGMOE_VERIFY_MIDLOOP BIGMOE_EXT_BUILD BIGMOE_MAXK
export EVALSUITE_BIGMOE_CODE="$HOME/EvalSuite/snap/bigmoe-6c64148"
PY=.venv/bin/python
KEYS_SHA=cedc284e5f66                            # results/r56/keys.txt (R-56's 36 prompts)
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
left() { echo $(( EVALSUITE_DEADLINE - $(date +%s) )); }
nrows() { if [ -f "results/r56b/$1.jsonl" ]; then wc -l < "results/r56b/$1.jsonl" | tr -d ' '; else echo 0; fi; }
free_gb() {   # NO-SWAP: free + speculative + purgeable + file-backed pages (vm_stat; orchestrator 10-30 23:5x, RULES.md:
  vm_stat | awk '/page size of/ {ps = $8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v = $NF; sub("[.]", "", v); s += v} END {printf "%d", s * ps / 1e9}'
}              # 'inactive' holds anonymous pages only swap can free; shared lock: peak + 8 GB)
SWAPTRIPS=0
lid_shut() {
  ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes || return 1
  (( $(system_profiler SPDisplaysDataType -detailLevel mini 2>/dev/null | grep -c "Resolution:") > 1 )) && return 1
  return 0
}
LASTWAIT=0
waitlog() { [ $(( $(date +%s) - LASTWAIT )) -ge 600 ] && { say "$*"; LASTWAIT=$(date +%s); }; }
calm() {
  while pgrep -f '^/bin/zsh [^ ]*soft_end\.sh' >/dev/null || [ -e "$HOME/.local/share/quiet/PAUSE" ] || lid_shut \
        || [ "$(free_gb)" -lt 0 ]; do                   # amendment 5: queue first; cli gates memory after the grant
    waitlog "waiting: soft_end.sh, PAUSE, a closed lid or < 46 GB no-swap ($(free_gb) GB)"
    sleep 30
  done
}
pin() {
  { shasum snap/leap-faff8e5/*.py
    (cd "$EVALSUITE_BIGMOE_CODE" && find bigmoe ext -type f ! -path '*__pycache__*' | sort | xargs shasum)
    find evalsuite -name '*.py' ! -path '*__pycache__*' | sort | xargs shasum
    (cd "$HOME/Leap" && shasum third_party/dflash_model_mlx.py && stat -f '%z %m' data/l120_ft.safetensors)
    ls "$HOME/BigMoE/.venv/lib/python3.12/site-packages" | grep -E '^(mlx|mlx_lm|mlx_metal|mtplx)-.*dist-info$'; } 2>&1 \
    | shasum | cut -c1-12
}
checkpin() {
  local p; p=$(pin)
  [ "$p" = "$PIN0" ] || { say "code changed (the snapshots, evalsuite, Leap's drafter, MLX / mtplx): pin $PIN0 -> $p: runner stops"; return 1; }
}
chunk() {   # chunk ARM [ENV=VALUE...]: one process, one quiet lock
  local v=$1 rc b; shift; b=$(nrows "$v")
  calm
  checkpin || exit 7
  [ "$(left)" -gt 1500 ] || { say "deadline near: $v not started: runner stops"; exit 6; }
  env "$@" ./bin/evalsuite gpu live16b ref --arm "$v" --keys results/r56/keys.txt --chunk-minutes 9 --C 192 --need 38 \
    --minutes 18 >> "$LOG" 2>&1; rc=$?   # amendment 5: a quiet ticket per chunk (FIFO), NO-SWAP >= peak + 4 after the grant
  say "$v: chunk rc=$rc ($(nrows "$v")/36 rows)"
  if [ "$rc" = 75 ]; then say "$v: yielded to a quiet ticket: asking again in 2 min"; sleep 120; YIELDED=1; return 0; fi
  if tail -4 "$LOG" | grep -q "can't tell which Claude session\|left the queue"; then
    say "the session that owns this runner's locks is gone: runner stops (relaunch it from a live session)"; exit 3
  fi
  if tail -4 "$LOG" | grep -q "swap guard"; then   # the stage was stopped; wait, then the gates again (amendment 3)
    SWAPTRIPS=$((SWAPTRIPS + 1)); [ "$SWAPTRIPS" -ge 6 ] && { say "swap guard: 6 trips: runner stops"; exit 3; }
    say "$v: swap guard trip $SWAPTRIPS: waiting 2 min, then the memory gate again"; sleep 120; YIELDED=1; return 0
  fi
  if [ "$rc" != 0 ] && [ "$(nrows "$v")" -le "$b" ] && tail -40 results/gpu_stage.log | grep -q "Traceback"; then
    say "$v: the stage failed (results/gpu_stage.log): runner stops"; exit 4
  fi
  return 0
}
NOPROG=0
progress() {
  if [ "${YIELDED:-0}" = 1 ]; then YIELDED=0; return 0; fi
  if [ "$2" -le "$1" ]; then
    NOPROG=$((NOPROG + 1)); [ "$NOPROG" -ge 3 ] && { say "three chunks without a new row: runner stops"; exit 3; }
    sleep 60
  else
    NOPROG=0
  fi
}

say "runner 31 start (pid $$): waiting for results/REF2B_DONE"
[ "$(shasum -a 256 results/r56/keys.txt | cut -c1-12)" = "$KEYS_SHA" ] || { say "results/r56/keys.txt changed: runner stops"; exit 2; }
while [ ! -e results/REF2B_DONE ]; do
  pgrep -f 'tools/day_runner30\.sh' >/dev/null || { sleep 5; [ -e results/REF2B_DONE ] || { say "runner 30 ended without REF2B_DONE: runner stops"; exit 2; }; }
  [ "$(left)" -gt 1500 ] || { say "deadline near before REF2B_DONE: runner stops"; exit 6; }
  sleep 60
done
PIN0=$(pin)
say "REF2B_DONE seen; code pin $PIN0"
while [ "$(nrows ref2b)" -lt 36 ]; do b=$(nrows ref2b); chunk ref2b; progress "$b" "$(nrows ref2b)"; done
while [ "$(nrows ref2b_gate16)" -lt 36 ]; do
  b=$(nrows ref2b_gate16); chunk ref2b_gate16 EVALSUITE_STAGE_ENV=BIGMOE_MAXK=160,BIGMOE_EXT_BUILD=build_next; progress "$b" "$(nrows ref2b_gate16)"
done
$PY tools/r56b.py > results/r56b.txt 2>&1
for v in ref2b ref2b_gate16; do say "$(grep -m1 "^$v: " results/r56b.txt)"; done
touch results/R56B_DONE
say "runner 31 end"
