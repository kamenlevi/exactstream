#!/bin/bash
# EvalSuite runner 35, 2026-11-01 (detached, from a live evalsuite session): R-56d, REF2-T + OOO + fused combine
# (docs/PREREG_r56d_2026-11-01.md). Arm ref2to_gate16 on R-56's 36 keys: per prompt its own AR (OOO on) to max_tokens
# (results/r56d/ar/*.json), then the live loop, every token and every kept row bitwise vs that AR, the AR vs results/ref2t
# (R-58's rows). Processes start no new prompt after 9 min of running, one quiet ticket each (re-queued at once), NO-SWAP
# >= peak + 4 GB after the grant (cli --need 34: R-56c's peak 33.4 GB). Then tools/r56b.py ref2to_gate16 ->
# results/r56d.txt (+ the AR vs R-56c's AR, OOO engaged) and results/R56D_DONE.
# BigMoE from the frozen snap/bigmoe-6c64148 (ext/build_next for the gate, as R-56b's gate arm). Code pin: the two
# snapshots, MoonShots' planes_engine (3c), evalsuite/, Leap's drafter, MLX / mtplx; a change stops the runner (exit 7).
# A swap trip waits 2 min (max 6); a harness error stops it (exit 4). Deadline Sun 2026-11-01 14:00. Stop it by pid only,
# and its cli child too.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day35_runner.log
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-01 14:00" +%s)
export PEER_QUEUE_TIMEOUT=14400
unset MTPLX_FUSED_GATE_UP BIGMOE_TIER BIGMOE_TIER_GLU BIGMOE_TIER_CL BIGMOE_TIER_COMBINE BIGMOE_OOO BIGMOE_VERIFY_MIDLOOP BIGMOE_EXT_BUILD BIGMOE_MAXK
export EVALSUITE_BIGMOE_CODE="$HOME/EvalSuite/snap/bigmoe-6c64148"
STAGEENV="BIGMOE_MAXK=160,BIGMOE_EXT_BUILD=build_next,BIGMOE_TIER=4433333333,BIGMOE_TIER_GLU=3c,BIGMOE_TIER_CL=112,BIGMOE_TIER_COMBINE=1,BIGMOE_OOO=1"
PY=.venv/bin/python
KEYS_SHA=cedc284e5f66                            # results/r56/keys.txt (R-56's 36 prompts)
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
left() { echo $(( EVALSUITE_DEADLINE - $(date +%s) )); }
nrows() { if [ -f results/r56d/ref2to_gate16.jsonl ]; then wc -l < results/r56d/ref2to_gate16.jsonl | tr -d ' '; else echo 0; fi; }
lid_shut() {
  ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes || return 1
  (( $(system_profiler SPDisplaysDataType -detailLevel mini 2>/dev/null | grep -c "Resolution:") > 1 )) && return 1
  return 0
}
LASTWAIT=0
waitlog() { [ $(( $(date +%s) - LASTWAIT )) -ge 600 ] && { say "$*"; LASTWAIT=$(date +%s); }; }
calm() {   # queue first; cli.gpu gates memory (NO-SWAP) after the grant
  while pgrep -f '^/bin/zsh [^ ]*soft_end\.sh' >/dev/null || [ -e "$HOME/.local/share/quiet/PAUSE" ] || lid_shut; do
    waitlog "waiting: soft_end.sh, PAUSE or a closed lid"
    sleep 30
  done
}
pin() {
  { shasum snap/leap-faff8e5/*.py
    (cd "$EVALSUITE_BIGMOE_CODE" && find bigmoe ext -type f ! -path '*__pycache__*' | sort | xargs shasum)
    (cd "$HOME/MoonShots/patches/planes_engine" && ls *.py | sort | xargs shasum)
    find evalsuite -name '*.py' ! -path '*__pycache__*' | sort | xargs shasum
    (cd "$HOME/Leap" && shasum third_party/dflash_model_mlx.py && stat -f '%z %m' data/l120_ft.safetensors)
    ls "$HOME/BigMoE/.venv/lib/python3.12/site-packages" | grep -E '^(mlx|mlx_lm|mlx_metal|mtplx)-.*dist-info$'; } 2>&1 \
    | shasum | cut -c1-12
}
SWAPTRIPS=0; NOPROG=0
PIN0=$(pin)
say "runner 35 start (pid $$); code pin $PIN0"
[ "$(shasum -a 256 results/r56/keys.txt | cut -c1-12)" = "$KEYS_SHA" ] || { say "results/r56/keys.txt changed: runner stops"; exit 2; }
while [ "$(nrows)" -lt 36 ]; do
  b=$(nrows)
  calm
  [ "$(pin)" = "$PIN0" ] || { say "code changed: pin $PIN0 -> $(pin): runner stops"; exit 7; }
  [ "$(left)" -gt 1500 ] || { say "deadline near: runner stops at $b/36 rows"; exit 6; }
  EVALSUITE_STAGE_ENV="$STAGEENV" ./bin/evalsuite gpu live16b ref --arm ref2to_gate16 --keys results/r56/keys.txt \
    --chunk-minutes 9 --C 192 --need 34 --minutes 18 >> "$LOG" 2>&1; rc=$?
  say "ref2to_gate16: chunk rc=$rc ($(nrows)/36 rows)"
  if tail -4 "$LOG" | grep -q "can't tell which Claude session\|left the queue"; then
    say "the session that owns this runner's locks is gone: runner stops (relaunch it from a live session)"; exit 3
  fi
  if tail -4 "$LOG" | grep -q "swap guard"; then
    SWAPTRIPS=$((SWAPTRIPS + 1)); [ "$SWAPTRIPS" -ge 6 ] && { say "swap guard: 6 trips: runner stops"; exit 3; }
    say "swap guard trip $SWAPTRIPS: waiting 2 min"; sleep 120; continue
  fi
  if [ "$rc" != 0 ] && [ "$(nrows)" -le "$b" ] && tail -40 results/gpu_stage.log | grep -q "Traceback"; then
    say "the stage failed (results/gpu_stage.log): runner stops"; exit 4
  fi
  if [ "$(nrows)" -le "$b" ]; then
    NOPROG=$((NOPROG + 1)); [ "$NOPROG" -ge 3 ] && { say "three chunks without a new row: runner stops"; exit 3; }
    sleep 60
  else
    NOPROG=0
  fi
done
$PY tools/r56b.py ref2to_gate16 > results/r56d.txt 2>&1
say "$(grep -m1 '^ref2to_gate16: ' results/r56d.txt)"
touch results/R56D_DONE
say "runner 35 end"
