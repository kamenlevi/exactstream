#!/bin/bash
# EvalSuite runner 32, 2026-10-31 (detached, from a live evalsuite session): R-58, REF2-T's same-scores test vs REF2b
# (docs/PREREG_r58_2026-10-31.md). Waits while runner 31 runs. Decodes all 210 core rows with variant ref2t (REF2b code
# snap/leap-faff8e5 + BigMoE's rank tiers 4433333333 / 3c / C_l 112, set through EVALSUITE_STAGE_ENV) at C 192, one quiet
# ticket per chunk (no new row after 12 min; night mode: re-queued at once), NO-SWAP >= peak + 4 GB after the grant
# (cli --need 37). Then the reports ref2t vs ref2b (the verdict) and vs ref.r2 (information), and results/R58_DONE.
# BigMoE from the frozen snap/bigmoe-6c64148. Code pin: the two snapshots, MoonShots' patches/planes_engine (the 3c
# kernel), evalsuite/, MLX / mlx_lm / mtplx; a change stops the runner (exit 7). A swap-guard trip waits 2 min (max 6);
# a harness error stops it (exit 4). Deadline Sat 2026-10-31 12:00. Stop it by pid only, and its cli child too.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day32_runner.log
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-31 12:00" +%s)
export PEER_QUEUE_TIMEOUT=14400
unset MTPLX_FUSED_GATE_UP BIGMOE_TIER BIGMOE_TIER_GLU BIGMOE_TIER_CL BIGMOE_TIER_COMBINE BIGMOE_VERIFY_MIDLOOP BIGMOE_EXT_BUILD BIGMOE_MAXK
export EVALSUITE_BIGMOE_CODE="$HOME/EvalSuite/snap/bigmoe-6c64148"
TIERENV="BIGMOE_TIER=4433333333,BIGMOE_TIER_GLU=3c,BIGMOE_TIER_CL=112"
PY=.venv/bin/python
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
left() { echo $(( EVALSUITE_DEADLINE - $(date +%s) )); }
nrows() { if [ -f results/ref2t/decode.jsonl ]; then wc -l < results/ref2t/decode.jsonl | tr -d ' '; else echo 0; fi; }
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
    ls "$HOME/BigMoE/.venv/lib/python3.12/site-packages" | grep -E '^(mlx|mlx_lm|mlx_metal|mtplx)-.*dist-info$'; } 2>&1 \
    | shasum | cut -c1-12
}
SWAPTRIPS=0; NOPROG=0
say "runner 32 start (pid $$): waiting while runner 31 runs"
while pgrep -f 'tools/day_runner31\.sh' >/dev/null; do sleep 60; done
PIN0=$(pin)
say "code pin $PIN0"
while [ "$(nrows)" -lt 210 ]; do
  b=$(nrows)
  calm
  [ "$(pin)" = "$PIN0" ] || { say "code changed: pin $PIN0 -> $(pin): runner stops"; exit 7; }
  [ "$(left)" -gt 1500 ] || { say "deadline near: runner stops at $b/210 rows"; exit 6; }
  EVALSUITE_STAGE_ENV="$TIERENV" ./bin/evalsuite gpu decode ref2t --sets math,ifeval,humaneval --chunk-minutes 12 --C 192 \
    --need 37 --minutes 17 >> "$LOG" 2>&1; rc=$?
  say "ref2t: chunk rc=$rc ($(nrows)/210 rows)"
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
./bin/evalsuite report --variant ref2t --decode --ref ref2b > results/report_ref2t_vs_ref2b.txt 2>&1
say "report ref2t vs ref2b (verdict): $(tail -1 results/report_ref2t_vs_ref2b.txt)"
./bin/evalsuite report --variant ref2t --decode --ref ref.r2 > results/report_ref2t_vs_ref.r2.txt 2>&1
say "report ref2t vs ref.r2 (information): $(tail -1 results/report_ref2t_vs_ref.r2.txt)"
touch results/R58_DONE
say "runner 32 end"
