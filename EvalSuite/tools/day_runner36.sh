#!/bin/bash
# EvalSuite runner 36, 2026-11-01 (detached, from a live evalsuite session): REF2c-T (docs/PREREG_ref2ct_2026-11-01.md).
# Part 1: all 210 core rows with variant ref2ct (snap/leap-91cd316, Leap's REF2c freeze: ref2.CFG att_single + gdn_inproj
# on REF2-T; BigMoE's rank tiers 4433333333 / 3c / C_l 112 through EVALSUITE_STAGE_ENV) at C 192; then the reports ref2ct
# vs ref2t (the verdict: tools/d4.py, decision 4's rule), vs ref2b and ref.r2 (information), and results/REF2CT_DONE.
# Part 2, only on a Part 1 PASS: R-56e, live16b arm ref2cto_gate16 (REF2c-T on the gate + OOO / fused combine) on R-56's
# 36 keys vs its own AR -> tools/r56b.py ref2cto_gate16 -> results/r56e.txt and results/R56E_DONE.
# Queues from 23:05 (NewIdeas' night order: BigMoE C45 -> MTP Head D-23 -> us; FIFO). One quiet ticket per 60-min chunk (amendment 3),
# re-queued at once; NO-SWAP >= peak + 4 GB after the grant (cli --need 34). BigMoE from the frozen snap/bigmoe-6c64148.
# Code pin (taken at launch): snap/leap-91cd316 (+ ms_gdn), BigMoE's snapshot, planes_engine's tierglu.py (the 3c kernel),
# evalsuite/, the verdict tools, MLX / mtplx; Part 2 adds Leap's drafter. A change stops the runner (exit 7). A swap trip
# waits 2 min (max 6); a harness error stops it (exit 4). Deadline Mon 2026-11-02 18:00 (amendment 2). Stop it by pid only, and its
# cli child too.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day36_runner.log
START=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-01 23:05" +%s)
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-02 18:00" +%s)
export PEER_QUEUE_TIMEOUT=21600
unset MTPLX_FUSED_GATE_UP MTPLX_FUSED_GDN_INPROJ BIGMOE_GDN_EXACT BIGMOE_TIER BIGMOE_TIER_GLU BIGMOE_TIER_CL \
  BIGMOE_TIER_COMBINE BIGMOE_OOO BIGMOE_VERIFY_MIDLOOP BIGMOE_EXT_BUILD BIGMOE_MAXK
export EVALSUITE_BIGMOE_CODE="$HOME/EvalSuite/snap/bigmoe-6c64148"
TIERENV="BIGMOE_TIER=4433333333,BIGMOE_TIER_GLU=3c,BIGMOE_TIER_CL=112"
LIVEENV="BIGMOE_MAXK=160,BIGMOE_EXT_BUILD=build_next,$TIERENV,BIGMOE_TIER_COMBINE=1,BIGMOE_OOO=1"
PY=.venv/bin/python
KEYS_SHA=cedc284e5f66                            # results/r56/keys.txt (R-56's 36 prompts)
DEC=results/ref2ct/decode.jsonl
LIVE=results/r56e/ref2cto_gate16.jsonl
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
left() { echo $(( EVALSUITE_DEADLINE - $(date +%s) )); }
count() { if [ -f "$1" ]; then wc -l < "$1" | tr -d ' '; else echo 0; fi; }
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
  { shasum snap/leap-91cd316/*.py snap/leap-91cd316/ms_gdn/*.py tools/d4.py tools/r56b.py
    (cd "$EVALSUITE_BIGMOE_CODE" && find bigmoe ext -type f ! -path '*__pycache__*' | sort | xargs shasum)
    shasum "$HOME/MoonShots/patches/planes_engine/tierglu.py"
    find evalsuite -name '*.py' ! -path '*__pycache__*' | sort | xargs shasum
    ls "$HOME/BigMoE/.venv/lib/python3.12/site-packages" | grep -E '^(mlx|mlx_lm|mlx_metal|mtplx)-.*dist-info$'; } 2>&1 \
    | shasum | cut -c1-12
}
drafter() { (cd "$HOME/Leap" && shasum third_party/dflash_model_mlx.py && stat -f '%z %m' data/l120_ft.safetensors) | shasum | cut -c1-12; }
SWAPTRIPS=0; NOPROG=0
after_chunk() {   # rc, rows before, rows now
  if tail -4 "$LOG" | grep -q "can't tell which Claude session\|left the queue"; then
    say "the session that owns this runner's locks is gone: runner stops (relaunch it from a live session)"; exit 3
  fi
  if tail -4 "$LOG" | grep -q "swap guard"; then
    SWAPTRIPS=$((SWAPTRIPS + 1)); [ "$SWAPTRIPS" -ge 6 ] && { say "swap guard: 6 trips: runner stops"; exit 3; }
    say "swap guard trip $SWAPTRIPS: waiting 2 min"; sleep 120; return
  fi
  if [ "$1" != 0 ] && [ "$3" -le "$2" ] && tail -40 results/gpu_stage.log | grep -q "Traceback"; then
    say "the stage failed (results/gpu_stage.log): runner stops"; exit 4
  fi
  if [ "$3" -le "$2" ]; then
    NOPROG=$((NOPROG + 1)); [ "$NOPROG" -ge 3 ] && { say "three chunks without a new row: runner stops"; exit 3; }
    sleep 60
  else
    NOPROG=0
  fi
}
PIN0=$(pin)
say "runner 36 start (pid $$); code pin $PIN0; queues from 23:05"
[ -n "$OLD_CLI" ] && { say "amendment 3 (60-min chunks): waiting for the previous runner's chunk (cli $OLD_CLI) to end"; while kill -0 "$OLD_CLI" 2>/dev/null; do sleep 10; done; }
while [ "$(date +%s)" -lt "$START" ]; do sleep 60; done
say "23:05: Part 1 (ref2ct decode, 210 rows)"
while [ "$(count $DEC)" -lt 210 ]; do
  b=$(count $DEC)
  calm
  [ "$(pin)" = "$PIN0" ] || { say "code changed: pin $PIN0 -> $(pin): runner stops"; exit 7; }
  [ "$(left)" -gt 5400 ] || { say "deadline near: runner stops at $b/210 rows"; exit 6; }
  EVALSUITE_STAGE_ENV="$TIERENV" ./bin/evalsuite gpu decode ref2ct --sets math,ifeval,humaneval --chunk-minutes 60 --C 192 \
    --need 34 --minutes 80 >> "$LOG" 2>&1; rc=$?
  n=$(count $DEC)
  say "ref2ct: chunk rc=$rc ($n/210 rows)"
  after_chunk "$rc" "$b" "$n"
done
./bin/evalsuite report --variant ref2ct --decode --ref ref2t > results/report_ref2ct_vs_ref2t.txt 2>&1
$PY tools/d4.py results/report_ref2ct_vs_ref2t.txt > results/ref2ct_d4.txt 2>&1; d4=$?
say "Part 1 verdict (decision 4, ref2ct vs ref2t): $(tail -1 results/ref2ct_d4.txt); standard table: $(tail -1 results/report_ref2ct_vs_ref2t.txt)"
./bin/evalsuite report --variant ref2ct --decode --ref ref2b > results/report_ref2ct_vs_ref2b.txt 2>&1
./bin/evalsuite report --variant ref2ct --decode --ref ref.r2 > results/report_ref2ct_vs_ref.r2.txt 2>&1
say "information: vs ref2b $(tail -1 results/report_ref2ct_vs_ref2b.txt); vs ref.r2 $(tail -1 results/report_ref2ct_vs_ref.r2.txt)"
touch results/REF2CT_DONE
[ "$d4" = 0 ] || { say "Part 1 is not a PASS (d4 rc $d4): no R-56e; runner 36 end"; exit 0; }
say "Part 2 (R-56e, ref2cto_gate16, 36 prompts)"
[ "$(shasum -a 256 results/r56/keys.txt | cut -c1-12)" = "$KEYS_SHA" ] || { say "results/r56/keys.txt changed: runner stops"; exit 2; }
DR0=$(drafter); NOPROG=0
while [ "$(count $LIVE)" -lt 36 ]; do
  b=$(count $LIVE)
  calm
  [ "$(pin)" = "$PIN0" ] && [ "$(drafter)" = "$DR0" ] || { say "code changed: pin $PIN0 -> $(pin), drafter $DR0 -> $(drafter): runner stops"; exit 7; }
  [ "$(left)" -gt 5400 ] || { say "deadline near: runner stops at $b/36 rows"; exit 6; }
  EVALSUITE_STAGE_ENV="$LIVEENV" ./bin/evalsuite gpu live16b ref --arm ref2cto_gate16 --keys results/r56/keys.txt \
    --chunk-minutes 60 --C 192 --need 34 --minutes 80 >> "$LOG" 2>&1; rc=$?
  n=$(count $LIVE)
  say "ref2cto_gate16: chunk rc=$rc ($n/36 rows)"
  after_chunk "$rc" "$b" "$n"
done
$PY tools/r56b.py ref2cto_gate16 > results/r56e.txt 2>&1
say "$(grep -m1 '^ref2cto_gate16: ' results/r56e.txt)"
touch results/R56E_DONE
say "runner 36 end"
