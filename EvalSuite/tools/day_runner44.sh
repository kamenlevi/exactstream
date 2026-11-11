#!/bin/bash
# EvalSuite runner 44, Thu 2026-11-05 (attached to a live evalsuite session): NAX6 Part 2, the live-loop exactness audit
# (docs/PREREG_nax6_2026-11-04.md, Part 2; the orchestrator 18:0x: right after runner 43, share). Arm ref2tnaxno_gate16
# (evalsuite/live16b.py): NAX6-N (variant ref2tnaxn: tier_nax_ooo + nax_native, on the CONVERTED store) + BigMoE's OOO and
# fused combine on the gate (MAXK 160; ext/build of snap/bigmoe-da8e7a6) on R-56's 36 keys. Per prompt: its own AR to
# max_tokens, then the live loop, every token and every kept row bitwise vs that AR, and the AR vs results/ref2tnax
# (NAX6's row-major Part 1 rows: NAX6-N == NAX6 at the model level). Rows -> results/r56n/; the verdict is the session's
# (tools/r56b.py, extended for the arm). It runs only after a NAX6 PASS (results/NAX6_DONE) with all 48 layers
# converted, once runner 43 has ended. Share lock per 9-min process ("MEM 34 GB"), the day rule's MEM wait, the code pin
# (exit 7), the store staying converted (exit 8), and no process starts unless start + its measured length ends by
# 16:00 (exit 6; the orchestrator Thu 07:5x: cutoffs bound a chunk's END). Stop it by pid only, and its cli child too.
# The by-day order (the orchestrator Thu 09:1x): Breakthrough's split code arms (~20 min) go between runner 43 and this
# one: it holds until results/q44/GO (Breakthrough touches it when they are done) or Q44_HOLD (default 14:50); each
# process still waits while another session's MEM job runs. "HH:MM" in results/q44/CUTOFF (<= 16:30) moves the END bound.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day44_runner.log
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-05 16:30" +%s)
export PEER_QUEUE_TIMEOUT=21600
unset MTPLX_FUSED_GATE_UP MTPLX_FUSED_GDN_INPROJ BIGMOE_GDN_EXACT BIGMOE_TIER BIGMOE_TIER_GLU BIGMOE_TIER_CL \
  BIGMOE_TIER_COMBINE BIGMOE_OOO BIGMOE_VERIFY_MIDLOOP BIGMOE_EXT_BUILD BIGMOE_MAXK BIGMOE_COMPILE_HC BIGMOE_HC_VREAD \
  BIGMOE_HC_PACK BIGMOE_NAX_NATIVE REF2_FATT_VERIFY REF2_FATT_R8 REF2_HC8_FAST REF2_HC8_V REF2_DENSE_NATIVE REF2_OOO_1C
export EVALSUITE_BIGMOE_CODE="$HOME/EvalSuite/snap/bigmoe-da8e7a6"
STAGEENV="BIGMOE_MAXK=160,BIGMOE_TIER=4433333333,BIGMOE_TIER_GLU=3c,BIGMOE_TIER_CL=112,BIGMOE_TIER_COMBINE=1,BIGMOE_OOO=1"
ARM=ref2tnaxno_gate16
KEYS=results/r56/keys.txt
R3IDX=$HOME/BigMoE/models/bare/records3/index.json
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
left() { echo $(( EVALSUITE_DEADLINE - $(date +%s) )); }
nrows() { [ -f results/r56n/$ARM.jsonl ] && wc -l < results/r56n/$ARM.jsonl | tr -d ' ' || echo 0; }
lid_shut() {
  ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes || return 1
  (( $(system_profiler SPDisplaysDataType -detailLevel mini 2>/dev/null | grep -c "Resolution:") > 1 )) && return 1
  return 0
}
LASTWAIT=0
waitlog() { [ $(( $(date +%s) - LASTWAIT )) -ge 600 ] && { say "$*"; LASTWAIT=$(date +%s); }; }
calm() {
  while pgrep -f '^/bin/zsh [^ ]*soft_end\.sh' >/dev/null || [ -e "$HOME/.local/share/quiet/PAUSE" ] || lid_shut; do
    waitlog "waiting: soft_end.sh, PAUSE or a closed lid"
    sleep 30
  done
}
full_converted() {   # every one of the 48 layers carries "layout", and index.json has not changed for 2 min
  [ $(( $(date +%s) - $(stat -f %m "$R3IDX") )) -ge 120 ] || return 1
  python3 -c 'import json, sys
L = json.load(open(sys.argv[1]))["layers"]
sys.exit(0 if len(L) == 48 and all("layout" in v for v in L.values()) else 1)' "$R3IDX"
}
mem_busy() {    # another session's shared run with MEM > 5 GB (our own reasons start "evalsuite: MEM")
  peer ls 2>/dev/null | grep "shared run" | grep -v "evalsuite: MEM" | grep -oE "MEM [0-9]+ GB" | awk '$2 > 5 {f = 1} END {exit !f}'
}
pin() {
  { shasum snap/leap-52cf2c9/*.py snap/leap-52cf2c9/ms_gdn/* "$KEYS"
    (cd "$EVALSUITE_BIGMOE_CODE" && find bigmoe ext -type f ! -path '*__pycache__*' | sort | xargs shasum)
    (cd "$HOME/MoonShots/patches/planes_engine" && ls *.py | sort | xargs shasum)
    find evalsuite -name '*.py' ! -path '*__pycache__*' | sort | xargs shasum
    (cd "$HOME/Leap" && shasum third_party/dflash_model_mlx.py && stat -f '%z %m' data/l120_ft.safetensors)
    ls "$HOME/BigMoE/.venv/lib/python3.12/site-packages" | grep -E '^(mlx|mlx_lm|mlx_metal|mtplx)-.*dist-info$'; } 2>&1 \
    | shasum | cut -c1-12
}
alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
CUTOFF0_HM=16:00                                  # processes END by it (the orchestrator Thu 07:5x), or results/q44/CUTOFF's
cutoff_hm() { local h; h=$(head -c 5 results/q44/CUTOFF 2>/dev/null)
  [[ "$h" =~ ^(0[0-9]|1[0-6]):[0-5][0-9]$ ]] && [[ ! "$h" > 16:30 ]] && echo "$h" || echo "$CUTOFF0_HM"; }
cutoff() { date -j -f "%Y-%m-%d %H:%M" "2026-11-05 $(cutoff_hm)" +%s; }
HOLD_HM=${Q44_HOLD:-14:50}; HOLD_AT=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-05 $HOLD_HM" +%s)
EST=1200                                          # a 9-min process + its load and last prompt, until measured
freegb() { .venv/bin/python -c "import sys; sys.path.insert(0, '.'); from evalsuite.cli import free_gb; print(int(free_gb()))"; }
pick_c() {      # the gate needs C >= MAXK 160: C 192 (need 34) or C 160 (need 30) by free memory, else wait (no lock held)
  local fg w=0
  while :; do
    fg=$(freegb)
    if [ "$fg" -ge 38 ]; then C=192; NEED=34; return 0; fi
    if [ "$fg" -ge 34 ]; then C=160; NEED=30; return 0; fi
    [ $((w % 10)) = 0 ] && say "waiting for memory: $fg GB NO-SWAP (C 160 needs 34)"
    w=$((w + 1)); sleep 60; calm
    [ "$(left)" -gt 900 ] || { say "deadline near while waiting for memory: runner stops"; exit 6; }
  done
}
SWAPTRIPS=0; NOPROG=0
PIN0=$(pin)
echo $$ > results/q44.pid; trap 'rm -f results/q44.pid' EXIT
say "runner 44 start (pid $$); code pin $PIN0; waits for NAX6_DONE (PASS), runner 43's end and a converted store"
while [ ! -e results/NAX6_DONE ] || alive results/q42.pid || alive results/q43.pid; do
  if [ ! -e results/NAX6_DONE ] && ! alive results/q42.pid; then
    sleep 5; [ -e results/NAX6_DONE ] || { say "runner 42 is gone without NAX6_DONE: Part 2 does not run"; exit 3; }
  fi
  sleep 60
done
case "$(cat results/NAX6_DONE)" in
  "D4X: PASS"*|"D4X: DIFFERS"*) ;;
  *) say "NAX6 '$(cat results/NAX6_DONE)': no PASS, so no Part 2"; exit 0 ;;
esac
w=0
until full_converted; do
  [ $((w % 10)) = 0 ] && say "waiting for a fully converted store (Part 2 runs NAX6-N)"; w=$((w + 1)); sleep 60
  [ "$(left)" -gt 3600 ] || { say "no converted store before the deadline: Part 2 waits for a new launch"; exit 6; }
done
[ -e results/q44/GO ] || [ "$(date +%s)" -ge "$HOLD_AT" ] || say "holding for Breakthrough's split code arms: until results/q44/GO or $HOLD_HM"
while [ ! -e results/q44/GO ] && [ "$(date +%s)" -lt "$HOLD_AT" ]; do sleep 30; done
say "hold over: $([ -e results/q44/GO ] && echo "results/q44/GO exists" || echo "the $HOLD_HM cap")"
say "go: $ARM on $KEYS ($(grep -c . "$KEYS") prompts), share, C 192 or 160 by free memory"
while [ "$(nrows)" -lt 36 ]; do
  b=$(nrows)
  calm
  w=0
  while mem_busy; do
    [ $((w % 10)) = 0 ] && say "another session's MEM job holds a shared lock: waiting (day rule)"
    w=$((w + 1)); sleep 60; calm
  done
  [ "$(pin)" = "$PIN0" ] || { say "code changed: pin $PIN0 -> $(pin): runner stops"; exit 7; }
  full_converted || { say "records3 is not fully converted any more: runner stops at $b/36 rows"; exit 8; }
  [ $(( $(date +%s) + EST )) -le "$(cutoff)" ] || { say "no process fits before $(cutoff_hm) (measured $((EST / 60)) min): runner stops at $b/36 rows"; exit 6; }
  pick_c
  t0=$(date +%s)
  EVALSUITE_STAGE_ENV="$STAGEENV" ./bin/evalsuite gpu live16b ref --arm "$ARM" --keys "$KEYS" \
    --chunk-minutes 9 --C "$C" --need "$NEED" --minutes 18 --share >> "$LOG" 2>&1; rc=$?
  d=$(( $(date +%s) - t0 )); [ "$d" -gt "$EST" ] && EST=$d    # the longest process so far bounds the next start
  say "$ARM: chunk rc=$rc ($(nrows)/36 rows, $((d / 60)) min)"
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
touch results/PART2_ROWS_DONE
say "runner 44 end: 36/36 rows (the verdict: tools/r56b.py $ARM)"
