#!/bin/bash
# EvalSuite runner 43, Thu 2026-11-05 (attached to a live evalsuite session): the FAST batch F1 + r8 + HC8-1P on the
# base NAX6's verdict sets (docs/PREREG_f1_2026-11-03.md revision 3).
# It waits for runner 42 to finish. results/NAX6_DONE's line picks the base:
# - D4X PASS or DIFFERS: the NAX6 base (control ref2tnax, batch ref2tnfh, core reference ref2tnax);
# - FAIL: REF2-T's (control ref2tf0, batch ref2tfh, core reference ref2t, stage-2 reference ref2t.r2);
# - VOID or no file once runner 42 is gone: it stops.
# The store (the orchestrator 18:0x): on a NAX6 PASS BigMoE converts records3 in place right after runner 42 (which
# ends with the records3 check 1). Q43_STORE (converted | rowmajor, set at launch) says where the NAX6-base batch runs.
# converted = wait until all 48 layers carry "layout" and the index has not moved for 2 min, then the records3 check 2
# (ref2tf0 on the converted store == check 1) and the NAX6-N bridge ($NAXC.r1 on the 3 step-0 keys == ref2tnax's
# Part 1 rows); either failing stops it (exit 5). rowmajor = run now; BigMoE converts after this runner. On REF2-T's
# base (NAX6 FAIL) there is no conversion: row-major. Every chunk checks the store is still as expected (exit 8).
# Order: the control decodes RULER 24 + agent 29; the batch decodes RULER + agent, then the core 210; tools/d4x.py
# gives the batch's core lines (results/f1_d4x.txt). A stage 2 runs here only if it fits before 14:00 and the base's
# IFX rows exist. The verdict (RULER line, agent flags, copy breaks, gates) is tools/f1_report.py's, run by the
# session. AR at C 192, share locks ("MEM 36 GB"), the day rule's MEM wait, the code
# pin (exit 7), the store's layout (exit 8), and no chunk starts unless start + its measured length ends by 14:15 (exit 6;
# the orchestrator Thu 07:5x: cutoffs bound a chunk's END). Stop it by pid only, and its cli child too.
# The by-day order (the orchestrator Thu 09:1x, 09:2x): after the conversion, Breakthrough's split arms and MTP Head's
# PORT check (CPU, MEM 12 GB) run first, one at a time. Before its first GPU job the runner holds until results/q43/GO
# or Q43_HOLD (default 11:10), then until no other session's MEM job has run for 3 min straight (a gap between the two
# short jobs does not let it in). "HH:MM" in results/q43/CUTOFF (<= 15:00) moves the END bound without a relaunch.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day43_runner.log
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-05 15:00" +%s)
export PEER_QUEUE_TIMEOUT=21600
unset MTPLX_FUSED_GATE_UP MTPLX_FUSED_GDN_INPROJ BIGMOE_GDN_EXACT BIGMOE_TIER BIGMOE_TIER_GLU BIGMOE_TIER_CL \
  BIGMOE_TIER_COMBINE BIGMOE_OOO BIGMOE_VERIFY_MIDLOOP BIGMOE_EXT_BUILD BIGMOE_MAXK BIGMOE_COMPILE_HC BIGMOE_HC_VREAD \
  BIGMOE_HC_PACK BIGMOE_NAX_NATIVE REF2_FATT_VERIFY REF2_FATT_R8 REF2_HC8_FAST REF2_HC8_V REF2_DENSE_NATIVE REF2_OOO_1C
export EVALSUITE_BIGMOE_CODE="$HOME/EvalSuite/snap/bigmoe-da8e7a6"
TIERENV="BIGMOE_TIER=4433333333,BIGMOE_TIER_GLU=3c,BIGMOE_TIER_CL=112"
PY=.venv/bin/python
C=192; NEED=$(( (131 * C + 10200 + 999) / 1000 ))   # per chunk: pick_c (Wed 22:5x: C 192 / 160 / 128 by free memory)
R3IDX=$HOME/BigMoE/models/bare/records3/index.json
STORE=${Q43_STORE:-rowmajor}                     # where the NAX6-base batch runs (see above)
NAXC=${Q43_NAXC:-ref2tnax}; NAXB=${Q43_NAXB:-ref2tnfh}   # the NAX6-base control / batch on that store
MODE=rowmajor
HOLD_HM=${Q43_HOLD:-11:10}; HOLD_AT=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-05 $HOLD_HM" +%s)
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
left() { echo $(( EVALSUITE_DEADLINE - $(date +%s) )); }
count() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
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
converted() {   # any layer with a "layout" entry, or a record size other than row-major records3's 2,932,736 B
  grep -q '"layout"' "$R3IDX" || [ "$(grep -o '"recsz": [0-9]*' "$R3IDX" | sort -u)" != '"recsz": 2932736' ]
}
full_converted() {   # every one of the 48 layers carries "layout", and index.json has not changed for 2 min
  [ $(( $(date +%s) - $(stat -f %m "$R3IDX") )) -ge 120 ] || return 1
  python3 -c 'import json, sys
L = json.load(open(sys.argv[1]))["layers"]
sys.exit(0 if len(L) == 48 and all("layout" in v for v in L.values()) else 1)' "$R3IDX"
}
store_ok() { if [ "$MODE" = converted ]; then full_converted; else ! converted; fi; }
mem_busy() {    # another session's shared run with MEM > 5 GB (our own reasons start "evalsuite: MEM")
  peer ls 2>/dev/null | grep "shared run" | grep -v "evalsuite: MEM" | grep -oE "MEM [0-9]+ GB" | awk '$2 > 5 {f = 1} END {exit !f}'
}
pin() {
  { shasum snap/leap-52cf2c9/*.py snap/leap-52cf2c9/ms_gdn/* tools/d4x.py tools/r3check.sh results/ifx/stage1_keys.txt \
      results/r3check_keys.txt results/q42/step0_keys.txt data/items/dec_math.jsonl data/items/dec_ifeval.jsonl data/items/dec_humaneval.jsonl \
      data/items/dec_ifevalx.jsonl data/items/dec_ruler.jsonl data/items/dec_agent.jsonl
    (cd "$EVALSUITE_BIGMOE_CODE" && find bigmoe ext -type f ! -path '*__pycache__*' | sort | xargs shasum)
    (cd "$HOME/MoonShots/patches/planes_engine" && ls *.py | sort | xargs shasum)
    find evalsuite -name '*.py' ! -path '*__pycache__*' | sort | xargs shasum
    ls "$HOME/BigMoE/.venv/lib/python3.12/site-packages" | grep -E '^(mlx|mlx_lm|mlx_metal|mtplx)-.*dist-info$'; } 2>&1 \
    | shasum | cut -c1-12
}
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
freegb() { .venv/bin/python -c "import sys; sys.path.insert(0, '.'); from evalsuite.cli import free_gb; print(int(free_gb()))"; }
pick_c() {      # C by free memory (tokens do not depend on C): the largest of 192 / 160 / 128 whose cli gate (need + 4) fits;
  local fg w=0  # waits (no lock held) while none fits; sets C and NEED
  while :; do
    fg=$(freegb)
    for C in 192 160 128; do
      NEED=$(( (131 * C + 10200 + 999) / 1000 ))
      [ "$fg" -ge $(( NEED + 4 )) ] && return 0
    done
    [ $((w % 10)) = 0 ] && say "waiting for memory: $fg GB NO-SWAP (C 128 needs $(( (131 * 128 + 10200 + 999) / 1000 + 4 )))"
    w=$((w + 1)); sleep 60; calm
    [ "$(left)" -gt 3600 ] || { say "deadline near while waiting for memory: runner stops"; exit 6; }
  done
}
CUTOFF0_HM=14:15                                  # chunks END by it (the orchestrator Thu 07:5x), or results/q43/CUTOFF's
cutoff_hm() { local h; h=$(head -c 5 results/q43/CUTOFF 2>/dev/null)
  [[ "$h" =~ ^(0[0-9]|1[0-5]):[0-5][0-9]$ ]] && [[ ! "$h" > 15:00 ]] && echo "$h" || echo "$CUTOFF0_HM"; }
cutoff() { date -j -f "%Y-%m-%d %H:%M" "2026-11-05 $(cutoff_hm)" +%s; }
EST=3000                                          # a 45-min chunk + load and its last row, until measured
fits() { [ $(( $(date +%s) + EST )) -le "$(cutoff)" ]; }
hold() {   # GO file, cap (epoch), why: wait for the file or the cap, then until no other session's MEM job (day rule) has
  local w=0 q=0   # run for 3 checks in a row, 1 min apart
  [ -e "$1" ] || [ "$(date +%s)" -ge "$2" ] || say "holding for $3: until $1 or $(date -r "$2" +%H:%M)"
  while [ ! -e "$1" ] && [ "$(date +%s)" -lt "$2" ]; do sleep 30; done
  say "hold over: $([ -e "$1" ] && echo "$1 exists" || echo "the $(date -r "$2" +%H:%M) cap"); then 3 min without another MEM job"
  while [ "$q" -lt 3 ]; do
    if mem_busy; then
      q=0; [ $((w % 10)) = 0 ] && say "another session's MEM job holds a shared lock: waiting (day rule)"; w=$((w + 1))
    else
      q=$((q + 1))
    fi
    [ "$q" -lt 3 ] && { sleep 60; calm; }
  done
  say "no other MEM job for 3 min: go"
}
run_rows() {   # run, rows wanted, stage args...: chunks under a share lock until the run has that many rows
  local run=$1 want=$2 b n rc w; shift 2
  NOPROG=0
  while [ "$(count results/$run/decode.jsonl)" -lt "$want" ]; do
    b=$(count results/$run/decode.jsonl)
    calm
    w=0
    while mem_busy; do
      [ $((w % 10)) = 0 ] && say "another session's MEM job holds a shared lock: waiting (day rule)"
      w=$((w + 1)); sleep 60; calm
    done
    [ "$(pin)" = "$PIN0" ] || { say "code changed: pin $PIN0 -> $(pin): runner stops"; exit 7; }
    store_ok || { say "records3 is not $MODE as this run needs (index.json): runner stops at $run $b/$want rows"; exit 8; }
    fits || { say "no chunk fits before $(cutoff_hm) (measured chunk $((EST / 60)) min): runner stops at $run $b/$want rows"; exit 6; }
    pick_c
    say "$run: chunk at C $C (--need $NEED; $(freegb) GB NO-SWAP)"
    t0=$(date +%s)
    EVALSUITE_STAGE_ENV="$TIERENV" ./bin/evalsuite gpu decode $run "$@" --chunk-minutes 45 --share --C "$C" \
      --need "$NEED" --minutes 60 >> "$LOG" 2>&1; rc=$?
    d=$(( $(date +%s) - t0 )); [ "$d" -gt "$EST" ] && EST=$d     # the longest chunk so far bounds the next start
    n=$(count results/$run/decode.jsonl)
    say "$run: chunk rc=$rc ($n/$want rows, $((d / 60)) min)"
    after_chunk "$rc" "$b" "$n"
  done
}
alive42() { [ -f results/q42.pid ] && kill -0 "$(cat results/q42.pid)" 2>/dev/null; }
PIN0=$(pin)
echo $$ > results/q43.pid; trap 'rm -f results/q43.pid' EXIT
say "runner 43 start (pid $$); code pin $PIN0; waits for results/NAX6_DONE (runner 42)"
while [ ! -e results/NAX6_DONE ]; do
  alive42 || { sleep 5; [ -e results/NAX6_DONE ] || { say "runner 42 is gone without NAX6_DONE: runner 43 stops"; exit 3; }; }
  sleep 60
done
while alive42; do sleep 30; done                  # runner 42 still writes its reports
V=$(cat results/NAX6_DONE)
case "$V" in
  "D4X: PASS"*|"D4X: DIFFERS"*) BASE=nax6; CTRL=$NAXC; BATCH=$NAXB; CORE=ref2tnax; IFXREF=ref2tnax; MODE=$STORE ;;
  "D4X: FAIL"*) BASE=ref2t; CTRL=ref2tf0; BATCH=ref2tfh; CORE=ref2t; IFXREF=ref2t.r2 ;;
  *) say "NAX6: '$V': no valid base for the batch: runner 43 stops"; exit 5 ;;
esac
say "NAX6 '$V': the batch runs on the $BASE base (control $CTRL, batch $BATCH, core reference $CORE), store $MODE"
if [ "$MODE" = converted ]; then
  w=0
  until full_converted; do
    [ $((w % 10)) = 0 ] && say "waiting for BigMoE's conversion (all 48 layers, index still for 2 min)"
    w=$((w + 1)); sleep 60
  done
  say "records3 is converted"
fi
hold results/q43/GO "$HOLD_AT" "the split arms + PORT (the orchestrator's by-day order)"
if [ "$MODE" = converted ]; then
  say "check 2: tools/r3check.sh snap/bigmoe-da8e7a6 ref2tf0.r2 converted ref2tf0.r1,ref2t.r2"
  pick_c; R3_C=$C tools/r3check.sh snap/bigmoe-da8e7a6 ref2tf0.r2 converted ref2tf0.r1,ref2t.r2 >> "$LOG" 2>&1; c2=$?
  echo "check 2 rc $c2 ($(date +%H:%M))" > results/CHECK2_DONE
  [ $c2 = 0 ] || { say "check 2 rc $c2: ref2tf0 on the converted store != check 1: runner stops (VOID there)"; exit 5; }
  say "check 2 PASS; the NAX6-N bridge: $CTRL.r1 on the step-0 keys vs ref2tnax's Part 1 rows"
  run_rows "$CTRL.r1" 3 --sets ifeval,humaneval --keys results/q42/step0_keys.txt
  $PY - "$CTRL.r1" <<'BRIDGE' >> "$LOG" 2>&1; br=$?
import json, sys
keys = open("results/q42/step0_keys.txt").read().split()
load = lambda v: {r["key"]: r for r in map(json.loads, open(f"results/{v}/decode.jsonl"))}
a, b = load(sys.argv[1]), load("ref2tnax")
bad = [k for k in keys if a[k]["completion_ids"] != b[k]["completion_ids"]]
print(f"NAX6-N bridge: {sys.argv[1]} (converted store) vs ref2tnax (row-major, Part 1): "
      + (f"DIFFERENT on {bad}" if bad else "token-identical"))
sys.exit(5 if bad else 0)
BRIDGE
  [ $br = 0 ] || { say "NAX6-N bridge failed: runner stops (VOID for converted-store runs)"; exit 5; }
fi
n0=$(count results/$CTRL/decode.jsonl)
run_rows "$CTRL" $((n0 + 53)) --sets ruler,agent
run_rows "$BATCH" 53 --sets ruler,agent
run_rows "$BATCH" 263 --sets math,ifeval,humaneval
$PY tools/d4x.py "$BATCH" "$CORE" "$IFXREF" > results/f1_d4x.txt 2>&1; d=$?
say "the batch's core lines: $(grep '^D4X:' results/f1_d4x.txt | tail -1) (d4x rc $d)"
if [ $d = 10 ]; then
  nx=$(grep -c '"key": "ifevalx/' "results/$IFXREF/decode.jsonl" 2>/dev/null || echo 0)
  if [ "$nx" -ge 220 ] && [ $(( $(date +%s) + 5760 )) -le "$(cutoff)" ]; then   # stage 2 (~1.6 h) must END by the cutoff
    run_rows "$BATCH" 483 --sets ifevalx --keys results/ifx/stage1_keys.txt
    $PY tools/d4x.py "$BATCH" "$CORE" "$IFXREF" > results/f1_d4x.txt 2>&1; d=$?
    say "after the batch's stage 2: $(grep '^D4X:' results/f1_d4x.txt | tail -1) (d4x rc $d)"
  else
    say "the batch's stage 2 does not fit today ($IFXREF has $nx IFX rows, $(( $(left) / 60 )) min left): left for later"
  fi
fi
touch results/F1_ROWS_DONE
say "runner 43 end"
