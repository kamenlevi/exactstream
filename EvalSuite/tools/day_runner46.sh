#!/bin/bash
# EvalSuite runner 46 (attached to a live evalsuite session): the rest of the FAST batch F1 + r8 + HC8-1P on the NAX6
# base (docs/PREREG_f1_2026-11-03.md revision 3 + the Thu 14:08 status note). Runner 43 stopped Thu 14:06 at its END
# bound with the control ref2tnaxn at 53/53 (RULER + agent) and the batch ref2tnfhn at 38/53. This runner finishes the
# batch's RULER + agent, then its core 210, then tools/d4x.py's core lines (results/f1_d4x.txt) and F1_ROWS_DONE. The
# verdict is tools/f1_report.py's, run by the session.
# Slot (the orchestrator Thu 14:0x): the Fri -> Sat night, from ~00:30 Sat, right after BigMoE's FAR-G1. It starts at
# R46_EARLIEST once R46_GO exists (default results/q46/GO; the order's previous job touches it, or the orchestrator).
# What runner 43 did first stays valid and is only re-checked here (no GPU): NAX6_DONE PASS, records3 check 2 rc 0,
# the NAX6-N bridge rows (ref2tnaxn.r1 == ref2tnax on the 3 step-0 keys). The store must be fully converted at every
# chunk (exit 8). The decode stage does not import relax.py / replay.py (stages.py imports them for their own stages
# only), so Friday's RELAX patch leaves this decode's code as Thursday's.
# Share locks ("MEM 36 GB"); C 192 / 160 / 128 by free memory (tokens do not depend on C); the day rule's MEM wait; the
# code pin (exit 7); no chunk starts unless start + the longest chunk so far ends by R46_END (exit 6; "HH:MM" in
# results/q46/CUTOFF, on R46_END's day and before EVALSUITE_DEADLINE, moves it). Stop it by pid only, and its cli child.
#   R46_EARLIEST="2026-11-07 00:30" R46_END="2026-11-07 09:30" ./tools/day_runner46.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day46_runner.log
EARLIEST_AT=${R46_EARLIEST:-2026-11-07 00:30}
END_AT=${R46_END:-2026-11-07 09:30}
END_DAY=${END_AT%% *}
GO=${R46_GO:-results/q46/GO}
EARLIEST=$(date -j -f "%Y-%m-%d %H:%M" "$EARLIEST_AT" +%s) || exit 2
CUTOFF0=$(date -j -f "%Y-%m-%d %H:%M" "$END_AT" +%s) || exit 2
export EVALSUITE_DEADLINE=$(( CUTOFF0 + 1800 ))
export PEER_QUEUE_TIMEOUT=21600
unset MTPLX_FUSED_GATE_UP MTPLX_FUSED_GDN_INPROJ BIGMOE_GDN_EXACT BIGMOE_TIER BIGMOE_TIER_GLU BIGMOE_TIER_CL \
  BIGMOE_TIER_COMBINE BIGMOE_OOO BIGMOE_VERIFY_MIDLOOP BIGMOE_EXT_BUILD BIGMOE_MAXK BIGMOE_COMPILE_HC BIGMOE_HC_VREAD \
  BIGMOE_HC_PACK BIGMOE_NAX_NATIVE REF2_FATT_VERIFY REF2_FATT_R8 REF2_HC8_FAST REF2_HC8_V REF2_DENSE_NATIVE REF2_OOO_1C
export EVALSUITE_BIGMOE_CODE="$HOME/EvalSuite/snap/bigmoe-da8e7a6"
TIERENV="BIGMOE_TIER=4433333333,BIGMOE_TIER_GLU=3c,BIGMOE_TIER_CL=112"
PY=.venv/bin/python
CTRL=ref2tnaxn; BATCH=ref2tnfhn; CORE=ref2tnax; IFXREF=ref2tnax
R3IDX=$HOME/BigMoE/models/bare/records3/index.json
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
full_converted() {   # every one of the 48 layers carries "layout", and index.json has not changed for 2 min
  [ $(( $(date +%s) - $(stat -f %m "$R3IDX") )) -ge 120 ] || return 1
  python3 -c 'import json, sys
L = json.load(open(sys.argv[1]))["layers"]
sys.exit(0 if len(L) == 48 and all("layout" in v for v in L.values()) else 1)' "$R3IDX"
}
store_ok() { local i; for i in 1 2 3 4; do full_converted && return 0; sleep 60; done; return 1; }
mem_busy() {    # another session's shared run with MEM > 5 GB (our own reasons start "evalsuite: MEM")
  peer ls 2>/dev/null | grep "shared run" | grep -v "evalsuite: MEM" | grep -oE "MEM [0-9]+ GB" | awk '$2 > 5 {f = 1} END {exit !f}'
}
pin() {
  { shasum snap/leap-52cf2c9/*.py snap/leap-52cf2c9/ms_gdn/* tools/d4x.py results/ifx/stage1_keys.txt \
      results/q42/step0_keys.txt data/items/dec_math.jsonl data/items/dec_ifeval.jsonl data/items/dec_humaneval.jsonl \
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
cutoff_hm() { local h; h=$(head -c 5 results/q46/CUTOFF 2>/dev/null); [[ "$h" =~ ^([01][0-9]|2[0-3]):[0-5][0-9]$ ]] && echo "$h"; }
cutoff() {      # the END bound: results/q46/CUTOFF's "HH:MM" on R46_END's day if it is before the deadline, else R46_END
  local h c; h=$(cutoff_hm)
  [ -n "$h" ] && c=$(date -j -f "%Y-%m-%d %H:%M" "$END_DAY $h" +%s) && [ "$c" -le $(( EVALSUITE_DEADLINE - 600 )) ] && { echo "$c"; return; }
  echo "$CUTOFF0"
}
EST=3000                                          # a 45-min chunk + load and its last row, until measured
fits() { [ $(( $(date +%s) + EST )) -le "$(cutoff)" ]; }
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
    store_ok || { say "records3 is not fully converted (index.json): runner stops at $run $b/$want rows"; exit 8; }
    fits || { say "no chunk fits before $(date -r "$(cutoff)" +%H:%M) (measured chunk $((EST / 60)) min): runner stops at $run $b/$want rows"; exit 6; }
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
case "$(cat results/NAX6_DONE 2>/dev/null)" in "D4X: PASS"*|"D4X: DIFFERS"*) ;; *) echo "no NAX6 PASS"; exit 5 ;; esac
case "$(cat results/CHECK2_DONE 2>/dev/null)" in "check 2 rc 0 "*) ;; *) echo "records3 check 2 did not pass"; exit 5 ;; esac
python3 - <<'BRIDGE' || { echo "the NAX6-N bridge rows are missing or differ"; exit 5; }
import json, sys
keys = open("results/q42/step0_keys.txt").read().split()
load = lambda v: {r["key"]: r for r in map(json.loads, open(f"results/{v}/decode.jsonl"))}
a, b = load("ref2tnaxn.r1"), load("ref2tnax")
sys.exit(0 if all(k in a and a[k]["completion_ids"] == b[k]["completion_ids"] for k in keys) else 1)
BRIDGE
[ "$(count results/$CTRL/decode.jsonl)" -ge 53 ] || { echo "the control $CTRL lacks its 53 RULER + agent rows"; exit 5; }
PIN0=$(pin)
echo $$ > results/q46.pid; trap 'rm -f results/q46.pid' EXIT
say "runner 46 start (pid $$); code pin $PIN0; from $EARLIEST_AT once $GO exists; END bound $END_AT; $BATCH at $(count results/$BATCH/decode.jsonl) rows"
while [ "$(date +%s)" -lt "$EARLIEST" ] || [ ! -e "$GO" ]; do sleep 60; done
say "go: the batch's RULER + agent, then its core 210 (control $CTRL 53/53 from runner 43; check 2 and the bridge PASS Thu 10:49)"
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
    say "the batch's stage 2 does not fit ($IFXREF has $nx IFX rows): left for later"
  fi
fi
touch results/F1_ROWS_DONE
say "runner 46 end"
