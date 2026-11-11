#!/bin/bash
# EvalSuite runner 42, Thu 2026-11-05 (attached to a live evalsuite session): NAX6 as the 45 GB EXACT reference, Part 1
# (docs/PREREG_nax6_2026-11-04.md; Kamen YES Wed 15:0x). After RELAX_DONE and BigMoE's Thu-morning ticket (the
# orchestrator 15:2x), once results/q42/GO exists (and RELAX_DONE, or runner 40's pid in results/q40.pid gone).
# Step 0: ref2tf0 (REF2-T on snap/leap-52cf2c9 + snap/bigmoe-da8e7a6, every switch off) decodes ifeval/3409,
# ifeval/2439, HumanEval/27; they must equal results/ref2t token for token, and no NAX6 / F1 / HC8 path may run in
# them, else VOID (exit 5). Part 1: ref2tnax (+ tier_nax_ooo) on the 210 core rows; the decode stage's switch guard
# checks every row ran NAX6. Verdict: tools/d4x.py ref2tnax ref2t ref2t.r2 -> results/nax6_d4.txt; on STAGE2
# (IFEval alone fails) ref2tnax decodes IFX's 220 and d4x runs again. Then results/NAX6_DONE (its line = the verdict),
# which runner 43 (F1 + HC8-1P) reads for its base; just before it, the records3 check 1 on the row-major store
# (tools/r3check.sh, ~10 min; the orchestrator 18:0x: BigMoE converts right after a NAX6 PASS).
# AR at C 192 under share locks ("MEM 36 GB": the day's one memory-heavy job; it waits while another session's
# shared run has MEM > 5 GB). Code pin at launch (exit 7 on a change); it stops (exit 8) if records3's layout changes;
# no chunk starts after Thu 14:00 (exit 6). Stop it by pid only, and its cli child too.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day42_runner.log
GO=results/q42/GO
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-05 15:00" +%s)
export PEER_QUEUE_TIMEOUT=21600
unset MTPLX_FUSED_GATE_UP MTPLX_FUSED_GDN_INPROJ BIGMOE_GDN_EXACT BIGMOE_TIER BIGMOE_TIER_GLU BIGMOE_TIER_CL \
  BIGMOE_TIER_COMBINE BIGMOE_OOO BIGMOE_VERIFY_MIDLOOP BIGMOE_EXT_BUILD BIGMOE_MAXK BIGMOE_COMPILE_HC BIGMOE_HC_VREAD \
  BIGMOE_HC_PACK BIGMOE_NAX_NATIVE REF2_FATT_VERIFY REF2_FATT_R8 REF2_HC8_FAST REF2_HC8_V REF2_DENSE_NATIVE REF2_OOO_1C
export EVALSUITE_BIGMOE_CODE="$HOME/EvalSuite/snap/bigmoe-da8e7a6"
TIERENV="BIGMOE_TIER=4433333333,BIGMOE_TIER_GLU=3c,BIGMOE_TIER_CL=112"
PY=.venv/bin/python
C=192; NEED=$(( (131 * C + 10200 + 999) / 1000 ))   # per chunk: pick_c (Wed 22:5x: C 192 / 160 / 128 by free memory)
K0=results/q42/step0_keys.txt
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
converted() {   # any layer with a "layout" entry, or a record size other than row-major records3's 2,932,736 B
  grep -q '"layout"' "$R3IDX" || [ "$(grep -o '"recsz": [0-9]*' "$R3IDX" | sort -u)" != '"recsz": 2932736' ]
}
mem_busy() {    # another session's shared run with MEM > 5 GB (our own reasons start "evalsuite: MEM")
  peer ls 2>/dev/null | grep "shared run" | grep -v "evalsuite: MEM" | grep -oE "MEM [0-9]+ GB" | awk '$2 > 5 {f = 1} END {exit !f}'
}
pin() {
  { shasum snap/leap-52cf2c9/*.py snap/leap-52cf2c9/ms_gdn/* tools/d4x.py tools/r3check.sh "$K0" results/ifx/stage1_keys.txt \
      results/r3check_keys.txt \
      data/items/dec_math.jsonl data/items/dec_ifeval.jsonl data/items/dec_humaneval.jsonl data/items/dec_ifevalx.jsonl
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
run_rows() {   # run, rows wanted, stage args...: chunks under a share lock until the run has that many rows
  local run=$1 want=$2 b n rc w; shift 2
  NOPROG=0
  while [ "$(count results/$run/decode.jsonl)" -lt "$want" ]; do
    b=$(count results/$run/decode.jsonl)
    calm
    w=0; while mem_busy; do [ $((w % 10)) = 0 ] && say "another session's MEM job holds a shared lock: waiting (day rule)"; w=$((w + 1)); sleep 60; calm; done
    [ "$(pin)" = "$PIN0" ] || { say "code changed: pin $PIN0 -> $(pin): runner stops"; exit 7; }
    converted && { say "records3 is not row-major any more (index.json): runner stops at $run $b/$want rows"; exit 8; }
    [ "$(left)" -gt 3600 ] || { say "deadline near (no chunk starts after 14:00): runner stops at $run $b/$want rows"; exit 6; }
    pick_c
    say "$run: chunk at C $C (--need $NEED; $(freegb) GB NO-SWAP)"
    EVALSUITE_STAGE_ENV="$TIERENV" ./bin/evalsuite gpu decode $run "$@" --chunk-minutes 45 --share --C "$C" --need "$NEED" \
      --minutes 60 >> "$LOG" 2>&1; rc=$?
    n=$(count results/$run/decode.jsonl)
    say "$run: chunk rc=$rc ($n/$want rows)"
    after_chunk "$rc" "$b" "$n"
  done
}
PIN0=$(pin)
echo $$ > results/q42.pid; trap 'rm -f results/q42.pid' EXIT
say "runner 42 start (pid $$); code pin $PIN0; waits for results/RELAX_DONE and $GO; C $C, --need $NEED"
alive40() { [ -f results/q40.pid ] && kill -0 "$(cat results/q40.pid)" 2>/dev/null; }   # runner 40 (RELAX)
while { [ ! -e results/RELAX_DONE ] && alive40; } || [ ! -e "$GO" ]; do sleep 60; done
[ -e results/RELAX_DONE ] || say "RELAX ended without RELAX_DONE (runner 40 stopped early): starting on BigMoE's GO"
say "go: step 0 (ref2tf0 on $(tr '\n' ' ' < "$K0"))"
run_rows ref2tf0 3 --sets ifeval,humaneval --keys "$K0"
$PY - "$K0" <<'EOF' >> "$LOG" 2>&1; r=$?
import json, sys
keys = open(sys.argv[1]).read().split()
load = lambda v: {r["key"]: r for r in map(json.loads, open(f"results/{v}/decode.jsonl"))}
a, b = load("ref2tf0"), load("ref2t")
bad = [k for k in keys if a[k]["completion_ids"] != b[k]["completion_ids"]]
paths = {k: {p: n for p, n in a[k].get("ref2", {}).get("calls", {}).items()
             if p in ("nax6", "nax6_ooo", "fatt_verify", "fatt_r8", "hc8_fast", "dense_native") and n} for k in keys}
used = {k: p for k, p in paths.items() if p}
print(f"step 0: ref2tf0 vs ref2t on {', '.join(keys)}: " + (f"DIFFERENT on {bad}" if bad else "token-identical")
      + (f"; switched paths ran: {used}" if used else "; no switched path ran"))
sys.exit(5 if bad or used else 0)
EOF
[ $r = 0 ] || { say "VOID: step 0 failed (see the line above): runner stops"; echo "D4X: VOID (step 0)" > results/NAX6_DONE; exit 5; }
say "step 0 PASS; Part 1: ref2tnax (NAX6) on the 210 core rows"
run_rows ref2tnax 210 --sets math,ifeval,humaneval
$PY tools/d4x.py ref2tnax ref2t ref2t.r2 > results/nax6_d4.txt 2>&1; d=$?
if [ $d = 10 ]; then
  say "NAX6: IFEval alone fails on the core: stage 2 (IFX's 220)"
  run_rows ref2tnax 430 --sets ifevalx --keys results/ifx/stage1_keys.txt
  $PY tools/d4x.py ref2tnax ref2t ref2t.r2 > results/nax6_d4.txt 2>&1; d=$?
fi
V=$(grep '^D4X:' results/nax6_d4.txt | tail -1)
say "NAX6 Part 1 verdict: ${V:-no verdict line (d4x rc $d)}"
./bin/evalsuite report --variant ref2tnax --decode --ref ref2t > results/report_ref2tnax_vs_ref2t.txt 2>&1
./bin/evalsuite report --variant ref2tnax --decode --ref ref2b > results/report_ref2tnax_vs_ref2b.txt 2>&1
say "information: the standard table vs ref2t: $(tail -1 results/report_ref2tnax_vs_ref2t.txt)"
calm                                              # records3 check 1 (row-major), before BigMoE's in-place conversion
say "records3 check 1: tools/r3check.sh snap/bigmoe-da8e7a6 ref2tf0.r1 rowmajor ref2t.r2"
pick_c; R3_C=$C tools/r3check.sh snap/bigmoe-da8e7a6 ref2tf0.r1 rowmajor ref2t.r2 >> "$LOG" 2>&1; c1=$?
echo "check 1 rc $c1 ($(date +%H:%M))" > results/CHECK1_DONE
say "check 1 rc $c1 (0 = token-identical to ref2t.r2; results/r3check.log)"
echo "${V:-D4X: none (rc $d)}" > results/NAX6_DONE
say "runner 42 end"
