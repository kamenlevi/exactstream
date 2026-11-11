#!/bin/bash
# EvalSuite runner 39, 2026-11-02 (detached, from a live evalsuite session): Q3 HC8 gate (docs/PREREG_hc8_2026-11-02.md).
# Kamen's day 9, Q3: variant ref2th8 = REF2-T (R-58's ref2t: snap/leap-faff8e5, BigMoE's rank tiers 4433333333 / 3c /
# C_l 112) with the HC mixers + PLE key/value at 8 bits (evalsuite/model.py hc8_fake) on the 210 core rows; the verdict:
# tools/d4.py on the report vs ref2t (decision 4's rule) -> results/hc8_d4.txt, then results/HC8_DONE.
# Waits for results/hc8/GO (Breakthrough's map passes HC8; Breakthrough, the orchestrator or EvalSuite touch it). Its first
# word sets C (default 96, for the day with Chrome open). Numerics do not depend on C; step 0 checks that: 2 REF2-T rows
# (scratch run ref2t.r4) at that C must be token-identical to R-58's, else VOID (exit 5).
# Share lock per 60-min chunk (day 9: quiet locks only for timings); NO-SWAP >= peak + 8 GB after the grant: cli --need
# 0.131 C + 10.2 (peak ~0.131 C + 6.2 GB: 14.5 GB at C 64). Code pin at launch; a change stops it (exit 7). Deadline Tue
# 2026-11-03 08:30: no chunk starts in its last 90 min. Stop it by pid only, and its cli child too.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day39_runner.log
GO=results/hc8/GO
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-03 08:30" +%s)
unset MTPLX_FUSED_GATE_UP MTPLX_FUSED_GDN_INPROJ BIGMOE_GDN_EXACT BIGMOE_TIER BIGMOE_TIER_GLU BIGMOE_TIER_CL \
  BIGMOE_TIER_COMBINE BIGMOE_OOO BIGMOE_VERIFY_MIDLOOP BIGMOE_EXT_BUILD BIGMOE_MAXK BIGMOE_COMPILE_HC BIGMOE_HC_VREAD \
  BIGMOE_HC_PACK
export EVALSUITE_BIGMOE_CODE="$HOME/EvalSuite/snap/bigmoe-6c64148"
TIERENV="BIGMOE_TIER=4433333333,BIGMOE_TIER_GLU=3c,BIGMOE_TIER_CL=112"
PY=.venv/bin/python
KR=results/ifx/repro_keys.txt                    # ifeval/3409, ifeval/2439 (tools/ifx.py repro compares them)
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
pin() {
  { shasum snap/leap-faff8e5/*.py tools/d4.py tools/ifx.py "$KR" data/items/dec_math.jsonl data/items/dec_ifeval.jsonl \
      data/items/dec_humaneval.jsonl
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
run_rows() {   # run, rows wanted, stage args...: chunks under a share lock until the run has that many rows
  local run=$1 want=$2 b n rc; shift 2
  NOPROG=0
  while [ "$(count results/$run/decode.jsonl)" -lt "$want" ]; do
    b=$(count results/$run/decode.jsonl)
    calm
    [ "$(pin)" = "$PIN0" ] || { say "code changed: pin $PIN0 -> $(pin): runner stops"; exit 7; }
    [ "$(left)" -gt 5400 ] || { say "deadline near: runner stops at $run $b/$want rows"; exit 6; }
    EVALSUITE_STAGE_ENV="$TIERENV" ./bin/evalsuite gpu decode $run "$@" --chunk-minutes 60 --share --C "$C" --need "$NEED" \
      --minutes 80 >> "$LOG" 2>&1; rc=$?
    n=$(count results/$run/decode.jsonl)
    say "$run: chunk rc=$rc ($n/$want rows)"
    after_chunk "$rc" "$b" "$n"
  done
}
PIN0=$(pin)
say "runner 39 start (pid $$); code pin $PIN0; waits for $GO"
while [ ! -e "$GO" ]; do sleep 60; done
C=$(awk 'NR == 1 {print $1}' "$GO"); [[ "$C" =~ ^[0-9]+$ ]] && [ "$C" -ge 48 ] && [ "$C" -le 256 ] || C=96
NEED=$(( (131 * C + 10200 + 999) / 1000 ))
say "go: C $C, --need $NEED (share lock; the cli gates on NO-SWAP >= $((NEED + 4)) GB)"
run_rows ref2t.r4 2 --sets ifeval --keys "$KR"
$PY tools/ifx.py repro ref2t.r4 >> "$LOG" 2>&1; r=$?
[ $r = 0 ] || { say "VOID: REF2-T at C $C does not reproduce R-58's rows (repro rc $r): runner stops"; exit 5; }
say "step 0: REF2-T at C $C reproduces R-58's 2 rows; step 1: ref2th8 on the 210 core rows"
run_rows ref2th8 210 --sets math,ifeval,humaneval
./bin/evalsuite report --variant ref2th8 --decode --ref ref2t > results/report_ref2th8_vs_ref2t.txt 2>&1
$PY tools/d4.py results/report_ref2th8_vs_ref2t.txt > results/hc8_d4.txt 2>&1; d4=$?
say "Q3 HC8 verdict (decision 4, ref2th8 vs ref2t): $(tail -1 results/hc8_d4.txt) (d4 rc $d4); standard table: $(tail -1 results/report_ref2th8_vs_ref2t.txt)"
./bin/evalsuite report --variant ref2th8 --decode --ref ref2b > results/report_ref2th8_vs_ref2b.txt 2>&1
say "information: vs ref2b $(tail -1 results/report_ref2th8_vs_ref2b.txt)"
touch results/HC8_DONE
say "runner 39 end"
