#!/bin/bash
# EvalSuite runner 38, 2026-11-02 (detached, from a live evalsuite session): REF2c-T IFX (docs/PREREG_ifx_2026-11-02.md).
# REF2c-T (scratch run ref2ct.r2) vs REF2-T (ref2t.r2) on IFEval's other 441 prompts (data/items/dec_ifevalx.jsonl),
# greedy at C 192 as Part 1 / R-58 ran the core rows. Each arm's first process re-decodes 2 core rows first
# (results/ifx/repro_keys.txt): VOID (exit 5) unless they are token-equal to the arm's saved rows.
# Stage 1: 220 prompts per arm (results/ifx/stage1_keys.txt), arms alternating, one quiet ticket per 60-min chunk; then
# tools/ifx.py stage1 (decision 4's rule on the 220 + the stopping rule): STOP -> that is the verdict; else stage 2 (the
# other 221) and tools/ifx.py final. Either way results/ifx.txt and results/IFX_DONE.
# Queues from Tue 00:30 once results/ifx/GO exists (BigMoE's night items are done; BigMoE, NewIdeas or EvalSuite touch
# it), or from Tue 05:30 without it (BigMoE holds 01:00-05:30; Kamen's schedule Mon 19:3x: big tests weekdays
# 00:30-15:00; relaunched Mon 20:5x, PREREG amendment 1). NO-SWAP >= peak + 4 GB after the grant (cli --need 34). Code pin as runner 36
# + snap/leap-faff8e5, MoonShots' planes_engine, tools/ifx.py, the items and key files; a change stops it (exit 7).
# Deadline Tue 2026-11-03 15:00: no chunk starts in its last 90 min; what is left runs another night (a new launch).
# Stop it by pid only, and its cli child too.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day38_runner.log
GO=results/ifx/GO
EARLIEST=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-03 00:30" +%s)
LATEST=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-03 05:30" +%s)
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-03 15:00" +%s)
export PEER_QUEUE_TIMEOUT=21600
unset MTPLX_FUSED_GATE_UP MTPLX_FUSED_GDN_INPROJ BIGMOE_GDN_EXACT BIGMOE_TIER BIGMOE_TIER_GLU BIGMOE_TIER_CL \
  BIGMOE_TIER_COMBINE BIGMOE_OOO BIGMOE_VERIFY_MIDLOOP BIGMOE_EXT_BUILD BIGMOE_MAXK
export EVALSUITE_BIGMOE_CODE="$HOME/EvalSuite/snap/bigmoe-6c64148"
TIERENV="BIGMOE_TIER=4433333333,BIGMOE_TIER_GLU=3c,BIGMOE_TIER_CL=112"
PY=.venv/bin/python
A=ref2ct.r2; B=ref2t.r2
K1=results/ifx/stage1_keys.txt; K2=results/ifx/stage2_keys.txt; KR=results/ifx/repro_keys.txt
RUN1=results/ifx/run1_keys.txt                   # stage 1 + the 2 reproduction keys (set ifeval runs first)
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
left() { echo $(( EVALSUITE_DEADLINE - $(date +%s) )); }
have() {   # rows the run $1 has for the keys in file $2
  [ -f "results/$1/decode.jsonl" ] || { echo 0; return; }
  grep -o '"key": "[^"]*"' "results/$1/decode.jsonl" | sed 's/^"key": "//; s/"$//' | grep -c -x -F -f "$2"
}
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
  { shasum snap/leap-91cd316/*.py snap/leap-91cd316/ms_gdn/*.py snap/leap-faff8e5/*.py tools/ifx.py "$K1" "$K2" "$KR" \
      data/items/dec_ifevalx.jsonl data/items/dec_ifeval.jsonl
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
REPRO_A=0; REPRO_B=0
stage() {   # key file, rows per arm, label: chunks (arms alternating) until both arms have every key's row
  local v b n rc r na nb
  NOPROG=0
  while :; do
    na=$(have $A "$1"); nb=$(have $B "$1")
    [ "$na" -ge "$2" ] && [ "$nb" -ge "$2" ] && return 0
    if { [ "$na" -le "$nb" ] && [ "$na" -lt "$2" ]; } || [ "$nb" -ge "$2" ]; then v=$A; else v=$B; fi
    calm
    [ "$(pin)" = "$PIN0" ] || { say "code changed: pin $PIN0 -> $(pin): runner stops"; exit 7; }
    [ "$(left)" -gt 5400 ] || { say "deadline near: runner stops ($3: $A $na, $B $nb of $2 rows)"; exit 6; }
    b=$(have $v "$1")
    EVALSUITE_STAGE_ENV="$TIERENV" ./bin/evalsuite gpu decode $v --sets ifeval,ifevalx --keys "$1" --chunk-minutes 60 \
      --C 192 --need 34 --minutes 80 >> "$LOG" 2>&1; rc=$?
    n=$(have $v "$1")
    say "$v ($3): chunk rc=$rc ($n/$2 rows)"
    if { [ "$v" = $A ] && [ $REPRO_A = 0 ]; } || { [ "$v" = $B ] && [ $REPRO_B = 0 ]; }; then
      $PY tools/ifx.py repro $v >> "$LOG" 2>&1; r=$?
      [ $r = 5 ] && { say "VOID: $v does not reproduce its saved rows: runner stops"; exit 5; }
      if [ $r = 0 ]; then
        if [ "$v" = $A ]; then REPRO_A=1; else REPRO_B=1; fi
      fi
    fi
    after_chunk "$rc" "$b" "$n"
  done
}
PIN0=$(pin)
cat "$K1" "$KR" > "$RUN1"
say "runner 38 start (pid $$); code pin $PIN0; queues from Tue 00:30 once $GO exists, or from Tue 05:30; deadline Tue 15:00"
while [ "$(date +%s)" -lt "$EARLIEST" ] || { [ ! -e "$GO" ] && [ "$(date +%s)" -lt "$LATEST" ]; }; do sleep 60; done
say "go ($( [ -e "$GO" ] && echo "$GO" || echo "Tue 05:30 without $GO")): stage 1 (220 prompts per arm + 2 reproduction rows)"
stage "$RUN1" 222 "stage 1"
[ $REPRO_A = 1 ] && [ $REPRO_B = 1 ] || { say "reproduction rows unchecked (A $REPRO_A, B $REPRO_B): runner stops"; exit 5; }
$PY tools/ifx.py stage1 > results/ifx_stage1.txt 2>&1; s=$?
say "stage 1: $(tail -1 results/ifx_stage1.txt) (rc $s)"
case $s in
  0) say "stage 2 (the other 221 prompts per arm)"
     stage "$K2" 221 "stage 2"
     $PY tools/ifx.py final > results/ifx.txt 2>&1; f=$?
     say "$(tail -1 results/ifx.txt) (rc $f)"
     [ $f -le 1 ] || { say "the final analysis failed (rc $f): runner stops"; exit 4; } ;;
  10|11) cp results/ifx_stage1.txt results/ifx.txt ;;
  *) say "the stage-1 analysis failed (rc $s): runner stops"; exit 4 ;;
esac
touch results/IFX_DONE
say "runner 38 end"
