#!/bin/bash
# EvalSuite runner 45, Fri 2026-11-06 (attached to a live evalsuite session): RELAX's Friday checks
> (a quote from the project owner, kept private)
# 1. The drift re-check: the night's arm B (Q1-UNION cf 0.9 + copy + the comment rule + HC8) on results/drift/keys_B.txt
#    (121: humanevalx 104 + agentlong 17), into results/relax_B/ beside the night's 124 rows. Then REF2-T's replay of the
#    new checks, tools/relax_report.py --arms B -> results/relax_drift_report.txt (the pooled long-code drift line), and
#    results/DRIFT_DONE.
# 2. E8 (E + HC8 on chat; HC8 is process-global as built: NewIdeas Thu 08:5x; the orchestrator confirmed it): arm E8 on
#    E's 124 keys + its replay, then the ship8 mode (B code + E8 chat) -> results/relax_e8_report.txt, results/E8_DONE.
# The store: Thursday's records3 conversion (after NAX6 Part 1's PASS) leaves a store the night's row-major code cannot
# read. The mode is picked once, at GO, and every chunk checks that it still holds (exit 8):
#  - converted (all 48 layers, index.json unchanged for 2 min): snap/bigmoe-da8e7a6 on ext/build (MAXK 160 is runtime)
#    and EVALSUITE_RELAX_VARIANT=ref2tf0 (REF2-T on snap/leap-52cf2c9, every switch off; converted layers take 3c-N even
#    with REF2 removed). The bridge runs first: B's replay process on the night's rows, which only runs its step 0 (the
#    control's gate-off rows must replay bitwise; ~5 min). Each G process's step 0 then checks HumanEval/27 against
#    results/ref2th8 token for token.
#  - row-major (no conversion happened): the night's code: snap/bigmoe-6c64148 on build_next, ref2t.
# Needs pending/apply_drift_e8.py applied (exit 2). Waits for results/q45/GO (the night batch's end) and for runners
# 41-44 (one lock per session). Share locks ("MEM <n> GB", the day's one > 5 GB job); C 192 or 160 by free memory (the
# gate needs C >= 160; tokens do not depend on C); a memory wait without a lock; the code pin (exit 7). A chunk starts
# only if start + the longest chunk so far ends by 15:00 (exit 6; the orchestrator: cutoffs bound ends). Stop it by pid
# only, and its cli child too.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day45_runner.log
GO=results/q45/GO
CUTOFF=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-06 15:00" +%s)
export EVALSUITE_DEADLINE=$CUTOFF
export PEER_QUEUE_TIMEOUT=21600
unset MTPLX_FUSED_GATE_UP BIGMOE_TIER BIGMOE_TIER_GLU BIGMOE_TIER_CL BIGMOE_TIER_COMBINE BIGMOE_OOO BIGMOE_VERIFY_MIDLOOP \
  BIGMOE_EXT_BUILD BIGMOE_MAXK BIGMOE_COMPILE_HC BIGMOE_HC_VREAD BIGMOE_HC_PACK EVALSUITE_RELAX_TAG EVALSUITE_RELAX_HASH \
  EVALSUITE_RELAX_GATE EVALSUITE_RELAX_VARIANT EVALSUITE_BIGMOE_CODE
TIERENV="BIGMOE_MAXK=160,BIGMOE_TIER=4433333333,BIGMOE_TIER_GLU=3c,BIGMOE_TIER_CL=112,BIGMOE_TIER_COMBINE=1,BIGMOE_OOO=1"
PY=.venv/bin/python
KB=results/drift/keys_B.txt
KE=results/relax/keys_E.txt
R3IDX=$HOME/BigMoE/models/bare/records3/index.json
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
nkeys() { grep -c . "$1"; }
have() {   # rows of arm $1 among the keys in file $2
  [ -f "results/relax_$1/decode.jsonl" ] || { echo 0; return; }
  python3 -c 'import json, sys
k = set(open(sys.argv[2]).read().split())
print(sum(json.loads(l)["key"] in k for l in open(sys.argv[1])))' "results/relax_$1/decode.jsonl" "$2"
}
replayed() {   # replays of arm $1 among the keys in file $2
  python3 -c 'import os, sys
k = set(open(sys.argv[2]).read().split())
d = sys.argv[1]
print(sum(f[:-9].replace("__", "/") in k for f in (os.listdir(d) if os.path.isdir(d) else []) if f.endswith(".jsonl.gz")))' \
    "results/relax_$1/replay" "$2"
}
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
store_ok() {    # the store is still as the mode needs (a converted index that was just rewritten gets 4 min to settle)
  [ "$MODE" = rowmajor ] && { ! converted; return; }
  local i; for i in 1 2 3 4; do full_converted && return 0; sleep 60; done; return 1
}
alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
busy4x() {      # runner 41 / 42 / 43 / 44 still alive: one lock per session (41 writes no pid file)
  pgrep -qf 'tools/day_runner41\.sh' && return 0
  local f; for f in results/q42.pid results/q43.pid results/q44.pid; do alive "$f" && return 0; done; return 1
}
mem_busy() {    # another session's shared run with MEM > 5 GB (our own reasons start "evalsuite: MEM")
  peer ls 2>/dev/null | grep "shared run" | grep -v "evalsuite: MEM" | grep -oE "MEM [0-9]+ GB" | awk '$2 > 5 {f = 1} END {exit !f}'
}
freegb() { .venv/bin/python -c "import sys; sys.path.insert(0, '.'); from evalsuite.cli import free_gb; print(int(free_gb()))"; }
EST=3000                                          # a 45-min chunk + load and its last prompt, until measured
fits() { [ $(( $(date +%s) + EST )) -le "$CUTOFF" ]; }
pick_c() {      # the gate needs C >= MAXK 160: C 192 (need 34) or 160 (need 30) by free memory, else wait (no lock held)
  local fg w=0
  while :; do
    fg=$(freegb)
    if [ "$fg" -ge 38 ]; then C=192; NEED=34; return 0; fi
    if [ "$fg" -ge 34 ]; then C=160; NEED=30; return 0; fi
    [ $((w % 10)) = 0 ] && say "waiting for memory: $fg GB NO-SWAP (C 160 needs 34)"
    w=$((w + 1)); sleep 60; calm
    fits || { say "no chunk fits before 15:00 while waiting for memory: runner stops"; exit 6; }
  done
}
pin() {         # both modes' code (frozen snapshots), the keys and items, evalsuite/, the drafter, the venv's packages
  { shasum snap/leap-faff8e5/*.py snap/leap-52cf2c9/*.py snap/leap-52cf2c9/ms_gdn/* "$KB" "$KE" \
      data/items/dec_humanevalx.jsonl data/items/dec_agentlong.jsonl data/items/dec_ifeval.jsonl data/items/dec_multi.jsonl
    for s in snap/bigmoe-6c64148 snap/bigmoe-da8e7a6; do (cd "$s" && find bigmoe ext -type f ! -path '*__pycache__*' | sort | xargs shasum); done
    (cd "$HOME/MoonShots/patches/planes_engine" && ls *.py | sort | xargs shasum)
    find evalsuite -name '*.py' ! -path '*__pycache__*' | sort | xargs shasum
    (cd "$HOME/Leap" && shasum third_party/dflash_model_mlx.py && stat -f '%z %m' data/l120_ft.safetensors)
    ls "$HOME/BigMoE/.venv/lib/python3.12/site-packages" | grep -E '^(mlx|mlx_lm|mlx_metal|mtplx)-.*dist-info$'; } 2>&1 \
    | shasum | cut -c1-12
}
SWAPTRIPS=0; NOPROG=0
after() {   # label, rc, before, now
  if tail -4 "$LOG" | grep -q "can't tell which Claude session\|left the queue"; then
    say "the session that owns this runner's locks is gone: runner stops (relaunch it from a live session)"; exit 3
  fi
  if tail -4 "$LOG" | grep -q "swap guard"; then
    SWAPTRIPS=$((SWAPTRIPS + 1)); [ "$SWAPTRIPS" -ge 6 ] && { say "swap guard: 6 trips: runner stops"; exit 3; }
    say "swap guard trip $SWAPTRIPS: waiting 2 min"; sleep 120; return
  fi
  [ "$2" = 5 ] && { say "$1: VOID (step 0): runner stops"; exit 5; }
  [ "$2" = 76 ] && { say "$1: the stage exited on a long PAUSE: waiting it out in bash"; return; }
  if [ "$2" != 0 ] && [ "$4" -le "$3" ] && tail -40 results/gpu_stage.log | grep -q "Traceback"; then
    say "$1: the stage failed (results/gpu_stage.log): runner stops"; exit 4
  fi
  [ "$4" -le "$3" ] && { NOPROG=$((NOPROG + 1)); [ "$NOPROG" -ge 3 ] && { say "$1: three chunks without progress: runner stops"; exit 3; }; sleep 60; } || NOPROG=0
}
chunk() {   # label, stage args...: one share-lock chunk with the day rule, the pin, the store, memory and the END bound
  local lab=$1 t0 d rc; shift
  calm
  local w=0
  while mem_busy; do
    [ $((w % 10)) = 0 ] && say "another session's MEM job holds a shared lock: waiting (day rule)"
    w=$((w + 1)); sleep 60; calm
  done
  [ "$(pin)" = "$PIN0" ] || { say "code changed: pin $PIN0 -> $(pin): runner stops"; exit 7; }
  store_ok || { say "records3 is no longer $MODE as this run's code reads it (index.json): runner stops"; exit 8; }
  fits || { say "$lab: no chunk fits before 15:00 (measured $((EST / 60)) min): runner stops"; exit 6; }
  pick_c
  t0=$(date +%s)
  EVALSUITE_STAGE_ENV="$STAGEENV" ./bin/evalsuite gpu "$@" --chunk-minutes 45 --C "$C" --need "$NEED" --minutes 60 \
    --share >> "$LOG" 2>&1; rc=$?
  d=$(( $(date +%s) - t0 )); [ "$d" -gt "$EST" ] && EST=$d
  RC=$rc
  say "$lab: chunk rc=$rc at C $C ($((d / 60)) min)"
}
grep -q '"humanevalx"' evalsuite/relax.py && grep -q EVALSUITE_RELAX_VARIANT evalsuite/replay.py || {
  say "relax.py / replay.py lack the Friday changes: run pending/apply_drift_e8.py first"; exit 2; }
PIN0=$(pin)
echo $$ > results/q45.pid; trap 'rm -f results/q45.pid' EXIT
say "runner 45 start (pid $$); code pin $PIN0; waits for $GO and runners 41-44; $(nkeys "$KB") drift keys, E8 on $(nkeys "$KE")"
while [ ! -e "$GO" ] || busy4x; do sleep 60; done
if converted; then
  until full_converted; do
    waitlog "records3 is partly converted or index.json just changed: waiting for all 48 layers, unchanged for 2 min"
    sleep 60; fits || { say "no chunk fits before 15:00 while waiting for the store: runner stops"; exit 6; }
  done
  MODE=converted; STAGEENV=$TIERENV
  export EVALSUITE_BIGMOE_CODE="$HOME/EvalSuite/snap/bigmoe-da8e7a6" EVALSUITE_RELAX_VARIANT=ref2tf0
else
  MODE=rowmajor; STAGEENV="$TIERENV,BIGMOE_EXT_BUILD=build_next"
  export EVALSUITE_BIGMOE_CODE="$HOME/EvalSuite/snap/bigmoe-6c64148"
fi
say "go: records3 $MODE: BigMoE code $EVALSUITE_BIGMOE_CODE, REF2-T variant ${EVALSUITE_RELAX_VARIANT:-ref2t}, stage env $STAGEENV"
if [ "$MODE" = converted ]; then   # the bridge: the new code must replay the night control's gate-off rows bitwise
  BRIDGED=0
  until [ "$BRIDGED" = 1 ]; do
    chunk "bridge" relax-replay ref --arm B
    [ "$RC" = 0 ] && BRIDGED=1 || after "bridge" "$RC" 0 0
  done
  say "bridge: $(grep -h 'RELAX replay B step 0' results/gpu_stage.log "$LOG" 2>/dev/null | tail -1 | cut -c1-170)"
fi
say "B on the drift keys"
NOPROG=0
while [ "$(have B "$KB")" -lt "$(nkeys "$KB")" ]; do
  b=$(have B "$KB")
  chunk "B drift" relax ref --arm B --keys "$KB"
  after "B drift" "$RC" "$b" "$(have B "$KB")"
done
say "B: $(have B "$KB")/$(nkeys "$KB") drift rows; the replay"
NOPROG=0
while [ "$(replayed B "$KB")" -lt "$(have B "$KB")" ]; do
  b=$(replayed B "$KB")
  chunk "replay B" relax-replay ref --arm B
  after "replay B" "$RC" "$b" "$(replayed B "$KB")"
done
$PY tools/relax_report.py --arms B --out results/relax_drift_report.txt > /dev/null 2>&1
say "the drift verdict: $(grep '^== long-code drift' results/relax_drift_report.txt | sed 's/^== //')"
touch results/DRIFT_DONE
say "E8 (E + HC8 on chat)"
cp "$KE" results/relax/keys_E8.txt
NOPROG=0
while [ "$(have E8 "$KE")" -lt "$(nkeys "$KE")" ]; do
  b=$(have E8 "$KE")
  chunk "E8" relax ref --arm E8 --keys results/relax/keys_E8.txt
  after "E8" "$RC" "$b" "$(have E8 "$KE")"
done
NOPROG=0
while [ "$(replayed E8 "$KE")" -lt "$(have E8 "$KE")" ]; do
  b=$(replayed E8 "$KE")
  chunk "replay E8" relax-replay ref --arm E8
  after "replay E8" "$RC" "$b" "$(replayed E8 "$KE")"
done
$PY tools/relax_report.py --arms control,B,E,E8 --score --out results/relax_e8_report.txt > /dev/null 2>&1
say "ship8 (B code + E8 chat): $(grep -- '-> ship8:' results/relax_e8_report.txt | head -1)"
touch results/E8_DONE
say "runner 45 end"
