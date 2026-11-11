#!/bin/bash
# EvalSuite runner 40 (DRAFT until the smoke passes; detached, from a live evalsuite session): RELAX's night
# (docs/PREREG_relax_2026-11-02.md, revision 2). G arms in order control, A, B, E, T3ALL, F, C, G (quiet tickets, C RELAX_C,
# default 256, --need from C: R-56d's 33.4 GB at C 192 + 0.133 GB per slot; Wed 22:5x: C 160 for arms and replays, NO-SWAP 36 GB), each to all its keys (results/relax/keys_<arm>.txt) in 60-min
# chunks; then the REF2-T replays of the arms with a gate, HC8 or other tiers (A, B, T3ALL, C, G; shared, C RELAX_RC, default 192); a memory wait before each chunk; then
# tools/relax_report.py --score -> results/relax_report.txt and results/RELAX_DONE.
# Window: from START once results/relax/GO exists (NewIdeas / the orchestrator assigns the slot), until END: no chunk
# starts in the last 90 min; what is left (G first, then C, then F: the PREREG's drop order) waits for a new launch.
# Kamen's schedule: big tests weekdays 00:30-15:00 only. NO-SWAP >= peak + 4 GB after the grant (cli). Code pin: the
# snapshots, MoonShots' planes_engine, evalsuite/ (relax.py, replay.py, relax_core.py), Leap's drafter, the keys,
# MLX / mtplx; a change stops it (exit 7), and so does records3's in-place conversion (exit 8). Stop it by pid only, and
# its cli child too.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day40_runner.log
GO=results/relax/GO
START=${RELAX_START:-"2026-11-05 00:30"}          # Thu 00:30 (or Wed 10:30 if NewIdeas orders it: RELAX_START=...)
END=${RELAX_END:-"2026-11-05 15:00"}
RC=${RELAX_C:-256}; RRC=${RELAX_RC:-192}     # arms / replays C (Wed 22:5x: NO-SWAP 36 GB -> C 160 for both, PREREG amendment)
need_of() { python3 -c "import math; print(math.ceil(33.4 + ($1 - 192) * 0.133))"; }   # R-56d's 33.4 GB at C 192, 0.133 / slot
freegb() { .venv/bin/python -c "import sys; sys.path.insert(0, '.'); from evalsuite.cli import free_gb; print(int(free_gb()))"; }
mem_wait() {    # wait (no lock held) until NO-SWAP >= need + 4, the cli's gate, so a memory wait never counts as a failed chunk
  local w=0
  while [ "$(freegb)" -lt $(( $1 + 4 )) ]; do
    [ $((w % 10)) = 0 ] && say "waiting for memory: $(freegb) GB NO-SWAP, need $(( $1 + 4 )) (C $2)"
    w=$((w + 1)); sleep 60; calm
    [ "$(left)" -gt "$3" ] || { say "deadline near while waiting for memory: runner stops"; exit 6; }
  done
}
EARLIEST=$(date -j -f "%Y-%m-%d %H:%M" "$START" +%s)
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "$END" +%s)
export PEER_QUEUE_TIMEOUT=21600
unset MTPLX_FUSED_GATE_UP BIGMOE_TIER BIGMOE_TIER_GLU BIGMOE_TIER_CL BIGMOE_TIER_COMBINE BIGMOE_OOO BIGMOE_VERIFY_MIDLOOP BIGMOE_EXT_BUILD BIGMOE_MAXK \
  EVALSUITE_RELAX_TAG EVALSUITE_RELAX_HASH EVALSUITE_RELAX_GATE
export EVALSUITE_BIGMOE_CODE="$HOME/EvalSuite/snap/bigmoe-6c64148"
STAGEENV="BIGMOE_MAXK=160,BIGMOE_EXT_BUILD=build_next,BIGMOE_TIER=4433333333,BIGMOE_TIER_GLU=3c,BIGMOE_TIER_CL=112,BIGMOE_TIER_COMBINE=1,BIGMOE_OOO=1"
PY=.venv/bin/python
ARMS="control A B E T3ALL F C G"
REPLAY="A B T3ALL C G"                            # E, F and the control run REF2-T's own numerics: no replay needed
STAGEENV3=${STAGEENV/BIGMOE_TIER=4433333333/BIGMOE_TIER=3333333333}   # T3ALL's G runs (its replay: REF2-T's tiers)
envof() { [ "$1" = T3ALL ] && echo "$STAGEENV3" || echo "$STAGEENV"; }
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
left() { echo $(( EVALSUITE_DEADLINE - $(date +%s) )); }
have() {   # G rows of arm $1
  [ -f "results/relax_$1/decode.jsonl" ] && wc -l < "results/relax_$1/decode.jsonl" | tr -d ' ' || echo 0
}
want() { wc -w < "results/relax/keys_$1.txt" | tr -d ' '; }
replayed() { ls "results/relax_$1/replay/" 2>/dev/null | grep -c 'jsonl.gz$'; }
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
R3IDX=$HOME/BigMoE/models/bare/records3/index.json
converted() {   # BigMoE's in-place records3 conversion (PLAN_records3_convert): the frozen engine reads row-major only
  grep -q '"layout"' "$R3IDX" || [ "$(grep -o '"recsz": [0-9]*' "$R3IDX" | sort -u)" != '"recsz": 2932736' ]
}
pin() {
  { shasum snap/leap-faff8e5/*.py results/relax/keys_*.txt
    (cd "$EVALSUITE_BIGMOE_CODE" && find bigmoe ext -type f ! -path '*__pycache__*' | sort | xargs shasum)
    (cd "$HOME/MoonShots/patches/planes_engine" && ls *.py | sort | xargs shasum)
    find evalsuite -name '*.py' ! -path '*__pycache__*' | sort | xargs shasum
    (cd "$HOME/Leap" && shasum third_party/dflash_model_mlx.py && stat -f '%z %m' data/l120_ft.safetensors)
    ls "$HOME/BigMoE/.venv/lib/python3.12/site-packages" | grep -E '^(mlx|mlx_lm|mlx_metal|mtplx)-.*dist-info$'; } 2>&1 \
    | shasum | cut -c1-12
}
SWAPTRIPS=0
after() {   # label, rc, before, now; returns 1 when the runner must stop
  if tail -4 "$LOG" | grep -q "can't tell which Claude session\|left the queue"; then
    say "the session that owns this runner's locks is gone: runner stops (relaunch it from a live session)"; exit 3
  fi
  if tail -4 "$LOG" | grep -q "swap guard"; then
    SWAPTRIPS=$((SWAPTRIPS + 1)); [ "$SWAPTRIPS" -ge 6 ] && { say "swap guard: 6 trips: runner stops"; exit 3; }
    say "swap guard trip $SWAPTRIPS: waiting 2 min"; sleep 120; return 0
  fi
  [ "$2" = 5 ] && { say "$1: VOID (step 0): runner stops"; exit 5; }
  [ "$2" = 76 ] && { say "$1: the stage exited on a long PAUSE: waiting it out in bash"; return 0; }
  if [ "$2" != 0 ] && [ "$4" -le "$3" ] && tail -40 results/gpu_stage.log | grep -q "Traceback"; then
    say "$1: the stage failed (results/gpu_stage.log): runner stops"; exit 4
  fi
  [ "$4" -le "$3" ] && { NOPROG=$((NOPROG + 1)); [ "$NOPROG" -ge 3 ] && { say "$1: three chunks without progress: runner stops"; exit 3; }; sleep 60; } || NOPROG=0
}
PIN0=$(pin)
say "runner 40 start (pid $$); code pin $PIN0; from $START once $GO exists, until $END"
[ -f evalsuite/relax.py ] && grep -q '"relax"' evalsuite/stages.py || { say "not wired: runner stops"; exit 2; }
while [ "$(date +%s)" -lt "$EARLIEST" ] || [ ! -e "$GO" ]; do sleep 60; done
say "go: G arms $ARMS"
for arm in $ARMS; do
  NOPROG=0
  while [ "$(have $arm)" -lt "$(want $arm)" ]; do
    calm
    converted && { say "records3 is converted (index.json): the frozen engine reads it row-major: runner stops"; exit 8; }
    [ "$(pin)" = "$PIN0" ] || { say "code changed: pin $PIN0 -> $(pin): runner stops"; exit 7; }
    [ "$(left)" -gt 5400 ] || { say "deadline near: runner stops (arm $arm at $(have $arm)/$(want $arm))"; exit 6; }
    b=$(have $arm)
    mem_wait "$(need_of $RC)" "$RC" 5400
    EVALSUITE_STAGE_ENV="$(envof $arm)" ./bin/evalsuite gpu relax ref --arm $arm --keys results/relax/keys_$arm.txt \
      --chunk-minutes 60 --C "$RC" --need "$(need_of $RC)" --minutes 80 >> "$LOG" 2>&1; rc=$?
    say "G $arm: chunk rc=$rc ($(have $arm)/$(want $arm) rows)"
    after "G $arm" $rc $b $(have $arm)
  done
done
say "replays"
for arm in $REPLAY; do
  NOPROG=0
  while [ "$(replayed $arm)" -lt "$(have $arm)" ]; do
    calm
    converted && { say "records3 is converted (index.json): the frozen engine reads it row-major: runner stops"; exit 8; }
    [ "$(pin)" = "$PIN0" ] || { say "code changed: pin $PIN0 -> $(pin): runner stops"; exit 7; }
    [ "$(left)" -gt 3600 ] || { say "deadline near: replays stop (arm $arm at $(replayed $arm)/$(have $arm))"; exit 6; }
    b=$(replayed $arm)
    mem_wait "$(need_of $RRC)" "$RRC" 3600
    EVALSUITE_STAGE_ENV="$STAGEENV" ./bin/evalsuite gpu relax-replay ref --arm $arm --chunk-minutes 60 --C "$RRC" \
      --need "$(need_of $RRC)" --minutes 80 --share >> "$LOG" 2>&1; rc=$?
    say "replay $arm: chunk rc=$rc ($(replayed $arm)/$(have $arm) prompts)"
    after "replay $arm" $rc $b $(replayed $arm)
  done
done
$PY tools/relax_report.py --score > results/relax_report_run.txt 2>&1; s=$?
say "report rc=$s: $(grep -- '->' results/relax_report.txt | tr '\n' ' ')"
touch results/RELAX_DONE
say "runner 40 end"
