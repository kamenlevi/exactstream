#!/bin/bash
# EvalSuite RELAX smoke (docs/PREREG_relax_2026-11-02.md, Runs): every arm on 2-3 short prompts + step 0, A again with
# the masking ids (Q1's), then REF2-T replays of control / A / B, then tools/relax_smoke_check.py -> results/relax_smoke.txt.
# Shared GPU (`gpu --share`, bits only, no timing), C 192 (peak ~33 GB: --need 34). Run after tools/relax_wire.sh.
# Stop it by pid only, and its cli child too.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/relax_smoke_runner.log
export PEER_QUEUE_TIMEOUT=7200
unset MTPLX_FUSED_GATE_UP BIGMOE_TIER BIGMOE_TIER_GLU BIGMOE_TIER_CL BIGMOE_TIER_COMBINE BIGMOE_OOO BIGMOE_VERIFY_MIDLOOP BIGMOE_EXT_BUILD BIGMOE_MAXK
export EVALSUITE_BIGMOE_CODE="$HOME/EvalSuite/snap/bigmoe-6c64148"
STAGEENV="BIGMOE_MAXK=160,BIGMOE_EXT_BUILD=build_next,BIGMOE_TIER=4433333333,BIGMOE_TIER_GLU=3c,BIGMOE_TIER_CL=112,BIGMOE_TIER_COMBINE=1,BIGMOE_OOO=1"
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
[ -f evalsuite/relax.py ] && grep -q '"relax"' evalsuite/stages.py || { say "not wired (tools/relax_wire.sh first)"; exit 2; }
calm() {
  while pgrep -f '^/bin/zsh [^ ]*soft_end\.sh' >/dev/null || [ -e "$HOME/.local/share/quiet/PAUSE" ]; do sleep 30; done
}
stage() {   # label, then the stage's args; env vars for the stage come from the caller
  local lab=$1; shift
  calm
  EVALSUITE_STAGE_ENV="$STAGEENV" ./bin/evalsuite gpu "$@" --chunk-minutes 15 --C 192 --need 34 --minutes 25 --share \
    >> "$LOG" 2>&1; local rc=$?
  say "$lab: rc=$rc"
  if [ $rc = 5 ]; then say "$lab: VOID (step 0): smoke stops"; exit 5; fi
  if [ $rc = 76 ]; then say "$lab: exited on a long PAUSE: again after it"; stage "$lab" "$@"; return; fi
  if [ $rc != 0 ] && tail -40 results/gpu_stage.log | grep -q "Traceback"; then say "$lab: stage failed: smoke stops"; exit 4; fi
}
say "relax smoke start (pid $$)"
for arm in control A B E F C G; do
  EVALSUITE_RELAX_TAG=smoke EVALSUITE_RELAX_HASH=1 stage "G $arm" relax ref --arm $arm --keys results/relax/smoke_keys_$arm.txt
done
STAGEENV_REF=$STAGEENV; STAGEENV=${STAGEENV/BIGMOE_TIER=4433333333/BIGMOE_TIER=3333333333}
EVALSUITE_RELAX_TAG=smoke EVALSUITE_RELAX_HASH=1 stage "G T3ALL" relax ref --arm T3ALL --keys results/relax/smoke_keys_T3ALL.txt
STAGEENV=$STAGEENV_REF
EVALSUITE_RELAX_TAG=smokemask EVALSUITE_RELAX_HASH=1 EVALSUITE_RELAX_GATE=mask stage "G A mask" relax ref --arm A \
  --keys results/relax/smoke_keys_A.txt
for arm in control A B T3ALL; do
  EVALSUITE_RELAX_TAG=smoke stage "replay $arm" relax-replay ref --arm $arm
done
.venv/bin/python tools/relax_smoke_check.py > results/relax_smoke.txt 2>&1; rc=$?
say "smoke check rc=$rc: $(tail -1 results/relax_smoke.txt)"
touch results/RELAX_SMOKE_DONE
