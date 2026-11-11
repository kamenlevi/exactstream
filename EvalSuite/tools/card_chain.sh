#!/bin/bash
# Monday's card chain (docs/PREREG_card_gates amendment 8; ~/Papers/MONDAY.md; NewIdeas 19:2x): once F2a stage 1 is
# done (the last line of ~/TrainLab/data/f2a/night.log says "F2a night DONE"), or at 03:55 at the latest (NewIdeas 19:5x), with no gap:
#   the card reference's leftover (CARD_CUTOFF 16:00), then the arms lcoh -> m1b -> zs32 (the 32 GB chat arm, C 160),
#   each with CARD_CUTOFF 15:30: gate() assumes 900 s per process, and the last process must END by 16:00 (the
#   orchestrator 19:4x: gates may run to 16:00; apps closed 23:00 -> Mon 17:00, so the arms keep MEM + 4).
# Tonight's runner (results/q61r.pid) ends first. A PAUSE guard runs beside each reference step (amendment 6).
# The chain stops if the reference is not complete. After an arm: rc 0 (a verdict, or rows done), 2 (its own config) or
# 6 (its own abort or the cutoff) -> the next arm; any other rc (swap, code pin, store, a stage failure) -> the chain stops.
# Dry run: CHAIN_DRY=1 prints the plan and the arms' dry runs, then stops before any GPU step.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
L=results/card_chain.log
log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$L"; }
F2A=$HOME/TrainLab/data/f2a/night.log; S1=$HOME/TrainLab/data/f2a/STAGE1_DONE
LATEST=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-09 03:55" +%s)
REFCUT="2026-11-09 16:00"; ARMCUT="2026-11-09 15:30"
BMS=$HOME/BigMoE/data/stack
ARMS="lcoh m1b zs32"
T0=$(date +%s); N0=$(wc -l < "$F2A" 2>/dev/null || echo 0)
# F2a stage 1's end (MTP Head: written after its peer unlock, on any exit after 23:00): its STAGE1_DONE file, newer than
# this chain's start, or a NEW night.log line that is exactly "[Day HH:MM:SS] F2a night DONE" (an armed line once quoted
# those words, NewIdeas 19:3x).
f2a_done() { { [ -f "$S1" ] && [ "$(stat -f %m "$S1")" -ge "$T0" ]; } \
  || tail -n +$((N0 + 1)) "$F2A" 2>/dev/null | grep -qE '^\[[A-Z][a-z]{2} [0-9:]{8}\] F2a night DONE$'; }
refcount() { python3 - <<'RC'
import json
keys = {l.strip() for l in open("results/card/card_ref_keys.txt") if l.strip()}
have = {json.loads(l)["key"] for l in open("results/ref2tnaxn.r6/decode.jsonl") if l.strip()}
print(len(keys & have))
RC
}
if [ "${CHAIN_DRY:-0}" = 1 ]; then
  log "DRY chain: F2a DONE now: $(f2a_done && echo yes || echo no); latest start $(date -r "$LATEST" '+%m-%d %H:%M'); reference $(refcount)/381"
  CARD_DRY=1 CARD_STAGE=ref CARD_TAG=ref CARD_WAIT_STRATA=1 CARD_CUTOFF="$REFCUT" tools/day_runner61.sh; log "DRY ref rc=$?"
  for a in $ARMS; do
    [ -s "$BMS/card_${a}_env.txt" ] || { log "DRY $a: no env file $BMS/card_${a}_env.txt"; continue; }
    CARD_DRY=1 CARD_TAG=$a CARD_FENV="$BMS/card_${a}_env.txt" CARD_CUTOFF="$ARMCUT" tools/day_runner61.sh; log "DRY $a rc=$?"
  done
  exit 0
fi
log "card chain (pid $$): after tonight's runner, then F2a night DONE or 03:55: ref leftover -> $ARMS"
while [ -s results/q61r.pid ] && kill -0 "$(cat results/q61r.pid)" 2>/dev/null; do sleep 60; done
until f2a_done || [ "$(date +%s)" -ge "$LATEST" ]; do sleep 20; done
log "start: $(f2a_done && echo "F2a night DONE" || echo "03:55 (no F2a DONE line)"); reference $(refcount)/381"
step() {        # name, env assignments...: runner 61 in the foreground of this chain; returns its exit code
  local name=$1 r g rc; shift
  log "step $name: $*"
  env "$@" tools/day_runner61.sh & r=$!
  if [ "$name" = ref ]; then tools/card_pause_guard.sh "$r" > /dev/null 2>&1 & g=$!; else g=""; fi
  wait "$r"; rc=$?
  [ -n "$g" ] && kill "$g" 2>/dev/null
  log "step $name: rc=$rc"
  return $rc
}
step ref CARD_STAGE=ref CARD_TAG=ref CARD_WAIT_STRATA=1 CARD_CUTOFF="$REFCUT"
n=$(refcount); [ "$n" = 381 ] || { log "chain stops: the reference has $n/381 rows"; exit 1; }
for a in $ARMS; do
  [ -s "$BMS/card_${a}_env.txt" ] || { log "arm $a: no env file: skipped"; continue; }
  step "$a" CARD_TAG="$a" CARD_FENV="$BMS/card_${a}_env.txt" CARD_CUTOFF="$ARMCUT"; rc=$?
  case "$rc" in 0|2|6) ;; *) log "chain stops after arm $a (rc $rc)"; exit "$rc" ;; esac
  [ "$(date +%s)" -lt "$(date -j -f "%Y-%m-%d %H:%M" "$ARMCUT" +%s)" ] || { log "chain ends: past $ARMCUT"; break; }
done
log "chain end"
