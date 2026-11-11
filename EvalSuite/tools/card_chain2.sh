#!/bin/bash
# Monday's card chain, restarted after lcoh's VOID at 06:37 (docs/PREREG_card_gates amendment 9). The reference is complete
# (381/381), so no reference step: lcoh resumes (CARD_RESUME=1: its kept fwd rows stay, its processes are checked against
# the 03:42 manifest; the 10 runs cut at the time budget and the whole rev order run now), then m1b -> zs32, each with
# CARD_CUTOFF 15:30 as in tools/card_chain.sh (amendment 8). After an arm: rc 0, 2 or 6 -> the next arm; any other rc
# (swap, code pin, store, VOID, a stage failure) -> the chain stops.
# Dry run: CHAIN_DRY=1 runs the arms' dry runs (no GPU step).
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
L=results/card_chain.log
log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$L"; }
ARMCUT="2026-11-09 15:30"
BMS=$HOME/BigMoE/data/stack
ARMS="lcoh m1b zs32"
refcount() { python3 - <<'RC'
import json
keys = {l.strip() for l in open("results/card/card_ref_keys.txt") if l.strip()}
have = {json.loads(l)["key"] for l in open("results/ref2tnaxn.r6/decode.jsonl") if l.strip()}
print(len(keys & have))
RC
}
n=$(refcount); [ "$n" = 381 ] || { log "chain2 stops: the reference has $n/381 rows"; exit 1; }
if [ "${CHAIN_DRY:-0}" = 1 ]; then
  for a in $ARMS; do
    r=0; [ "$a" = lcoh ] && r=1
    CARD_DRY=1 CARD_RESUME=$r CARD_TAG=$a CARD_FENV="$BMS/card_${a}_env.txt" CARD_CUTOFF="$ARMCUT" tools/day_runner61.sh
    log "DRY chain2 $a rc=$?"
  done
  exit 0
fi
log "card chain2 (pid $$): restart after lcoh's VOID at 06:37 (amendment 9); reference $n/381; $ARMS (lcoh resumes)"
for a in $ARMS; do
  [ -s "$BMS/card_${a}_env.txt" ] || { log "arm $a: no env file: skipped"; continue; }
  r=0; [ "$a" = lcoh ] && r=1
  log "step $a: CARD_RESUME=$r CARD_TAG=$a CARD_FENV=$BMS/card_${a}_env.txt CARD_CUTOFF=$ARMCUT"
  CARD_RESUME=$r CARD_TAG="$a" CARD_FENV="$BMS/card_${a}_env.txt" CARD_CUTOFF="$ARMCUT" tools/day_runner61.sh; rc=$?
  log "step $a: rc=$rc"
  case "$rc" in 0|2|6) ;; *) log "chain2 stops after arm $a (rc $rc)"; exit "$rc" ;; esac
  [ "$(date +%s)" -lt "$(date -j -f "%Y-%m-%d %H:%M" "$ARMCUT" +%s)" ] || { log "chain2 ends: past $ARMCUT"; break; }
done
log "chain2 end"
