#!/bin/bash
# PREREG_card_gates amendment 10: when card_chain2's step for arm $1 ends rc 6 on its 15:30 cutoff (the runner's "does
# not fit before the cutoff" or "no memory before the cutoff" stop; no DONE file), resume that arm at once with
# CARD_CUTOFF "$2" (the queue owner's OK, NewIdeas 07:22: m1b to 15:45, so its last process starts by 15:30).
# Any other end (rc 0, VOID, swap, a pin, a 'no free slot' INCOMPLETE, past the new cutoff) -> no resume.
# chain2 may start its next arm meanwhile: that runner's first gate stops on the same 15:30 cutoff before any lock or
# process, so the two never share the GPU.
# Test: EXT_CL=<chain log> EXT_ML=<arm log> EXT_PID=<pid> EXT_LOG=<log> EXT_DRY=1 tools/card_extend_cutoff.sh m1b "<cutoff>"
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
A=${1:?usage: card_extend_cutoff.sh TAG YYYY-mm-dd HH:MM}
NEWCUT=${2:?usage: card_extend_cutoff.sh TAG YYYY-mm-dd HH:MM}
CL=${EXT_CL:-results/card_chain.log}
ML=${EXT_ML:-results/day61_${A}_runner.log}
CP=${EXT_PID:-39264}
L=${EXT_LOG:-results/card_extend.log}
log() { echo "$(date '+%m-%d %H:%M:%S') extend $A: $*" | tee -a "$L"; }
cut_s=$(date -j -f "%Y-%m-%d %H:%M" "$NEWCUT" +%s 2>/dev/null) || { log "the cutoff '$NEWCUT' does not parse"; exit 2; }
[ -s "$HOME/BigMoE/data/stack/card_${A}_env.txt" ] || { log "no env file for $A"; exit 2; }
n=$(wc -l < "$CL")
log "armed (pid $$): waits for chain2's step $A to end; on a cutoff stop it resumes $A with CARD_CUTOFF $NEWCUT"
until tail -n +$((n + 1)) "$CL" | grep -q "step $A: rc="; do
  kill -0 "$CP" 2>/dev/null || { log "chain2 ($CP) ended before step $A did: no resume"; exit 1; }
  sleep 15
done
rc=$(tail -n +$((n + 1)) "$CL" | grep -o "step $A: rc=[0-9]*" | tail -1 | cut -d= -f2)
why=$(grep "runner stops" "$ML" | tail -1)
log "step $A ended rc=$rc; the runner's last stop line: ${why:-none}"
[ "$rc" = 6 ] || { log "not a cutoff stop: no resume"; exit 0; }
echo "$why" | grep -qE "does not fit before the cutoff|no memory before the cutoff" || { log "rc 6 but not a cutoff stop: no resume"; exit 0; }
[ $(( $(date +%s) + 900 )) -le "$cut_s" ] || { log "no process fits before $NEWCUT any more: no resume"; exit 0; }
[ -e "results/CARD_${A}_DONE" ] && { log "results/CARD_${A}_DONE exists: no resume"; exit 0; }
log "resume: CARD_RESUME=1 CARD_TAG=$A CARD_CUTOFF=$NEWCUT"
if [ "${EXT_DRY:-0}" = 1 ]; then log "DRY: would run tools/day_runner61.sh"; exit 0; fi
CARD_RESUME=1 CARD_TAG="$A" CARD_FENV="$HOME/BigMoE/data/stack/card_${A}_env.txt" CARD_CUTOFF="$NEWCUT" tools/day_runner61.sh
rc=$?
log "resumed $A: rc=$rc"
exit "$rc"
