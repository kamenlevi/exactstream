#!/bin/sh
# L1-20 chain (Leap, Sat 2026-10-24): while the session sleeps on its usage pacer, wait for batch 1 (l120_b1.sh) to end,
# check it produced every feature file, then queue a quiet lock and run batch 2 (training + scoring) under run_locked.sh.
# Log: ~/Leap/logs/l120_chain.log
cd "$HOME/Leap" || exit 1
LOG=$HOME/Leap/logs/l120_chain.log
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
log "waiting for batch 1"
while ! tail -1 logs/locked.log | grep -q "job done"; do sleep 30; done
want=$(wc -l < data/l120_windows.jsonl 2>/dev/null || echo 0)
have=$(ls data/l120/*.npz 2>/dev/null | wc -l)
log "batch 1 ended: $(tail -1 logs/locked.log); features $have / $want"
if [ "$want" -eq 0 ] || [ "$have" -lt "$want" ]; then log "incomplete: not starting batch 2"; exit 1; fi
log "queueing the lock for batch 2"
peer lock quiet --queue 120m "Leap L1-20 b2: block-32 DFlash2 fine-tune (<= 50 min) + replay" >> "$LOG" 2>&1 || { log "no lock"; exit 1; }
log "lock held: running batch 2"
"$HOME/Leap/run_locked.sh" 75 sh "$HOME/Leap/scripts/l120_b2.sh"
log "batch 2 ended: $(tail -1 logs/locked.log)"
