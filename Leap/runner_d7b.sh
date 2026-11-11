#!/bin/sh
# Leap DAY9 (Mon 2026-11-02 11:2x): D7b (r85b_d7b.py, ~5 GB, ~4 min) on one quiet ticket. Attached (session background).
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
log() { echo "[d7b] $* $(date '+%H:%M:%S')"; }
PEER_QUEUE_TIMEOUT=14400 peer lock quiet --queue 15m "Leap DAY9 #6: D7b mixed-precision drafter (which parts stay bf16), ~5 GB, ~4 min" >/dev/null 2>&1 || { log "no grant"; exit 1; }
log "quiet lock"
perl -e 'alarm shift; exec @ARGV' 600 $PY scripts/r85b_d7b.py > logs/r85b_d7b.out 2>&1
log "r85b_d7b: rc $?"
peer unlock >/dev/null 2>&1
log "done"
