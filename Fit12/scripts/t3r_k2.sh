#!/bin/sh
# Fit12 T3R K2 (2026-10-25; NewIdeas' bet T3R 19:58, ~/NewIdeas/QUEUE.md row T3R; pre-registered in PLAN.md 'T3R'): CPU only,
# no lock, <= 2 threads (fanwatch down since 13:56: NewIdeas' fan rule); fit_t3r_k2.py steps aside between experts while
# anyone holds or waits for quiet, or PAUSE exists -> data/t3r/k2.txt (+ k2_partial.txt per layer).
# Launch (background): cd /Users/kamen/Fit12 && CLAUDE_PID=23445 sh /Users/kamen/Fit12/scripts/t3r_k2.sh   (F12_T3R_SMOKE=1: 1 expert)
F=$HOME/Fit12; OUT=$F/data/t3r; LOG=$F/logs/summary.txt
mkdir -p "$OUT"
log() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
N=k2; [ "$F12_T3R_SMOKE" = 1 ] && N=k2_smoke
log "T3R $N start (no lock, 2 threads)"
env PYTHONDONTWRITEBYTECODE=1 CLAUDE_PID=${CLAUDE_PID:-23445} VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2 B3_THREADS=2 \
  nice -n 5 $HOME/BigMoE/.venv/bin/python -B $F/scripts/fit_t3r_k2.py > "$OUT/$N.txt" 2>&1
rc=$?
log "T3R $N done (rc $rc): $(grep -c DECISIVE "$OUT/$N.txt") decisive lines$(grep -E 'Error|Traceback' "$OUT/$N.txt" | tail -1 | cut -c1-200)"
