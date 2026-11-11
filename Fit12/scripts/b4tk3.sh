#!/bin/sh
# Fit12 (2026-10-25): B4 (read-path microbench) -> TK3 (top-p NaN fix check) -> B3b (out-of-sample trellis check, CPU, all
# cores; then B8's foresight-capped Belady replay and the fixed RD floor) in ONE quiet lock; B3b runs only with enough lock time left; unlock at the end.
# Launch (background task) after the quiet-lock grant: cd ~/Fit12 && CLAUDE_PID=23445 sh ~/Fit12/scripts/b4tk3.sh
F=$HOME/Fit12; LOG=$F/logs/summary.txt; PY=$HOME/BigMoE/.venv/bin/python; OUT=$F/data/b3
LOCKF=$HOME/.claude-peers/locks/${CLAUDE_PID:-23445}.json
log() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
left() { /usr/bin/python3 -c 'import json, sys, time
try: d = json.load(open(sys.argv[1]))
except Exception: print(0); sys.exit()
print(int((d.get("until", 0) - time.time()) / 60) if d.get("mode") == "quiet" else 0)' "$LOCKF"; }
B4_KEEP=1 sh $F/scripts/b4.sh
TK3_KEEP=1 CLAUDE_PID=${CLAUDE_PID:-23445} sh $F/scripts/tk3.sh
export PYTHONDONTWRITEBYTECODE=1 VECLIB_MAXIMUM_THREADS=8 OMP_NUM_THREADS=8 B3_THREADS=12
if [ "$(left)" -ge 9 ]; then
  log "B3b start ($(left) min of the lock left)"
  $PY -B $F/scripts/fit_b3.py oos > "$OUT/b3b_oos.txt" 2>&1; log "B3b oos rc $?: $(tail -2 $OUT/b3b_oos.txt | tr '\n' ' ' | cut -c1-300)"
else log "B3b skipped: $(left) min of the lock left"; fi
if [ "$(left)" -ge 4 ]; then
  $PY -B $F/scripts/fit_b8.py > "$F/data/b8/b8.txt" 2>&1; log "B8 rc $?: $(grep -c "^A=" $F/data/b8/b8.txt) rows"
else log "B8 skipped: $(left) min of the lock left"; fi
if [ "$(left)" -ge 3 ]; then
  $PY -B $F/scripts/fit_b3.py test > "$OUT/b3_test2.txt" 2>&1
  $PY -B $F/scripts/fit_b3.py rd > "$OUT/b3_rd.txt" 2>&1; log "B3 rd rc $?: $(tail -1 $OUT/b3_test2.txt | cut -c1-200)"
fi
peer unlock > /dev/null 2>&1
log "b4tk3 done"
