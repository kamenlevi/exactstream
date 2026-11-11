#!/bin/sh
# Fit12 (2026-10-25 07:12): the rest of the 07:08 quiet window after TK3 (b4tk3.sh stopped after B4's build failure):
# wait for TK3's runner to end, then B4 (fixed build) -> B3b (out-of-sample, rows deduped) -> B8 -> RD floor; unlock.
# Launch (background task): cd ~/Fit12 && CLAUDE_PID=23445 sh ~/Fit12/scripts/post_tk3.sh <tk3 runner pid>
F=$HOME/Fit12; LOG=$F/logs/summary.txt; PY=$HOME/BigMoE/.venv/bin/python; OUT=$F/data/b3
LOCKF=$HOME/.claude-peers/locks/${CLAUDE_PID:-23445}.json
log() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
left() { /usr/bin/python3 -c 'import json, sys, time
try: d = json.load(open(sys.argv[1]))
except Exception: print(0); sys.exit()
print(int((d.get("until", 0) - time.time()) / 60) if d.get("mode") == "quiet" else 0)' "$LOCKF"; }
while kill -0 "$1" 2>/dev/null; do sleep 2; done
log "post_tk3: TK3 runner gone, $(left) min of the lock left"
[ "$(left)" -ge 3 ] && B4_KEEP=1 sh $F/scripts/b4.sh
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
log "post_tk3 done"
