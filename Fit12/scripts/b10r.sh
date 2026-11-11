#!/bin/sh
# Fit12 B10-r (2026-10-25; NewIdeas' final 2-bit test, ~/NewIdeas/QUEUE.md row B10-r; pre-registered in PLAN.md 'B10-r').
# GPU part under the quiet lock (~4 min): capture R0..R9 (broad calibration + validation, fit_b10r_text.py) and U (fragility
# judge) at B10a's 9 layers -> data/b10rcap, only once the SSD is <= 66 C; stop the capture at >= 73 C (BigMoE's ssd_gate
# rule). Then UNLOCK, and run the CPU analysis without a lock at 4 cores (fit_b3.py b10r: it steps aside between experts while
# anyone holds or waits for quiet, or PAUSE exists) -> data/b10r/b10r.txt (+ b10r_partial.txt per layer).
# Launch (background) right after a quiet-lock grant: cd /Users/kamen/Fit12 && CLAUDE_PID=23445 sh /Users/kamen/Fit12/scripts/b10r.sh
F=$HOME/Fit12; OUT=$F/data/b10r; LOG=$F/logs/summary.txt
PY=$HOME/BigMoE/.venv/bin/python; S=$F/scripts/fit_b12.py
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1 BT_N=128 F12_ASYNC=1 BIGMOE_PF_GROUP=64
LOCKF=$HOME/.claude-peers/locks/${CLAUDE_PID:-23445}.json
log() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
keepalive() { $PY -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF"; }
ssd_t() { tail -1 "$HOME/Library/Logs/fanwatch.log" 2>/dev/null | grep -o 'ssd [0-9]*' | awk '{print $2}'; }
swapmb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}'; }
keepalive || { log "B10-r: no lock"; exit 1; }
SW0=$(swapmb)
log "B10-r lock taken; SSD $(ssd_t) C, swap $SW0 MB"
i=0
while t=$(ssd_t); [ -n "$t" ] && [ "$t" -gt 66 ] && [ $i -lt 120 ]; do sleep 10; i=$((i + 1)); [ $((i % 12)) -eq 0 ] && keepalive; done
t=$(ssd_t)
if [ -n "$t" ] && [ "$t" -gt 66 ]; then log "B10-r: SSD still $t C after 20 min: no capture"; peer unlock >/dev/null 2>&1; exit 1; fi
log "B10-r capture start (SSD ${t:-?} C after $((i * 10)) s of cooling)"
env F12_V3F=$F/v3k BIGMOE_JOINT=1 BIGMOE_SPIN_G2=1.05 BIGMOE_SPIN_PAIRS=256 F12_CAP=$F/data/b10rcap \
  F12_CAP_LAYERS=2,7,12,18,24,30,36,41,46 F12_CAP_TEXTS=R0,R1,R2,R3,R4,R5,R6,R7,R8,R9,U $PY $S b10_cap > "$OUT/b10r_cap.txt" 2>&1 &
P=$!
i=0; rc=0
while kill -0 $P 2>/dev/null && [ $i -lt 900 ]; do
  sleep 5; i=$((i + 5))
  t=$(ssd_t)
  if [ -n "$t" ] && [ "$t" -ge 73 ]; then log "B10-r: SSD $t C >= 73 C during the capture: stopping it"; kill $P; sleep 5; kill -9 $P 2>/dev/null; rc=73; break; fi
  if [ $(( $(swapmb) - SW0 )) -gt 2048 ]; then log "B10-r: swap grew > 2 GB during the capture: stopping it"; kill $P; sleep 5; kill -9 $P 2>/dev/null; rc=127; break; fi
  [ $((i % 120)) -eq 0 ] && { keepalive || { log "B10-r: lost the lock during the capture: stopping it"; kill $P; sleep 5; kill -9 $P 2>/dev/null; rc=125; break; }; }
done
kill -0 $P 2>/dev/null && { kill -9 $P; log "B10-r: capture TIMEOUT"; rc=124; }
wait $P; w=$?; [ $rc -eq 0 ] && rc=$w
peer unlock >/dev/null 2>&1
n=$(ls $F/data/b10rcap/*/ids_L*.npy 2>/dev/null | wc -l | tr -d ' ')
log "B10-r capture done (rc $rc, $i s): $n of 99 layer files; SSD $(ssd_t) C; quiet lock released$(grep -E 'Error|Traceback' "$OUT/b10r_cap.txt" | tail -1 | cut -c1-200)"
[ "$rc" -eq 0 ] && [ "$n" -eq 99 ] || exit 1
log "B10-r analysis start (no lock, 4 cores, steps aside for the GPU queue)"
env CLAUDE_PID=${CLAUDE_PID:-23445} VECLIB_MAXIMUM_THREADS=4 OMP_NUM_THREADS=4 B3_THREADS=4 $PY -B $F/scripts/fit_b3.py b10r > "$OUT/b10r.txt" 2>&1
log "B10-r analysis done (rc $?): $(grep -c DECISIVE "$OUT/b10r.txt") decisive lines$(grep -E 'Error|Traceback' "$OUT/b10r.txt" | tail -1 | cut -c1-200)"
