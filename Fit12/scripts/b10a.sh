#!/bin/sh
# Fit12 B10a (2026-10-25): capture 8 fit documents (~32K tokens) + 1 judge text at 9 layers (GPU, ~6-8 min), then the held-out trellis analysis
# (CPU, ~30 min at the 4-core cap of RULES.md, inside the quiet window; per-layer partial results in data/b10/b10a_partial.txt), then B8's fixed replay (~1 min). Pre-registered in PLAN.md (B10a).
# Launch (background task) on a quiet-lock grant: cd ~/Fit12 && CLAUDE_PID=23445 sh ~/Fit12/scripts/b10a.sh
F=$HOME/Fit12; OUT=$F/data/b10; LOG=$F/logs/summary.txt
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
left() { $PY -I -S -c 'import json, sys, time; print(max(60, int(json.load(open(sys.argv[1]))["until"] - time.time() - 150)))' "$LOCKF"; }   # seconds of lock left minus B8's 150 s
swapmb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}'; }
usedgb() { vm_stat | awk '/page size of/ {ps=$8} /Anonymous pages/ {a=$3} /Pages wired/ {w=$4} /occupied by compressor/ {c=$5} END {printf "%d", (a+w+c)*ps/1e9}'; }
SW0=$(swapmb)
step() {   # step NAME SECONDS_LIMIT NEED_GB CMD...: skipped unless anonymous + wired + compressor + NEED_GB <= 56 GB
  name=$1; lim=$2; need=$3; shift 3
  keepalive || { log "lock lost before $name"; return 125; }
  u=$(usedgb); [ $((u + need)) -le 56 ] || { log "memory: anon+wired+compr $u GB + $need GB > 56: $name skipped"; return 126; }
  log "start $name (anon+wired+compr $u GB, swap $(swapmb) MB)"
  ( "$@" > "$OUT/$name.txt" 2>&1 ) & P=$!
  i=0; rc=0
  while kill -0 $P 2>/dev/null && [ $i -lt "$lim" ]; do
    sleep 5; i=$((i + 5))
    if [ $(( $(swapmb) - SW0 )) -gt 2048 ]; then log "swap grew > 2 GB during $name: stopping it"; kill $P; sleep 5; kill -9 $P 2>/dev/null; rc=127; break; fi
    if [ $((i % 120)) -eq 0 ]; then keepalive || { log "lost the lock during $name: stopping it"; kill $P; sleep 5; kill -9 $P 2>/dev/null; rc=125; break; }; fi
  done
  kill -0 $P 2>/dev/null && { kill -9 $P; log "TIMEOUT $name"; rc=124; }
  wait $P; w=$?; [ $rc -eq 0 ] && rc=$w
  log "done $name (rc $rc): $(grep -cE '^\[.*(code|prose|math|write) k=' "$OUT/$name.txt") items$(grep -E 'Error|Traceback|timed out' "$OUT/$name.txt" | tail -1 | cut -c1-200)"
  return $rc
}
keepalive || { log "TK1: no lock"; exit 1; }
log "B10a lock taken; swap $SW0 MB"
K=$F/v3k
ls $K/ext/build/bigmoe_ext*.so >/dev/null 2>&1 || { log "B10a: v3k ext not built"; peer unlock; exit 1; }
V="env F12_V3F=$K BIGMOE_JOINT=1 BIGMOE_SPIN_G2=1.05 BIGMOE_SPIN_PAIRS=256"
step b10_cap 900 16 $V F12_CAP=$F/data/b10cap F12_CAP_LAYERS=2,7,12,18,24,30,36,41,46 F12_CAP_TEXTS=A0,A1,A2,A3,A4,A5,A6,A7,B $PY $S b10_cap
ls $F/data/b10cap/B/ids_L46.npy >/dev/null 2>&1 && step b10a "$(left)" 8 env PYTHONDONTWRITEBYTECODE=1 VECLIB_MAXIMUM_THREADS=4 OMP_NUM_THREADS=4 B3_THREADS=4 $PY -B $F/scripts/fit_b3.py b10a
step b8 300 4 env PYTHONDONTWRITEBYTECODE=1 $PY -B $F/scripts/fit_b8.py
cp $OUT/b8.txt $F/data/b8/b8_fixed.txt 2>/dev/null
log "B10a done; swap $(swapmb) MB"
peer unlock >/dev/null 2>&1
