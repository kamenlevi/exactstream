#!/bin/sh
# Fit12 O4 (2026-10-24): fold check of /Users/kamen/Fit12/patches/b1v2_joint_v3f.diff = Breakthrough's v3f + B1 v2 only, in a
# fresh copy (~/Fit12/v3k, configured and built here). Exact: shas = U1's u_nat, JOINT off and on. PLAN.md (O4).
# Launch (background task) on a quiet-lock grant: cd ~/Fit12 && CLAUDE_PID=23445 sh ~/Fit12/scripts/o4.sh
F=$HOME/Fit12; OUT=$F/data/o4; LOG=$F/logs/summary.txt
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
keepalive || { log "O4: no lock"; exit 1; }
log "O4 lock taken; swap $SW0 MB"
K=$F/v3k
( cd $K && nice -n 5 cmake -S ext -B ext/build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$PY -DNB_CMAKE_DIR=$HOME/BigMoE/.nb215/nanobind/cmake > ext/configure.log 2>&1 \
  && nice -n 5 cmake --build ext/build -j 2 > ext/build.log 2>&1 ) || { log "O4: v3k does not build"; peer unlock; exit 1; }
grep -q "BIGMOE_JOINT" $K/ext/build/bigmoe_ext*.so || { log "O4: built ext lacks BIGMOE_JOINT"; peer unlock; exit 1; }
$PY -m py_compile $S $K/bigmoe_v3f/gated.py $K/bigmoe_v3f/stream.py || { log "O4: scripts do not compile"; peer unlock; exit 1; }
A4=code:3,prose:3,math:3,write:3
A2=code:3,prose:3
step k_off_a 400 14 env F12_V3F=$K BIGMOE_JOINT=0 F12_ARMS=$A4 $PY $S k_off_a
step k_on_a 400 14 env F12_V3F=$K BIGMOE_JOINT=1 F12_ARMS=$A4 $PY $S k_on_a
step k_off_b 400 14 env F12_V3F=$K BIGMOE_JOINT=0 F12_ARMS=$A4 $PY $S k_off_b
step k_on_b 400 14 env F12_V3F=$K BIGMOE_JOINT=1 F12_ARMS=$A4 $PY $S k_on_b
step k_59 400 14 env F12_V3F=$K BIGMOE_JOINT=1 B12_CM=59 B12_CL=20 F12_HCPRE=1 F12_EMBED=1 F12_ARMS=$A2 $PY $S k_59
log "O4 done; swap $(swapmb) MB"
peer unlock >/dev/null 2>&1
