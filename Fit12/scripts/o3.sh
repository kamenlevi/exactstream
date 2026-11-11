#!/bin/sh
# Fit12 O3 (2026-10-24): MoonShots' exact GDN conv+norm (F12_GDNX=1, gdn_exact_patch) on B12LL, on the B1 v2 stack and on
# stock v3f. Exact: decode shas must equal U1's u_nat on every arm. Pre-registered in PLAN.md (O3).
# Launch (background task) on a quiet-lock grant: cd ~/Fit12 && CLAUDE_PID=23445 sh ~/Fit12/scripts/o3.sh
F=$HOME/Fit12; OUT=$F/data/o3; LOG=$F/logs/summary.txt
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
keepalive || { log "O3: no lock"; exit 1; }
log "O3 lock taken; swap $SW0 MB"
$PY -m py_compile $S $F/scripts/fit_ooo.py $F/scripts/fit_trim.py || { log "O3: scripts do not compile"; peer unlock; exit 1; }
A4=code:3,prose:3,math:3,write:3
A2=code:3,prose:3
J=$F/v3j
V="env F12_V3F=$J BIGMOE_JOINT=1"
step o3_base_a 400 14 $V F12_ARMS=$A4 $PY $S o3_base_a
step o3_gdn_a 400 14 $V F12_GDNX=1 F12_ARMS=$A4 $PY $S o3_gdn_a
step o3_base_b 400 14 $V F12_ARMS=$A4 $PY $S o3_base_b
step o3_gdn_b 400 14 $V F12_GDNX=1 F12_ARMS=$A4 $PY $S o3_gdn_b
# fold check on Breakthrough's stock v3f (U1 = 21.3 / 13.0 / 16.8 / 14.5)
step o3_ref_gdn 400 14 env F12_GDNX=1 F12_ARMS=$A4 $PY $S o3_ref_gdn
# the stack: C_m 59 + both trims + B1 v2 + exact GDN
step o3_59 400 14 $V F12_GDNX=1 B12_CM=59 B12_CL=20 F12_HCPRE=1 F12_EMBED=1 F12_ARMS=$A2 $PY $S o3_59
log "O3 done; swap $(swapmb) MB"
peer unlock >/dev/null 2>&1
