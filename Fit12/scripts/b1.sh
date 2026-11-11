#!/bin/sh
# Fit12 B1 (2026-10-24): one read burst per layer (~/Fit12/v3j, BIGMOE_JOINT=1) vs the sequential base -> sidecar service.
# Exact: decode shas must equal U1's u_nat (Breakthrough v3f) on all four prompts, JOINT on and off. Pre-registered in PLAN.md (B1).
# Launch (background task) on a quiet-lock grant: CLAUDE_PID=23445 sh ~/Fit12/scripts/b1.sh
F=$HOME/Fit12; OUT=$F/data/b1; LOG=$F/logs/summary.txt
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
keepalive || { log "B1: no lock"; exit 1; }
log "B1 lock taken; swap $SW0 MB"
$PY -m py_compile $S $F/v3j/bigmoe_v3f/gated.py $F/v3j/bigmoe_v3f/stream.py || { log "B1: scripts do not compile"; peer unlock; exit 1; }
ls $F/v3j/ext/build/bigmoe_ext*.so >/dev/null 2>&1 || { log "B1: v3j ext not built"; peer unlock; exit 1; }
A4=code:3,prose:3,math:3,write:3
J=$F/v3j
step b1_ref 400 14 env F12_ARMS=$A4 $PY $S b1_ref
step b1_off_a 400 14 env F12_V3F=$J BIGMOE_JOINT=0 BIGMOE_CALLLOG=$OUT/clog_off_a.csv F12_ARMS=$A4 $PY $S b1_off_a
step b1_on_a 400 14 env F12_V3F=$J BIGMOE_JOINT=1 BIGMOE_CALLLOG=$OUT/clog_on_a.csv F12_ARMS=$A4 $PY $S b1_on_a
step b1_off_b 400 14 env F12_V3F=$J BIGMOE_JOINT=0 BIGMOE_CALLLOG=$OUT/clog_off_b.csv F12_ARMS=$A4 $PY $S b1_off_b
step b1_on_b 400 14 env F12_V3F=$J BIGMOE_JOINT=1 BIGMOE_CALLLOG=$OUT/clog_on_b.csv F12_ARMS=$A4 $PY $S b1_on_b
log "B1 done; swap $(swapmb) MB"
# T1 follow-up: t_spd59 (C_m 59 + both trims, fit_b12) died in its warm-up with 47 spin-gate timeouts; fit_pf decoded the
# same config cleanly. Which part does it: rerun, then no trims / hc_pre only / embed only, and both trims at C_m 51.
A2=code:3,prose:3
step i59both 400 14 env B12_CM=59 B12_CL=20 F12_HCPRE=1 F12_EMBED=1 F12_ARMS=$A2 $PY $S i59both
step i59plain 400 14 env B12_CM=59 B12_CL=20 F12_ARMS=$A2 $PY $S i59plain
step i59hc 400 14 env B12_CM=59 B12_CL=20 F12_HCPRE=1 F12_ARMS=$A2 $PY $S i59hc
step i59emb 400 14 env B12_CM=59 B12_CL=20 F12_EMBED=1 F12_ARMS=$A2 $PY $S i59emb
step i51both 400 14 env F12_HCPRE=1 F12_EMBED=1 F12_ARMS=$A2 $PY $S i51both
log "T1 follow-up done; swap $(swapmb) MB"
peer unlock >/dev/null 2>&1
