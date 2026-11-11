#!/bin/sh
# Fit12 RES2 (2026-10-24, Kamen 17:2x: 87.5 %, then 100 % if 448 shows room): C_m 448, then C_m 512 only if 448 kept
# available memory (free + inactive + speculative) > 6 GB throughout; both with BIGTEST. (RES1 header kept below.)
# Fit12 RES1 (2026-10-24): gate = 45 GB free to start c378 / c512 (c512 needs ~59: the orchestrator's watcher freezes
# apps; macOS compresses; both swap guards watch). Step A' (C_m 192 -> 288 -> 378, <= 45 GB) then Step A (C_m 512 = every MSB-3 base part resident,
# ~59 GB, Kamen's one-time OK 16:10) on B12LL + PR1 + tau 0.7 (exact w.r.t. U2: LSB for ranks 0-1 streamed as today, C_l
# small). Each process runs the 4 prompts twice (round 2 = warm pools). Guards: free-memory gate per arm (need + 3 GB), swap
# abort at +2 GB over the batch start, the lock keepalive; Step A writes its python PID to ~/.local/share/quiet/BIGTEST at
# launch (the orchestrator's watcher freezes apps, logs swap / pressure, SIGTERMs on its limits) and removes it at exit.
# Launch (background task) on a quiet-lock grant: cd ~/Fit12 && CLAUDE_PID=23445 sh ~/Fit12/scripts/bigres2.sh
F=$HOME/Fit12; OUT=$F/data/res2; LOG=$F/logs/summary.txt
PY=$HOME/BigMoE/.venv/bin/python; S=$F/scripts/fit_b12.py
BT=$HOME/.local/share/quiet/BIGTEST
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
availgb() { vm_stat | awk '/page size of/ {ps=$8} /Pages free/ {f=$3} /Pages inactive/ {i=$3} /Pages speculative/ {s=$3} END {printf "%.1f", (f+i+s)*ps/1e9}'; }
usedgb() { vm_stat | awk '/page size of/ {ps=$8} /Anonymous pages/ {a=$3} /Pages wired/ {w=$4} /occupied by compressor/ {c=$5} END {printf "%d", (a+w+c)*ps/1e9}'; }
SW0=$(swapmb)
bigstep() {   # bigstep NAME SECONDS_LIMIT NEED_GB BIGTEST(0/1) CMD...: runs only if 64 - (anon+wired+compressor) >= NEED + 3
  name=$1; lim=$2; need=$3; bt=$4; shift 4
  keepalive || { log "lock lost before $name"; return 125; }
  av=$(availgb); avi=${av%.*}
  [ "$avi" -ge "$need" ] || { log "memory: available $av GB < $need: $name skipped"; return 126; }
  log "start $name (available $av GB, anon+wired+compr $(usedgb) GB, swap $(swapmb) MB, gate $need GB)"
  MINAV=$av
  ( exec "$@" > "$OUT/$name.txt" 2>&1 ) & P=$!
  [ "$bt" = 1 ] && { echo "$P" > "$BT"; log "$name: BIGTEST <- pid $P"; }
  i=0; rc=0
  while kill -0 $P 2>/dev/null && [ $i -lt "$lim" ]; do
    sleep 5; i=$((i + 5))
    a=$(availgb); MINAV=$(echo "$a $MINAV" | awk '{print ($1 < $2) ? $1 : $2}')
    if [ $(( $(swapmb) - SW0 )) -gt 2048 ]; then log "swap grew > 2 GB during $name: stopping it"; kill $P; sleep 5; kill -9 $P 2>/dev/null; rc=127; break; fi
    if [ $((i % 120)) -eq 0 ]; then keepalive || { log "lost the lock during $name: stopping it"; kill $P; sleep 5; kill -9 $P 2>/dev/null; rc=125; break; }; fi
  done
  kill -0 $P 2>/dev/null && { kill -9 $P; log "TIMEOUT $name"; rc=124; }
  wait $P; w=$?; [ $rc -eq 0 ] && rc=$w
  [ "$bt" = 1 ] && { rm -f "$BT"; log "$name: BIGTEST removed"; }
  log "done $name (rc $rc, swap $(swapmb) MB, min available $MINAV GB): $(grep -cE '^\[.*(code|prose|math|write) k=' "$OUT/$name.txt") items$(grep -E 'Error|Traceback|timed out' "$OUT/$name.txt" | tail -1 | cut -c1-200)"
  return $rc
}
keepalive || { log "RES2: no lock"; exit 1; }
log "RES2 lock taken; swap $SW0 MB, available $(availgb) GB"
K=$F/v3k
A8=code:3,prose:3,math:3,write:3,code:3,prose:3,math:3,write:3
V="env F12_V3F=$K BIGMOE_JOINT=1 BIGMOE_SPIN_G2=1.05 BIGMOE_SPIN_PAIRS=256 F12_PR1=1 F12_WTAU=0.7"
bigstep res_c448 1200 34 1 $V B12_CM=448 B12_CL=48 F12_ARMS=$A8 $PY $S res_c448; r448=$?
M448=$MINAV
if [ "$r448" -eq 0 ] && [ "$(echo "$M448" | awk '{print ($1 > 6) ? 1 : 0}')" = 1 ]; then
  log "RES2: c448 kept available >= $M448 GB (> 6): trying c512"
  bigstep res_c512 1200 34 1 $V B12_CM=512 B12_CL=48 F12_ARMS=$A8 $PY $S res_c512 || log "RES2: c512 rc $?"
else
  log "RES2: c512 skipped (c448 rc $r448, min available $M448 GB)"
fi
rm -f "$BT"
log "RES2 done; swap $(swapmb) MB"
peer unlock >/dev/null 2>&1
