#!/bin/sh
# Fit12 X1 + T1 (2026-10-24, NewIdeas QUEUE L1-02 + L1-03, then L2-02 + L2-03): DFlash2's REAL 8-row pass with exact (natural, unbudgeted) verify at
# 24-34 GB residency vs MTP k=3 on the same prompts. Pre-registered in ~/Fit12/PLAN.md (X1).
# Launch (background task) on a quiet-lock grant: CLAUDE_PID=23445 sh ~/Fit12/scripts/x1.sh
F=$HOME/Fit12; OUT=$F/data/x1; LOG=$F/logs/summary.txt
PY=$HOME/BigMoE/.venv/bin/python; S=$F/scripts/fit_b12.py; D=$F/scripts/fit_dfl2.py
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1 BT_N=256 F12_ASYNC=1 BIGMOE_PF_GROUP=64 F12_SAVE=$OUT
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
usedgb() { vm_stat | awk '/page size of/ {ps=$8} /Anonymous pages/ {a=$3} /Pages wired/ {w=$4} /occupied by compressor/ {c=$5} END {printf "%d", (a+w+c)*ps/1e9}'; }   # file cache excluded
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
  wait $P; [ $rc -eq 0 ] && rc=$?
  log "done $name (rc $rc): $(grep -cE '^\[.*(code|lru|csv|tf[cg][0-9]|p2k|p8k)(:| k=)|BITWISE|MISMATCH' "$OUT/$name.txt") items$(grep -E 'Error|Traceback' "$OUT/$name.txt" | tail -1 | cut -c1-200)"
  return $rc
}
keepalive || { log "X1: no lock"; exit 1; }
log "X1 lock taken; swap $SW0 MB"
$PY -m py_compile $S $D || { log "X1: scripts do not compile"; peer unlock; exit 1; }
# Kamen's quiet window: no GPU work before 03:45 (orchestrator 02:50); hold the lock alive until then
NB=$(date -j -f "%Y-%m-%d %H:%M:%S" "${F12_NOT_BEFORE:-2026-10-24 03:45:00}" +%s)
while [ "$(date +%s)" -lt "$NB" ]; do sleep 20; keepalive || { log "X1: lock lost while waiting for 03:45"; exit 1; }; done
log "X1: start time reached"
A=code,lru,csv,tfc0,tfc1,tfg0,tfg1
A3=code:3,lru:3,csv:3,tfc0:3,tfc1:3,tfg0:3,tfg1:3
step x_mtp192 900 30 env F12_FRAC=0.375 F12_ARMS=$A3 $PY $S x_mtp192
step x_df192 900 30 env F12_FRAC=0.375 F12_VW=8 F12_ARMS=$A $PY $D x_df192
# Leap's T16/T8 (02:38): oracle drafts = x_df192's greedy outputs; MAXK 80: 8 rows frugal, 16 rows v3 path; MAXK 70: both v3 path
O=code,tfc0,tfc1,tfg0,tfg1
step x_or192f 600 30 env F12_FRAC=0.375 F12_ORACLE=x_df192 F12_VWS=8,16 F12_ARMS=$O BIGMOE_MAXK=80 $PY $D x_or192f
step x_or192v 600 30 env F12_FRAC=0.375 F12_ORACLE=x_df192 F12_VWS=8,16 F12_ARMS=$O BIGMOE_MAXK=70 $PY $D x_or192v
step x_df256 900 38 env F12_FRAC=0.5 F12_VW=8 F12_ARMS=$A $PY $D x_df256
log "X1 done; swap $(swapmb) MB"
# T1 (L2-02 / L2-03): exact memory trims for C_m 59 (fit_trim.py), footprints per phase (fit_pf.py sampler) + decode shas
PF=$F/scripts/fit_pf.py; E=$F/scripts/fit_embed.py
$PY -m py_compile $PF $E $F/scripts/fit_trim.py || { log "T1: scripts do not compile"; peer unlock; exit 1; }
step t_embcheck 300 4 $PY $E
step t_base59 600 14 env B12_CM=59 B12_CL=20 $PY $PF t_base59
step t_hc59 600 14 env B12_CM=59 B12_CL=20 F12_HCPRE=1 $PY $PF t_hc59
step t_both59 600 14 env B12_CM=59 B12_CL=20 F12_HCPRE=1 F12_EMBED=1 $PY $PF t_both59
step t_both59s2k 600 14 env B12_CM=59 B12_CL=20 F12_HCPRE=1 F12_EMBED=1 F12_SMOE=2048 $PY $PF t_both59s2k
step t_both51 600 14 env F12_HCPRE=1 F12_EMBED=1 $PY $PF t_both51
step t_snap59 600 14 env B12_CM=59 B12_CL=20 F12_HCPRE=1 F12_EMBED=1 F12_FPSNAP=1 F12_PROMPTS=p8k $PY $PF t_snap59
step t_spd51 600 14 env BT_N=128 F12_SAVE= F12_ARMS=code:3,prose:3 $PY $S t_spd51
step t_spd59 600 14 env BT_N=128 F12_SAVE= B12_CM=59 B12_CL=20 F12_HCPRE=1 F12_EMBED=1 F12_ARMS=code:3,prose:3 $PY $S t_spd59
log "T1 done; swap $(swapmb) MB"
peer unlock >/dev/null 2>&1
