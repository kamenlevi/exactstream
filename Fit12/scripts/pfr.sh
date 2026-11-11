#!/bin/sh
# Fit12 PFR T1 (2026-10-25; pre-registered in PLAN.md 'PFR pre-registration'): one quiet lock, three fresh processes on the
# v3p engine (Fit12 copy of v3k + ext.reset_layer / invalidate_tables), prompts p2k + p8k (8,474 tokens), 32 decode tokens:
#   ref51  F12_PFR=0 (shipped prefill: s_moe 4096, pools resident), C_m 51 / C_l 20
#   pfr51  F12_PFR=1 (trunk pools released for the prefill, one MoE chunk up to 16,384 tokens, pools re-created), C_m 51
#   pfr59  F12_PFR=1 at C_m 59 / C_l 20 (the decode-headroom part)
# SSD gate from the health monitor (~/Library/Logs/health.csv ssd_t; fanwatch is down): start each arm <= 66 C, stop at
# >= 73 C; swap guard +2 GB; 12 min per arm. Launch (background) right after a quiet-lock grant:
#   cd /Users/kamen/Fit12 && CLAUDE_PID=23445 sh /Users/kamen/Fit12/scripts/pfr.sh
F=$HOME/Fit12; OUT=$F/data/pfr; LOG=$F/logs/summary.txt; PY=$HOME/BigMoE/.venv/bin/python
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1 F12_V3F=$F/v3p
LOCKF=$HOME/.claude-peers/locks/${CLAUDE_PID:-23445}.json
log() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
keepalive() { $PY -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF"; }
ssd_t() { tail -1 "$HOME/Library/Logs/health.csv" | awk -F, -v now="$(date +%s)" '{ if (now - $1 < 180) printf "%d", $6 }'; }
swapmb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}'; }
keepalive || { log "PFR T1: no lock"; exit 1; }
SW0=$(swapmb)
log "PFR T1 lock taken; SSD $(ssd_t) C, swap $SW0 MB"
for arm in ref51 pfr51 pfr59; do
  i=0
  while t=$(ssd_t); { [ -z "$t" ] || [ "$t" -gt 66 ]; } && [ $i -lt 90 ]; do sleep 10; i=$((i + 1)); [ $((i % 12)) -eq 0 ] && keepalive; done
  t=$(ssd_t)
  if [ -z "$t" ] || [ "$t" -gt 66 ]; then log "PFR T1: SSD '${t:-unknown}' C before $arm after 15 min: stopping"; break; fi
  case $arm in
    ref51) E="F12_PFR=0" ;;
    pfr51) E="F12_PFR=1" ;;
    pfr59) E="F12_PFR=1 B12_CM=59 B12_CL=20" ;;
  esac
  log "PFR T1 $arm start (SSD $t C; $E)"
  env $E $PY -B $F/scripts/fit_pfr.py $arm > "$OUT/$arm.txt" 2>&1 &
  P=$!; j=0; rc=0
  while kill -0 $P 2>/dev/null && [ $j -lt 720 ]; do
    sleep 5; j=$((j + 5)); t=$(ssd_t)
    if [ -n "$t" ] && [ "$t" -ge 73 ]; then log "PFR T1: SSD $t C during $arm: stopping"; kill $P; sleep 5; kill -9 $P 2>/dev/null; rc=73; break; fi
    if [ $(( $(swapmb) - SW0 )) -gt 2048 ]; then log "PFR T1: swap +2 GB during $arm: stopping"; kill $P; sleep 5; kill -9 $P 2>/dev/null; rc=127; break; fi
    [ $((j % 120)) -eq 0 ] && { keepalive || { log "PFR T1: lost the lock during $arm"; kill $P; sleep 5; kill -9 $P 2>/dev/null; rc=125; break; }; }
  done
  kill -0 $P 2>/dev/null && { kill -9 $P; log "PFR T1: $arm TIMEOUT"; rc=124; }
  wait $P; w=$?; [ $rc -eq 0 ] && rc=$w
  log "PFR T1 $arm done (rc $rc, $j s, swap $(swapmb) MB): $(grep -c ' sha ' "$OUT/$arm.txt") prompts$(grep -E 'Error|Traceback' "$OUT/$arm.txt" | tail -1 | cut -c1-200)"
  [ "$rc" -eq 0 ] || break
done
peer unlock >/dev/null 2>&1
log "PFR T1 end; quiet lock released; SSD $(ssd_t) C, swap $(swapmb) MB (start $SW0 MB)"
