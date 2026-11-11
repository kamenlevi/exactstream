#!/bin/sh
# Fit12 PFR T2 (2026-10-25; pre-registered in PLAN.md 'PFR T2 pre-registration'): PFR at 64K on the AR path, one quiet lock,
# four fresh processes on the v3p engine (fit_pfr64.py, N 65000, 32 greedy decode steps):
#   ref64 (s_moe 4096) | pfr64 (PFR cap 16384) | lane64 (L2-10 lane, s_moe 4096) | lanepfr64 (lane + PFR)
# SSD gate from the health monitor (~/Library/Logs/health.csv ssd_t): start each arm <= 66 C, stop at >= 73 C; swap guard
# +2 GB; 12 min per arm; an arm is skipped when < 8 min of the lock is left. Launch (background) right after a grant:
#   cd /Users/kamen/Fit12 && CLAUDE_PID=23445 sh /Users/kamen/Fit12/scripts/pfr64.sh
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
left() { $PY -I -S -c 'import json, sys, time; print(int(json.load(open(sys.argv[1])).get("until", 0) - time.time()))' "$LOCKF" 2>/dev/null || echo 0; }
ssd_t() { tail -1 "$HOME/Library/Logs/health.csv" | awk -F, -v now="$(date +%s)" '{ if (now - $1 < 180) printf "%d", $6 }'; }
swapmb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}'; }
keepalive || { log "PFR T2: no lock"; exit 1; }
SW0=$(swapmb)
log "PFR T2 lock taken ($(left) s left); SSD $(ssd_t) C, swap $SW0 MB"
for arm in ref64 pfr64 lane64 lanepfr64; do
  i=0
  while t=$(ssd_t); { [ -z "$t" ] || [ "$t" -gt 66 ]; } && [ $i -lt 90 ]; do sleep 10; i=$((i + 1)); [ $((i % 12)) -eq 0 ] && keepalive; done
  t=$(ssd_t)
  if [ -z "$t" ] || [ "$t" -gt 66 ]; then log "PFR T2: SSD '${t:-unknown}' C before $arm after 15 min: stopping"; break; fi
  if [ "$(left)" -lt 480 ]; then log "PFR T2: < 8 min of lock left before $arm: stopping"; break; fi
  case $arm in
    ref64) E="F12_PFR=0" ;;
    pfr64) E="F12_PFR=1" ;;
    lane64) E="F12_PFR=0 F12_LANE=1" ;;
    lanepfr64) E="F12_PFR=1 F12_LANE=1" ;;
  esac
  log "PFR T2 $arm start (SSD $t C; $E)"
  env $E $PY -B $F/scripts/fit_pfr64.py $arm > "$OUT/$arm.txt" 2>&1 &
  P=$!; j=0; rc=0
  while kill -0 $P 2>/dev/null && [ $j -lt 720 ]; do
    sleep 5; j=$((j + 5)); t=$(ssd_t)
    if [ -n "$t" ] && [ "$t" -ge 73 ]; then log "PFR T2: SSD $t C during $arm: stopping"; kill $P; sleep 5; kill -9 $P 2>/dev/null; rc=73; break; fi
    if [ $(( $(swapmb) - SW0 )) -gt 2048 ]; then log "PFR T2: swap +2 GB during $arm: stopping"; kill $P; sleep 5; kill -9 $P 2>/dev/null; rc=127; break; fi
    [ $((j % 120)) -eq 0 ] && { keepalive || { log "PFR T2: lost the lock during $arm"; kill $P; sleep 5; kill -9 $P 2>/dev/null; rc=125; break; }; }
  done
  kill -0 $P 2>/dev/null && { kill -9 $P; log "PFR T2: $arm TIMEOUT"; rc=124; }
  wait $P; w=$?; [ $rc -eq 0 ] && rc=$w
  log "PFR T2 $arm done (rc $rc, $j s, swap $(swapmb) MB, SSD $(ssd_t) C): $(grep -c ' sha ' "$OUT/$arm.txt") decodes$(grep -E 'Error|Traceback' "$OUT/$arm.txt" | tail -1 | cut -c1-200)"
  [ "$rc" -eq 0 ] || break
done
peer unlock >/dev/null 2>&1
log "PFR T2 end; quiet lock released; SSD $(ssd_t) C, swap $(swapmb) MB (start $SW0 MB)"
