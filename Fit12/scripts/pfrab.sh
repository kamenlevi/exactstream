#!/bin/sh
# Fit12 PFR A/B (2026-10-26; pre-registered in PLAN.md "PFR A/B pre-registration" + "PFR A/B additions" (08:09)): which part of PFR
# broke bit-exactness in T2. Fresh processes on the v3p engine (fit_pfr64.py, AR path, 32 greedy steps; KV hash at the end
# of the prefill, first-step logits hash + .npy, 32-token sha), arms from $ARMS (the first is the reference):
#   r8 ref s_moe 4096 / s8 s_moe 8474 pools resident / c8 PFR exactly 4096 / p8 PFR cap 16384 (N 8474);
#   r64 / s64 / c64 / a64 (PFR-a, 512-aligned 16384 chunks) (N 65000)
# NewIdeas 03:03: every arm whose RESULT (kv, logits, sha32, sha8) differs from the reference's is run once more as <arm>b,
# and then so is the reference (REPEAT=0: no repeats; REPREF=0: not the reference).
# SSD gate from the health monitor (~/Library/Logs/health.csv ssd_t): start each arm <= 66 C, stop at >= 73 C; swap guard
# +2 GB; 12 min per arm; an arm is skipped when less lock is left than it needs (150 s at 8.5K, 480 s at 64K). Launch
# (background) right after a grant:
#   cd /Users/kamen/Fit12 && ARMS="r8 s8 c8 p8" CLAUDE_PID=23445 sh /Users/kamen/Fit12/scripts/pfrab.sh
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
res() { grep ' RESULT ' "$OUT/ab_$1.txt" 2>/dev/null | sed 's/.* RESULT //'; }
env_of() {
  case ${1%b} in
    r8) echo "F12_N=8474 F12_PFR=0" ;;
    s8) echo "F12_N=8474 F12_PFR=0 F12_SMOE=8474" ;;
    c8) echo "F12_N=8474 F12_PFR=1 F12_PFR_CAP=4096 F12_PFR_BAL=0" ;;
    p8) echo "F12_N=8474 F12_PFR=1" ;;
    r64) echo "F12_PFR=0" ;;
    s64) echo "F12_PFR=0 F12_SMOE=16250" ;;
    c64) echo "F12_PFR=1 F12_PFR_CAP=4096 F12_PFR_BAL=0" ;;
    a64) echo "F12_PFR=1 F12_PFR_CAP=16384 F12_PFR_BAL=0" ;;   # PFR-a: 512-aligned chunks 16384 x 3 + 15848
    a64m) echo "F12_PFR=1 F12_PFR_CAP=16384 F12_PFR_BAL=0 F12_KVHASH=0" ;;   # a64 without the KV hash (memory)
  esac
}
run_arm() {
  arm=$1; E=$(env_of "$arm")
  [ -n "$E" ] || { log "PFR AB: unknown arm $arm"; return 2; }
  case $arm in *64*) need=480 ;; *) need=150 ;; esac
  i=0
  while t=$(ssd_t); { [ -z "$t" ] || [ "$t" -gt 66 ]; } && [ $i -lt 90 ]; do sleep 10; i=$((i + 1)); [ $((i % 12)) -eq 0 ] && keepalive; done
  t=$(ssd_t)
  if [ -z "$t" ] || [ "$t" -gt 66 ]; then log "PFR AB: SSD '${t:-unknown}' C before $arm after 15 min: stopping"; return 3; fi
  if [ "$(left)" -lt $need ]; then log "PFR AB: < $need s of lock left before $arm: stopping"; return 4; fi
  log "PFR AB $arm start (SSD $t C; $(peer top 2>/dev/null | head -1 | grep -o 'GPU [0-9]*%') before it, top process $(ps -Ao pcpu,comm -r | sed -n 2p | awk '{n = split($2, a, "/"); print $1 "% " a[n]}'); $E)"
  env $E F12_LOGITS="$OUT/ab_${arm}_logits.npy" $PY -B $F/scripts/fit_pfr64.py $arm > "$OUT/ab_$arm.txt" 2>&1 &
  P=$!; j=0; rc=0
  while kill -0 $P 2>/dev/null && [ $j -lt 720 ]; do
    sleep 5; j=$((j + 5)); t=$(ssd_t)
    if [ -n "$t" ] && [ "$t" -ge 73 ]; then log "PFR AB: SSD $t C during $arm: stopping"; kill $P; sleep 5; kill -9 $P 2>/dev/null; rc=73; break; fi
    if [ $(( $(swapmb) - SW0 )) -gt 2048 ]; then log "PFR AB: swap +2 GB during $arm: stopping"; kill $P; sleep 5; kill -9 $P 2>/dev/null; rc=127; break; fi
    [ $((j % 120)) -eq 0 ] && { keepalive || { log "PFR AB: lost the lock during $arm"; kill $P; sleep 5; kill -9 $P 2>/dev/null; rc=125; break; }; }
  done
  kill -0 $P 2>/dev/null && { kill -9 $P; log "PFR AB: $arm TIMEOUT"; rc=124; }
  wait $P; w=$?; [ $rc -eq 0 ] && rc=$w
  log "PFR AB $arm done (rc $rc, $j s, swap $(swapmb) MB, SSD $(ssd_t) C): $(res "$arm")$(grep -E 'Error|Traceback' "$OUT/ab_$arm.txt" | tail -1 | cut -c1-200)"
  return $rc
}
keepalive || { log "PFR AB: no lock"; exit 1; }
SW0=$(swapmb)
log "PFR AB lock taken ($(left) s left); SSD $(ssd_t) C, swap $SW0 MB"
ok=1
for a in ${ARMS:-r8 s8 c8 p8}; do run_arm "$a" || { ok=0; break; }; done
if [ $ok -eq 1 ] && [ "${REPEAT:-1}" = 1 ]; then
  set -- ${ARMS:-r8 s8 c8 p8}; ref=$1; shift; rr=$(res "$ref"); rep=""
  for a in "$@"; do [ "$(res "$a")" = "$rr" ] || rep="$rep $a"; done
  rref=""; [ -n "$rep" ] && [ "${REPREF:-1}" = 1 ] && rref=$ref    # REPREF=0: the reference is not repeated
  [ -n "$rep" ] && log "PFR AB: differ from $ref:$rep; repeating them${rref:+ and $rref}"
  for a in $rep $rref; do
    run_arm "${a}b" || break
    if [ "$(res "${a}b")" = "$(res "$a")" ]; then log "PFR AB repeat $a: same as its first run"; else log "PFR AB repeat $a: DIFFERS from its first run"; fi
  done
fi
peer unlock >/dev/null 2>&1
log "PFR AB end; quiet lock released; SSD $(ssd_t) C, swap $(swapmb) MB (start $SW0 MB)"
