#!/bin/sh
# Breakthrough Wed 2026-10-21 22:15: B8 (NewIdeas: frugal 5 % on v3 + cold MSB-3 ranks 2-3 half a layer late) and B12 (agreed:
# frugal 10 % on v3, bf16 mixers, no late): speed, RAM, GSM8K-20 + KLD of late ranks 2-3 (easy) and HC 8-bit.
B=$HOME/Breakthrough; OUT=$B/data/day; LOG=$B/data/expS/summary.txt
PY=$HOME/BigMoE/.venv/bin/python; S=$B/scripts
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1 PEER_QUEUE_TIMEOUT=10800
LOCKF=$HOME/.claude-peers/locks/$CLAUDE_PID.json
log() { echo "$(date +%H:%M) $*" >> "$LOG"; }
keepalive() { $PY -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF"; }
step() {   # step NAME SECONDS_LIMIT CMD...
  name=$1; lim=$2; shift 2
  keepalive || { log "lock lost before $name"; return 125; }
  log "start $name"
  ( "$@" > "$OUT/$name.txt" 2>&1 ) & P=$!
  i=0; rc=0
  while kill -0 $P 2>/dev/null && [ $i -lt "$lim" ]; do
    sleep 5; i=$((i + 5))
    if [ $((i % 120)) -eq 0 ]; then keepalive || { log "lost the lock during $name: stopping it"; kill $P; sleep 5; kill -9 $P 2>/dev/null; rc=125; break; }; fi
  done
  kill -0 $P 2>/dev/null && { kill -9 $P; log "TIMEOUT $name"; rc=124; }
  wait $P; [ $rc -eq 0 ] && rc=$?
  log "done $name (rc $rc): $(grep -E '^\[|^bt_|^125B|^[A-Za-z_0-9]+ +C=|Error|Traceback|bigmoe:' "$OUT/$name.txt" | tail -2 | tr '\n' ' ' | cut -c1-400)"
  return $rc
}
kld() {   # kld VAR FRAC: skip a variant whose file exists (tf_frugal_var tags non-10 % runs _rNN)
  rt=""; [ "$2" = "0.10" ] || rt=_r$(printf %02d $(echo "$2" | awk '{print int($1 * 100 + 0.5)}'))
  [ -f $B/data/tfvar/$1${rt}_e8.npz ] && return 0
  step tfvar8_$1$rt 1200 env BIGMOE_MAXK=80 $PY $S/tf_frugal_var.py $1 $2 tf_ifeval,tf_multi 8
}
keepalive || { log "no lock to reuse (b8): take one first with peer lock quiet --queue 50m"; exit 1; }   # launched once my session holds the lock
g=0
while { [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -lt 70 ] || [ "$(ps -axo rss=,pid= | awk -v me=$$ '$1 > 8000000 && $2 != me' | wc -l)" -gt 0 ]; } && [ $g -lt 120 ]; do
  sleep 5; g=$((g + 1)); [ $((g % 30)) -eq 0 ] && keepalive; done
log "lock taken (b8)"
step b8_check 300 env B12_LATE3=2,4 $PY $S/bt_b12.py 0.05 check
if grep -q "^check" $OUT/b8_check.txt; then
  for PR in code prose; do
    step b8_$PR 300 env PROMPT=$PR B12_LATE3=2,4 $PY $S/bt_b12.py 0.05 speed
    step b8_nolate_$PR 300 env PROMPT=$PR $PY $S/bt_b12.py 0.05 speed
  done
  step b8_gsm20 900 env B12_LATE3=2,4 $PY $S/bt_b12.py 0.05 gsm20
fi
step b12p_gsm20 900 env $PY $S/bt_b12.py 0.10 gsm20
kld base_l2t4_Sgc 0.05
# 23:32 (dd in place): NewIdeas' "B8-fast" = frugal 5 % on v3 waiting for ranks 0-2 (no phase B): speed, RAM, GSM8K-20
step b8f_check 300 env B12_W=3 $PY $S/bt_b12.py 0.05 check
if grep -q "^check" $OUT/b8f_check.txt; then
  for PR in code prose; do step b8f_$PR 300 env PROMPT=$PR B12_W=3 $PY $S/bt_b12.py 0.05 speed; done
  step b8f_gsm20 900 env B12_W=3 $PY $S/bt_b12.py 0.05 gsm20
fi
for v in exact_hc8 exact_hc8_Sgc; do kld $v 0.10; done
peer unlock > /dev/null 2>&1; log "lock released (b8)"
exit 0
