#!/bin/sh
# Breakthrough Mon 2026-10-26: QUEUE L2-11 (NewIdeas PORT GO 12:30): bt_b12's B12_PFR A/B before it becomes the default.
# v3l engine, AR path, shipped env, no lane; fresh process per arm: r8 / p8 (8,474 tokens, PFR off / on), r64 / p64
# (65,000 tokens). Exact = p equals r on the KV hash after the prefill, the first-step logits hash and the 32-token sha,
# at both lengths. Also shown: r8 / r64 against Fit12's v3p references (r8 ecd389371d / c57fe16fd1 / 142abd5d7e,
# r64 2227a92414 / 4dcabb778f / dc67a9270b), i.e. v3l = v3i = v3p on this path.
B=$HOME/Breakthrough; OUT=$B/data/pfr; LOG=$B/data/pfr/summary.txt
PY=$HOME/BigMoE/.venv/bin/python; S=$B/scripts
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1
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
  wait $P; w=$?; [ $rc -eq 0 ] && rc=$w
  log "done $name (rc $rc): $(grep -E 'RESULT|prefill [0-9]+ tokens|Error|Traceback|B12_PFR skipped' "$OUT/$name.txt" | tail -3 | cut -c1-230 | tr '\n' ' ')"
  return $rc
}
res() { grep -o 'RESULT kv [0-9a-f]* logits [0-9a-f]* sha32 [0-9a-f]*' "$OUT/$1.txt" 2>/dev/null | cut -d' ' -f2-; }
cmp_arms() {   # cmp_arms REF ARM
  a=$(res "$1"); b=$(res "$2")
  if [ -n "$a" ] && [ "$a" = "$b" ]; then log "A/B $2 = $1: EXACT ($a)"; else log "A/B $2 vs $1: DIFFERS ('$b' vs '$a')"; fi
}
keepalive || { log "no lock to reuse (pfr)"; exit 1; }
log "lock taken (pfr)"
export B12_ENGINE=v3l
step r8 240 env B12_PFR=0 PA_N=8474 $PY $S/pfr_ab.py r8
step p8 240 env B12_PFR=1 PA_N=8474 $PY $S/pfr_ab.py p8
cmp_arms r8 p8
log "r8 vs Fit12 v3p r8 (kv logits sha32 = ecd389371d c57fe16fd1 142abd5d7e): $(res r8)"
step r64 600 env B12_PFR=0 PA_N=65000 $PY $S/pfr_ab.py r64
step p64 600 env B12_PFR=1 PA_N=65000 $PY $S/pfr_ab.py p64
cmp_arms r64 p64
log "r64 vs Fit12 v3p r64 (kv logits sha32 = 2227a92414 4dcabb778f dc67a9270b): $(res r64)"
log "pfr done"
