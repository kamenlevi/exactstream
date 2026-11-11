#!/bin/sh
# Breakthrough D1 quality (2026-10-20 night): frugal KLD estimator variants on the Eval suite's teacher-forced subset
# (ifeval + multi, every 4th sequence), after day_simtab.sh (pid 4215, same CLAUDE_PID): partial renormalisation
# beta 0.25 / 0.5, wait-by-mass 50 / 70 % without renormalisation, SERE substitution + wait-by-mass 50 / 70 %.
# Correctness-only (KLD), fine during the silent night. One quiet lock; active_at keepalive; memory gate 70 % free.
B=$HOME/Breakthrough; OUT=$B/data/expS; LOG=$OUT/summary.txt
PY=$HOME/BigMoE/.venv/bin/python; S=$B/scripts
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1 EVALSUITE_FRUGAL=1 PEER_QUEUE_TIMEOUT=5400
LOCKF=$HOME/.claude-peers/locks/$CLAUDE_PID.json
log() { echo "$(date +%H:%M) $*" >> "$LOG"; }
keepalive() { $PY -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF"; }
step() {   # step NAME SECONDS_LIMIT CMD...  -> the command's exit code (124 on timeout, 125 on a lost lock)
  name=$1; lim=$2; shift 2
  keepalive || { log "lock lost before $name"; return 125; }
  log "start $name"
  ( "$@" > "$OUT/$name.txt" 2>&1 ) & P=$!
  i=0; rc=0
  while kill -0 $P 2>/dev/null && [ $i -lt "$lim" ]; do
    sleep 5; i=$((i + 5))
    if [ $((i % 150)) -eq 0 ]; then keepalive || { log "lost the lock during $name: stopping it"; kill $P; sleep 5; kill -9 $P 2>/dev/null; rc=125; break; }; fi
  done
  kill -0 $P 2>/dev/null && { kill -9 $P; log "TIMEOUT $name"; rc=124; }
  wait $P; [ $rc -eq 0 ] && rc=$?
  log "done $name (rc $rc): $(grep -E '^[a-z_0-9]+ +C=|Error|Traceback|bigmoe:' "$OUT/$name.txt" | tr '\n' ' ' | cut -c1-600)"
  return $rc
}
while kill -0 4215 2>/dev/null; do sleep 20; done   # day_simtab.sh (same CLAUDE_PID)
peer lock quiet --queue 60m "Breakthrough: frugal KLD estimator variants, teacher-forced subset (~40 min)" > /dev/null 2>&1 || { log "no lock (tfvar2)"; exit 1; }
g=0
while { [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -lt 70 ] || [ "$(ps -axo rss=,pid= | awk -v me=$$ '$1 > 8000000 && $2 != me' | wc -l)" -gt 0 ]; } && [ $g -lt 120 ]; do
  sleep 5; g=$((g + 1)); [ $((g % 30)) -eq 0 ] && keepalive; done
[ $g -ge 120 ] && { log "memory busy for 10 min: giving the lock back (tfvar2)"; peer unlock > /dev/null 2>&1; exit 1; }
log "lock taken (tfvar2)"
for v in b25 b50 m50_nr m70_nr sere_m50 sere_m70; do
  step tfvar_$v 1500 env BIGMOE_MAXK=80 $PY $S/tf_frugal_var.py $v 0.10 tf_ifeval,tf_multi 4
done
peer unlock > /dev/null 2>&1; log "lock released (tfvar2)"
