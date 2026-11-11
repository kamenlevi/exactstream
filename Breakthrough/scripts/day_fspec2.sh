#!/bin/sh
# Breakthrough D1 x speculation (2026-10-20 evening): build fork125bt's native loader, then frugal wait-top-2 with
# MTP / prompt-lookup drafts on the 125B: speed at 10 % and 5 % residency (code; prose at 10 %), then GSM8K 0-49 at
# 10 % with spec 3:3:1. One quiet lock for the batch; active_at keepalive; memory gate; stops if the smoke run fails.
B=$HOME/Breakthrough; OUT=$B/data/expS; LOG=$OUT/summary.txt
PY=$HOME/BigMoE/.venv/bin/python; S=$B/scripts; CM=$(command -v cmake || echo /opt/homebrew/bin/cmake)
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1 PEER_QUEUE_TIMEOUT=5400
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
  log "done $name (rc $rc): $(grep -E '^\[|^125B|^peak|^bt_frugal|Error|Traceback|bigmoe:' "$OUT/$name.txt" | tr '\n' ' ' | cut -c1-900)"
  return $rc
}
held() { $PY -I -S -c 'import json, sys, time
d = json.load(open(sys.argv[1])); sys.exit(0 if d.get("mode") == "quiet" and d.get("until", 0) > time.time() else 1)' "$LOCKF" 2>/dev/null; }
while kill -0 52695 2>/dev/null; do sleep 20; done   # day_fspec.sh (same CLAUDE_PID: its lock would look like ours)
held || peer lock quiet --queue 60m "Breakthrough: frugal + MTP GSM8K-150 at 10 % and 5 % (~35 min)" > /dev/null 2>&1 || { log "no lock (fspec)"; exit 1; }
g=0
while { [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -lt 85 ] || [ "$(ps -axo rss=,pid= | awk -v me=$$ '$1 > 8000000 && $2 != me' | wc -l)" -gt 0 ]; } && [ $g -lt 120 ]; do
  sleep 5; g=$((g + 1)); [ $((g % 30)) -eq 0 ] && keepalive; done
[ $g -ge 120 ] && { log "memory busy for 10 min: giving the lock back"; peer unlock > /dev/null 2>&1; exit 1; }
log "lock taken (fspec); free $(df -g / | tail -1 | awk '{print $4}') GB"
if ! ls $B/fork125bt/ext/build/bigmoe_ext*.so > /dev/null 2>&1; then
  ( cd $B/fork125bt/ext && $CM -S . -B build -DCMAKE_BUILD_TYPE=Release -DNB_CMAKE_DIR=$HOME/BigMoE/.nb215/nanobind/cmake \
      -DPython_EXECUTABLE=$PY > build_cmake.log 2>&1 && $CM --build build -j 6 > build.log 2>&1 )
  ls $B/fork125bt/ext/build/bigmoe_ext*.so > /dev/null 2>&1 || { log "ext build FAILED (fork125bt/ext/build.log)"; peer unlock > /dev/null 2>&1; exit 1; }
  log "ext built"
fi
# batch A: GSM8K 0-149 at 10 % (spec 3:3:1, resumes 0-49) and at 5 % (spec 3:3:0)
step fs_gsm10_150 2700 env OFFSET=0 $PY $S/bt_frugal_spec.py 0.10 gsm150 3:3:1
step fs_gsm05_150 2700 env OFFSET=0 $PY $S/bt_frugal_spec.py 0.05 gsm150 3:3:0
peer unlock > /dev/null 2>&1; log "lock released (fspec2 A)"
# batch B (new lock): frugal + planes emulation, plane_sim layout MSB-2 H=42 F=5 at the 5 % budget (plan 4422222222, eager)
peer lock quiet --queue 60m "Breakthrough: frugal + 2-bit planes emulation, GSM8K-150 (~30 min)" > /dev/null 2>&1 || { log "no lock (fspec2 B)"; exit 1; }
log "lock taken (fspec2 B)"
step fp_h42_150 2700 env OFFSET=0 BT_C=42 BT_PLAN=4422222222 BT_TAG=p2_h42 $PY $S/bt_frugal_spec.py 0.08 gsm150 ar2
peer unlock > /dev/null 2>&1; log "lock released (fspec2 B)"
