#!/bin/sh
# Breakthrough experiment A2, frugal mode (BT_NK=2: missing ranks >= 2 are never read from the SSD, as in
# scripts/waitr_sim.py). Starts when night_expA.sh (pid in data/expA/runner.pid) has exited.
# Breakthrough experiment A1 (detached, evening gap 2026-10-19; no new job after 02:30, before the main-track night runners).
# Question: does "wait for the top-2 ranks, substitute the rest from resident experts" keep the 125B's GSM8K answers at
# 10 % and 20 % expert residency (C = 51 / 102 of 512 per layer)? 20 % is what 12 GB gives once experts are at ~2.5 bits.
# Harness: MoEPrefetch's waittop125.py (copied as bt_waittop125.py, writes only to data/expA), fork125 read-only.
# Reference: MoEPrefetch data/w125_gsm_eager.json (BigMoE eager exact, same harness, 147/150).
# Lock discipline copied from MoonShots day_runner9: quiet lock --queue per job, released right after; memory gate
# >= 85 % free and no other process > 8 GB; active_at keepalive every 3 min (job stopped if the lock is lost); hard limits.
OUT=$HOME/Breakthrough/data/expA; mkdir -p "$OUT"
PY=$HOME/BigMoE/.venv/bin/python
S=$HOME/Breakthrough/scripts/bt_waittop125.py
export PEER_QUEUE_TIMEOUT=14400 PYTHONDONTWRITEBYTECODE=1
log() { echo "$(date +%H:%M) $*" >> "$OUT/summary.txt"; }
DL=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-20 02:30" +%s)   # no new job after 02:30
LOCKF=$HOME/.claude-peers/locks/$CLAUDE_PID.json
keepalive() { $PY -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
if sys.argv[2] == "check": sys.exit(0 if time.time() - max(d.get("taken", 0), d.get("active_at", 0)) < 360 else 1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF" "${1:-stamp}"; }
job() {   # job NAME SECONDS_LIMIT CMD...
  name=$1; lim=$2; shift 2
  [ "$(date +%s)" -lt "$DL" ] || { log "skip $name (past 02:30)"; return; }
  if keepalive check; then keepalive; log "lock already held for $name"
  else peer lock quiet --queue 90m "Breakthrough expA: $name" > /dev/null 2>&1 || { log "no lock for $name"; return; }; fi
  g=0
  while { [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -lt 85 ] || [ "$(ps -axo rss=,pid= | awk -v me=$$ '$1 > 8000000 && $2 != me' | wc -l)" -gt 0 ]; } && [ $g -lt 120 ]; do
    sleep 5; g=$((g + 1)); [ $((g % 36)) -eq 0 ] && keepalive; done
  if [ $g -ge 120 ]; then log "skip $name (memory busy)"; peer unlock > /dev/null 2>&1; return; fi
  log "start $name"
  ( "$@" >> "$OUT/$name.txt" 2>&1 ) & P=$!
  i=0
  while kill -0 $P 2>/dev/null && [ $i -lt "$lim" ]; do
    sleep 5; i=$((i + 5))
    if [ $((i % 180)) -eq 0 ]; then
      keepalive || { log "lost the quiet lock during $name: stopping it"; kill $P 2>/dev/null; sleep 5; kill -9 $P 2>/dev/null; break; }
    fi
  done
  kill -0 $P 2>/dev/null && { kill -9 $P; log "TIMEOUT $name"; }
  wait $P 2>/dev/null
  peer unlock > /dev/null 2>&1
  log "done $name: $(grep -E '^(125B|Traceback|[A-Za-z]*Error)' "$OUT/$name.txt" | tail -2 | tr '\n' ' ' | cut -c1-400)"
}
compare() {   # compare TAG: answers vs the exact eager reference
  $PY -I -c '
import json, sys
ref = json.load(open("/Users/kamen/MoEPrefetch/data/w125_gsm_eager.json"))
noise = json.load(open("/Users/kamen/MoEPrefetch/data/w125_gsm_duprow_hc.json"))
d = json.load(open(sys.argv[1])); n = len(d["ok"])
same = sum(a == b for a, b in zip(d["answers"], ref["answers"][:n]))
nz = sum(a == b for a, b in zip(noise["answers"][:n], ref["answers"][:n]))
only_ref = [i for i in range(n) if ref["ok"][i] and not d["ok"][i]]; only_new = [i for i in range(n) if d["ok"][i] and not ref["ok"][i]]
acc, racc = sum(d["ok"]), sum(ref["ok"][:n])
print("%s: GSM8K %d/%d vs exact %d/%d; answers identical %d/%d (noise arm %d/%d); exact-only right %s, new-only right %s" % (sys.argv[2], acc, n, racc, n, same, n, nz, n, only_ref, only_new))
' "$OUT/w125_gsm_$1.json" "$1" >> "$OUT/summary.txt" 2>&1; }
R1=$(cat "$OUT/runner.pid" 2>/dev/null)
while [ -n "$R1" ] && kill -0 "$R1" 2>/dev/null; do sleep 30; done
log "night_expA2 (frugal) started (pid $$); free $(df -g / | tail -1 | awk '{print $4}') GB"
job speed_w2f_r20  600 env BT_NK=2 $PY $S 0.20 w2 speed
for c in 1 2 3; do job gsm_w2f_r20_part$c 1500 env BT_NK=2 W125_TAG=w2f_r20 W125_CHUNK=50 $PY $S 0.20 w2 gsm150; done
compare w2f_r20
job speed_w2f_r10  600 env BT_NK=2 $PY $S 0.10 w2 speed
for c in 1 2 3; do job gsm_w2f_r10_part$c 1500 env BT_NK=2 W125_TAG=w2f_r10 W125_CHUNK=50 $PY $S 0.10 w2 gsm150; done
compare w2f_r10
log "night_expA2 finished; free $(df -g / | tail -1 | awk '{print $4}') GB"
