#!/bin/bash
# b44 = R-44 (a) M-01 A/B (NewIdeas 10-28): S = BIGMOE_M01=1 on records3m/ (records3 with the per-group fields re-packed
# at a fixed width, exact; the ext unpacks them into the same pool slots: equal pools, MSB part 132 -> 125 blocks, LSB part
# 47 -> 44), A = the shipped stack on records3. One 10-min quiet lock: a spec-only ram24_probe of S (lifetime footprint),
# then runs A1 S1 S2 A2 (prompt 4 of each domain, 400 tok), both arms on ext/build_m01. Lines (AB_LINES=m01): PASS every
# sha equal AND rusage SSD bytes/tok <= -2.5 % (default on NewIdeas' word); KILL any mismatch.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
set -m   # each run is its own process group: time-up or a lost lock stops the whole tree (RULES, orchestrator 20:5x)
cd "$HOME/BigMoE" || exit 1
OUT="$HOME/BigMoE/data/b44"; mkdir -p "$OUT"
LOG="$OUT/b44.log"
PY="$HOME/BigMoE/.venv/bin/python"
A_ENV="BIGMOE_MTP_PF_GROUP=48"; P_ENV="BIGMOE_MTP_PF_GROUP=48 BIGMOE_M01=1"
REF="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12"
LOCKF="$HOME/.claude-peers/locks/${CLAUDE_PID}.json"
keepalive() { "$PY" -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF"; }
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
source "$HOME/BigMoE/scripts/ssd_gate.sh"
run() {  # run NAME TIMEOUT OUTFILE CMD...  (memory gate at most 30 s: the lock is only 10 min)
  local name=$1 to=$2 of=$3; shift 3
  keepalive || { say "lost the lock before $name: stopping the batch"; exit 1; }
  while [ -e "$HOME/.local/share/quiet/PAUSE" ]; do sleep 5; keepalive || { say "lock ended during a PAUSE"; exit 1; }; done
  ssd_wait 66
  local i=0
  while { [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -lt 60 ] || [ "$(ps -axo rss=,pid= | awk -v me=$$ '$1 > 8000000 && $2 != me' | wc -l)" -gt 0 ]; } && [ $i -lt 6 ]; do sleep 5; i=$((i+1)); done
  [ $i -ge 6 ] && say "memory busy at the start of $name (ran anyway after 30 s)"
  say "run $name"
  perl -e 'alarm shift; exec @ARGV' "$to" "$@" >> "$OUT/$of" 2>&1 &
  local jp=$!
  while kill -0 $jp 2>/dev/null; do
    for k in $(seq 36); do kill -0 $jp 2>/dev/null || break; sleep 5; done
    kill -0 $jp 2>/dev/null || break
    keepalive || { say "lost the lock during $name: stopping"; kill -- -$jp; sleep 5; kill -9 -- -$jp 2>/dev/null; exit 1; }
  done
  wait $jp; local rc=$?; kill -9 -- -$jp 2>/dev/null; say "done $name rc=$rc"   # the job's whole group (RULES)
  ssd_check || say "SSD hot after $name"
  return $rc
}
say "b44 start (CLAUDE_PID=$CLAUDE_PID, S = $P_ENV)"
if [ -z "$HAVE_LOCK" ]; then
  PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 10m "BigMoE b44 (R-44 M-01 A/B): footprint probe + exact bytes A/B, records3m vs records3 (24 GB v3, ~8 min)" >> "$LOG" 2>&1 || { say "no lock"; exit 1; }
fi
X="$REF BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_FRSPEC=1 BIGMOE_SCOUT_BUNDLE=1 MTPLX_QWEN4_VERIFY_GLUE=1 BIGMOE_MTP_STREAM=1"
V3="BIGMOE_TIER=4433333333 BIGMOE_TIER_GLU=3c BIGMOE_TIER_COMBINE=1 BIGMOE_OOO=1"
G="BIGMOE_GDN_EXACT=1"
PG="BIGMOE_TIER_PF_GROUP=64 BIGMOE_PF_NOLOGITS=1"
FULL="$G $V3 $PG BIGMOE_EVICT_LFU=0.99 BIGMOE_VERIFY_ROWS_SDPA=1"
J="BIGMOE_JOINT=1"
AB="BIGMOE_TRACE=1 S10B_IDX=4 S10B_X_IDX=9"
export BIGMOE_EXT_BUILD=build_m01; B=ext/build_m01                     # both arms on the same binary
(cd $B && make 2>&1 | tail -3) >> "$LOG" 2>&1
if [ ! "$B/bigmoe_ext.cpython-312-darwin.so" -nt ext/bigmoe_ext.cpp ]; then   # never run a stale ext
  say "BUILD FAILED in $B: stopping"; [ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1; exit 1; fi
say "ext built in $B"
run probeS 150 probeS.txt env $X $FULL $J $P_ENV RAM24_MODE=spec $PY scripts/ram24_probe.py "$OUT/probeS.json"
PEAK=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["mem"]["spec decode 400 tok (incl. its prefill), peak since reset"]["lifetime_peak_footprint"])' "$OUT/probeS.json" 2>>"$LOG")
say "S lifetime peak (spec mode, code prompt 0): $PEAK GB (shipped A, b41c: 23.16)"
for R in A1 S1 S2 A2; do
  T="$OUT/$R"; mkdir -p "$T"
  ARM="$A_ENV"; [ "${R:0:1}" = "S" ] && ARM="$P_ENV"
  run "$R" 200 "$R.txt" env $X $FULL $J $AB $ARM BIGMOE_S10B=1 BIGMOE_S10B_MISS=$T/miss.bin $PY scripts/s10b_trace.py 0.25 400 $T || { say "$R failed: stopping the batch"; break; }
done
[ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1
say "traces: $(du -sh "$OUT" | cut -f1)"
peer wait --timeout 5h >> "$LOG" 2>&1   # CPU scoring only when nobody measures
env AB_LINES=m01 OMP_NUM_THREADS=4 taskpolicy -b $PY scripts/r27_ab.py "$OUT" > "$OUT/ab.txt" 2>&1; say "ab rc=$?"
say "b44 end"
