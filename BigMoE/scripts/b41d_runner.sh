#!/bin/bash
# b41d = R-38 re-sized A/B (NewIdeas 10-28: the 24 GB line = lifetime peak phys_footprint; b41c: shipped + MTP_PF_GROUP
# = 23.16 GB in spec mode on code prompt 0 -> ~0.69 GB of pool room): one 10-min quiet lock. First a spec-only ram24_probe of
# S = MTP_PF_GROUP 48 + C_m 152 / C_l 47 (its lifetime peak feeds the scorer), then runs A1 S1 S2 A2 (prompt 4 of each domain,
# 400 tok; A = the new shipped stack incl. MTP_PF_GROUP 48). R-38 lines: PASS S lifetime peak <= 23.9 GB, all shas equal,
# bytes/tok <= -5 %; KILL peak > 24.0 or bytes better than -2 % only; else BETWEEN.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
set -m   # each run is its own process group: time-up or a lost lock stops the whole tree (RULES, orchestrator 20:5x)
cd "$HOME/BigMoE" || exit 1
OUT="$HOME/BigMoE/data/b41d"; mkdir -p "$OUT"
LOG="$OUT/b41d.log"
PY="$HOME/BigMoE/.venv/bin/python"
P_ENV="BIGMOE_MTP_PF_GROUP=48 BIGMOE_TIER_CM=152 BIGMOE_TIER_CL=47"; A_ENV="BIGMOE_MTP_PF_GROUP=48"
REF="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12"
LOCKF="$HOME/.claude-peers/locks/${CLAUDE_PID}.json"
keepalive() { "$PY" -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF"; }
left_s() { "$PY" -I -S -c 'import json, sys, time; print(int(json.load(open(sys.argv[1])).get("until", 0) - time.time()))' "$LOCKF" 2>/dev/null || echo 0; }
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
say "b41d start (CLAUDE_PID=$CLAUDE_PID, S = $P_ENV)"
if [ -z "$HAVE_LOCK" ]; then
  PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 10m "BigMoE b41d (R-38 re-sized A/B): spec-only footprint probe of S (C_m 152 / C_l 47) + bytes A/B vs shipped (24 GB v3, ~8 min)" >> "$LOG" 2>&1 || { say "no lock"; exit 1; }
fi
X="$REF BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_FRSPEC=1 BIGMOE_SCOUT_BUNDLE=1 MTPLX_QWEN4_VERIFY_GLUE=1 BIGMOE_MTP_STREAM=1"
V3="BIGMOE_TIER=4433333333 BIGMOE_TIER_GLU=3c BIGMOE_TIER_COMBINE=1 BIGMOE_OOO=1"
G="BIGMOE_GDN_EXACT=1"
PG="BIGMOE_TIER_PF_GROUP=64 BIGMOE_PF_NOLOGITS=1"
FULL="$G $V3 $PG BIGMOE_EVICT_LFU=0.99 BIGMOE_VERIFY_ROWS_SDPA=1"
J="BIGMOE_JOINT=1"
AB="BIGMOE_TRACE=1 S10B_IDX=4 S10B_X_IDX=9"
if lsof 2>/dev/null | grep -q "BigMoE/ext/build/bigmoe_ext"; then export BIGMOE_EXT_BUILD=build_next; B=ext/build_next; else B=ext/build; fi
(cd $B && make 2>&1 | tail -3) >> "$LOG" 2>&1
if [ ! "$B/bigmoe_ext.cpython-312-darwin.so" -nt ext/bigmoe_ext.cpp ]; then   # never run a stale ext
  say "BUILD FAILED in $B: stopping"; [ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1; exit 1; fi
say "ext built in $B"
ok=1
run probeS 150 probeS.txt env $X $FULL $J $P_ENV RAM24_MODE=spec $PY scripts/ram24_probe.py "$OUT/probeS.json"
PEAK=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["mem"]["spec decode 400 tok (incl. its prefill), peak since reset"]["lifetime_peak_footprint"])' "$OUT/probeS.json" 2>>"$LOG")
say "S lifetime peak (spec mode, code prompt 0): $PEAK GB"
for R in A1 S1 S2 A2; do
  T="$OUT/$R"; mkdir -p "$T"
  ARM="$A_ENV"; [ "${R:0:1}" = "S" ] && ARM="$P_ENV"
  run "$R" 200 "$R.txt" env $X $FULL $J $AB $ARM BIGMOE_S10B=1 BIGMOE_S10B_MISS=$T/miss.bin $PY scripts/s10b_trace.py 0.25 400 $T || { say "$R failed: stopping the batch"; ok=0; break; }
done
if ! lsof 2>/dev/null | grep -q "BigMoE/ext/build/bigmoe_ext"; then (cd ext/build && make 2>&1 | tail -1) >> "$LOG" 2>&1; say "ext/build updated"; fi
[ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1
say "traces: $(du -sh "$OUT" | cut -f1)"
peer wait --timeout 5h >> "$LOG" 2>&1   # CPU scoring only when nobody measures
env AB_LINES=r38 PEAK_GB=$PEAK OMP_NUM_THREADS=4 taskpolicy -b $PY scripts/r27_ab.py "$OUT" > "$OUT/ab.txt" 2>&1; say "ab rc=$?"
say "b41d end"
