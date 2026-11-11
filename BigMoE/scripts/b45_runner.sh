#!/bin/bash
# b45 = R-44 (b) POM + LP1 A/B (NewIdeas 10-29): A = the 24 GB default (M-01 on records3m), P = A + BIGMOE_POM=4
# (MoonShots' M-03: misses used only at gate ranks >= 4 read records3p's top two code planes; a later rank < 4 use
# upgrades), L = P + BIGMOE_POM_LP1=1 (M-07: a rank-1 use whose LSB is not resident runs at tier 3, nothing read). Equal
# pools, order A1 P1 L1 L2 P2 A2, prompt 4 of each domain, 400 tok, all on ext/build_m01; a spec-only ram24_probe of L.
# Lines (scripts/r44b_ab.py): P vs A PASS rusage bytes/tok <= -12 % with tok/s not worse, KILL a saving under 6 %;
# L vs P PASS a further <= -6 % with tok/s not worse, KILL a saving under 3 %. Both non-exact -> Eval suite, then Kamen.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
set -m   # each run is its own process group: time-up or a lost lock stops the whole tree (RULES, orchestrator 20:5x)
cd "$HOME/BigMoE" || exit 1
OUT="$HOME/BigMoE/data/b45"; mkdir -p "$OUT"
LOG="$OUT/b45.log"
PY="$HOME/BigMoE/.venv/bin/python"
A_ENV="BIGMOE_MTP_PF_GROUP=48 BIGMOE_M01=1"; P_ENV="$A_ENV BIGMOE_POM=4"; L_ENV="$P_ENV BIGMOE_POM_LP1=1"
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
run() {  # run NAME TIMEOUT OUTFILE CMD...  (memory gate at most 30 s: the lock is only 15 min)
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
say "b45 start (CLAUDE_PID=$CLAUDE_PID, P = $P_ENV, L = $L_ENV)"
if [ -z "$HAVE_LOCK" ]; then
  PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 15m "BigMoE b45 (R-44 b POM + LP1 A/B): footprint probe + A P L L P A bytes A/B (24 GB v3, ~12 min)" >> "$LOG" 2>&1 || { say "no lock"; exit 1; }
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
if [ -z "${NOPROBE:-}" ]; then   # NOPROBE=1: skip the footprint probe (lock time)
run probeL 150 probeL.txt env $X $FULL $J $L_ENV RAM24_MODE=spec $PY scripts/ram24_probe.py "$OUT/probeL.json"
PEAK=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["mem"]["spec decode 400 tok (incl. its prefill), peak since reset"]["lifetime_peak_footprint"])' "$OUT/probeL.json" 2>>"$LOG")
say "L lifetime peak (spec mode, code prompt 0): $PEAK GB (M-01 default, b44: 23.10)"
else PEAK=; fi
for R in A1 P1 L1 L2 P2 A2; do
  T="$OUT/$R"; mkdir -p "$T"
  ARM="$A_ENV"; [ "${R:0:1}" = "P" ] && ARM="$P_ENV"; [ "${R:0:1}" = "L" ] && ARM="$L_ENV"
  run "$R" 200 "$R.txt" env $X $FULL $J $AB $ARM BIGMOE_S10B=1 BIGMOE_S10B_MISS=$T/miss.bin $PY scripts/s10b_trace.py 0.25 400 $T || { say "$R failed: stopping the batch"; break; }
done
[ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1
say "traces: $(du -sh "$OUT" | cut -f1)"
peer wait --timeout 5h >> "$LOG" 2>&1   # CPU scoring only when nobody measures
env PEAK_GB=$PEAK OMP_NUM_THREADS=4 taskpolicy -b $PY scripts/r44b_ab.py "$OUT" > "$OUT/ab.txt" 2>&1; say "ab rc=$?"
say "b45 end"
