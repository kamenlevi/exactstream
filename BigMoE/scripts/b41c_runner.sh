#!/bin/bash
# b41c (R-38 re-size, NewIdeas Wed ~09:10: the 24 GB line is the LIFETIME peak phys_footprint, ri_lifetime_max_phys_footprint
# via RUSAGE_INFO_V4, over load + prefill + decode, worst prompt = code prompt 0; line <= 23.9 GB, 24.0 hard): three
# scripts/ram24_probe.py processes under one quiet lock: A0 = the old shipped stack (C_m 147 / C_l 46, no MTP_PF_GROUP),
# A1 = + BIGMOE_MTP_PF_GROUP=48 (the new shipped default), S = A1 + pools grown to put the lifetime peak at ~23.85 GB
# (sized here from A1's peak: C_m / C_l scaled by 1 + headroom / pool bytes). The bytes A/B (A1 vs S) follows separately.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
set -m   # each run is its own process group: time-up or a lost lock stops the whole tree (RULES)
cd "$HOME/BigMoE" || exit 1
OUT="$HOME/BigMoE/data/b41c"; mkdir -p "$OUT"
LOG="$OUT/b41c.log"
PY="$HOME/BigMoE/.venv/bin/python"
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
say "b41c start (CLAUDE_PID=$CLAUDE_PID)"
if [ -z "$HAVE_LOCK" ]; then
  PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 10m "BigMoE b41c (R-38 re-size): lifetime peak footprint probes A0 / A1 / S on code prompt 0 (24 GB v3, ~3 min)" >> "$LOG" 2>&1 || { say "no lock"; exit 1; }
fi
X="$REF BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_FRSPEC=1 BIGMOE_SCOUT_BUNDLE=1 MTPLX_QWEN4_VERIFY_GLUE=1 BIGMOE_MTP_STREAM=1"
V3="BIGMOE_TIER=4433333333 BIGMOE_TIER_GLU=3c BIGMOE_TIER_COMBINE=1 BIGMOE_OOO=1"
G="BIGMOE_GDN_EXACT=1"
PG="BIGMOE_TIER_PF_GROUP=64 BIGMOE_PF_NOLOGITS=1"
FULL="$G $V3 $PG BIGMOE_EVICT_LFU=0.99 BIGMOE_VERIFY_ROWS_SDPA=1"
J="BIGMOE_JOINT=1"
if lsof 2>/dev/null | grep -q "BigMoE/ext/build/bigmoe_ext"; then export BIGMOE_EXT_BUILD=build_next; B=ext/build_next; else B=ext/build; fi
(cd $B && make 2>&1 | tail -3) >> "$LOG" 2>&1
if [ ! "$B/bigmoe_ext.cpython-312-darwin.so" -nt ext/bigmoe_ext.cpp ]; then   # never run a stale ext
  say "BUILD FAILED in $B: stopping"; [ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1; exit 1; fi
say "ext built in $B"
run A0 150 A0.txt env $X $FULL $J $PY scripts/ram24_probe.py "$OUT/A0.json"
run A1 150 A1.txt env $X $FULL $J BIGMOE_MTP_PF_GROUP=48 $PY scripts/ram24_probe.py "$OUT/A1.json"
SZ=$("$PY" -c '
import json, sys
p = json.load(open(sys.argv[1]))["mem"]["ar decode 128 tok (incl. its prefill), peak since reset"]["lifetime_peak_footprint"]
extra = (23.85 - p) * 1e9                                   # bytes of headroom under the line (decimal GB, as Fit12)
f = 1 + extra / (48 * (147 * 2162688 + 46 * 770048))
print(int(147 * f), int(46 * f), round(p, 3)) if extra > 0 else print(0, 0, round(p, 3))' "$OUT/A1.json" 2>>"$LOG")
say "A1 lifetime peak -> S sizes C_m C_l (A1 GB): $SZ"
set -- $SZ
if [ -n "$1" ] && [ "$1" -gt 147 ]; then
  run S 150 S.txt env $X $FULL $J BIGMOE_MTP_PF_GROUP=48 BIGMOE_TIER_CM=$1 BIGMOE_TIER_CL=$2 $PY scripts/ram24_probe.py "$OUT/S.json"
else
  say "no headroom under 23.85 GB after A1: S not run"
fi
if ! lsof 2>/dev/null | grep -q "BigMoE/ext/build/bigmoe_ext"; then (cd ext/build && make 2>&1 | tail -1) >> "$LOG" 2>&1; say "ext/build updated"; fi
[ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1
say "runs done"
say "b41c end"
