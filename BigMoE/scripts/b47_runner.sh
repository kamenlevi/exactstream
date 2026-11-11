#!/bin/bash
# b47 = the official 24 GB number (ram24_probe: lifetime peak phys_footprint, spec mode, code prompt 0) on the
# bounce pool build (N: ext/build_next, as in b46; build_m01 + build are rebuilt here too) vs the pre-fix binary (O: ext/build_tl), order N1 O1 O2 N2.
# b44's M-01 default: 23.10 GB. b46 hinted -0.3 GB in the s10b harness. Exact build: only memory can move.
# usage: HAVE_LOCK=1 nohup bash scripts/b47_runner.sh > data/b47/runner.out 2>&1 &
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
set -m   # each run is its own process group: time-up or a lost lock stops the whole tree (RULES, orchestrator 20:5x)
cd "$HOME/BigMoE" || exit 1
OUT="$HOME/BigMoE/data/b47"; mkdir -p "$OUT"
LOG="$OUT/b47.log"
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
say "b47 start (CLAUDE_PID=$CLAUDE_PID)"
if [ -z "$HAVE_LOCK" ]; then
  PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 8m "BigMoE b47: ram24_probe old vs new bounce pool (exact), 4 x 30 s" >> "$LOG" 2>&1 || { say "no lock"; exit 1; }
fi
X="$REF BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_FRSPEC=1 BIGMOE_SCOUT_BUNDLE=1 MTPLX_QWEN4_VERIFY_GLUE=1 BIGMOE_MTP_STREAM=1"
V3="BIGMOE_TIER=4433333333 BIGMOE_TIER_GLU=3c BIGMOE_TIER_COMBINE=1 BIGMOE_OOO=1"
FULL="BIGMOE_GDN_EXACT=1 $V3 BIGMOE_TIER_PF_GROUP=64 BIGMOE_PF_NOLOGITS=1 BIGMOE_EVICT_LFU=0.99 BIGMOE_VERIFY_ROWS_SDPA=1"
A_ENV="BIGMOE_JOINT=1 BIGMOE_MTP_PF_GROUP=48 BIGMOE_M01=1"
for BB in ext/build_m01 ext/build; do                  # the fix is verified exact (b46): the shared builds take it now
  if lsof 2>/dev/null | grep -q "BigMoE/$BB/bigmoe_ext"; then say "$BB in use: not rebuilt"
  else (cd $BB && make -j 4 2>&1 | grep -E 'error|Built target bigmoe_ext' | head -3) >> "$LOG" 2>&1
    [ "$BB/bigmoe_ext.cpython-312-darwin.so" -nt ext/bigmoe_ext.cpp ] && say "$BB rebuilt with the bounce pool" || say "$BB BUILD FAILED"; fi
done
for R in N1 O1 O2 N2; do
  BLD=build_next; [ "${R:0:1}" = "O" ] && BLD=build_tl
  run "probe$R" 150 "probe$R.txt" env $X $FULL $A_ENV RAM24_MODE=spec BIGMOE_EXT_BUILD=$BLD $PY scripts/ram24_probe.py "$OUT/probe$R.json" || say "probe$R failed"
done
[ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1
say "lock released"
for R in N1 O1 O2 N2; do
  say "probe$R lifetime peak (spec mode, code prompt 0): $("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["mem"]["spec decode 400 tok (incl. its prefill), peak since reset"]["lifetime_peak_footprint"])' "$OUT/probe$R.json" 2>>"$LOG") GB"
done
say "b47 end"
