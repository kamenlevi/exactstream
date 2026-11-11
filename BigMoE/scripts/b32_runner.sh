#!/bin/bash
# b32 (S10b deployable score, NewIdeas GO 10-26 09:08): b31 again + BIGMOE_S10B_RES (ext: the residency of pools L+1..L+3
# at every JIT prefetch decision), so the predictors are scored on the top-16 among NON-resident experts. b31 was:
# (S10b, NewIdeas GO 10-25 15:27; slim spec 10-26 00:xx): the hidden-state trace for TrainLab's long-lead expert
# predictor. One run of the shipped 24 GB v3 decode (JOINT, adaptive width, MTP) over 4 domains x 5 prompts, 400 tokens
# each, with BIGMOE_S10B=1 (bigmoe/s10b.py: every layer's routed ids per row, MoE inputs of 8 spread predictor layers)
# and BIGMOE_S10B_MISS (ext: per (pass, layer) unique ids + not-resident flags). Layout: scripts/s10b_trace.py docstring.
# b31 was 439 MB; b32 adds res.bin (~47 MB); S10B_X_IDX=4 (from the launch env) would keep x for the held-out prompts
# only (~70 MB instead of ~350). -> data/b32/s10b/ (deleted after the verdict); b31's trace must be gone first (NewIdeas:
# disk floor ~100 GB). Then, after the unlock, scripts/s10b_check.py (CPU, 4 threads): consistency + the plain router's
# recall@16 of the misses, over all 512 and DEPLOYABLE (top-16 among the experts not resident at the decision).
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
OUT="$HOME/BigMoE/data/b32"; mkdir -p "$OUT"
LOG="$OUT/b32.log"
PY="$HOME/BigMoE/.venv/bin/python"
REF="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12"
LOCKF="$HOME/.claude-peers/locks/${CLAUDE_PID}.json"
# keepalive (MoonShots' fix): peer counts a lock as active from max(taken, active_at); stamp active_at in our lock file
# (atomic replace), as peer's own hook does for a session's commands. Exit 1 = lock gone / not quiet / expired.
keepalive() { "$PY" -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF"; }
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
source "$HOME/BigMoE/scripts/ssd_gate.sh"
# ONE quiet lock for the whole batch (RULES.md "One lock per batch, not per job"); keepalive every 3 min; memory gate per job.
run() {  # run NAME TIMEOUT OUTFILE CMD...
  local name=$1 to=$2 of=$3; shift 3
  keepalive || { say "lost the lock before $name: stopping the batch"; exit 1; }
  while [ -e "$HOME/.local/share/quiet/PAUSE" ]; do sleep 5; keepalive; done   # quiet mode: never start a job into a pause
  ssd_wait 66
  local i=0
  while { [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -lt 60 ] || [ "$(ps -axo rss=,pid= | awk -v me=$$ '$1 > 8000000 && $2 != me' | wc -l)" -gt 0 ]; } && [ $i -lt 120 ]; do sleep 5; i=$((i+1)); [ $((i % 36)) -eq 0 ] && keepalive; done
  [ $i -ge 120 ] && { say "skip $name (memory busy)"; return; }
  say "run $name"
  perl -e 'alarm shift; exec @ARGV' "$to" "$@" >> "$OUT/$of" 2>&1 &
  local jp=$!
  while kill -0 $jp 2>/dev/null; do
    for k in $(seq 36); do kill -0 $jp 2>/dev/null || break; sleep 5; done
    kill -0 $jp 2>/dev/null || break
    keepalive || { say "lost the lock during $name: stopping"; kill $jp; sleep 5; kill -9 $jp 2>/dev/null; exit 1; }
  done
  wait $jp; say "done $name rc=$?"
  ssd_check || say "SSD hot after $name (batch has one arm: nothing more to stop)"
}
say "b32 start"
# disk: the trace is ~0.6 GB; Kamen's floor is ~100 GB free (NewIdeas 00:xx: don't wait for 110, keep it <= ~1 GB)
free_gb() { df -k / | awk 'NR==2 {printf "%.1f", $4 / 1048576}'; }
if [ -z "$HAVE_LOCK" ]; then
  peer lock quiet --queue 30m "BigMoE b32: S10b re-capture with resident bitmaps (24 GB v3 decode, 20 prompts x 400 tok, ~8 min)" >> "$LOG" 2>&1 || { say "no lock"; exit 1; }
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
if [ -d "$HOME/BigMoE/data/b31/s10b" ]; then    # NewIdeas 09:08: delete b31's trace first
  say "b31's trace is still on disk: not writing a second one"; [ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1; exit 1; fi
if awk -v f="$(free_gb)" 'BEGIN { exit !(f < 100.6) }'; then   # checked at grant time: the queue may take hours
  say "disk $(free_gb) GB free < 100.6: not writing the trace"; [ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1; exit 1; fi
T="$OUT/s10b"; mkdir -p "$T"
run s10b 1800 trace.txt env $X $FULL $J BIGMOE_S10B=1 BIGMOE_S10B_MISS=$T/miss.bin BIGMOE_S10B_RES=$T/res.bin ${S10B_X_IDX:+S10B_X_IDX=$S10B_X_IDX} $PY scripts/s10b_trace.py 0.25 400 $T
if ! lsof 2>/dev/null | grep -q "BigMoE/ext/build/bigmoe_ext"; then (cd ext/build && make 2>&1 | tail -1) >> "$LOG" 2>&1; say "ext/build updated"; fi
[ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1
say "trace: $(du -sh "$T" | cut -f1), disk $(free_gb) GB free"
# CPU check without the lock (RULES: CPU-only jobs <= 4 cores)
OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 OPENBLAS_NUM_THREADS=4 taskpolicy -b "$PY" scripts/s10b_check.py "$T" >> "$OUT/check.txt" 2>&1
say "check rc=$?"
say "b32 end"
