#!/bin/bash
# b24: b23's skipped arms (memory gate at 85 % free blocked them; 24 GB runs need ~40 %: gate 60 % here). b23: opt-in fast mode = per-pass expert budget b16 + budget-aware JIT (TrainLab, NON-EXACT) on the full v3 stack at 24 GB
# (+ GDN, PF_GROUP, LFU, VERIFY_ROWS): natural vs b16+JIT, 256 tokens, MTP 3:3:0 and adaptive. -> data/b23/.
# b9: OoO (MoEPrefetch) adopted: ext post_ids partial-hit row (all modes) + BIGMOE_OOO; v1/v3b shas, A/B, GSM chunk 2.
# b8: tier prefill = the reference call shape (compact scratch); exact via 3c (4444444444) recheck, v3 shas, v3b GSM8K chunk.
# v3 validation (b7): GSM8K 0-471 with the 3-bit engine (3c + combine), one CHUNK per lock, resumable (data/v3/). (2026-10-20 evening): records/ -> records3/ conversion (MoonShots make_records3.py --delete-src),
# then guard_sim, exact mode on records3 (BIGMOE_RECORDS=3, v1 shas), loop retest, first BIGMOE_TIER=4433333333 runs.
# One quiet lock for the batch. Results -> data/b4/.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
OUT="$HOME/BigMoE/data/b24"; mkdir -p "$OUT"
LOG="$OUT/b24.log"
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
# ONE quiet lock for the whole batch (RULES.md "One lock per batch, not per job"); keepalive every 3 min; memory gate per job.
run() {  # run NAME TIMEOUT OUTFILE CMD...
  local name=$1 to=$2 of=$3; shift 3
  keepalive || { say "lost the lock before $name: stopping the batch"; exit 1; }
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
}
say "b24 start"
if [ -z "$HAVE_LOCK" ]; then
  peer lock quiet --queue 120m "BigMoE b24: fast-mode arms b23 skipped (~15 min)" >> "$LOG" 2>&1 || { say "no lock"; exit 1; }
fi
X="$REF BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_FRSPEC=1 BIGMOE_SCOUT_BUNDLE=1 MTPLX_QWEN4_VERIFY_GLUE=1 BIGMOE_MTP_STREAM=1"
V3="BIGMOE_TIER=4433333333 BIGMOE_TIER_GLU=3c BIGMOE_TIER_COMBINE=1 BIGMOE_OOO=1"
SPEC="$HOME/SpecDecode/scripts/test_spec125.py"
CF="3:5:0,3:7:1,a:2.0"
G="BIGMOE_GDN_EXACT=1"
PG="BIGMOE_TIER_PF_GROUP=64 BIGMOE_PF_NOLOGITS=1"
NG="BIGMOE_LOOP_GUARD=0 MTPLX_LOOP_GUARD=0"
E3="BIGMOE_RECORDS=3"
FULL="$G $V3 $PG BIGMOE_EVICT_LFU=0.99 BIGMOE_VERIFY_ROWS_SDPA=1"
run b16_code 900 fast.txt env $X $FULL BIGMOE_BUDGET=b16 BIGMOE_BUDGET_JIT=1 PROMPT=code $PY $SPEC 0.25 256 3:3:0,a:2.0
run nat_edit 900 fast.txt env $X $FULL PROMPT=edit $PY $SPEC 0.25 256 3:3:0,a:2.0
for P in prose math430; do
  run "nat_$P" 900 fast.txt env $X $FULL PROMPT=$P $PY $SPEC 0.25 256 3:3:0,a:2.0
  run "b16_$P" 900 fast.txt env $X $FULL BIGMOE_BUDGET=b16 BIGMOE_BUDGET_JIT=1 PROMPT=$P $PY $SPEC 0.25 256 3:3:0,a:2.0
done
[ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1
say "b24 end"
