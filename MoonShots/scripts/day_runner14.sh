#!/bin/sh
# MoonShots day_runner14 (Wed 10-21 daytime, account-1): one quiet lock for the round-32 GPU tests, in value order.
# Same discipline as day_runner13 (memory guard, active_at keepalive, a single unlock at the end). Launch it from the
# session WHILE holding the granted quiet lock; its first job reuses it ("lock already held").
# Results: ~/MoonShots/data/day14/*.txt, one line per job in summary.txt.
OUT=$HOME/MoonShots/data/day14; mkdir -p "$OUT"
PY=$HOME/BigMoE/.venv/bin/python
E=$HOME/MoonShots/patches/planes_engine
S=$HOME/MoonShots/scripts
export PEER_QUEUE_TIMEOUT=14400
log() { echo "$(date +%H:%M) $*" >> "$OUT/summary.txt"; }
LOCKF=$HOME/.claude-peers/locks/$CLAUDE_PID.json
keepalive() { $PY -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
if sys.argv[2] == "check": sys.exit(0 if time.time() - max(d.get("taken", 0), d.get("active_at", 0)) < 360 else 1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF" "${1:-stamp}"; }
job() {   # job NAME SECONDS_LIMIT CMD...   (sets JP = the job's pid)
  name=$1; lim=$2; shift 2
  if keepalive check; then keepalive; log "lock already held for $name"
  else peer lock quiet --queue 120m "MoonShots day_runner14: $name" > /dev/null 2>&1 || { log "no lock for $name"; return; }; fi
  g=0
  while { [ "$(memory_pressure | tail -1 | grep -o "[0-9]*")" -lt 25 ] || [ "$(ps -axo rss=,pid= | awk -v me=$$ '$1 > 8000000 && $2 != me' | wc -l)" -gt 0 ]; } && [ $g -lt 120 ]; do
    sleep 5; g=$((g + 1)); [ $((g % 36)) -eq 0 ] && keepalive; done
  if [ $g -ge 120 ]; then log "skip $name (memory busy)"; return; fi
  log "start $name"
  ( "$@" > "$OUT/$name.txt" 2>&1 ) & JP=$!
  i=0
  while kill -0 $JP 2>/dev/null && [ $i -lt "$lim" ]; do
    sleep 5; i=$((i + 5))
    if [ $((i % 180)) -eq 0 ]; then
      keepalive || { log "lost the quiet lock during $name: stopping it"; kill $JP 2>/dev/null; sleep 5; kill -9 $JP 2>/dev/null; break; }
    fi
  done
  kill -0 $JP 2>/dev/null && { kill -9 $JP; log "TIMEOUT $name"; }
  wait $JP 2>/dev/null
  log "done $name: $(grep -E '^(M=|ALL|PASS|FAIL|code|math|chat|Traceback|[A-Za-z]*Error)' "$OUT/$name.txt" | tr '\n' ' ' | cut -c1-900)"
}
log "day_runner14 started (pid $$); free $(df -g / | tail -1 | awk '{print $4}') GB"
REF_ENV=${REF_ENV:-"BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12 BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_FRSPEC=1 BIGMOE_SCOUT_BUNDLE=1 MTPLX_QWEN4_VERIFY_GLUE=1 BIGMOE_MTP_STREAM=1 BIGMOE_RECORDS=3 MTPLX_NGRAM_PREWARM=off MTPLX_FAMILY_CAPTURE_COMMIT=1 MTPLX_DROP_EVENTS=1 MTPLX_LAZY_MTP_HISTORY_APPEND=1 MTPLX_DEFER_REPAIR_EVAL=1 MTPLX_COMPILED_VERIFY=0 MTPLX_QWEN4EXP_COMPILE=0 MTPLX_COMPILE_AR_FORWARD=0 MTPLX_COMPILED_TARGET_PREFIX=0"}   # BigMoE 08:28: the exact reference (24 GB mode)
# 1. exact GDN prework + gated norm: speed vs the eager chain, then model-level logits identity (ref / exact / mtplx)
job gdn_bench       300 $PY $E/bench_gdn_exact.py
job gdn_model       900 env $REF_ENV $PY $S/gdn_exact_model125.py 0.25
# 2. out-of-order experts: cost of the split dispatches
job ooo_split       300 $PY $E/bench_split.py
# 3. 3c load/geometry sweep (synthetic pools, ~1 min each)
job tg_ld0          300 env TG_ONLY=stock4,3c,3c+comb $PY $E/bench_tierglu.py
job tg_ld1          300 env TG_LD=1 TG_ONLY=3c,3c+comb $PY $E/bench_tierglu.py
job tg_ra8          300 env TG_RA=8 TG_ONLY=3c,3c+comb $PY $E/bench_tierglu.py
job tg_rb8          300 env TG_RB=8 TG_ONLY=3c $PY $E/bench_tierglu.py
job tg_nsg4         300 env TG_NSG=4 TG_ONLY=3c $PY $E/bench_tierglu.py
# 4. the dependent-kernel chain under the reference's flags
job chain_ref       600 env $REF_ENV $PY $S/export_pass_graph125.py 0.25 $HOME/MoonShots/data/pass_graph125_ref.dot
# 5. tree width cost in experts (model load ~1 min, ~3-5 min); inter-pass split (serialized drafts)
job sibling_union   900 $PY $S/sibling_union125.py 0.25 48 6
job draft_sync      900 env DRAFT_SYNC=1 PROFILE_OUT=$HOME/MoonShots/data/pass_profile125_draftsync.json $PY $S/pass_profile125.py 0.5 160
peer unlock > /dev/null 2>&1   # one lock per batch
log "day_runner14 finished; free $(df -g / | tail -1 | awk '{print $4}') GB"
