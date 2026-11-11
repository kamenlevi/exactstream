#!/bin/sh
# Breakthrough Mon 2026-10-26: QUEUE L2-06t-dx (NewIdeas GO 03:10): 8-bit KV distractor check. Per length (32K, 60K) one
# document with 6 interleaved 4-step variable chains; each chain asked from the same state right after the document
# (the cache is rolled back before every question, item 0 repeated at the end as the fork check). bf16 vs kv:64 in the
# shipped env + the prefill lane. Line (12 items per arm): kv8 passes only if (a) gold names within 2 of bf16's,
# (b) distractor names <= bf16's + 2, (c) no item naming a (near-)whole other chain where bf16 doesn't. Decode tok/s per
# arm at 60K reported as information. RE-RUN (NewIdeas 12:30; the 09:44 run was INVALID): exact buffer copies restored
# before every item, a 2K restore self-test per arm first (a FAIL stops the run), cap 160 tokens, tag _g64_lane_r2.
# RUN 3 (NewIdeas GO 14:08; run 2's self-test failed at 14:00 with the cache restored bit-exactly): the decode pools are
# also emptied before every item (single-row passes add cold ranks late, so their bits depend on the pools), tag
# _g64_lane_r3. Steps are not timed while a quiet window holds them stopped; a step stops if swap grows > 2 GB or the
# health monitor says ACT (Leap's run_locked.sh guards).
B=$HOME/Breakthrough; OUT=$B/data/kv8; LOG=$B/data/kv8/summary.txt
PY=$HOME/BigMoE/.venv/bin/python; S=$B/scripts
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1
LOCKF=$HOME/.claude-peers/locks/$CLAUDE_PID.json
HEALTH=$HOME/.local/share/quiet/HEALTH.md
log() { echo "$(date +%H:%M) $*" >> "$LOG"; }
keepalive() { $PY -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF"; }
swap_mb() { sysctl -n vm.swapusage | awk '{for (i = 1; i <= NF; i++) if ($i == "used") {v = $(i + 2); sub("M", "", v); print int(v)}}'; }
tree() { echo "$1"; for c in $(pgrep -P "$1"); do tree "$c"; done; }
paused() { for p in $(tree $P); do case "$(ps -o stat= -p $p 2>/dev/null)" in *T*) return 0;; esac; done; return 1; }
stop() { log "$1: stopping $name"; T=$(tree $P); kill $T 2>/dev/null; sleep 5; kill -9 $T 2>/dev/null; }
step() {   # step NAME SECONDS_LIMIT CMD...
  name=$1; lim=$2; shift 2
  keepalive || { log "lock lost before $name"; return 125; }
  log "start $name"
  SW0=$(swap_mb)
  ( "$@" > "$OUT/$name.txt" 2>&1 ) & P=$!
  i=0; rc=0
  while kill -0 $P 2>/dev/null && [ $i -lt "$lim" ]; do
    sleep 5
    if paused; then continue; fi
    i=$((i + 5))
    if [ $((i % 120)) -eq 0 ]; then keepalive || { stop "lost the lock"; rc=125; break; }; fi
    SW=$(swap_mb); if [ $((SW - SW0)) -gt 2048 ]; then stop "swap grew ${SW0} -> ${SW} MB"; rc=126; break; fi
    if grep -q '^\*\*Verdict: ACT' "$HEALTH" 2>/dev/null; then stop "health monitor at ACT"; rc=127; break; fi
  done
  kill -0 $P 2>/dev/null && { stop "TIMEOUT"; rc=124; }
  wait $P; w=$?; [ $rc -eq 0 ] && rc=$w
  log "done $name (rc $rc): $(grep -E '^long L=|^kv8_long (build|selftest)|Error|Traceback|bigmoe:' "$OUT/$name.txt" | tail -4 | cut -c1-200 | tr '\n' ' ')"
  return $rc
}
keepalive || { log "no lock to reuse (kvdx)"; exit 1; }
log "lock taken (kvdx)"
export KV8L_SET=_dx KV8L_LENS=32000,60000
step dx_build 120 $PY $S/kv8_long.py build || exit 1
export B12_CACHE_MB=64 BIGMOE_TIER_PF_GROUP=64 BIGMOE_PF_GROUP=64 BIGMOE_ASYNC_EVERY=1
step dx_run 2700 env KV8L_TAG=_g64_lane_r3 KV8L_LANE=1 KV8L_ARMS=bf16,kv:64 $PY $S/kv8_long.py run || { log "dx_run stopped: no score (see dx_run.txt)"; exit 1; }
step dx_score 120 env KV8L_TAG=_g64_lane_r3 $PY $S/kv8_long.py score
log "kvdx done"
