#!/bin/bash
# b48 = FR-Spec head no-cycle (NewIdeas 20:4x GO; Breakthrough's B12_FRS_NOCYCLE ported as BIGMOE_FRS_NOCYCLE, default 1):
# the 24 GB default on ext/build_m01 (bounce pool), b45's 4 prompts + S10B_GAPS=4:12 (a 16-token generate after each
# 12-s idle gap), arms FIX (default) and NOFIX (BIGMOE_FRS_NOCYCLE=0). PASS (NewIdeas): every sha = the default's
# (b45 A 4633d70e77 5f2c200e62 8f02448059 2ae4ee95ae; gap generate 697b780955) AND footprint growth <= 0.02 GB per
# request over the gaps. usage: HAVE_LOCK=1 nohup bash scripts/b48_runner.sh > data/b48/runner.out 2>&1 &
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
set -m   # each run is its own process group: time-up or a lost lock stops the whole tree (RULES, orchestrator 20:5x)
cd "$HOME/BigMoE" || exit 1
OUT="$HOME/BigMoE/data/b48"; mkdir -p "$OUT"
LOG="$OUT/b48.log"
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
say "b48 start (CLAUDE_PID=$CLAUDE_PID)"
if [ -z "$HAVE_LOCK" ]; then
  PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 10m "BigMoE b48: FR-Spec head no-cycle (exact, memory): 2 arms x (4 prompts + 4 idle gaps)" >> "$LOG" 2>&1 || { say "no lock"; exit 1; }
fi
X="$REF BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_FRSPEC=1 BIGMOE_SCOUT_BUNDLE=1 MTPLX_QWEN4_VERIFY_GLUE=1 BIGMOE_MTP_STREAM=1"
V3="BIGMOE_TIER=4433333333 BIGMOE_TIER_GLU=3c BIGMOE_TIER_COMBINE=1 BIGMOE_OOO=1"
FULL="BIGMOE_GDN_EXACT=1 $V3 BIGMOE_TIER_PF_GROUP=64 BIGMOE_PF_NOLOGITS=1 BIGMOE_EVICT_LFU=0.99 BIGMOE_VERIFY_ROWS_SDPA=1"
A_ENV="BIGMOE_JOINT=1 BIGMOE_MTP_PF_GROUP=48 BIGMOE_M01=1"
AB="BIGMOE_TRACE=1 S10B_IDX=4 S10B_X_IDX=9 S10B_GAPS=4:12"
B=ext/build_m01
if [ ! "$B/bigmoe_ext.cpython-312-darwin.so" -nt ext/bigmoe_ext.cpp ]; then   # never run a stale ext
  say "STALE ext in $B: stopping"; [ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1; exit 1; fi
for R in FIX NOFIX; do
  T="$OUT/$R"; mkdir -p "$T"
  ARM=""; [ "$R" = "NOFIX" ] && ARM="BIGMOE_FRS_NOCYCLE=0"
  run "$R" 300 "$R.txt" env $X $FULL $A_ENV $AB $ARM BIGMOE_EXT_BUILD=build_m01 BIGMOE_S10B=1 BIGMOE_S10B_MISS=$T/miss.bin $PY scripts/s10b_trace.py 0.25 400 $T || say "$R failed"
done
[ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1
say "lock released"
OMP_NUM_THREADS=2 taskpolicy -b $PY - "$OUT" > "$OUT/ab.txt" 2>&1 <<'PYEOF'
import json, os, sys
D = sys.argv[1]
B45A = "4633d70e77 5f2c200e62 8f02448059 2ae4ee95ae".split()
print("b48 FR-Spec head no-cycle: shas vs the default, phys_footprint now after each request (GB)")
for R in ("FIX", "NOFIX"):
    try:
        m = json.load(open(os.path.join(D, R, "meta.json")))
    except Exception as e:
        print(R, "no meta:", e); continue
    sh = [p["sha"] for p in m["prompts"]]
    g = m.get("gaps", [])
    gs = sorted({x["sha"] for x in g})
    print(f"{R}: shas {' '.join(sh)} == b45 A: {sh == B45A}; gap shas {gs} == ['697b780955']: {gs == ['697b780955']}; "
          f"lifetime peak {max(p['lifetime_fp_gb'] for p in m['prompts']):.3f} GB")
    print("    per prompt fp now: " + "  ".join(f"{p['fp_now_gb']:.4f}" for p in m["prompts"]))
    for x in g:
        print(f"    gap {x['gap']:2d}: after gap {x['fp_after_gap']:.4f}  after gen {x['fp_after_gen']:.4f}  bounce {x['bounce']}")
    if len(g) >= 2:
        d = [b["fp_after_gen"] - a["fp_after_gen"] for a, b in zip(g, g[1:])]
        print("    growth per request over the gaps: " + " ".join(f"{1e3 * v:+.1f}" for v in d) +
              f" MB; max {1e3 * max(d):+.1f} MB, mean {1e3 * sum(d) / len(d):+.1f} MB  (PASS line <= +20 MB)")
PYEOF
say "ab rc=$?"
say "b48 end"
