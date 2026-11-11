#!/bin/bash
# b46 = bounce pool fix (NewIdeas 10-29 GO, after Fit12's R-45T): the ext's per-thread read buffers (thread_local,
# never freed) become one shared pool. Exact by construction; this checks it and measures the leak, in one quiet lock:
# (1) build ext/build_next (the fix); ext/build_tl is a copy of the pre-fix build_m01 binary (17:56, as in b45);
# (2) loader tests bitwise on build_next (M-01 / records3 / POM paths against records3);
# (3) ext-only gap test (scripts/test_bounce_pool.py, 5 cycles of 64 reads, 12-s idle gaps): old vs new footprint;
# (4) the 24 GB default (M-01) on build_next: b45's 4 prompts (every sha must equal b45 A's), then S10B_GAPS=4:12
#     (a 16-token generate after each 12-s idle gap); the same on build_tl as the control. ext/build and
#     ext/build_m01 take the fix at their next make (not here: the lock is 15 min).
# usage: HAVE_LOCK=1 nohup bash scripts/b46_runner.sh > data/b46/runner.out 2>&1 &
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
set -m   # each run is its own process group: time-up or a lost lock stops the whole tree (RULES, orchestrator 20:5x)
cd "$HOME/BigMoE" || exit 1
OUT="$HOME/BigMoE/data/b46"; mkdir -p "$OUT"
LOG="$OUT/b46.log"
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
say "b46 start (CLAUDE_PID=$CLAUDE_PID)"
if [ -z "$HAVE_LOCK" ]; then
  PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 15m "BigMoE b46: bounce pool fix (exact) build + bitwise + footprint over idle gaps (~11 min)" >> "$LOG" 2>&1 || { say "no lock"; exit 1; }
fi
X="$REF BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_FRSPEC=1 BIGMOE_SCOUT_BUNDLE=1 MTPLX_QWEN4_VERIFY_GLUE=1 BIGMOE_MTP_STREAM=1"
V3="BIGMOE_TIER=4433333333 BIGMOE_TIER_GLU=3c BIGMOE_TIER_COMBINE=1 BIGMOE_OOO=1"
FULL="BIGMOE_GDN_EXACT=1 $V3 BIGMOE_TIER_PF_GROUP=64 BIGMOE_PF_NOLOGITS=1 BIGMOE_EVICT_LFU=0.99 BIGMOE_VERIFY_ROWS_SDPA=1"
A_ENV="BIGMOE_JOINT=1 BIGMOE_MTP_PF_GROUP=48 BIGMOE_M01=1"
AB="BIGMOE_TRACE=1 S10B_IDX=4 S10B_X_IDX=9 S10B_GAPS=4:12"
B=ext/build_next                                     # my own staging build dir (b25-b41d A/Bs): incremental make
(cd $B && make -j 4 2>&1 | grep -E 'error|Built target bigmoe_ext' | head -5) >> "$LOG" 2>&1
if [ ! "$B/bigmoe_ext.cpython-312-darwin.so" -nt ext/bigmoe_ext.cpp ]; then   # never run a stale ext
  say "BUILD FAILED in $B: stopping"; [ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1; exit 1; fi
say "ext built in $B"
run t3m 120 t3m.txt env BIGMOE_M01=1 BIGMOE_EXT_BUILD=build_next $PY scripts/test_records3m.py 0,23,47 96
run t3 120 t3.txt env BIGMOE_EXT_BUILD=build_next $PY scripts/test_records3.py 23 96
run t3p 120 t3p.txt env BIGMOE_M01=1 BIGMOE_POM=4 BIGMOE_POM_LP1=1 BIGMOE_PF_JIT=1 BIGMOE_EXT_BUILD=build_next $PY scripts/test_records3p.py 0,23,47
run gap_tl 150 gap_tl.txt env BIGMOE_M01=1 BIGMOE_EXT_BUILD=build_tl $PY scripts/test_bounce_pool.py 5 12
run gap_bp 150 gap_bp.txt env BIGMOE_M01=1 BIGMOE_EXT_BUILD=build_next $PY scripts/test_bounce_pool.py 5 12
for R in NEW OLD; do
  T="$OUT/$R"; mkdir -p "$T"
  BLD=build_next; [ "$R" = "OLD" ] && BLD=build_tl
  run "$R" 300 "$R.txt" env $X $FULL $A_ENV $AB BIGMOE_EXT_BUILD=$BLD BIGMOE_S10B=1 BIGMOE_S10B_MISS=$T/miss.bin $PY scripts/s10b_trace.py 0.25 400 $T || say "$R failed"
done
[ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1
say "lock released"
OMP_NUM_THREADS=2 taskpolicy -b $PY - "$OUT" > "$OUT/ab.txt" 2>&1 <<'EOF'
import json, os, sys
D = sys.argv[1]
B45A = "4633d70e77 5f2c200e62 8f02448059 2ae4ee95ae".split()
print("b46 bounce pool fix: shas vs b45 A, and phys_footprint now across idle gaps (GB)")
for R in ("NEW", "OLD"):
    try:
        m = json.load(open(os.path.join(D, R, "meta.json")))
    except Exception as e:
        print(R, "no meta:", e); continue
    sh = [p["sha"] for p in m["prompts"]]
    g = m.get("gaps", [])
    print(f"{R}: shas {' '.join(sh)} == b45 A: {sh == B45A}; lifetime peak {max(p['lifetime_fp_gb'] for p in m['prompts']):.3f} GB")
    for x in g:
        print(f"    gap {x['gap']:2d}: before {x['fp_before_gap']:.4f}  after gap {x['fp_after_gap']:.4f}  after gen "
              f"{x['fp_after_gen']:.4f}  lifetime {x['lifetime_fp_gb']:.4f}  sha {x['sha']}  bounce {x['bounce']}")
    if len(g) >= 2:
        d = g[-1]["fp_after_gen"] - g[0]["fp_after_gen"]
        print(f"    after-gen footprint, last gap - warm: {1e3 * d:+.1f} MB over {len(g) - 1} gaps ({1e3 * d / (len(g) - 1):+.1f} MB / gap)")
for f in ("t3m.txt", "t3.txt", "t3p.txt"):
    try:
        print(f, "|", open(os.path.join(D, f)).read().strip().split("\n")[-1])
    except Exception as e:
        print(f, "missing:", e)
for f in ("gap_tl.txt", "gap_bp.txt"):
    try:
        print(f, "|", [l for l in open(os.path.join(D, f)) if l.startswith("footprint after")][-1].strip())
    except Exception as e:
        print(f, "missing:", e)
EOF
say "ab rc=$?"
say "b46 end"
