#!/bin/bash
# R-52 SCRATCH24 (NewIdeas 10-29 order (3); 24 GB TTFT, after the 45 GB items): scripts/r52_ttft.py on the shipped 24 GB
# stack (ext/build_m01 as shipped: R-55's .cpp change does not touch MAXK <= 92), one process per arm, BIGMOE_PF_REUSE
# 0 1 2 0. PASS (NewIdeas): shas equal across arms AND 8K TTFT -5 % or better AND lifetime peak within +0.1 GB.
# usage: HAVE_LOCK=1 nohup bash scripts/r52_runner.sh > data/r52/runner.out 2>&1 &
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
set -m   # each run is its own process group: time-up or a lost lock stops the whole tree (RULES, orchestrator 20:5x)
cd "$HOME/BigMoE" || exit 1
OUT="$HOME/BigMoE/data/r52"; mkdir -p "$OUT"
LOG="$OUT/r52.log"
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
say "r52 start (CLAUDE_PID=$CLAUDE_PID)"
if [ -z "$HAVE_LOCK" ]; then
  PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 12m "BigMoE R-52: 24 GB TTFT, tier prefill buffers kept (exact), 4 x ~90 s" >> "$LOG" 2>&1 || { say "no lock"; exit 1; }
fi
X="$REF BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_FRSPEC=1 BIGMOE_SCOUT_BUNDLE=1 MTPLX_QWEN4_VERIFY_GLUE=1 BIGMOE_MTP_STREAM=1"
V3="BIGMOE_TIER=4433333333 BIGMOE_TIER_GLU=3c BIGMOE_TIER_COMBINE=1 BIGMOE_OOO=1"
FULL="BIGMOE_GDN_EXACT=1 $V3 BIGMOE_TIER_PF_GROUP=64 BIGMOE_PF_NOLOGITS=1 BIGMOE_EVICT_LFU=0.99 BIGMOE_VERIFY_ROWS_SDPA=1"
A_ENV="BIGMOE_JOINT=1 BIGMOE_MTP_PF_GROUP=48 BIGMOE_M01=1"
n=0
for A in 0 1 2 0; do
  n=$((n+1))
  run "arm$A.$n" 240 "arm$A.$n.txt" env $X $FULL $A_ENV BIGMOE_PF_REUSE=$A BIGMOE_EXT_BUILD=build_m01 $PY scripts/r52_ttft.py "$OUT/arm$A.$n.json" || say "arm $A ($n) failed"
done
[ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1
say "lock released"
OMP_NUM_THREADS=2 taskpolicy -b $PY - "$OUT" > "$OUT/r52_screen.txt" 2>&1 <<'PYEOF2'
import glob, json, os, statistics as S, sys
D = sys.argv[1]
R = {}
for f in sorted(glob.glob(os.path.join(D, "arm*.json"))):
    d = json.load(open(f))
    if "decode" in d:
        R.setdefault(d["arm"], []).append(d)
print("R-52 SCRATCH24: shipped 24 GB stack, one process per arm (BIGMOE_PF_REUSE 0 1 2 0); TTFT from a fresh cache (1-token spec125.generate)")
for a, runs in sorted(R.items()):
    for d in runs:
        print(f"  arm {a}: TTFT {d['prompt8k_tokens']} tok {d['ttft8k_s']} s, {d['prompt_tokens']} tok {d['ttft_s']} s; decode {d['decode']}; "
              f"lifetime peak {d['lifetime_peak_gb']} GB, now {d['fp_now_gb']} GB; load {d['load_s']} s")
if 0 in R:
    b8 = S.mean(min(d["ttft8k_s"]) for d in R[0]); bs = S.mean(S.median(d["ttft_s"]) for d in R[0])
    bp = max(d["lifetime_peak_gb"] for d in R[0]); sha0 = {d["decode"]["sha"] for d in R[0]}
    for a in (1, 2):
        if a not in R:
            continue
        a8 = S.mean(min(d["ttft8k_s"]) for d in R[a]); as_ = S.mean(S.median(d["ttft_s"]) for d in R[a])
        ap = max(d["lifetime_peak_gb"] for d in R[a]); sh = {d["decode"]["sha"] for d in R[a]} | sha0
        ok = len(sh) == 1
        v = "FAIL (shas differ)" if not ok else "PASS" if a8 <= 0.95 * b8 and ap <= bp + 0.1 else "KILL" if a8 > 0.99 * b8 else "BETWEEN"
        print(f"R-52 arm {a} vs 0: 8K TTFT {a8:.2f} vs {b8:.2f} s ({a8 / b8 - 1:+.1%}); short TTFT {as_:.3f} vs {bs:.3f} s ({as_ / bs - 1:+.1%}); "
              f"lifetime peak {ap:.3f} vs {bp:.3f} GB ({ap - bp:+.3f}); shas {sorted(sh)}: {v} (PASS: shas equal AND 8K TTFT -5 % AND peak <= +0.1 GB)")
PYEOF2
say "ab rc=$?"
say "r52 end"
