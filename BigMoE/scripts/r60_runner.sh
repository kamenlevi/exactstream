#!/bin/bash
# R-60 MTPDROP (NewIdeas 10-31 / 11-01: screen only on an idle shared slot): MTP Head's trained D-22 head (b3) as the 24 GB
# stack's draft head via BIGMOE_MTP_HEAD (bigmoe/loader.py: bf16 dense parts, MoE block + streamed experts kept). b45's prompts
# (S10B_IDX=4, 400 tokens), arms DEF / D22 under a SHARED lock (rough timing); PASS (NewIdeas): shas identical AND tokens
# per pass +3 % or better -> one quiet confirmation; KILL under +1 %. usage: CLAUDE_PID=.. bash scripts/r60_runner.sh
# (as a background task)
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
set -m   # each run is its own process group: time-up or a lost lock stops the whole tree (RULES, orchestrator 20:5x)
cd "$HOME/BigMoE" || exit 1
OUT="$HOME/BigMoE/data/r60"; mkdir -p "$OUT"
LOG="$OUT/r60.log"
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
say "r60 start (CLAUDE_PID=$CLAUDE_PID)"
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
if [ -n "$R60_QUIET" ]; then                              # the quiet confirmation (NewIdeas: PASS -> one quiet run)
  PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 15m "BigMoE R-60 confirmation: D-22 MTP head on the 24 GB stack, tok/s + official peak" >> "$LOG" 2>&1 || { say "no lock"; exit 1; }
  i=0
else
for i in $(seq 120); do                                  # shared lock + NO-SWAP >= peak 24 + 8 (shared), up to 1 h
  if [ ! -e "$HOME/.local/share/quiet/PAUSE" ] && [ "$(noswap)" -ge 32 ] && \
     peer lock share 20m --cpu 300 --gpu 60 "BigMoE R-60: D-22 MTP head on the 24 GB stack, shas + tokens per pass (shared, rough)" >> "$LOG" 2>&1; then
    break; fi
  sleep 30
done
[ $i -ge 120 ] && { say "no shared slot with memory in 1 h"; exit 1; }
fi
X="$REF BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_FRSPEC=1 BIGMOE_SCOUT_BUNDLE=1 MTPLX_QWEN4_VERIFY_GLUE=1 BIGMOE_MTP_STREAM=1"
V3="BIGMOE_TIER=4433333333 BIGMOE_TIER_GLU=3c BIGMOE_TIER_COMBINE=1 BIGMOE_OOO=1"
FULL="BIGMOE_GDN_EXACT=1 $V3 BIGMOE_TIER_PF_GROUP=64 BIGMOE_PF_NOLOGITS=1 BIGMOE_EVICT_LFU=0.99 BIGMOE_VERIFY_ROWS_SDPA=1"
A_ENV="BIGMOE_JOINT=1 BIGMOE_MTP_PF_GROUP=48 BIGMOE_M01=1"
AB="BIGMOE_TRACE=1 S10B_IDX=4 S10B_X_IDX=9"
HEADF="BIGMOE_MTP_HEAD=$HOME/TrainLab/ckpt/d22_b3.safetensors"
arm_env() { case "$1" in D22) echo "$HEADF BIGMOE_MTP_HEAD_COMPILE=0";; D22C) echo "$HEADF";; D22B8C) echo "$HEADF BIGMOE_MTP_HEAD_BITS=8";; *) echo "";; esac; }
for R in ${R60_ARMS:-DEF D22}; do                       # R-60b: R60_ARMS="DEF D22C D22B8C" (compiled prepare; 8-bit g64)
  T="$OUT/$R"; mkdir -p "$T"
  ARM=$(arm_env $R)
  say "run $R"
  perl -e 'alarm shift; exec @ARGV' 600 env $X $FULL $A_ENV $AB $ARM BIGMOE_EXT_BUILD=build_m01 BIGMOE_S10B=1 BIGMOE_S10B_MISS=$T/miss.bin $PY scripts/s10b_trace.py 0.25 400 $T >> "$OUT/$R.txt" 2>&1
  say "done $R rc=$?"
done
if [ -n "$R60_QUIET" ] && [ -z "$R60_NOPROBE" ]; then        # the official 24 GB number (spec mode, code prompt 0) per arm
  for R in ${R60_ARMS:-DEF D22}; do
    ARM=$(arm_env $R)
    perl -e 'alarm shift; exec @ARGV' 200 env $X $FULL $A_ENV $ARM RAM24_MODE=spec BIGMOE_EXT_BUILD=build_m01 $PY scripts/ram24_probe.py "$OUT/probe$R.json" >> "$OUT/probe$R.txt" 2>&1
    say "probe $R rc=$?: lifetime peak $("$PY" -c 'import json, sys; m = json.load(open(sys.argv[1]))["mem"]; print(max(v["lifetime_peak_footprint"] for v in m.values()))' "$OUT/probe$R.json" 2>&1) GB"
  done
fi
peer unlock >> "$LOG" 2>&1
say "lock released"
OMP_NUM_THREADS=2 taskpolicy -b $PY - "$OUT" > "$OUT/r60_screen.txt" 2>&1 <<'PYEOF2'
import json, os, sys
D = sys.argv[1]
B45A = "4633d70e77 5f2c200e62 8f02448059 2ae4ee95ae".split()
print("R-60 MTPDROP at 24 GB (shared GPU, rough): b45's prompts (prompt 4 per domain, 400 tokens), the shipped 24 GB stack")
M = {}
for R in [r for r in os.environ.get("R60_ARMS", "DEF D22").split() if os.path.isdir(os.path.join(D, r))]:
    try:
        M[R] = json.load(open(os.path.join(D, R, "meta.json")))
    except Exception as e:
        print(R, "no meta:", e)
if "DEF" in M and len(M) >= 2:
    tp = {}
    for R, m in M.items():
        sh = [p["sha"] for p in m["prompts"]]
        tp[R] = [p["n_tok"] / max(p["passes"], 1) for p in m["prompts"]]
        print(f"  {R}: shas {' '.join(sh)} (== b45 A: {sh == B45A}); tokens per pass {' '.join(f'{x:.2f}' for x in tp[R])} "
              f"(mean {sum(tp[R]) / len(tp[R]):.3f}); tok/s {' '.join(f'{p['tok_s']:.1f}' for p in m['prompts'])}; lifetime peak "
              f"{max(p.get('lifetime_fp_gb', 0) for p in m['prompts']):.3f} GB")
    ts = {R: [p["tok_s"] for p in m["prompts"]] for R, m in M.items()}
    for R in [r for r in M if r != "DEF"]:
        same = [p["sha"] for p in M["DEF"]["prompts"]] == [p["sha"] for p in M[R]["prompts"]]
        g = sum(tp[R]) / sum(tp["DEF"]) - 1
        gt = sum(ts[R]) / sum(ts["DEF"]) - 1
        v = "FAIL (tokens changed)" if not same else "PASS" if g >= 0.03 else "KILL" if g < 0.01 else "BETWEEN"
        print(f"R-60 {R} vs DEF (PASS: shas identical AND tokens per pass +3 %; KILL under +1 %): {v} (tokens per pass {g:+.1%}, "
              f"tok/s {gt:+.1%}; shas identical {same})")
PYEOF2
say "r60 end"
