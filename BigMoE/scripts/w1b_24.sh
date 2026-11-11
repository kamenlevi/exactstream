#!/bin/bash
# W1B at 24 GB (docs/WALL1_BENCH.md; docs/WED_TESTS.md item 1): today's shipped exact 24 GB stack (b48_runner's env: v3
# tiers 4433333333 / 3c + COMBINE + OOO, GDN_EXACT, TIER_PF_GROUP 64, EVICT_LFU 0.99, VERIFY_ROWS_SDPA, JOINT, MTP
# stream + MTP_PF_GROUP 48, M-01 records3m; FR-Spec + Scout drafts; C 128 = FRAC 0.25) on W1B's prompts (S45's six,
# EvalSuite's prompt ids via S10B_KEYS), N 256, through scripts/s10b_trace.py on ext/build_m01:
#   w24   untraced: tok/s, SSD MB per token (disk_read / n_tok), lifetime footprint (<= 24 GB)
#   w24t  BIGMOE_TRACE=1: gate wait (exposed SSD), SSD busy, reads per layer (s10b's drain_trace), per prompt
# The sprint's 24 GB baseline row; the line's target is 1.5x its tok/s. Exact: each prompt's sha must be the same in both
# arms (the trace changes no token).
# usage: inside turn45 (HAVE_LOCK=1 KEEP_LOCK=1); ext/build_m01 rebuilt from the current source first (b48's stale guard)
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
set -m
cd "$HOME/BigMoE" || exit 1
OUT="$HOME/BigMoE/data/w1b24"; mkdir -p "$OUT"
LOG="$OUT/w1b24.log"
PY="$HOME/BigMoE/.venv/bin/python"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
B=ext/build_m01
[ "$B/bigmoe_ext.cpython-312-darwin.so" -nt ext/bigmoe_ext.cpp ] || { say "STALE ext in $B (rebuild it from the current source): STOP"; exit 2; }
KEYS="humaneval/HumanEval/68,math500/test/prealgebra/1640.json,humaneval/HumanEval/91,math500/test/algebra/694.json"
KEYS="$KEYS,math500/test/algebra/1282.json,gsm8k/708"
REF="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12"
X="$REF BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_FRSPEC=1 BIGMOE_SCOUT_BUNDLE=1 MTPLX_QWEN4_VERIFY_GLUE=1 BIGMOE_MTP_STREAM=1"
V3="BIGMOE_TIER=4433333333 BIGMOE_TIER_GLU=3c BIGMOE_TIER_COMBINE=1 BIGMOE_OOO=1"
FULL="BIGMOE_GDN_EXACT=1 $V3 BIGMOE_TIER_PF_GROUP=64 BIGMOE_PF_NOLOGITS=1 BIGMOE_EVICT_LFU=0.99 BIGMOE_VERIFY_ROWS_SDPA=1"
A_ENV="BIGMOE_JOINT=1 BIGMOE_MTP_PF_GROUP=48 BIGMOE_M01=1"
say "W1B 24 GB start (pid $$, HAVE_LOCK=$HAVE_LOCK)"
for R in w24 w24t; do
  T="$OUT/$R"; mkdir -p "$T"
  ARM=""; [ "$R" = w24t ] && ARM="BIGMOE_TRACE=1"
  while [ -e "$HOME/.local/share/quiet/PAUSE" ]; do sleep 10; done
  say "run $R"
  perl -e 'alarm shift; exec @ARGV' 900 env $X $FULL $A_ENV $ARM S10B_KEYS=$KEYS BIGMOE_EXT_BUILD=build_m01 BIGMOE_S10B=1 \
    BIGMOE_S10B_MISS=$T/miss.bin $PY scripts/s10b_trace.py 0.25 256 $T >> "$OUT/$R.txt" 2>&1
  say "done $R rc=$?"
done
"$PY" - "$OUT" >> "$LOG" 2>&1 <<'PYEOF'
import json, os, sys
D = sys.argv[1]
m = {r: json.load(open(os.path.join(D, r, "meta.json"))) for r in ("w24", "w24t") if os.path.exists(os.path.join(D, r, "meta.json"))}
for r, x in m.items():
    P = x["prompts"]
    tps = [p["tok_s"] for p in P]; mb = [p["disk_read"] / 1e6 / max(1, p["n_tok"]) for p in P]
    print(f"{r}: tok/s mean {sum(tps) / len(tps):.1f} (per prompt {' '.join(f'{t:.1f}' for t in tps)}); SSD MB per token "
          f"{sum(mb) / len(mb):.1f}; lifetime fp max {max(p['lifetime_fp_gb'] for p in P):.2f} GB")
    if r == "w24t":
        g = [p["trace"].get("gate_wait_s", 0) / max(1, p["passes"]) for p in P]
        b = [p["trace"].get("ssd_busy_s", 0) / max(1, p["passes"]) for p in P]
        print(f"   per pass: exposed (gate wait) {1e3 * sum(g) / len(g):.1f} ms, SSD busy {1e3 * sum(b) / len(b):.1f} ms")
if len(m) == 2:
    sa = [p["sha"] for p in m["w24"]["prompts"]]; sb = [p["sha"] for p in m["w24t"]["prompts"]]
    print(f"shas equal across the arms (the trace changes no token): {sa == sb}")
PYEOF
say "W1B 24 GB end (report above)"
