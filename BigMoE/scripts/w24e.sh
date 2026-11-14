#!/bin/bash
# W24e (BigMoE, Tue 10 Nov 02:0x; the orchestrator ~02:0x: "(A) plus (B), not (C) ... Run (B) with NewIdeas after the gate
# ends. The smaller prefill chunk only carries the gate's label if its tokens are identical to PFCHUNK=1024 on those prompts;
# otherwise Wednesday ships (A). Include the first-token cost"). w13b exactly as gated (data/stack/card_w24d_env.txt, blob
# e06f495e, its env line verbatim) with smaller prefill chunks, on the card gate's longest prompts (prompt tokens [K] from the
# card runs): gpqad/recnTTKdBzfuoZ7w7 2,801; lcb6/abc394_g 1,465; lcb6/abc400_g 1,102; lcb6/arc192_a 942;
# gpqad/rec6sE2CRtD4drtHg 919 (24.81 GB in the gate); lcb6/3705 726. CAP 128, R57_LOGHASH=1. Arms, one process each, in
# this order: pc1024 (as gated), pc512, pc256, pc1024b (as gated again: determinism + drift). Rule: docs/W24_DECISION.md
# section W24e. Uses ~25 GB: AFTER the w24 card gate, under a quiet ticket (turn45.sh gives HAVE_LOCK / KEEP_LOCK).
# usage: bash scripts/turn45.sh 26 30 "BigMoE W24e: ..." bash scripts/w24e.sh      (W24E_DEADLINE=HHMM: last arm start)
cd "$HOME/BigMoE" || exit 1
D=data/w24; LOG=$D/w24.log; PY=/opt/homebrew/opt/python@3.12/bin/python3.12
RS=data/r57/snap/c45_runner_rel1.sh; SCR=data/r57/snap/w24/r57_screen_w24.py; F=data/stack/card_w24d_env.txt
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; [ -z "$W24E_DRY" ] || echo "$*"; }
[ "$(git hash-object $F | cut -c1-8)" = e06f495e ] || { say "W24e not run: $F is not blob e06f495e (the gated w13b)"; exit 2; }
git diff --quiet HEAD -- data/r57/snap/w24 bigmoe "$RS" || { say "W24e not run: snapshot w24 / bigmoe / runner not committed"; exit 2; }
ENVL=$(sed -n 2p $F | sed 's/,R57_QOM_SETS=humaneval,mbpp,lcb6$//')     # the runner splits its env list on commas: the
echo "$ENVL" | grep -q 'R57_QOM_SETS' && { say "W24e not run: R57_QOM_SETS left in the env line"; exit 2; }   # sets go below
KEYS="gpqad/recnTTKdBzfuoZ7w7,lcb6/abc394_g,lcb6/abc400_g,lcb6/arc192_a,gpqad/rec6sE2CRtD4drtHg,lcb6/3705"
ARMS="pc1024:1024 pc512:512 pc256:256 pc1024b:1024"
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
if [ -n "$W24E_DRY" ]; then
  for a in $ARMS; do say "dry: ${a%%:*}g_100 = card_w24d env ($(echo "$ENVL" | tr ',' '\n' | grep -c =) keys) + R57_PFCHUNK=${a#*:}; keys $KEYS"; done
  exit 0
fi
A=$(avail_gb); awk -v a="$A" 'BEGIN {exit !(a >= 30.0)}' || { say "W24e not run: NO-SWAP $A GB < 30"; exit 0; }
export C45_OUT="$PWD/$D" C45_SWAP0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}') C45_ALARM=2400 \
  MG=4 C45_DEADLINE=${W24E_DEADLINE:-$(date -v+60M +%H%M)} C45_SCREEN="$SCR" C45_REFTOK= R57_MODMAN=1
say "W24e start (pid $$, NO-SWAP $A GB, w13b as gated = $F e06f495e, arms $ARMS)"
TAGS=""
for a in $ARMS; do
  t=${a%%:*}g_100
  rm -f data/r57/${t}_* $D/${t}.txt
  R57_KEYS=$KEYS C45_CAP=128 R57_LOGHASH=1 R57_QOM_SETS=humaneval,mbpp,lcb6 C45_ARMS="$t:100:$ENVL,R57_PFCHUNK=${a#*:}" /bin/bash "$RS"
  say "$t rc=$?"; TAGS="$TAGS $t"
done
"$PY" -I - $TAGS > $D/w24e_report.txt <<'EOF'
import json, sys
tags = sys.argv[1:]
R, T = {}, {}
for t in tags:
    try:
        R[t] = {r["key"]: r for r in json.load(open(f"data/r57/{t}_runs.json"))}
        T[t] = json.load(open(f"data/r57/{t}_tokens.json"))
    except (OSError, ValueError):
        print(f"  {t}: MISSING")
ref = "pc1024g_100"
print("W24e (data/w24; w13b as gated + R57_PFCHUNK; CAP 128): per prompt, first token (pfs) / lifetime peak after prefill "
      "(mem[1]) / max lifetime (mem[-1]); tokens + logits hashes vs pc1024 (as gated)")
for t in tags:
    if t not in R:
        continue
    pk = max(max(r["mem"][1], r["mem"][-1]) for r in R[t].values())
    same = sum(T[t].get(k) == T[ref].get(k) for k in T[t]) if ref in T else -1
    lh = sum(R[t][k].get("lh") == R[ref][k].get("lh") for k in R[t] if k in R.get(ref, {})) if ref in R else -1
    bad = [k for k, r in R[t].items() if (r.get("bad") or "").strip() or r.get("cut")]
    print(f"  {t}: max lifetime peak {pk:.3f} GB{' (OVER 24.0)' if pk > 24.0 else ''}; tokens == pc1024 on {same}/{len(T[t])} prompts, "
          f"logits hashes on {lh}/{len(R[t])}; bad / cut rows: {bad or 'none'}")
    for k, r in sorted(R[t].items(), key=lambda kv: -kv[1]["P"]):
        print(f"    {k:28s} P {r['P']:>5}: first token {r['pfs']:.2f} s, peak after prefill {r['mem'][1]:.3f} GB, MLX prefill "
              f"{r['mem'][0]:.2f} GB, tokens {'==' if T[t].get(k) == T.get(ref, {}).get(k) else 'DIFFER'}")
EOF
say "W24e report rc=$? ($D/w24e_report.txt)"
git add $D/w24e_report.txt $D/w24.log $(for t in $TAGS; do ls data/r57/${t}_* $D/${t}.txt 2>/dev/null; done) 2>/dev/null
> (a quote from the project owner, kept private)

$(cat $D/w24e_report.txt)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011DvdV6eUB4NgBbVkyVFPrT" && say "W24e data committed: $(git log -1 --format=%h)"
