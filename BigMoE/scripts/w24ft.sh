#!/bin/bash
# Wed 11 Nov 14:2x. W24FT: the 24 GB first-token arms (Kamen 14:1x via the orchestrator: use the idle GPU hour; the first-token
# job, docs/FIRST_TOKEN.md). The shipped 24 GB card (w13b as gated: data/stack/card_w24d_env.txt line 2, blob e06f495e) on
# snapshot w24ft = snapshot w24 (rel1's screen + R57_ROWCAP) + BigMoE bc04e37's P3 fold in pflm.py + ff0215d's P2 v0 in the
# screen (both applied as patches; nothing else differs). The 12 N45 IFEval prompts (data/n45/keys.txt line 1), CAP 256,
# one process per arm, in this order:
#   w3c1 the card as gated (control; its prefill = tier.prefill_call, no PF-D), w3d + PF-D v4 + read-ahead (exact by
#   construction, new at 24 GB), w3f + R57_PFD_FOLD=1 (P3), w3p + R57_P2=256 (P2 v0, LOSSY), w3c2 the control again (drift).
# REPORT data/w24/w24ft_report.txt: per arm TTFT mean / median / range, PF-D timers, P2 chunks, tokens + logits hash vs w3c1
# (w3f also vs w3d), lifetime peak.
# usage (a harness background task, under a share lock with MEM in its reason): bash scripts/w24ft.sh   (W24FT_DRY=1: checks only)
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
D=data/w24; mkdir -p $D
LOG="$PWD/$D/w24ft.log"
PY=$HOME/BigMoE/.venv/bin/python
RS=data/r57/snap/c45_runner_rel1.sh; SCR=data/r57/snap/w24ft/r57_screen_w24ft.py; F=data/stack/card_w24d_env.txt
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; [ -z "$W24FT_DRY" ] || echo "$*"; }
[ "$(git hash-object $F | cut -c1-8)" = e06f495e ] || { say "W24FT not run: $F is not blob e06f495e (the gated w13b)"; exit 2; }
git diff --quiet HEAD -- data/r57/snap/w24ft bigmoe "$RS" && [ -z "$(git ls-files --others --exclude-standard -- data/r57/snap/w24ft)" ] \
  || { say "W24FT not run: snapshot w24ft / bigmoe / runner not committed"; exit 2; }
grep -q 'R57_PFD_FOLD' data/r57/snap/w24ft/pflm.py && grep -q 'def prefill_p2' "$SCR" || { say "W24FT not run: no fold / P2 in w24ft"; exit 2; }
ENVL=$(sed -n 2p $F | sed 's/,R57_QOM_SETS=humaneval,mbpp,lcb6$//')
echo "$ENVL" | grep -q 'R57_QOM_SETS' && { say "W24FT not run: R57_QOM_SETS left in the env line"; exit 2; }
KEYS=$(sed -n 1p data/n45/keys.txt)
[ "$(echo "$KEYS" | tr ',' '\n' | grep -c '^ifeval/')" = 12 ] || { say "W24FT not run: data/n45/keys.txt line 1 is not 12 IFEval keys"; exit 2; }
PFD="R57_PFD=1,R57_PFD_V=4,R57_PFD_AHEAD=1"
ARMS=${W24FT_ARMS:-"w3c1: w3d:$PFD w3f:$PFD,R57_PFD_FOLD=1 w3p:R57_P2=256 w3c2:"}   # 14:2x: W24FT_ARMS for extra arms
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
if [ -n "$W24FT_DRY" ]; then
  for a in $ARMS; do say "dry: ${a%%:*}_100 = card_w24d env ($(echo "$ENVL" | tr ',' '\n' | grep -c =) keys)${a#*:}+ [${a#*:}]"; done
  exit 0
fi
A=$(avail_gb); awk -v a="$A" 'BEGIN {exit !(a >= 28)}' || { say "W24FT not run: NO-SWAP $A GB < 28"; exit 0; }
export C45_OUT="$PWD/$D" C45_SWAP0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}') C45_ALARM=2400 \
  MG=4 C45_DEADLINE=${W24FT_DEADLINE:-$(date -v+60M +%H%M)} C45_SCREEN="$SCR" C45_REFTOK= R57_MODMAN=1
say "W24FT start (pid $$, NO-SWAP $A GB, w13b as gated = $F e06f495e, snapshot $(git log -1 --format=%h -- data/r57/snap/w24ft))"
TAGS=""
for a in $ARMS; do
  t=${a%%:*}_100; x=${a#*:}
  rm -f data/r57/${t}_* $D/${t}.txt
  R57_KEYS=$KEYS C45_CAP=256 R57_LOGHASH=1 R57_PFSTAT=1 R57_QOM_SETS=humaneval,mbpp,lcb6 C45_ARMS="$t:100:$ENVL${x:+,$x}" /bin/bash "$RS"
  say "$t rc=$?"; TAGS="$TAGS $t"
done
"$PY" -I - $TAGS > $D/${W24FT_REPORT:-w24ft}_report.txt 2>&1 <<'EOF'
import json, re, statistics as S, sys
tags = sys.argv[1:]
peak = {}
try:
    for ln in open("data/w24/c45.log"):
        m = re.search(r"done (\S+) rc=\d+:.*lifetime peak ([\d.]+) GB", ln)
        if m:
            peak[m.group(1)] = float(m.group(2))
except OSError:
    pass
def load(t):
    try:
        return ({r["key"]: r for r in json.load(open(f"data/r57/{t}_runs.json"))},
                {k: list(v.values())[0][0] for k, v in json.load(open(f"data/r57/{t}_tokens.json")).items()})
    except (OSError, ValueError, KeyError):
        return {}, {}
TM = ("tP", "tJ", "tPred", "tB", "tC", "tG", "tM")
L = ["W24FT (24 GB card w13b as gated, snapshot w24ft; the 12 N45 IFEval prompts, CAP 256, under a share lock by day): TTFT = pfs; "
     "PF-D timers per prompt (medians, s); tokens and logits hash vs the control"]
def arm(t, ctl, nm):
    R, T = load(t); R0, T0 = load(ctl)
    if t not in tags:
        return
    if not R:
        L.append(f"  {t} {nm}: MISSING"); return
    ks = sorted(R)
    tt = [R[k]["pfs"] for k in ks if R[k].get("pfs")]
    pd = [R[k].get("pfd") or {} for k in ks]
    tm = "; ".join(f"{x} {S.median([p.get(x, 0.0) for p in pd]):.3f}" for x in TM) if any(pd) else "no PF-D"
    N = sum(r["n"] for r in R.values()); Dd = sum(r["n"] / r["tps"] for r in R.values() if r.get("tps"))
    same = sum(1 for k in ks if T.get(k) == T0.get(k))
    hs = sum(sum(1 for a_, b_ in zip(R[k].get("lh") or [], (R0.get(k) or {}).get("lh") or []) if a_ == b_) for k in ks)
    hn = sum(min(len(R[k].get("lh") or []), len((R0.get(k) or {}).get("lh") or [])) for k in ks)
    p2 = sum((R[k].get("p2") or 0) for k in ks)
    gbs = [(R[k].get("pfst") or {}).get("gb") for k in ks]; gbs = [g for g in gbs if g is not None]
    L.append(f"  {t} {nm}: TTFT mean {S.mean(tt):.2f} s, median {S.median(tt):.2f} ({min(tt):.2f}-{max(tt):.2f}); timers {tm}; "
             f"P2 chunks {p2}; prefill SSD {S.mean(gbs) if gbs else float('nan'):.2f} GB per prompt; whole decode "
             f"{N / Dd if Dd else float('nan'):.1f} tok/s" + ("" if t == ctl else f"; tokens == {ctl} {same}/{len(ks)}, logits hash {hs}/{hn}")
             + f"; lifetime peak {peak.get(t, float('nan')):.2f} GB")
arm("w3c1_100", "w3c1_100", "the card as gated (control), 1st")
arm("w3d_100", "w3c1_100", "+ PF-D v4 + read-ahead (exact by construction), 2nd")
arm("w3f_100", "w3c1_100", "+ PF-D v4 + read-ahead + R57_PFD_FOLD=1 (P3), 3rd")
arm("w3f_100", "w3d_100", "  (P3 vs PF-D alone)")
arm("w3p_100", "w3c1_100", "+ R57_P2=256 (P2 v0, LOSSY), 4th")
arm("w3c2_100", "w3c1_100", "the control again (drift), 5th")
arm("w3p13_100", "w3c1_100", "+ R57_P2=256, R57_P2_ROWS=13 (the card's ROWCAP: 130 ids <= 141 base slots, the sync-free gate), LOSSY")
arm("w3p13n_100", "w3c1_100", "+ R57_P2=256, R57_P2_ROWS=13, R57_P2_VERIFY=0 (plain forwards), LOSSY")
arm("w4c1_100", "w4c1_100", "W24T: the card as gated (control), 1st")
arm("w4p1_100", "w4c1_100", "W24T: card + P2 (13 rows, plain; the gated arm), 2nd")
arm("w4p2_100", "w4p1_100", "W24T: the same P2 again, 3rd (vs the 2nd: determinism)")
arm("w4c2_100", "w4c1_100", "W24T: the card again, 4th (vs the 1st: drift)")
TT = {t: [r["pfs"] for r in load(t)[0].values() if r.get("pfs")] for t in ("w4c1_100", "w4c2_100", "w4p1_100", "w4p2_100") if t in tags}
c_, p_ = TT.get("w4c1_100", []) + TT.get("w4c2_100", []), TT.get("w4p1_100", []) + TT.get("w4p2_100", [])
if c_ and p_:
    L.append(f"POOLED W24T (ABBA): card median {S.median(c_):.2f} s (mean {S.mean(c_):.2f}, {min(c_):.2f}-{max(c_):.2f}); P2 median "
             f"{S.median(p_):.2f} s (mean {S.mean(p_):.2f}, {min(p_):.2f}-{max(p_):.2f}); P2 / card {S.median(p_) / S.median(c_):.3f} "
             f"(median), {S.mean(p_) / S.mean(c_):.3f} (mean). Strata on these prompts: 1.46 s (D) / 1.65 s (B) median")
print("\n".join(L))
EOF
say "report rc=$? ($D/${W24FT_REPORT:-w24ft}_report.txt)"
touch $D/W24FT_DONE
