#!/bin/bash
# Wed 11 Nov: P3, the exact first-token step (docs/FIRST_TOKEN.md; Kamen 11:1x via the orchestrator: first token much faster
# than Strata's, cheaply). The 45 GB chat as shipped (zs + PF-D v4 + read-ahead) against the same + R57_PFD_FOLD=1: the next
# layer's forecast evaluated with this layer's routing ids (one sync, not two), its B + C staging arrays in one eval, the
# assemble pass with the call (one eval, not two). Same reads, same values: tokens and logits hash must equal the control.
# The 12 N45 IFEval prompts (data/n45/keys.txt line 1), zs, CAP 256, one process per arm, ABBA against the order bias
# (memory: back-to-back arms favour the second): from 14:5x ABCCBA = p3c1 control, p3f1 fold, p2p1 / p2p2 P2 plain (LOSSY),
# p3f2 fold, p3c2 control; then FAST (LOSSY, info)
# without and with the fold: p3fa, p3ff. Then P2 v0 (LOSSY, a first screen): prompts <= 256 tokens through the decode's
# check path in 16-row chunks (R57_P2=256; p2v verify scope + full-accept commit, the first token a plain forward; p2p plain
# forwards under the decode arm), vs p3c1.
# REPORT data/p3/p3_report.txt: per arm TTFT mean / median / range, the PF-D timers per prompt (medians), SSD per prompt, tokens
# and logits hash vs p3c1 (FAST arms vs p3fa), lifetime peak; pooled control vs pooled fold.
# usage (a harness background task): NOTBEFORE=00:30 NOTAFTER=05:30 bash scripts/turn45.sh 34 35 "P3 first-token A/B" /bin/bash scripts/p3.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "${P3_ROOT:-$HOME/BigMoE}" || exit 1
D=data/p3; mkdir -p $D
LOG="$PWD/$D/p3.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
PY=${P3_PY:-$HOME/BigMoE/.venv/bin/python}
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
A=${P3_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2: not run"; exit 0; }
[ -n "$P3_DRYRUN" ] || sleep 20
A=${P3_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2 at the 20 s look-again: not run"; exit 0; }
BUN="BIGMOE_PAUSE_EVERY_MS=50,R57_S0CARRY=1,REF2_SB_FIRST=1,REF2_PLE_LATE=1,REF2_COMMIT_SUB=9,R57_DQ1=1"
L=$(grep -v '^#' data/stack/s5p_env.txt | grep . | tail -1 | sed 's/,R57_QOM_SETS=humaneval,mbpp$//')
ZS="$L,R57_QOM_MTP=2,REF2_NAX_COH=1,R57_SB1=1"
V4="$ZS,R57_PFD=1,R57_PFD_V=4,R57_PFD_AHEAD=1"
FA="$ZS,R57_PFD=1,R57_PFD_FAST=1,R57_PFD_AHEAD=1"
KI=$(sed -n 1p data/n45/keys.txt)
[ "$(echo "$KI" | tr ',' '\n' | grep -c '^ifeval/')" = 12 ] || { say "data/n45/keys.txt line 1 is not 12 IFEval keys: not run"; exit 2; }
FR=$(bash scripts/deep_freeze.sh p3) || { say "freeze failed: $FR"; exit 2; }
SD=data/r57/snap/p3
grep -q 'R57_PFD_FOLD' "$SD/pflm.py" && grep -q 'def stage_many' "$SD/pflm.py" && grep -q 'PFD_ON' "$SD/r57_screen_p3.py" \
  && grep -q 'def prefill_p2' "$SD/r57_screen_p3.py" \
  || { say "the frozen pflm.py has no P3 fold (R57_PFD_FOLD, stage_many) or the screen no PF-D: not run"; exit 2; }
[ -n "$P3_DRYRUN" ] && { echo "dry run: $FR; ZS $(echo "$ZS" | tr ',' '\n' | grep -c =) K=V, KI $(echo "$KI" | tr ',' '\n' | grep -c .)"; exit 0; }
"$PY" scripts/pfd_unit.py 6 >> "$LOG" 2>&1 || { say "pfd_unit.py FAILED: not run"; exit 4; }
"$PY" scripts/pfd_fold_unit.py 6 >> "$LOG" 2>&1 || { say "pfd_fold_unit.py FAILED (P3 staging != the records): not run"; exit 4; }
git add data/r57/snap/frozen_p3 ':!data/r57/snap/frozen_p3/**/__pycache__/**' "$SD" ":!$SD/__pycache__/**" data/r57/snap/c45_runner_p3.sh \
> (a quote from the project owner, kept private)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011DvdV6eUB4NgBbVkyVFPrT" || { say "snapshot commit failed: not run"; exit 3; }
SNAP=$(git rev-parse --short HEAD)
RUN="/bin/bash data/r57/snap/c45_runner_p3.sh"
export C45_OUT="$PWD/$D" C45_SWAP0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}') C45_ALARM=2400 \
  MG=4 C45_DEADLINE=$(date -v+60M +%H%M) C45_SCREEN="$SD/r57_screen_p3.py" C45_REFTOK= R57_QOM_SETS=humaneval,mbpp R57_MODMAN=1
rm -f $D/DONE
say "P3 start (pid $$, NO-SWAP $A GB, snapshot $SNAP)"
TAGS=""
arm() {   # tag cap keys qom_sets env
  rm -f data/r57/${1}_256_* $D/${1}_256.txt
  R57_KEYS=$3 C45_CAP=$2 R57_PFSTAT=1 R57_LOGHASH=1 R57_QOM_SETS=$4 C45_ARMS="${1}_256:256:$5" $RUN
  say "$1 rc=$?"; TAGS="$TAGS $1"
}
QZ="humaneval,mbpp,ifeval,multi"
# 14:5x (the orchestrator): ABCCBA, so the fold AND P2 (plain, its faster form at 24 GB) each get an early and a late slot
arm p3c1 256 "$KI" $QZ "$V4"
arm p3f1 256 "$KI" $QZ "$V4,R57_PFD_FOLD=1"
arm p2p1 256 "$KI" $QZ "$V4,R57_P2=256,R57_P2_VERIFY=0"
arm p2p2 256 "$KI" $QZ "$V4,R57_P2=256,R57_P2_VERIFY=0"
arm p3f2 256 "$KI" $QZ "$V4,R57_PFD_FOLD=1"
arm p3c2 256 "$KI" $QZ "$V4"
arm p3fa 256 "$KI" $QZ "$FA"
arm p3ff 256 "$KI" $QZ "$FA,R57_PFD_FOLD=1"
arm p2v 256 "$KI" $QZ "$V4,R57_P2=256"
"$PY" -I - "$SNAP" "$TAGS" >> "$LOG" 2>&1 <<'EOF'
import json, re, sys, statistics as S
snap = sys.argv[1]; tags = sys.argv[2].split() if len(sys.argv) > 2 else []
peak = {}
try:
    for ln in open("data/p3/c45.log"):
        m = re.search(r"done (\S+) rc=\d+:.*lifetime peak ([\d.]+) GB", ln)
        if m:
            peak[m.group(1)] = float(m.group(2))
except OSError:
    pass
def load(t):
    try:
        return ({r["key"]: r for r in json.load(open(f"data/r57/{t}_256_runs.json"))},
                {k: list(v.values())[0][0] for k, v in json.load(open(f"data/r57/{t}_256_tokens.json")).items()})
    except (OSError, ValueError, KeyError):
        return {}, {}
TM = ("tP", "tJ", "tPred", "tB", "tC", "tG", "tM")
L = [f"P3 (data/p3; snapshot {snap}; the 12 N45 IFEval prompts, zs, CAP 256, quiet): TTFT = pfs; PF-D timers per prompt "
     f"(medians, s): tP the wrapped calls, tJ the folded id + forecast sync, tPred the forecast's own work, tB / tC the waits "
     f"for read rows, tG the assemble eval (with the fold: inside tM), tM the call; tokens and logits hash vs the arm's control"]
TT = {}
def arm(t, ctl, nm):
    R, T = load(t)
    R0, T0 = load(ctl)
    if t not in tags:
        return
    if not R:
        L.append(f"  {t} {nm}: MISSING")
        return
    ks = sorted(R)
    ttft = [R[k]["pfs"] for k in ks if R[k].get("pfs")]
    TT[t] = ttft
    pd = [R[k].get("pfd") or {} for k in ks]
    tm = "; ".join(f"{x} {S.median([p.get(x, 0.0) for p in pd]):.3f}" for x in TM)
    gbs = [(R[k].get("pfst") or {}).get("gb") for k in ks]
    gbs = [g for g in gbs if g is not None]
    same = sum(1 for k in ks if T.get(k) == T0.get(k))
    hs = sum(sum(1 for a_, b_ in zip(R[k].get("lh") or [], (R0.get(k) or {}).get("lh") or []) if a_ == b_) for k in ks)
    hn = sum(min(len(R[k].get("lh") or []), len((R0.get(k) or {}).get("lh") or [])) for k in ks)
    tok = "" if t == ctl else f"; tokens == {ctl} {same}/{len(ks)}, logits hash {hs}/{hn}"
    fold = sum(p.get("fold", 0) for p in pd)
    p2 = sum((R[k].get("p2") or 0) for k in ks)
    L.append(f"  {t} {nm}: TTFT mean {S.mean(ttft):.2f} s, median {S.median(ttft):.2f} ({min(ttft):.2f}-{max(ttft):.2f}); "
             f"timers {tm}; folded forecasts {fold}; P2 chunks {p2}; SSD {S.mean(gbs) if gbs else float('nan'):.1f} GB per prompt{tok}; "
             f"lifetime peak {peak.get(f'{t}_256', float('nan')):.2f} GB")
    for k in ks:
        r = R[k]
        L.append(f"      {k}: P {r['P']}, TTFT {r['pfs']:.2f} s, tP {(r.get('pfd') or {}).get('tP', float('nan')):.3f}"
                 + ("" if t == ctl else f", tokens {'==' if T.get(k) == T0.get(k) else 'DIFF'}"))
arm("p3c1", "p3c1", "control: zs + PF-D v4 + read-ahead (the 45 GB chat as shipped), 1st")
arm("p3f1", "p3c1", "P3: + R57_PFD_FOLD=1, 2nd")
arm("p2p1", "p3c1", "P2 v0 (LOSSY): R57_P2=256, 16-row plain forwards under the decode arm, 3rd")
arm("p2p2", "p3c1", "P2 v0 (LOSSY): the same, 4th")
arm("p3f2", "p3c1", "P3: + R57_PFD_FOLD=1, 5th")
arm("p3c2", "p3c1", "control again, 6th")
arm("p3fa", "p3fa", "FAST (LOSSY) + read-ahead, 7th (the FAST arms' own control)")
arm("p3ff", "p3fa", "FAST (LOSSY) + read-ahead + R57_PFD_FOLD=1, 8th")
arm("p2v", "p3c1", "P2 v0 (LOSSY): R57_P2=256, 16-row verify chunks + full-accept commits, 9th")
c = TT.get("p3c1", []) + TT.get("p3c2", [])
f = TT.get("p3f1", []) + TT.get("p3f2", [])
if c and f:
    L.append(f"POOLED (ABCCBA): control TTFT median {S.median(c):.2f} s (mean {S.mean(c):.2f}), fold median {S.median(f):.2f} s "
             f"(mean {S.mean(f):.2f}): {100 * (S.median(f) / S.median(c) - 1):+.1f} % median, {100 * (S.mean(f) / S.mean(c) - 1):+.1f} % mean")
q = TT.get("p2p1", []) + TT.get("p2p2", [])
if c and q:
    L.append(f"POOLED (ABCCBA): P2 plain median {S.median(q):.2f} s (mean {S.mean(q):.2f}): {100 * (S.median(q) / S.median(c) - 1):+.1f} % median "
             f"vs the control, {100 * (S.mean(q) / S.mean(c) - 1):+.1f} % mean; Strata at 45 GB on these prompts 0.83 s median")
open("data/p3/p3_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? ($D/p3_report.txt)"
touch $D/DONE
