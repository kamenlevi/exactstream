#!/bin/bash
# Fri 6 Nov ~08:0x (the orchestrator 07:5x, first on the GPU): the FAST-NAX engine A/B. Leap's part B (87c9009): experts at
# 16 rows ~25 ms with FAST-NAX (R57_NAX=1 REF2_NAX_1CQ=1) vs ~60 ms for today's 3c-N one-call. Arms on the frozen screen
# (scripts/r57_screen.py, flags only), 12 HumanEval, quiet lock, A-B-B-A: qall / nax / naxb / qallb, the split OFF.
# qall = today's FAST all-in (as allin_wed.sh / fri_split.sh: QOM + D17 + D7b code on NAX + HC8 + DNR + hc_b + SMALL_NATIVE +
# one-call OOO). C by NO-SWAP (peak + 4; NAX given 1.5 GB more): 224 >= 45.2, 208 >= 43.1, 192 >= 41.0, 176 >= 38.9, 160 >= 36.7.
# Passes overlapping a fanwatch freeze are dropped from the pair means and listed.
# Line (pre-registered Fri 08:0x, before arming): PASS = pair-mean geo steady tok/s >= +8 % vs qall AND kept per check within
# 1 %; KILL = < +3 %; else BETWEEN. Below C 224: the verdict is INDICATIVE.
# usage: NOTAFTER=hh:mm NOWAIT45=1 GAP0=0 GAP=0 MARGIN=4 bash scripts/turn45.sh 33 30 "..." /bin/bash scripts/fri_nax.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "${FN_ROOT:-$HOME/BigMoE}" || exit 1
mkdir -p data/fnx data/r57/snap
LOG="$PWD/data/fnx/fnx.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
A=${FN_AVAIL:-$(avail_gb)}
fits() { awk -v a="$A" -v n="$1" 'BEGIN {exit !(n <= a)}'; }
CC=; for c in 224:45.2 208:43.1 192:41.0 176:38.9 160:36.7; do fits ${c#*:} && { CC=${c%%:*}; break; }; done
[ -z "$CC" ] && { say "NO-SWAP $A GB < 36.7: nothing fits, nothing runs"; exit 0; }
cp scripts/r57_screen.py data/r57/snap/r57_screen_fnx.py && cp scripts/c45_runner.sh data/r57/snap/c45_runner_fnx.sh || exit 2
RUN="/bin/bash data/r57/snap/c45_runner_fnx.sh"
export C45_OUT="$PWD/data/fnx" C45_SWAP0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}') C45_ALARM=900 \
  MG=${FN_MG:-4} C45_DEADLINE=${FN_DEADLINE:-$(date -v+25M +%H%M)} C45_SCREEN=data/r57/snap/r57_screen_fnx.py C45_REFTOK=
K="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79"
K="$K,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,humaneval/HumanEval/56,humaneval/HumanEval/141"
K="$K,humaneval/HumanEval/101,humaneval/HumanEval/63"
QA="BIGMOE_EXT_BUILD=build_qom,R57_REFTOK=,R57_FAST=QOM,R57_D17=1,R57_D7B=code,R57_HC8=1,REF2_HC8_FAST=1,R57_DNR=1,R57_HCB=1"
QA="$QA,REF2_SMALL_NATIVE=1,R57_D7BNAX=1,REF2_OOO_1C=1"
rm -f data/r57/nx*_${CC}_*
say "FAST-NAX A/B start (pid $$, NO-SWAP $A GB -> C $CC, deadline $C45_DEADLINE)"
for a in "nxq:$QA" "nxn:$QA,R57_NAX=1,REF2_NAX_1CQ=1" "nxnb:$QA,R57_NAX=1,REF2_NAX_1CQ=1" "nxqb:$QA"; do
  R57_KEYS=$K C45_ARMS="${a%%:*}_$CC:$CC:${a#*:}" $RUN
  say "${a%%:*} rc=$?"
done
"${FN_PY:-$HOME/BigMoE/.venv/bin/python}" -I - "$CC" "$(date +%F)" "${FN_FANLOG:-$HOME/Library/Logs/fanwatch.log}" >> "$LOG" 2>&1 <<'EOF'
import datetime as D, json, math, re, statistics as S, sys
C, day, fanlog = sys.argv[1], sys.argv[2], sys.argv[3]
g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")
win, t0 = [], None
try:
    for ln in open(fanlog):
        m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (froze|cool)", ln)
        if m and m.group(1).startswith(day):
            t = D.datetime.fromisoformat(m.group(1))
            if m.group(2) == "froze":
                t0 = t
            elif t0 is not None:
                win.append((t0, t)); t0 = None
except OSError:
    pass
arm_t = {}
for ln in open("data/fnx/c45.log"):
    m = re.match(r"(\d\d:\d\d:\d\d) (run|done) (\S+)", ln)
    if m:
        arm_t.setdefault(m.group(3), {})[m.group(2)] = D.datetime.fromisoformat(f"{day} {m.group(1)}")
hit = {t for t, v in arm_t.items() if "run" in v and "done" in v and any(a < v["done"] and b > v["run"] for a, b in win)}
P = {"qall": ("nxq", "nxqb"), "nax": ("nxn", "nxnb")}
runs = {}
for t in sum(P.values(), ()):
    try:
        runs[t] = {r["key"]: r for r in json.load(open(f"data/r57/{t}_{C}_runs.json"))}
    except OSError:
        runs[t] = {}
ok = lambda t, k: k in runs[t] and "PAUSED" not in runs[t][k].get("bad", "") and f"{t}_{C}" not in hit
kept = lambda r: r["kept"] if isinstance(r["kept"], (int, float)) else S.mean(r["kept"])
def pm(ps, f):
    out = {}
    for k in set().union(*[runs[t] for t in ps]):
        v = [f(runs[t][k]) for t in ps if ok(t, k)]
        if v:
            out[k] = sum(v) / len(v)
    return out
def ratio(a, b):
    ks = [k for k in a if k in b]
    return (g([a[k] / b[k] for k in ks]), len(ks)) if ks else (float("nan"), 0)
L = [f"FAST-NAX A/B, C {C}, 12 HumanEval (data/fnx): qall / nax / naxb / qallb (A-B-B-A); freezes dropped: {sorted(hit) or 'none'}"]
for b, ps in P.items():
    tp = pm(ps, lambda r: r["tpsw"])
    L.append(f"{b:4s} geo steady {g(list(tp.values())):6.1f} tok/s over {len(tp)}; passes " +
             ", ".join(f"{t} {g([r['tpsw'] for k, r in runs[t].items() if ok(t, k)]):.1f}" for t in ps) +
             f"; kept per check {g(list(pm(ps, kept).values())):.2f}")
r_, n_ = ratio(pm(P["nax"], lambda r: r["tpsw"]), pm(P["qall"], lambda r: r["tpsw"]))
k_, _ = ratio(pm(P["nax"], kept), pm(P["qall"], kept))
q1, q2 = pm(("nxq",), lambda r: r["tpsw"]), pm(("nxqb",), lambda r: r["tpsw"])
d_, _ = ratio(q2, q1)
v = "MISSING" if n_ == 0 else ("PASS" if r_ >= 1.08 and abs(k_ - 1) <= 0.01 else "KILL" if r_ < 1.03 else "BETWEEN")
L.append(f"drift: qall second pass vs first {100 * (d_ - 1):+.1f} %")
L.append(f"Verdict FAST-NAX (pre-registered Fri 08:0x){' INDICATIVE at C < 224' if int(C) < 224 else ''}: {v}: tok/s "
         f"{100 * (r_ - 1):+.1f} %, kept {100 * (k_ - 1):+.1f} % vs qall over {n_} (PASS >= +8 % AND kept within 1 %; KILL < +3 %)")
open("data/fnx/fnx_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? (data/fnx/fnx_report.txt)"
say "FAST-NAX A/B end"
