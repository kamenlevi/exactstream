#!/bin/bash
# Sat 7 Nov (NewIdeas QUEUE S2, 00:4x; Leap's smoke PASSED 00:3x): S2 vs code_env (stackb) on the 12 HumanEval, quiet, C 256
# (stackb's shipping C: its C 256 line PASSed), palindrome b1 / s1 / s2 / b2. On scripts/r57_screen_fast.py with Leap's S2 diff
# (97a23bf), the outside code frozen at the start. Needs NO-SWAP >= 47.2 GB (C 256's peak 42.8 + 4), else not run.
#   B  = stackb = the stack + ship B span (R57_RELAX_SPAN=1, R57_RELAX_SPAN_RULE=R2:2:0.25) + MATCH g2 at 16 rows (code_env)
#   S2 = B + R57_SBN=2 (submit every 2 layers: Leap's glue fix) + D7BNAX_32=2 (single 32-row draft kernel) + R57_DSUB=1 (the
#        drafter's screened head) + R57_SVH=1 (the screened verifier head: FAST) [R57_SV_CHECK=1 for the smoke counters]
# Line (NewIdeas Sat 00:4x, pre-registered before arming): pair-mean geo steady tok/s, S2 vs B:
#   PASS >= +2 %, KILL < 0 %, else BETWEEN. Tokens: per prompt, S2's passes vs B's (B1 vs B2 printed as the floor: QOM's outputs
#   can differ run to run). Equal on all 12 in both pairs -> no gate needed; otherwise S2 joins S4's gate (EvalSuite).
#   Beside: the median cycle ms per arm and S2 - B (Leap's range -1.3..-3.6 ms), kept per check, tokens per task, lifetime peaks.
# usage: NOTAFTER=hh:mm WAITMEM=44 NOWAIT45=1 GAP0=0 GAP=0 MARGIN=4 bash scripts/turn45.sh 33 20 "..." /bin/bash scripts/s2_ab.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "${S2_ROOT:-$HOME/BigMoE}" || exit 1
mkdir -p data/s2 data/r57/snap
LOG="$PWD/data/s2/s2.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
PY=${S2_PY:-$HOME/BigMoE/.venv/bin/python}
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
A=${S2_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2: C 256 does not fit, not run"; exit 0; }
CC=256
FZ="$PWD/data/r57/snap/frozen_s2"; rm -rf "$FZ"; mkdir -p "$FZ"
cp scripts/r57_screen_fast.py data/r57/snap/r57_screen_s2.py && cp scripts/relax_delta.py scripts/relax_span.py scripts/inert_span.py \
  data/r57/snap/ && cp scripts/c45_runner.sh data/r57/snap/c45_runner_s2.sh || exit 2
[ -n "$S2_DRYRUN" ] || { cp -R "$HOME/Leap/scripts" "$FZ/Leap_scripts" && cp -R "$HOME/Leap/third_party" "$FZ/Leap_third_party" \
  && cp -R "$HOME/TrainLab/sticky" "$FZ/TrainLab_sticky" && cp -R "$HOME/Breakthrough/keepset" "$FZ/Breakthrough_keepset" || exit 2; }
find "$FZ" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
sed -i '' -e "s#\"~/Leap/scripts\"#\"$FZ/Leap_scripts\"#g" -e "s#\"~/Leap/third_party\"#\"$FZ/Leap_third_party\"#g" \
  -e "s#\"~/TrainLab/sticky\"#\"$FZ/TrainLab_sticky\"#g" -e "s#\"/Users/kamen/Breakthrough/keepset\"#\"$FZ/Breakthrough_keepset\"#g" \
  data/r57/snap/r57_screen_s2.py
say "S2 A/B start (pid $$, NO-SWAP $A GB, C $CC); frozen ($(grep -c frozen_s2 data/r57/snap/r57_screen_s2.py) screen paths); ref2 / session_vocab md5 $(md5 -q "$HOME/Leap/scripts/ref2.py" "$HOME/Leap/scripts/session_vocab.py" 2>/dev/null | tr '\n' ' ')"
RUN="/bin/bash data/r57/snap/c45_runner_s2.sh"
export C45_OUT="$PWD/data/s2" C45_SWAP0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}') C45_ALARM=1200 \
  MG=4 C45_DEADLINE=$(date -v+45M +%H%M) C45_SCREEN=data/r57/snap/r57_screen_s2.py C45_REFTOK=
K="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79"
K="$K,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,humaneval/HumanEval/56,humaneval/HumanEval/141"
K="$K,humaneval/HumanEval/101,humaneval/HumanEval/63"
QAC="BIGMOE_EXT_BUILD=build_qom,R57_REFTOK=,R57_FAST=QOM,R57_D17=1,R57_D7B=code,R57_HC8=1,REF2_HC8_FAST=1,R57_DNR=1,R57_HCB=1"
QAC="$QAC,REF2_SMALL_NATIVE=1,R57_D7BNAX=1,REF2_OOO_1C=1,R57_QCNT1C=1"
STACK="R57_NAX=1,REF2_NAX_1CQ=1,REF2_DN_KS=1,REF2_GDN_LAZY=1,D7BNAX_32=1,R57_FASTTAX=1"
B="$QAC,$STACK,R57_RELAX_SPAN=1,R57_RELAX_SPAN_RULE=R2:2:0.25,R57_COPY=g2,R57_RELAX_LOG=$PWD/data/s2/b_relax.jsonl"   # = stackb
S="$B,R57_SBN=2,D7BNAX_32=2,R57_DSUB=1,R57_SVH=1,R57_SV_CHECK=1"   # later entries win (D7BNAX_32=2 over the stack's =1)
for t in s2b1 s2s1 s2s2 s2b2; do rm -f data/r57/${t}_${CC}_* data/s2/${t}_${CC}.txt; done
for a in "s2b1:$B" "s2s1:$S" "s2s2:$S" "s2b2:$B"; do
  R57_KEYS=$K C45_ARMS="${a%%:*}_$CC:$CC:${a#*:}" $RUN
  say "${a%%:*} rc=$?"
done
"$PY" -I - "$CC" "$(date +%F)" "${S2_FANLOG:-$HOME/Library/Logs/fanwatch.log}" >> "$LOG" 2>&1 <<'EOF'
import datetime as D, json, math, re, statistics as S, sys
C, day, fanlog = sys.argv[1], sys.argv[2], sys.argv[3]
g = lambda xs: (lambda ys: math.exp(sum(map(math.log, ys)) / len(ys)) if ys else float("nan"))([x for x in xs if x and x > 0])
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
arm_t, peak = {}, {}
for ln in open("data/s2/c45.log"):
    m = re.match(r"(\d\d:\d\d:\d\d) (run|done) (\S+)", ln)
    if m:
        arm_t.setdefault(m.group(3), {})[m.group(2)] = D.datetime.fromisoformat(f"{day} {m.group(1)}")
    m = re.search(r"done (\S+) rc=\d+:.*lifetime peak ([\d.]+) GB", ln)
    if m:
        peak[m.group(1)] = float(m.group(2))
hit = {t for t, v in arm_t.items() if "run" in v and "done" in v and any(a < v["done"] and b > v["run"] for a, b in win)}
P = {"B (stackb)": ("s2b1", "s2b2"), "S2": ("s2s1", "s2s2")}
runs, toks = {}, {}
for t in sum(P.values(), ()):
    try:
        runs[t] = {r["key"]: r for r in json.load(open(f"data/r57/{t}_{C}_runs.json"))}
        toks[t] = {k: list(v.values())[0][0] for k, v in json.load(open(f"data/r57/{t}_{C}_tokens.json")).items()}
    except (OSError, ValueError):
        runs[t], toks[t] = {}, {}
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
L = [f"S2 A/B, C {C} (data/s2): B (stackb) / S2 / S2 / B; freeze windows {[(a.strftime('%H:%M:%S'), b.strftime('%H:%M:%S')) for a, b in win]}; "
     f"passes dropped for a freeze: {sorted(hit) or 'none'}"]
for b, ps in P.items():
    tp = pm(ps, lambda r: r["tpsw"])
    L.append(f"{b:10s} geo steady {g(list(tp.values())):6.1f} tok/s over {len(tp)}; passes " +
             ", ".join(f"{t} {g([r['tpsw'] for k, r in runs[t].items() if ok(t, k)]):.1f}" for t in ps) +
             f"; kept per check {g(list(pm(ps, kept).values())):.2f}; tokens per task "
             f"{S.median(pm(ps, lambda r: r['n']).values()) if pm(ps, lambda r: r['n']) else float('nan'):.0f}; lifetime peak "
             f"{max([peak.get(f'{t}_{C}', 0.0) for t in ps]) or float('nan'):.1f} GB")
r_, n_ = ratio(pm(P["S2"], lambda r: r["tpsw"]), pm(P["B (stackb)"], lambda r: r["tpsw"]))
v = "MISSING" if n_ == 0 else "PASS" if r_ >= 1.02 else "KILL" if r_ < 1.00 else "BETWEEN"
L.append(f"Verdict S2 (pre-registered NewIdeas Sat 00:4x): {v}: {100 * (r_ - 1):+.1f} % vs B (stackb) over {n_} (PASS >= +2 %, KILL < 0 %)")
cyc = {b: S.median([r["cyc"] for t in ps for k, r in runs[t].items() if ok(t, k) and isinstance(r.get("cyc"), (int, float))] or [float("nan")])
       for b, ps in P.items()}
L.append(f"median cycle per run: B {cyc['B (stackb)']:.1f} ms, S2 {cyc['S2']:.1f} ms, S2 - B {cyc['S2'] - cyc['B (stackb)']:+.1f} ms "
         f"(Leap's range -1.3..-3.6)")
same = lambda a, b: sum(1 for k in toks[a] if k in toks[b] and toks[a][k] == toks[b][k])
pairs = (("s2b1", "s2s1"), ("s2b2", "s2s2"))
ident = all(same(a, b) == 12 for a, b in pairs)
L.append(f"tokens identical per prompt: S2 vs B {[same(a, b) for a, b in pairs]} of 12 (floor B1 vs B2: {same('s2b1', 's2b2')}, S2 vs S2: "
         f"{same('s2s1', 's2s2')}) -> {'no gate needed (NewIdeas)' if ident else 'S2 joins S4 (EvalSuite gate)'}")
for t in P["S2"]:
    try:
        sv = [ln.strip() for ln in open(f"data/s2/{t}_{C}.txt") if "SESSION-VOCAB" in ln]
        if sv:
            L.append(f"  {t}: {sv[-1][:200]}")
    except OSError:
        pass
open("data/s2/s2_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? (data/s2/s2_report.txt)"
say "S2 A/B end"
