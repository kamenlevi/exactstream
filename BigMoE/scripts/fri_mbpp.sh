#!/bin/bash
# Fri 6 Nov 15:4x (NewIdeas 15:2x; Kamen extended today's big-test window to 17:00): NEVER-WAIT and SPLIT-NAX vs the stack by
# day, 12 HumanEval, C 224 by NO-SWAP (below 224 INDICATIVE), palindrome sts / ssp / snv / snvb / sspb / stsb; then, if the
# deadline allows, GOAL's MBPP12 CD16 speed pair mq / mc / mcb / mqb (qall vs qall + CD16, 12 MBPP, CAP 384). ENDS by 16:10
# (EvalSuite's MBPP reference 16:15-16:55): no new arm after C45_DEADLINE (default 16:02).
# Lines (pre-registered before any of these runs; NewIdeas Fri 10:1x / 13:5x, unchanged; GOAL MBPP12.md section 4):
#   SPLIT-NAX  = stack + R57_QOM_SPLIT=0.99: net pair-mean geo steady tok/s vs the stack on code: PASS >= +8 %, KILL < +3 %,
#                else BETWEEN; kept beside (no clause). Speed only tonight whatever it scores (NewIdeas).
#   NEVER-WAIT = stack + R57_NEVERWAIT=1 + REF2_NEVERWAIT=1: the same line. On a PASS it joins tonight's gated stack on the code
#                sets only (NewIdeas 15:2x). Per-check medians and never-wait counters beside.
#   MBPP12     = qall + R57_INERT=CD16 vs qall, 12 MBPP: PASS K ratio >= 1.30 AND tokens per task <= 1.05 AND test verdicts
#                equal on >= 11 / 12 (no net loss > 1); KILL K < 1.15 or tokens >= 1.15 or >= 2 tasks flip to fail; runs cut at
#                the cap are dropped. This script prints K and tokens; the verdicts come after (EvalSuite tools/mbpp_score.py).
# usage: NOTAFTER=15:55 GAP0=0 GAP=0 MARGIN=4 bash scripts/turn45.sh 33 30 "..." /bin/bash scripts/fri_day.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "${FD_ROOT:-$HOME/BigMoE}" || exit 1
mkdir -p data/fday data/r57/snap
LOG="$PWD/data/fday/fmbpp.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
STACK="R57_NAX=1,REF2_NAX_1CQ=1,REF2_DN_KS=1,REF2_GDN_LAZY=1,D7BNAX_32=1,R57_INERT=CD16,R57_FASTTAX=1"   # = tonight's STACK
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
A=${FD_AVAIL:-$(avail_gb)}
fits() { awk -v a="$A" -v n="$1" 'BEGIN {exit !(n <= a)}'; }
CC=; for c in 224:44.0 208:41.9 192:39.7 176:37.6 160:35.4; do fits ${c#*:} && { CC=${c%%:*}; break; }; done   # Fri C 224 runs peaked 38.4-39.4 GB; stack ~40 + 4
[ -z "$CC" ] && { say "NO-SWAP $A GB < 35.4: nothing fits, nothing runs"; exit 0; }
cp scripts/r57_screen_fast.py data/r57/snap/r57_screen_fday.py && cp scripts/relax_delta.py scripts/relax_span.py scripts/inert_span.py \
  data/r57/snap/ && cp scripts/c45_runner.sh data/r57/snap/c45_runner_fday.sh || exit 2
FZ="$PWD/data/r57/snap/frozen_fday"; rm -rf "$FZ"; mkdir -p "$FZ"       # the outside code frozen, as sat_night.sh does
cp -R "$HOME/Leap/scripts" "$FZ/Leap_scripts" && cp -R "$HOME/Leap/third_party" "$FZ/Leap_third_party" \
  && cp -R "$HOME/TrainLab/sticky" "$FZ/TrainLab_sticky" && cp -R "$HOME/Breakthrough/keepset" "$FZ/Breakthrough_keepset" || exit 2
find "$FZ" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
sed -i '' -e "s#\"~/Leap/scripts\"#\"$FZ/Leap_scripts\"#g" -e "s#\"~/Leap/third_party\"#\"$FZ/Leap_third_party\"#g" \
  -e "s#\"~/TrainLab/sticky\"#\"$FZ/TrainLab_sticky\"#g" -e "s#\"/Users/kamen/Breakthrough/keepset\"#\"$FZ/Breakthrough_keepset\"#g" \
  data/r57/snap/r57_screen_fday.py
say "frozen outside code ($(grep -c frozen_fday data/r57/snap/r57_screen_fday.py) screen paths): ref2 / d7b_nax / qom_keep md5 $(md5 -q "$HOME/Leap/scripts/ref2.py" "$HOME/Leap/scripts/d7b_nax.py" "$HOME/Breakthrough/keepset/qom_keep.py" | tr '\n' ' ')"
RUN="/bin/bash data/r57/snap/c45_runner_fday.sh"
export C45_OUT="$PWD/data/fday" C45_SWAP0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}') C45_ALARM=600 \
  MG=4 C45_DEADLINE=${FD_DEADLINE:-1602} C45_SCREEN=data/r57/snap/r57_screen_fday.py C45_REFTOK=
KC="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79"
KC="$KC,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,humaneval/HumanEval/56,humaneval/HumanEval/141"
KC="$KC,humaneval/HumanEval/101,humaneval/HumanEval/63"
KM="mbpp/16,mbpp/100,mbpp/146,mbpp/161,mbpp/183,mbpp/208,mbpp/224,mbpp/246,mbpp/268,mbpp/296,mbpp/305,mbpp/319"   # GOAL's 12
QAC="BIGMOE_EXT_BUILD=build_qom,R57_REFTOK=,R57_FAST=QOM,R57_D17=1,R57_D7B=code,R57_HC8=1,REF2_HC8_FAST=1,R57_DNR=1,R57_HCB=1"
QAC="$QAC,REF2_SMALL_NATIVE=1,R57_D7BNAX=1,REF2_OOO_1C=1,R57_QCNT1C=1"   # = tonight's QAC
NWE="R57_NEVERWAIT=1,REF2_NEVERWAIT=1"
MQ="$QAC"   # R57_QOM_SETS goes in the environment: c45_runner splits arm envs on commas (15:49 rc=127)
for t in mq mc mcb mqb; do rm -f data/r57/d${t}_${CC}_*; done
say "MBPP12 re-run start (pid $$, NO-SWAP $A GB -> C $CC, deadline $C45_DEADLINE)"
for a in "dmq:$MQ" "dmc:$MQ,R57_INERT=CD16,R57_INERT_LOG=$PWD/data/fday/mc_inert.jsonl" \
         "dmcb:$MQ,R57_INERT=CD16,R57_INERT_LOG=$PWD/data/fday/mcb_inert.jsonl" "dmqb:$MQ"; do
  R57_QOM_SETS=humaneval,mbpp C45_CAP=384 R57_KEYS=$KM C45_ARMS="${a%%:*}_$CC:$CC:${a#*:}" $RUN
  say "${a%%:*} rc=$?"
done
"${FD_PY:-$HOME/BigMoE/.venv/bin/python}" -I - "$CC" "$(date +%F)" "${FD_FANLOG:-$HOME/Library/Logs/fanwatch.log}" >> "$LOG" 2>&1 <<'EOF'
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
arm_t = {}
for ln in open("data/fday/c45.log"):
    m = re.match(r"(\d\d:\d\d:\d\d) (run|done) (\S+)", ln)
    if m:
        arm_t.setdefault(m.group(3), {})[m.group(2)] = D.datetime.fromisoformat(f"{day} {m.group(1)}")
hit = {t for t, v in arm_t.items() if "run" in v and "done" in v and any(a < v["done"] and b > v["run"] for a, b in win)}
P = {"stack": ("dsts", "dstsb"), "ssp": ("dssp", "dsspb"), "snv": ("dsnv", "dsnvb"), "mq": ("dmq", "dmqb"), "mc": ("dmc", "dmcb")}
runs = {}
for t in sum(P.values(), ()):
    try:
        runs[t] = {r["key"]: r for r in json.load(open(f"data/r57/{t}_{C}_runs.json"))}
    except (OSError, ValueError):
        runs[t] = {}
ok = lambda t, k: k in runs[t] and "PAUSED" not in runs[t][k].get("bad", "") and f"{t}_{C}" not in hit and not (k.startswith("mbpp") and runs[t][k].get("cut"))
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
def pcm(ts, f):
    xs = []
    for t in ts:
        if f"{t}_{C}" in hit:
            continue
        try:
            for e in json.load(open(f"data/r57/{t}_{C}_checks.json")):
                if not e.get("paused"):
                    xs += [x for x in f(e)[2:] if x is not None]
        except (OSError, ValueError):
            pass
    return S.median(xs) if xs else float("nan")
ind = " INDICATIVE at C < 224" if int(C) < 224 else ""
L = [f"FRI DAY RUN, C {C} (data/fday): sts / ssp / snv / snvb / sspb / stsb on 12 HumanEval, then mq / mc / mcb / mqb on 12 MBPP "
     f"(CAP 384); freeze windows {[(a.strftime('%H:%M:%S'), b.strftime('%H:%M:%S')) for a, b in win]}; passes dropped for a "
     f"freeze: {sorted(hit) or 'none'}"]
for b, ps in P.items():
    tp = pm(ps, lambda r: r["tpsw"])
    L.append(f"{b:5s} geo steady {g(list(tp.values())):6.1f} tok/s over {len(tp)}; passes " +
             ", ".join(f"{t} {g([r['tpsw'] for k, r in runs[t].items() if ok(t, k)]):.1f}" for t in ps) +
             f"; kept per check {g(list(pm(ps, kept).values())):.2f}")
for nm, b, when in (("SPLIT-NAX", "ssp", "Fri 10:1x"), ("NEVER-WAIT", "snv", "Fri 13:5x")):
    r_, n_ = ratio(pm(P[b], lambda r: r["tpsw"]), pm(P["stack"], lambda r: r["tpsw"]))
    k_, _ = ratio(pm(P[b], kept), pm(P["stack"], kept))
    v_ = "MISSING" if n_ == 0 else ("PASS" if r_ >= 1.08 else "KILL" if r_ < 1.03 else "BETWEEN")
    L.append(f"Verdict {nm} (pre-registered {when}, NewIdeas){ind}: {v_}: tok/s {100 * (r_ - 1):+.1f} % vs the stack on code over "
             f"{n_}, kept {100 * (k_ - 1):+.1f} % (PASS >= +8 %, KILL < +3 %)")
for b in ("stack", "snv"):
    L.append(f"  per check (checks >= 2, medians) {b:5s}: check {pcm(P[b], lambda e: e.get('ck', [])):.1f} ms, gate service "
             f"{pcm(P[b], lambda e: e.get('svc', [])):.1f} ms, misses {pcm(P[b], lambda e: e.get('miss', [])):.0f}, stalled layer "
             f"calls {pcm(P[b], lambda e: e.get('stl', [])):.1f}, eval wait {pcm(P[b], lambda e: e.get('wt', [])):.1f} ms")
L.append("  snv never-wait counters per check (medians): " + ", ".join(
    f"{n} {pcm(P['snv'], lambda e, i=i: [x[i] for x in e.get('nw', [])]):.1f}"
    for i, n in enumerate(("late groups", "late uses", "late mass", "last-layer late uses (dropped or waited)", "stalls"))))
kr, nk = ratio(pm(P["mc"], kept), pm(P["mq"], kept))
tr, _ = ratio(pm(P["mc"], lambda r: r["n"]), pm(P["mq"], lambda r: r["n"]))
sr, _ = ratio(pm(P["mc"], lambda r: r["tpsw"]), pm(P["mq"], lambda r: r["tpsw"]))
cut = sorted({k for t in P["mq"] + P["mc"] for k, r in runs[t].items() if r.get("cut")})
L.append(f"MBPP12 speed part (GOAL's line; verdicts pending){ind}: K ratio {kr:.3f} over {nk} tasks (PASS >= 1.30, KILL < 1.15), "
         f"tokens per task {tr:.3f} (PASS <= 1.05, KILL >= 1.15), tok/s {100 * (sr - 1):+.1f} %; cut at the cap: {cut or 'none'}")
open("data/fday/fmbpp_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? (data/fday/fmbpp_report.txt)"
say "MBPP12 re-run end"
