#!/bin/bash
# Sat 7 Nov (NewIdeas QUEUE S1): NW v3's engine A/B vs the gated stack. Armed only after Breakthrough's nw3 map verdict.
# On scripts/r57_screen_fast.py (Leap's ref2 with NW v3, dcd5f4f8, live, off by default; the r57 NW side gated per set (nwp_):
# NW runs only on prompts whose set has QOM's FAST rule), the outside code frozen at the start. C 256 (needs NO-SWAP >= 47.2 GB).
#   B  = the v1 candidate: S1_BASE=stackb_s2 (default: the stack + ship B span + MATCH g2 at 16 rows + S2) or stack_s2 (no B
#        span, no g2: the fallback if runner 49 rolls B back)
#   NW = B + R57_NEVERWAIT=1 + REF2_NEVERWAIT=1
# 0. A 2-prompt check of NW first (CAP 32): NW has never run with g2 / span / SBN 2 / SVH / DSUB. Not clean -> stop (no A/B).
# 1. Palindrome b1 / n1 / n2 / b2 on the 12 HumanEval.
# Line (NewIdeas QUEUE S1, pre-registered before arming): pair-mean geo steady tok/s, NW vs B: PASS >= +8 %, KILL < +3 %,
#   else BETWEEN. Beside: the median cycle, kept per check, tokens per task, lifetime peaks, NW v3's counters per check (8 fields:
#   late groups, late uses, late mass, L47 drops, L47 rank-0 stalls, forced, optional dropped now, sidecars 3-bit) and token
#   identity per prompt (information: QOM's outputs differ run to run). NW v3 is lossy (3-bit sidecars): a PASS goes to a gate.
# usage: NOTAFTER=hh:mm WAITMEM=48 NOWAIT45=1 GAP0=0 GAP=0 MARGIN=4 bash scripts/turn45.sh 33 20 "..." /bin/bash scripts/s1_ab.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "${S1_ROOT:-$HOME/BigMoE}" || exit 1
mkdir -p data/s1 data/r57/snap
LOG="$PWD/data/s1/s1.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
PY=${S1_PY:-$HOME/BigMoE/.venv/bin/python}
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
A=${S1_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2: C 256 does not fit, not run"; exit 0; }
CC=256
FZ="$PWD/data/r57/snap/frozen_s1"; rm -rf "$FZ"; mkdir -p "$FZ"
SD=data/r57/snap/s1; mkdir -p "$SD"     # own dir: never touch data/r57/snap/*.py while EvalSuite's runs load them
cp scripts/r57_screen_fast.py "$SD/r57_screen_s1.py" && cp scripts/relax_delta.py scripts/relax_span.py scripts/inert_span.py \
  "$SD/" && cp scripts/c45_runner.sh data/r57/snap/c45_runner_s1.sh || exit 2
[ -n "$S1_DRYRUN" ] || { cp -R "$HOME/Leap/scripts" "$FZ/Leap_scripts" && cp -R "$HOME/Leap/third_party" "$FZ/Leap_third_party" \
  && cp -R "$HOME/TrainLab/sticky" "$FZ/TrainLab_sticky" && cp -R "$HOME/Breakthrough/keepset" "$FZ/Breakthrough_keepset" || exit 2; }
find "$FZ" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
sed -i '' -e "s#\"~/Leap/scripts\"#\"$FZ/Leap_scripts\"#g" -e "s#\"~/Leap/third_party\"#\"$FZ/Leap_third_party\"#g" \
  -e "s#\"~/TrainLab/sticky\"#\"$FZ/TrainLab_sticky\"#g" -e "s#\"/Users/kamen/Breakthrough/keepset\"#\"$FZ/Breakthrough_keepset\"#g" \
  "$SD/r57_screen_s1.py"
RUN="/bin/bash data/r57/snap/c45_runner_s1.sh"
export C45_OUT="$PWD/data/s1" C45_SWAP0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}') C45_ALARM=1200 \
  MG=4 C45_DEADLINE=$(date -v+45M +%H%M) C45_SCREEN="$SD/r57_screen_s1.py" C45_REFTOK=
K="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79"
K="$K,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,humaneval/HumanEval/56,humaneval/HumanEval/141"
K="$K,humaneval/HumanEval/101,humaneval/HumanEval/63"
QAC="BIGMOE_EXT_BUILD=build_qom,R57_REFTOK=,R57_FAST=QOM,R57_D17=1,R57_D7B=code,R57_HC8=1,REF2_HC8_FAST=1,R57_DNR=1,R57_HCB=1"
QAC="$QAC,REF2_SMALL_NATIVE=1,R57_D7BNAX=1,REF2_OOO_1C=1,R57_QCNT1C=1"
ST="R57_NAX=1,REF2_NAX_1CQ=1,REF2_DN_KS=1,REF2_GDN_LAZY=1,D7BNAX_32=2,R57_FASTTAX=1,R57_SBN=2,R57_DSUB=1,R57_SVH=1,REF2_KEEP_CACHES=1"   # the stack + S2 + KEEP (v1 candidate since 01:4x)
case "${S1_BASE:-stackb_s2}" in
  stackb_s2) B="$QAC,$ST,R57_RELAX_SPAN=1,R57_RELAX_SPAN_RULE=R2:2:0.25,R57_COPY=g2,R57_RELAX_LOG=$PWD/data/s1/b_relax.jsonl" ;;
  stack_s2)  B="$QAC,$ST" ;;
  *) say "S1_BASE=${S1_BASE}: unknown"; exit 2 ;;
esac
NW="$B,R57_NEVERWAIT=1,REF2_NEVERWAIT=1"
for t in s1pf s1b1 s1n1 s1n2 s1b2; do rm -f data/r57/${t}_${CC}_* data/s1/${t}_${CC}.txt; done
say "S1 A/B start (pid $$, NO-SWAP $A GB, C $CC, base ${S1_BASE:-stackb_s2}); ref2 md5 $(md5 -q "$HOME/Leap/scripts/ref2.py")"
R57_KEYS="humaneval/HumanEval/68,humaneval/HumanEval/16" C45_CAP=32 C45_ARMS="s1pf_$CC:$CC:$NW" $RUN
n=$(grep -o '"key"' "data/r57/s1pf_${CC}_runs.json" 2>/dev/null | wc -l | tr -d ' ')
bad=$(grep -cE 'Traceback|AssertionError|Error:' "data/s1/s1pf_${CC}.txt" 2>/dev/null)
[ "${n:-0}" = 2 ] && [ "${bad:-1}" = 0 ] || { say "NW check NOT CLEAN (ran ${n:-0} / 2, error lines ${bad:-none}): no A/B"; exit 0; }
say "NW check CLEAN (2 / 2): the A/B runs"
for a in "s1b1:$B" "s1n1:$NW" "s1n2:$NW" "s1b2:$B"; do
  R57_KEYS=$K C45_ARMS="${a%%:*}_$CC:$CC:${a#*:}" $RUN
  say "${a%%:*} rc=$?"
done
"$PY" -I - "$CC" "$(date +%F)" "${S1_FANLOG:-$HOME/Library/Logs/fanwatch.log}" >> "$LOG" 2>&1 <<'EOF'
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
for ln in open("data/s1/c45.log"):
    m = re.match(r"(\d\d:\d\d:\d\d) (run|done) (\S+)", ln)
    if m:
        arm_t.setdefault(m.group(3), {})[m.group(2)] = D.datetime.fromisoformat(f"{day} {m.group(1)}")
    m = re.search(r"done (\S+) rc=\d+:.*lifetime peak ([\d.]+) GB", ln)
    if m:
        peak[m.group(1)] = float(m.group(2))
hit = {t for t, v in arm_t.items() if "run" in v and "done" in v and any(a < v["done"] and b > v["run"] for a, b in win)}
P = {"B": ("s1b1", "s1b2"), "NW": ("s1n1", "s1n2")}
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
L = [f"S1 A/B (NW v3), C {C} (data/s1): B / NW / NW / B; freeze windows {[(a.strftime('%H:%M:%S'), b.strftime('%H:%M:%S')) for a, b in win]}; "
     f"passes dropped for a freeze: {sorted(hit) or 'none'}"]
for b, ps in P.items():
    tp = pm(ps, lambda r: r["tpsw"])
    L.append(f"{b:3s} geo steady {g(list(tp.values())):6.1f} tok/s over {len(tp)}; passes " +
             ", ".join(f"{t} {g([r['tpsw'] for k, r in runs[t].items() if ok(t, k)]):.1f}" for t in ps) +
             f"; kept per check {g(list(pm(ps, kept).values())):.2f}; tokens per task "
             f"{S.median(pm(ps, lambda r: r['n']).values()) if pm(ps, lambda r: r['n']) else float('nan'):.0f}; median cycle "
             f"{S.median([r['cyc'] for t in ps for k, r in runs[t].items() if ok(t, k) and isinstance(r.get('cyc'), (int, float))] or [float('nan')]):.1f} ms; "
             f"lifetime peak {max([peak.get(f'{t}_{C}', 0.0) for t in ps]) or float('nan'):.1f} GB")
r_, n_ = ratio(pm(P["NW"], lambda r: r["tpsw"]), pm(P["B"], lambda r: r["tpsw"]))
pkn = max([peak.get(f"{t}_{C}", 0.0) for t in P["NW"]]) or float("nan")
v = "MISSING" if n_ == 0 else "KILL" if r_ < 1.03 or pkn > 45.0 else "PASS" if r_ >= 1.08 else "BETWEEN"
L.append(f"Verdict S1 NW v3 (pre-registered, NewIdeas QUEUE S1 + Sat 01:5x peak clause): {v}: {100 * (r_ - 1):+.1f} % vs B over {n_}, "
         f"NW lifetime peak {pkn:.1f} GB (PASS >= +8 % AND peak <= 45.0 GB; KILL < +3 % or peak > 45.0 GB)")
def pcm(ts, i):
    xs = []
    for t in ts:
        try:
            for e in json.load(open(f"data/r57/{t}_{C}_checks.json")):
                if not e.get("paused"):
                    xs += [x[i] for x in e.get("nw", [])[2:] if len(x) > i]
        except (OSError, ValueError):
            pass
    return S.median(xs) if xs else float("nan")
names = ("late groups", "late uses", "late mass", "L47 drops", "L47 stalls", "forced", "optional dropped now", "sidecars 3-bit")
L.append("NW v3 counters per check (medians, checks >= 2): " + ", ".join(f"{n} {pcm(P['NW'], i):.1f}" for i, n in enumerate(names)))
same = lambda a, b: sum(1 for k in toks[a] if k in toks[b] and toks[a][k] == toks[b][k])
L.append(f"tokens identical per prompt (information): NW vs B {[same('s1b1', 's1n1'), same('s1b2', 's1n2')]} of 12; floor B1 vs B2 "
         f"{same('s1b1', 's1b2')}")
open("data/s1/s1_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? (data/s1/s1_report.txt)"
say "S1 A/B end"
