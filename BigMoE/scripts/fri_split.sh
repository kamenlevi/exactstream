#!/bin/bash
# Fri 6 Nov early morning (the orchestrator 01:4x, decision (b)): the split engine A/B again, on the path that ships: the
# all-in's one-call OOO (REF2_OOO_1C=1), with Leap's switch that drops a strict row's optional uses before the post (rank-0
# id, weight 0; off by default). Fri 00:52's s99 / s90 arms never ran the split: the one-call path bypassed qom_keep_mx.
# Arms on the frozen screen, 12 HumanEval, one C by NO-SWAP (as fri_night.sh), a palindrome with three qall passes so the
# noise floor reads from qall-vs-qall pairs:
#   qall / s99 / s90 / qallm / s90b / s99b / qallb
# FS_SPLIT99 / FS_SPLIT90 = Leap's switch for T 0.99 / 0.9 (required: without them nothing runs).
# Freezes: arm passes that overlap a fanwatch freeze window (~/Library/Logs/fanwatch.log 'froze' -> 'cool') are dropped
# from the pair means and listed (Fri 00:52's freeze was not flagged PAUSED).
# Line (pre-registered Thu 10:40, unchanged): split PASS = pair-mean geo steady tok/s >= +8 % vs qall AND kept per check
# within 1 %; else NOT PASS. The C caveat as fri_night.sh (< 160: indicative only). Must END by 10:15 (q45 / runner 45).
# usage: FS_SPLIT99=... FS_SPLIT90=... NOTBEFORE=hh:mm NOTAFTER=09:30 NOWAIT45=1 GAP0=0 GAP=0 MARGIN=4 bash scripts/turn45.sh \
#   28 45 "BigMoE: split A/B on the one-call path ..." /bin/bash scripts/fri_split.sh  -> data/fsp/fsp.log, fsp_report.txt
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "${FS_ROOT:-$HOME/BigMoE}" || exit 1
mkdir -p data/fsp data/r57/snap
LOG="$PWD/data/fsp/fsp.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
FS_SPLIT99=${FS_SPLIT99-R57_QOM_SPLIT=0.99}; FS_SPLIT90=${FS_SPLIT90-R57_QOM_SPLIT=0.9}   # Leap's ref2 eb43fcc2: the switch acts on
[ -n "$FS_SPLIT99" ] && [ -n "$FS_SPLIT90" ] || { say "REFUSED: FS_SPLIT99 / FS_SPLIT90 empty"; exit 2; }   # the one-call path
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
A=${FS_AVAIL:-$(avail_gb)}
fits() { awk -v a="$A" -v n="$1" 'BEGIN {exit !(n <= a)}'; }
CC=; for c in 224:43.7 208:41.6 192:39.5 176:37.4 160:35.2 144:33.1 128:31.0; do fits ${c#*:} && { CC=${c%%:*}; break; }; done   # qall peak + 4
[ -z "$CC" ] && { say "NO-SWAP $A GB < 31.0: nothing fits, nothing runs"; exit 0; }
cp scripts/r57_screen.py data/r57/snap/r57_screen_fsp.py && cp scripts/c45_runner.sh data/r57/snap/c45_runner_fsp.sh || exit 2
RUN="/bin/bash data/r57/snap/c45_runner_fsp.sh"
export C45_OUT="$PWD/data/fsp" C45_SWAP0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}') C45_ALARM=900 \
  MG=${FS_MG:-4} C45_DEADLINE=${FS_DEADLINE:-$(date -v+35M +%H%M)} C45_SCREEN=data/r57/snap/r57_screen_fsp.py C45_REFTOK=
K="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79"
K="$K,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,humaneval/HumanEval/56,humaneval/HumanEval/141"
K="$K,humaneval/HumanEval/101,humaneval/HumanEval/63"
QA="BIGMOE_EXT_BUILD=build_qom,R57_REFTOK=,R57_FAST=QOM,R57_D17=1,R57_D7B=code,R57_HC8=1,REF2_HC8_FAST=1,R57_DNR=1,R57_HCB=1"
QA="$QA,REF2_SMALL_NATIVE=1,R57_D7BNAX=1,REF2_OOO_1C=1,R57_QCNT1C=1"   # the one-call counters in EVERY arm (equal cost)
say "split A/B start (pid $$, NO-SWAP $A GB -> C $CC, deadline $C45_DEADLINE, switches '$FS_SPLIT99' / '$FS_SPLIT90')"
for a in "sqall:$QA" "ss99:$QA,$FS_SPLIT99" "ss90:$QA,$FS_SPLIT90" "sqallm:$QA" "ss90b:$QA,$FS_SPLIT90" "ss99b:$QA,$FS_SPLIT99" "sqallb:$QA"; do
  R57_KEYS=$K C45_ARMS="${a%%:*}_$CC:$CC:${a#*:}" $RUN
  say "${a%%:*} rc=$?"
done
"${FS_PY:-$HOME/BigMoE/.venv/bin/python}" -I - "$CC" "$(date +%F)" "${FS_FANLOG:-$HOME/Library/Logs/fanwatch.log}" >> "$LOG" 2>&1 <<'EOF'
import datetime as D, json, math, re, statistics as S, sys
C, day, fanlog = sys.argv[1], sys.argv[2], sys.argv[3]
g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")
ts = lambda hms: D.datetime.fromisoformat(f"{day} {hms}")
win = []                                                     # fanwatch freeze windows today: 'froze' -> the next 'cool'
try:
    t0 = None
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
arm_t = {}                                                   # c45.log: 'HH:MM:SS run tag' / 'HH:MM:SS done tag'
for ln in open("data/fsp/c45.log"):
    m = re.match(r"(\d\d:\d\d:\d\d) (run|done) (\S+)", ln)
    if m:
        arm_t.setdefault(m.group(3), {})[m.group(2)] = ts(m.group(1))
hit = {t for t, v in arm_t.items() if "run" in v and "done" in v and any(a < v["done"] and b > v["run"] for a, b in win)}
P = {"qall": ("sqall", "sqallm", "sqallb"), "s99": ("ss99", "ss99b"), "s90": ("ss90", "ss90b")}
runs, chk = {}, {}
for ts_ in sum(P.values(), ()):
    try:
        runs[ts_] = {r["key"]: r for r in json.load(open(f"data/r57/{ts_}_{C}_runs.json"))}
    except OSError:
        runs[ts_] = {}
    try:
        chk[ts_] = json.load(open(f"data/r57/{ts_}_{C}_checks.json"))
    except OSError:
        chk[ts_] = []
def cnt(b, i, key="q1c"):   # per-check mean of counter i over the arm's clean passes (first check excluded)
    xs = [c[i] for t in P[b] if f"{t}_{C}" not in hit for e in chk[t] if not e.get("paused") for c in e.get(key, [])[1:]]
    return S.mean(xs) if xs else float("nan")
ok = lambda t, k: k in runs[t] and "PAUSED" not in runs[t][k].get("bad", "") and f"{t}_{C}" not in hit
kept = lambda r: r["kept"] if isinstance(r["kept"], (int, float)) else S.mean(r["kept"])
def pm(passes, f):   # per prompt: the mean over the passes that ran clean
    out = {}
    for k in set().union(*[runs[t] for t in passes]):
        v = [f(runs[t][k]) for t in passes if ok(t, k)]
        if v:
            out[k] = sum(v) / len(v)
    return out
def ratio(a, b):
    ks = [k for k in a if k in b]
    return (g([a[k] / b[k] for k in ks]), len(ks)) if ks else (float("nan"), 0)
L = [f"SPLIT A/B on the one-call path, C {C}, 12 HumanEval (data/fsp): qall / s99 / s90 / qallm / s90b / s99b / qallb; "
     f"freeze windows {[(a.strftime('%H:%M:%S'), b.strftime('%H:%M:%S')) for a, b in win]}; passes dropped for a freeze: "
     f"{sorted(hit) or 'none'}"]
for b, ps in P.items():
    tp = pm(ps, lambda r: r["tpsw"])
    L.append(f"{b:5s} geo steady {g(list(tp.values())):6.1f} tok/s over {len(tp)}; passes " +
             ", ".join(f"{t} {g([r['tpsw'] for k, r in runs[t].items() if ok(t, k)]):.1f}" for t in ps) +
             f"; kept per check {g(list(pm(ps, kept).values())):.2f}; groups computed per check {cnt(b, 2):.0f} (one-call, "
             f"post-drop); split drops per check {cnt(b, 3):.0f}; strict rows per check {cnt(b, 3, 'qc'):.1f}")
for b in ("s99", "s90"):
    L.append(f"split {b} engaged: groups computed {cnt(b, 2) / cnt('qall', 2):.3f}x qall's (Breakthrough's replay x0.72 / x0.63 at "
             f"C 160 full blocks); split drops per check {cnt(b, 3):.0f}")
if C == "192":
    L.append(f"C3-R8: groups computed per check, qall {cnt('qall', 2):.0f} vs exact 3,708 (Fri 00:52's fexr at C 192, 'uses' = "
             f"computed for exact) = {cnt('qall', 2) / 3708:.3f}x (audit: >= 0.85x)")
q13 = pm(("sqall", "sqallb"), lambda r: r["tpsw"]); qm = pm(("sqallm",), lambda r: r["tpsw"])
r_, n_ = ratio(qm, q13)
L.append(f"noise floor: qall middle pass vs its end passes {100 * (r_ - 1):+.1f} % over {n_} (the split arms' pair means "
         f"are centred on the middle pass)")
qa, qk = pm(P["qall"], lambda r: r["tpsw"]), pm(P["qall"], kept)
IND = f" INDICATIVE ONLY at C {C} < 160" if int(C) < 160 else ""
for b in ("s99", "s90"):
    r_, n_ = ratio(pm(P[b], lambda r: r["tpsw"]), qa); k_, _ = ratio(pm(P[b], kept), qk)
    v = "MISSING" if n_ == 0 else "PASS" if r_ >= 1.08 and abs(k_ - 1) <= 0.01 else "NOT PASS"
    L.append(f"Verdict split {b} (pre-registered Thu 10:40){IND}: {v}: tok/s {100 * (r_ - 1):+.1f} %, kept {100 * (k_ - 1):+.1f} % "
             f"vs qall over {n_} (PASS = >= +8 % AND kept within 1 %)")
open("data/fsp/fsp_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? (data/fsp/fsp_report.txt)"
say "split A/B end"
