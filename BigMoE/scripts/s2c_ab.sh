#!/bin/bash
# Sat 7 Nov 01:3x (NewIdeas: S2c PASSED in Leap's loop): REF2_KEEP_CACHES=1 (Leap, live ref2 ddbc0621, default off; self_test
# PASS): ref2.remove() before every draft no longer drops REF2's per-weight caches (_SN SMALL-N padded copies, _DN, _HC8 views), so
# checks stop re-padding / re-packing them. Identity-validated caches: same arrays, same bits by construction. One quiet ticket:
#   0. EXACT bitwise proof (NewIdeas): the NAX6-N exact env (R57_PRESET=45 defaults + R57_NAX=1, no R57_FAST), C 224, 3 prompts
#      (HumanEval/68, HumanEval/16, ifeval/1220), CAP 64, keep off (ex0) vs keep on (ex1). Line: tokens equal on all 3 = PROOF
#      PASS; any difference = PROOF FAIL (KEEP is not bitwise; it does not enter v1).
#   1. The engine A/B at C 256 (needs NO-SWAP >= 47.2 GB): stackb + S2 (B) vs B + REF2_KEEP_CACHES=1 (K), the 12 HumanEval,
#      palindrome b1 / k1 / k2 / b2. Line (NewIdeas Sat 01:3x, pre-registered before arming): pair-mean geo steady tok/s, K vs B:
#      PASS >= +4 % AND lifetime peak <= 45.0 GB; KILL < +1.5 % or peak > 45.0 GB; else BETWEEN. Median cycle ms beside.
# usage: NOTAFTER=hh:mm WAITMEM=48 NOWAIT45=1 GAP0=0 GAP=0 MARGIN=4 bash scripts/turn45.sh 33 20 "..." /bin/bash scripts/s2c_ab.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "${SC_ROOT:-$HOME/BigMoE}" || exit 1
mkdir -p data/s2c data/r57/snap
LOG="$PWD/data/s2c/s2c.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
PY=${SC_PY:-$HOME/BigMoE/.venv/bin/python}
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
A=${SC_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2: C 256 does not fit, not run"; exit 0; }
FZ="$PWD/data/r57/snap/frozen_s2c"; rm -rf "$FZ"; mkdir -p "$FZ"
SD=data/r57/snap/s2c; mkdir -p "$SD"     # own dir: never touch data/r57/snap/*.py while EvalSuite's runs load them
cp scripts/r57_screen_fast.py "$SD/r57_screen_s2c.py" && cp scripts/relax_delta.py scripts/relax_span.py scripts/inert_span.py \
  "$SD/" && cp scripts/c45_runner.sh data/r57/snap/c45_runner_s2c.sh || exit 2
[ -n "$SC_DRYRUN" ] || { cp -R "$HOME/Leap/scripts" "$FZ/Leap_scripts" && cp -R "$HOME/Leap/third_party" "$FZ/Leap_third_party" \
  && cp -R "$HOME/TrainLab/sticky" "$FZ/TrainLab_sticky" && cp -R "$HOME/Breakthrough/keepset" "$FZ/Breakthrough_keepset" || exit 2; }
find "$FZ" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
sed -i '' -e "s#\"~/Leap/scripts\"#\"$FZ/Leap_scripts\"#g" -e "s#\"~/Leap/third_party\"#\"$FZ/Leap_third_party\"#g" \
  -e "s#\"~/TrainLab/sticky\"#\"$FZ/TrainLab_sticky\"#g" -e "s#\"/Users/kamen/Breakthrough/keepset\"#\"$FZ/Breakthrough_keepset\"#g" \
  "$SD/r57_screen_s2c.py"
RUN="/bin/bash data/r57/snap/c45_runner_s2c.sh"
export C45_OUT="$PWD/data/s2c" C45_SWAP0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}') C45_ALARM=1200 \
  MG=4 C45_DEADLINE=$(date -v+45M +%H%M) C45_SCREEN="$SD/r57_screen_s2c.py" C45_REFTOK=
say "S2c start (pid $$, NO-SWAP $A GB); ref2 md5 $(md5 -q "$HOME/Leap/scripts/ref2.py"); keep_caches in ref2: $(grep -c REF2_KEEP_CACHES "$HOME/Leap/scripts/ref2.py")"
for t in sce0 sce1; do rm -f data/r57/${t}_224_* data/s2c/${t}_224.txt; done
for t in scb1 sck1 sck2 scb2; do rm -f data/r57/${t}_256_* data/s2c/${t}_256.txt; done
# 0. the exact bitwise proof (NAX6-N exact env, keep off / on)
EX="R57_NAX=1,R57_REFTOK="
for a in "sce0:$EX" "sce1:$EX,REF2_KEEP_CACHES=1"; do
  R57_KEYS="humaneval/HumanEval/68,humaneval/HumanEval/16,ifeval/1220" C45_CAP=64 C45_ARMS="${a%%:*}_224:224:${a#*:}" $RUN
  say "${a%%:*} rc=$?"
done
# 1. the engine A/B at C 256
K="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79"
K="$K,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,humaneval/HumanEval/56,humaneval/HumanEval/141"
K="$K,humaneval/HumanEval/101,humaneval/HumanEval/63"
QAC="BIGMOE_EXT_BUILD=build_qom,R57_REFTOK=,R57_FAST=QOM,R57_D17=1,R57_D7B=code,R57_HC8=1,REF2_HC8_FAST=1,R57_DNR=1,R57_HCB=1"
QAC="$QAC,REF2_SMALL_NATIVE=1,R57_D7BNAX=1,REF2_OOO_1C=1,R57_QCNT1C=1"
ST="R57_NAX=1,REF2_NAX_1CQ=1,REF2_DN_KS=1,REF2_GDN_LAZY=1,D7BNAX_32=2,R57_FASTTAX=1,R57_SBN=2,R57_DSUB=1,R57_SVH=1"   # the stack + S2
B="$QAC,$ST,R57_RELAX_SPAN=1,R57_RELAX_SPAN_RULE=R2:2:0.25,R57_COPY=g2,R57_RELAX_LOG=$PWD/data/s2c/b_relax.jsonl"   # stackb + S2
KK="$B,REF2_KEEP_CACHES=1"
for a in "scb1:$B" "sck1:$KK" "sck2:$KK" "scb2:$B"; do
  R57_KEYS=$K C45_ARMS="${a%%:*}_256:256:${a#*:}" $RUN
  say "${a%%:*} rc=$?"
done
"$PY" -I - "$(date +%F)" "${SC_FANLOG:-$HOME/Library/Logs/fanwatch.log}" >> "$LOG" 2>&1 <<'EOF'
import datetime as D, json, math, re, statistics as S, sys
day, fanlog = sys.argv[1], sys.argv[2]
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
for ln in open("data/s2c/c45.log"):
    m = re.match(r"(\d\d:\d\d:\d\d) (run|done) (\S+)", ln)
    if m:
        arm_t.setdefault(m.group(3), {})[m.group(2)] = D.datetime.fromisoformat(f"{day} {m.group(1)}")
    m = re.search(r"done (\S+) rc=\d+:.*lifetime peak ([\d.]+) GB", ln)
    if m:
        peak[m.group(1)] = float(m.group(2))
hit = {t for t, v in arm_t.items() if "run" in v and "done" in v and any(a < v["done"] and b > v["run"] for a, b in win)}
def load(t, c):
    try:
        r = {x["key"]: x for x in json.load(open(f"data/r57/{t}_{c}_runs.json"))}
        tk = {k: list(v.values())[0][0] for k, v in json.load(open(f"data/r57/{t}_{c}_tokens.json")).items()}
        return r, tk
    except (OSError, ValueError):
        return {}, {}
L = [f"S2c (REF2_KEEP_CACHES=1), data/s2c; freeze windows {[(a.strftime('%H:%M:%S'), b.strftime('%H:%M:%S')) for a, b in win]}; "
     f"passes dropped for a freeze: {sorted(hit) or 'none'}"]
(r0, t0_), (r1, t1_) = load("sce0", 224), load("sce1", 224)
ks = sorted(set(t0_) & set(t1_))
eq = [k for k in ks if t0_[k] == t1_[k]]
pv = "PROOF PASS" if len(ks) == 3 and len(eq) == 3 else "PROOF FAIL" if ks else "PROOF MISSING"
L.append(f"EXACT bitwise proof (NAX6-N exact env, C 224, CAP 64, keep off vs on): {pv}: tokens equal on {len(eq)} of {len(ks)} "
         f"({', '.join(f'{k.split(chr(47))[-1]} {len(t0_[k])} tok' for k in ks)}); peaks {peak.get('sce0_224')} / {peak.get('sce1_224')} GB")
runs = {t: load(t, 256)[0] for t in ("scb1", "sck1", "sck2", "scb2")}
ok = lambda t, k: k in runs[t] and "PAUSED" not in runs[t][k].get("bad", "") and f"{t}_256" not in hit
kept = lambda r: r["kept"] if isinstance(r["kept"], (int, float)) else S.mean(r["kept"])
def pm(ps, f):
    out = {}
    for k in set().union(*[runs[t] for t in ps]):
        v = [f(runs[t][k]) for t in ps if ok(t, k)]
        if v:
            out[k] = sum(v) / len(v)
    return out
def ratio(a, b):
    kk = [k for k in a if k in b]
    return (g([a[k] / b[k] for k in kk]), len(kk)) if kk else (float("nan"), 0)
P = {"B (stackb+S2)": ("scb1", "scb2"), "K (+KEEP)": ("sck1", "sck2")}
pk = {}
for b, ps in P.items():
    tp = pm(ps, lambda r: r["tpsw"])
    pk[b] = max([peak.get(f"{t}_256", 0.0) for t in ps]) or float("nan")
    L.append(f"{b:14s} geo steady {g(list(tp.values())):6.1f} tok/s over {len(tp)}; passes " +
             ", ".join(f"{t} {g([r['tpsw'] for k, r in runs[t].items() if ok(t, k)]):.1f}" for t in ps) +
             f"; kept per check {g(list(pm(ps, kept).values())):.2f}; median cycle "
             f"{S.median([r['cyc'] for t in ps for k, r in runs[t].items() if ok(t, k) and isinstance(r.get('cyc'), (int, float))] or [float('nan')]):.1f} ms; "
             f"lifetime peak {pk[b]:.1f} GB")
r_, n_ = ratio(pm(P["K (+KEEP)"], lambda r: r["tpsw"]), pm(P["B (stackb+S2)"], lambda r: r["tpsw"]))
v = ("MISSING" if n_ == 0 else "KILL" if r_ < 1.015 or pk["K (+KEEP)"] > 45.0 else "PASS" if r_ >= 1.04 else "BETWEEN")
L.append(f"Verdict S2c KEEP (pre-registered NewIdeas Sat 01:3x): {v}: {100 * (r_ - 1):+.1f} % vs B over {n_}, lifetime peak "
         f"{pk['K (+KEEP)']:.1f} GB (PASS >= +4 % AND <= 45.0 GB; KILL < +1.5 % or > 45.0 GB)")
L.append(f"v1 rule (NewIdeas): KEEP joins v1 only with {pv} AND the speed PASS -> {'JOINS' if pv == 'PROOF PASS' and v == 'PASS' else 'does not join'}")
open("data/s2c/s2c_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? (data/s2c/s2c_report.txt)"
say "S2c end"
