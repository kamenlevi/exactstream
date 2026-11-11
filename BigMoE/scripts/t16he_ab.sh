#!/bin/bash
# Sat 7 Nov 14:xx (MTP Head 14:1x; NewIdeas PLAN 220 item 6; pre-registered in ~/TrainLab/REPORT.md 'T16-HE', bccd23c):
# T16-HE, T16's HumanEval engine A/B on v1' + wired. Runs on S5's committed snapshot (data/r57/snap/s5/r57_screen_s5.py +
# frozen_s5, no live outside code) and only after a CLEAN S5. C 256, S1's 12 HumanEval, one process per arm, B / T / T / B.
#   B = v1' + wired (final_env dd7ad0f + R57_WIRED=1; R57_QOM_SETS=humaneval,mbpp in the environment, as S5)
#   T = B - R57_RELAX_SPAN=1 - R57_RELAX_SPAN_RULE=R2:2:0.25 (the screen's INERT / SPAN assert; span B made 0 relaxed accepts on
#       these 12) + R57_INERT=T16 + R57_INERT_LOG=data/t16he/<pass>_inert.jsonl
# LINE (MTP Head's, verbatim in substance), pair-mean geo WHOLE DECODE, T vs B:
#   PASS >= +3 % AND T lifetime peak <= 45.0 GB AND median tokens per task T / B <= 1.05 AND no T prompt reaches the cap where
#   both B passes ended by themselves. KILL < +1 %, or peak > 45.0 GB, or such a runaway. Else BETWEEN (no gate by Tuesday).
#   Beside: median cycle, full period, check 0, kept per check, tokens per task, T16 rule-kept rows (runs' inert counters + the
#   INERT log), tokens identical T vs B next to the B1-vs-B2 floor.
# PASS -> data/stack/t16_env.txt (final_env's format, the S5 snapshot's screen) committed, for EvalSuite's S4-style gate vs NAX6-N.
# usage: AFTER='[ -f data/s5/CLEAN ]' NOTAFTER=hh:mm NOWAIT45=1 GAP0=0 GAP=0 MARGIN=4 bash scripts/turn45.sh 33 20 "..." /bin/bash scripts/t16he_ab.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "${T6_ROOT:-$HOME/BigMoE}" || exit 1
mkdir -p data/t16he
LOG="$PWD/data/t16he/t16he.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
PY=${T6_PY:-$HOME/BigMoE/.venv/bin/python}
SD=data/r57/snap/s5
[ -f data/s5/CLEAN ] || { say "S5 is not CLEAN (data/s5/CLEAN missing): not run"; exit 0; }
git diff --quiet HEAD -- "$SD" data/r57/snap/frozen_s5 data/r57/snap/c45_runner_s5.sh 2>/dev/null && \
  [ -z "$(git ls-files --others --exclude-standard -- "$SD" data/r57/snap/frozen_s5 | grep -v __pycache__ | head -1)" ] || {
  say "the S5 snapshot differs from its commit: not run"; exit 2; }
SNAP=$(git log -1 --format=%h -- "$SD/r57_screen_s5.py")
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
A=${T6_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2: C 256 does not fit, not run"; exit 0; }
[ -n "$T6_DRYRUN" ] || sleep 20
A=${T6_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2 at the 20 s look-again: not run"; exit 0; }
CC=256
RUN="/bin/bash data/r57/snap/c45_runner_s5.sh"
export C45_OUT="$PWD/data/t16he" C45_SWAP0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}') C45_ALARM=1200 \
  MG=4 C45_DEADLINE=$(date -v+45M +%H%M) C45_SCREEN="$SD/r57_screen_s5.py" C45_REFTOK= R57_QOM_SETS=humaneval,mbpp R57_MODMAN=1
K="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79"
K="$K,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,humaneval/HumanEval/56,humaneval/HumanEval/141"
K="$K,humaneval/HumanEval/101,humaneval/HumanEval/63"
B=$(grep -v '^#' data/stack/final_env.txt | grep . | tail -1 | sed 's/,R57_QOM_SETS=humaneval,mbpp$//')
echo "$B" | grep -q 'NEVERWAIT\|R57_QOM_SETS' && { say "final_env.txt is not v1': not run"; exit 2; }
B="$B,R57_WIRED=1"
echo "$B" | tr ',' '\n' | grep -qx 'R57_RELAX_SPAN=1' && echo "$B" | tr ',' '\n' | grep -qx 'R57_RELAX_SPAN_RULE=R2:2:0.25' || {
  say "B lacks span B's two keys: not run"; exit 2; }
T0=$(echo "$B" | tr ',' '\n' | grep -vx 'R57_RELAX_SPAN=1' | grep -vx 'R57_RELAX_SPAN_RULE=R2:2:0.25' | paste -sd, -),R57_INERT=T16
[ -n "$T6_DRYRUN" ] && { echo "dry run: snapshot $SNAP; B $(echo "$B" | tr ',' '\n' | grep -c =) K=V, T $(echo "$T0" | tr ',' '\n' | grep -c =) K=V (+ INERT_LOG)"; exit 0; }
for t in t6b1 t6t1 t6t2 t6b2; do rm -f data/r57/${t}_${CC}_* data/t16he/${t}_${CC}.txt data/t16he/${t}_inert.jsonl; done
rm -f data/t16he/DONE
say "T16-HE A/B start (pid $$, NO-SWAP $A GB, C $CC, snapshot $SNAP)"
for a in "t6b1:$B" "t6t1:$T0,R57_INERT_LOG=$PWD/data/t16he/t6t1_inert.jsonl" "t6t2:$T0,R57_INERT_LOG=$PWD/data/t16he/t6t2_inert.jsonl" "t6b2:$B"; do
  R57_KEYS=$K C45_ARMS="${a%%:*}_$CC:$CC:${a#*:}" $RUN
  say "${a%%:*} rc=$?"
done
"$PY" -I - "$CC" "$(date +%F)" "${T6_FANLOG:-$HOME/Library/Logs/fanwatch.log}" >> "$LOG" 2>&1 <<'EOF'
import datetime as D, json, math, re, statistics as S, sys
C, day, fanlog = sys.argv[1], sys.argv[2], sys.argv[3]
CAP = 256
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
for ln in open("data/t16he/c45.log"):
    m = re.match(r"(\d\d:\d\d:\d\d) (run|done) (\S+)", ln)
    if m:
        arm_t.setdefault(m.group(3), {})[m.group(2)] = D.datetime.fromisoformat(f"{day} {m.group(1)}")
    m = re.search(r"done (\S+) rc=\d+:.*lifetime peak ([\d.]+) GB", ln)
    if m:
        peak[m.group(1)] = float(m.group(2))
hit = {t for t, v in arm_t.items() if "run" in v and "done" in v and any(a < v["done"] and b > v["run"] for a, b in win)}
P = {"B": ("t6b1", "t6b2"), "T": ("t6t1", "t6t2")}
runs, toks, live = {}, {}, 0
for t in sum(P.values(), ()):
    try:
        R = {r["key"]: r for r in json.load(open(f"data/r57/{t}_{C}_runs.json"))}
        for c in json.load(open(f"data/r57/{t}_{C}_checks.json")):
            r = R.get(c["key"])
            if r is None:
                continue
            dec, c0 = r["n"] / r["tps"], c.get("c0") or []
            r["fp"] = sum(c["kept"][2:]) / (dec - c0[2] / 1e3) if len(c0) > 2 and dec > c0[2] / 1e3 else None
            r["ck0"] = c["bld"][0] + c["wt"][0] if c["bld"] else None
            r["kpc"] = S.mean(c["kept"][2:]) if len(c["kept"]) > 2 else None
        runs[t] = R
        toks[t] = {k: list(v.values())[0][0] for k, v in json.load(open(f"data/r57/{t}_{C}_tokens.json")).items()}
        live += sum(1 for _, f in json.load(open(f"data/r57/{t}_{C}_modules.json")) if "/BigMoE/data/r57/snap/" not in f)
    except (OSError, ValueError, KeyError) as e:
        print(f"{t}: unreadable ({e})"); runs[t], toks[t] = {}, {}
ok = lambda t, k: k in runs[t] and "PAUSED" not in runs[t][k].get("bad", "") and f"{t}_{C}" not in hit
def pm(ps, f):
    out = {}
    for k in set().union(*[runs[t] for t in ps]):
        v = [f(runs[t][k]) for t in ps if ok(t, k) and f(runs[t][k]) is not None]
        if v:
            out[k] = sum(v) / len(v)
    return out
def ratio(a, b):
    ks = [k for k in a if k in b]
    return (g([a[k] / b[k] for k in ks]), len(ks)) if ks else (float("nan"), 0)
L = [f"T16-HE A/B (T16 on HumanEval, v1' + wired), C {C} (data/t16he): B / T / T / B; freeze windows "
     f"{[(a.strftime('%H:%M:%S'), b.strftime('%H:%M:%S')) for a, b in win]}; passes dropped for a freeze: {sorted(hit) or 'none'}; "
     f"live outside modules: {live}"]
for b, ps in P.items():
    pas = lambda f: ", ".join(f"{t} {g([f(r) for k, r in runs[t].items() if ok(t, k) and f(r)]):.1f}" for t in ps)
    L.append(f"{b}: WHOLE DECODE {g(list(pm(ps, lambda r: r['tps']).values())):6.1f} tok/s ({pas(lambda r: r['tps'])}); full period "
             f"{g(list(pm(ps, lambda r: r.get('fp')).values())):6.1f} ({pas(lambda r: r.get('fp'))})")
    L.append(f"    check 0 {S.median(pm(ps, lambda r: r.get('ck0')).values() or [float('nan')]):.0f} ms; median cycle "
             f"{S.median([r['cyc'] for t in ps for k, r in runs[t].items() if ok(t, k) and isinstance(r.get('cyc'), (int, float))] or [float('nan')]):.1f} ms; "
             f"kept per check {g(list(pm(ps, lambda r: r.get('kpc')).values())):.2f}; tokens per task "
             f"{S.median(pm(ps, lambda r: r['n']).values() or [float('nan')]):.0f}; lifetime peak "
             f"{max([peak.get(f'{t}_{C}', 0.0) for t in ps]) or float('nan'):.1f} GB")
rw, n_ = ratio(pm(P["T"], lambda r: r["tps"]), pm(P["B"], lambda r: r["tps"]))
rf, _ = ratio(pm(P["T"], lambda r: r.get("fp")), pm(P["B"], lambda r: r.get("fp")))
tpk = max([peak.get(f"{t}_{C}", 0.0) for t in P["T"]]) or float("nan")
nb, nt = pm(P["B"], lambda r: r["n"]), pm(P["T"], lambda r: r["n"])
nr = (S.median(nt.values()) / S.median(nb.values())) if nb and nt else float("nan")
run_ = sorted(k for k in nt if all(runs[t].get(k, {}).get("n", CAP) < CAP for t in P["B"])
              and any(runs[t].get(k, {}).get("n", 0) >= CAP for t in P["T"]))
ruled = sum((r.get("inert") or {}).get("ruled", 0) for t in P["T"] for r in runs[t].values())
opened = sum((r.get("inert") or {}).get("open", 0) for t in P["T"] for r in runs[t].values())
logn = 0
for t in P["T"]:
    try:
        logn += sum(1 for _ in open(f"data/t16he/{t}_inert.jsonl"))
    except OSError:
        pass
v = ("MISSING" if n_ == 0 else "KILL" if rw < 1.01 or tpk > 45.0 or run_ else
     "PASS" if rw >= 1.03 and nr <= 1.05 else "BETWEEN")
L.append(f"Verdict T16-HE (pre-registered, MTP Head ~/TrainLab/REPORT.md T16-HE bccd23c): {v}: WHOLE DECODE {100 * (rw - 1):+.1f} % vs B over "
         f"{n_}; T lifetime peak {tpk:.1f} GB; median tokens per task T / B {nr:.3f}; runaways (T at the cap {CAP} where both B ended) "
         f"{run_ or 'none'} (PASS >= +3 % AND peak <= 45.0 AND tokens <= 1.05 AND no runaway; KILL < +1 % or peak > 45.0 or a runaway)")
L.append(f"Beside: full period {100 * (rf - 1):+.1f} %; T16 rule-kept rows {ruled} (rows in spans {opened}; INERT log lines {logn}); "
         f"tokens identical T vs B {[sum(1 for k in toks['t6b1'] if toks['t6b1'][k] == toks['t6t1'].get(k)), sum(1 for k in toks['t6b2'] if toks['t6b2'][k] == toks['t6t2'].get(k))]} "
         f"of 12; floor B1 vs B2 {sum(1 for k in toks['t6b1'] if toks['t6b1'][k] == toks['t6b2'].get(k))}")
open("data/t16he/t16he_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
if v == "PASS":
    open("data/t16he/PASS", "w").write(L[-2] + "\n")
EOF
say "report rc=$? (data/t16he/t16he_report.txt)"
if [ -f data/t16he/PASS ]; then
  printf '%s\n' "# T16 candidate for EvalSuite's S4-style gate vs NAX6-N (BigMoE t16he_ab.sh, snapshot $SNAP): Sat 256, MEM 47 GB, screen data/r57/snap/s5/r57_screen_s5.py; v1' + wired with span B off and R57_INERT=T16 (T16-HE PASS). One comma-separated env line:" \
    "$T0,R57_QOM_SETS=humaneval,mbpp" > data/stack/t16_env.txt
  git add data/stack/t16_env.txt && say "t16_env.txt written (snapshot $SNAP)"
fi
git add data/t16he data/r57/t6b1_${CC}_* data/r57/t6b2_${CC}_* data/r57/t6t1_${CC}_* data/r57/t6t2_${CC}_* 2>/dev/null
> (a quote from the project owner, kept private)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011DvdV6eUB4NgBbVkyVFPrT"
touch data/t16he/DONE
say "T16-HE A/B end"
