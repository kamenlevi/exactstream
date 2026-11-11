#!/bin/bash
# Sat 7 Nov 05:1x (NewIdeas Sat 04:5x: "S1's +9.0 % comes from the steady metric's timing window"): S1b, NW v3 re-test on
# the FULL PERIOD, with NW's per-layer counters off. Checked by BigMoE before arming (docs/WED_TESTS.md, Sat 05:1x):
#   S1 (data/r57/s1{b1,n1,n2,b2}_256): tpsw = sum(kept[2:]) / sum(c2 - c0) NW +9.0 %, but NW spent 4.71 / 4.78 ms per check
#   outside c0..c2 vs B's 2.37 / 2.33: full period +4.8 % (NewIdeas' cut +4.3 %), whole decode +3.3 % -> BETWEEN, not PASS.
#   Cause (code): ref2.NWCNT as a list adds a sort + 8 sums per layer call (ref2.py:935), not ancestors of the logits, so
#   all of it runs at the post-c2 readback (screen line 1355) while the host waits. (B synced there too in S1: R57_QCNT1C=1.)
# Arms (C 256, the 12 HumanEval of S1, palindrome b1 / n1 / n2 / b2; production-like: no R57_QCNT1C, no R57_RELAX_LOG):
#   B  = stackb + S2 + KEEP (final_env 1403cd6 without NW)
#   NW = B + R57_NEVERWAIT=1 + REF2_NEVERWAIT=1 + R57_NWCNT=0 (counters off; engagement = the run's host-side nwk counters)
# 0. A 2-prompt check of the NW arm first (CAP 32): 2 / 2 runs, no error lines, nwk late calls > 0 on both. Else no A/B.
# LINE (NewIdeas Sat 04:5x, written here before arming): pair-mean geo FULL-PERIOD tok/s (per prompt: sum(kept[2:]) /
#   (decode wall - c0 of check 2), i.e. checks >= 2 over all their wall time), NW vs B:
#   PASS >= +8 % AND NW lifetime peak <= 45.0 GB -> NW stays in v1 with R57_NWCNT=0 (EvalSuite says whether S4's verdict on
#   4ef3734, NW with counters, carries to a counters-only change); KILL < +3 % or peak > 45.0 GB; else BETWEEN.
#   Not PASS -> v1 = stackb + S2 + KEEP (NewIdeas' rule). Beside: whole decode (tok/s over the decode), tpsw (S1's metric),
#   time outside c0..c2 per check, median cycle, kept per check, tokens per task, peaks, nwk, token identity (information).
# usage: NOTAFTER=hh:mm WAITMEM=48 NOWAIT45=1 GAP0=0 GAP=0 MARGIN=4 bash scripts/turn45.sh 33 20 "..." /bin/bash scripts/s1b_ab.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "${S1B_ROOT:-$HOME/BigMoE}" || exit 1
mkdir -p data/s1b data/r57/snap
LOG="$PWD/data/s1b/s1b.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
PY=${S1B_PY:-$HOME/BigMoE/.venv/bin/python}
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
A=${S1B_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2: C 256 does not fit, not run"; exit 0; }
[ -n "$S1B_DRYRUN" ] || sleep 20                         # look again (gated waiters start together): a job loading now shows here
A=${S1B_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2 at the 20 s look-again: not run"; exit 0; }
CC=256
FZ="$PWD/data/r57/snap/frozen_s1b"; rm -rf "$FZ"; mkdir -p "$FZ"
SD=data/r57/snap/s1b; mkdir -p "$SD"     # own dir: never touch data/r57/snap/*.py while EvalSuite's runs load them
cp scripts/r57_screen_fast.py "$SD/r57_screen_s1b.py" && cp scripts/relax_delta.py scripts/relax_span.py scripts/inert_span.py \
  "$SD/" && cp scripts/c45_runner.sh data/r57/snap/c45_runner_s1b.sh || exit 2
[ -n "$S1B_DRYRUN" ] || { cp -R "$HOME/Leap/scripts" "$FZ/Leap_scripts" && cp -R "$HOME/Leap/third_party" "$FZ/Leap_third_party" \
  && cp -R "$HOME/TrainLab/sticky" "$FZ/TrainLab_sticky" && cp -R "$HOME/Breakthrough/keepset" "$FZ/Breakthrough_keepset" || exit 2; }
find "$FZ" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
sed -i '' -e "s#\"~/Leap/scripts\"#\"$FZ/Leap_scripts\"#g" -e "s#\"~/Leap/third_party\"#\"$FZ/Leap_third_party\"#g" \
  -e "s#\"~/TrainLab/sticky\"#\"$FZ/TrainLab_sticky\"#g" -e "s#\"/Users/kamen/Breakthrough/keepset\"#\"$FZ/Breakthrough_keepset\"#g" \
  "$SD/r57_screen_s1b.py"
RUN="/bin/bash data/r57/snap/c45_runner_s1b.sh"
export C45_OUT="$PWD/data/s1b" C45_SWAP0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}') C45_ALARM=1200 \
  MG=4 C45_DEADLINE=$(date -v+45M +%H%M) C45_SCREEN="$SD/r57_screen_s1b.py" C45_REFTOK=
K="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79"
K="$K,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,humaneval/HumanEval/56,humaneval/HumanEval/141"
K="$K,humaneval/HumanEval/101,humaneval/HumanEval/63"
QA="BIGMOE_EXT_BUILD=build_qom,R57_REFTOK=,R57_FAST=QOM,R57_D17=1,R57_D7B=code,R57_HC8=1,REF2_HC8_FAST=1,R57_DNR=1,R57_HCB=1"
QA="$QA,REF2_SMALL_NATIVE=1,R57_D7BNAX=1,REF2_OOO_1C=1"
ST="R57_NAX=1,REF2_NAX_1CQ=1,REF2_DN_KS=1,REF2_GDN_LAZY=1,D7BNAX_32=2,R57_FASTTAX=1,R57_SBN=2,R57_DSUB=1,R57_SVH=1,REF2_KEEP_CACHES=1"
B="$QA,$ST,R57_RELAX_SPAN=1,R57_RELAX_SPAN_RULE=R2:2:0.25,R57_COPY=g2,R57_QOM_SETS=humaneval"   # final_env 1403cd6 minus NW
NW="$B,R57_NEVERWAIT=1,REF2_NEVERWAIT=1,R57_NWCNT=0"
[ -n "$S1B_DRYRUN" ] && { echo "dry run: $(grep -c frozen_s1b "$SD/r57_screen_s1b.py") screen paths; B $(echo "$B" | tr ',' '\n' | grep -c =) K=V, NW $(echo "$NW" | tr ',' '\n' | grep -c =) K=V"; exit 0; }
for t in s1bpf s1bb1 s1bn1 s1bn2 s1bb2 s1bnd; do rm -f data/r57/${t}_${CC}_* data/s1b/${t}_${CC}.txt; done
rm -f data/s1b/DONE
say "S1b A/B start (pid $$, NO-SWAP $A GB, C $CC); ref2 md5 $(md5 -q "$HOME/Leap/scripts/ref2.py"), screen md5 $(md5 -q scripts/r57_screen_fast.py)"
R57_KEYS="humaneval/HumanEval/68,humaneval/HumanEval/16" C45_CAP=32 C45_ARMS="s1bpf_$CC:$CC:$NW" $RUN
n=$(grep -o '"key"' "data/r57/s1bpf_${CC}_runs.json" 2>/dev/null | wc -l | tr -d ' ')
bad=$(grep -cE 'Traceback|AssertionError|Error:' "data/s1b/s1bpf_${CC}.txt" 2>/dev/null)
eng=$("$PY" -I -c 'import json, sys; r = json.load(open(sys.argv[1])); print(sum(1 for x in r if (x.get("nwk") or [0, 0])[1] > 0))' \
  "data/r57/s1bpf_${CC}_runs.json" 2>/dev/null)
[ "${n:-0}" = 2 ] && [ "${bad:-1}" = 0 ] && [ "${eng:-0}" = 2 ] || {
  say "NW check NOT CLEAN (ran ${n:-0} / 2, error lines ${bad:-none}, engaged ${eng:-none} / 2): no A/B"; touch data/s1b/DONE; exit 0; }
say "NW check CLEAN (2 / 2, engaged 2 / 2): the A/B runs"
for a in "s1bb1:$B" "s1bn1:$NW" "s1bn2:$NW" "s1bb2:$B"; do
  R57_KEYS=$K C45_ARMS="${a%%:*}_$CC:$CC:${a#*:}" $RUN
  say "${a%%:*} rc=$?"
done
# 5th arm, information only (NewIdeas Sat 05:1x, for EvalSuite's carry-over question): NW with its counters built but read once
# after the decode clock (R57_NWCNT=2): late groups per check at ~counters-off timing, vs S1's (read per check) median 18.0.
ND="$B,R57_NEVERWAIT=1,REF2_NEVERWAIT=1,R57_NWCNT=2"
R57_KEYS=$K C45_ARMS="s1bnd_$CC:$CC:$ND" $RUN
say "s1bnd rc=$?"
# 6th step, information (NewIdeas Sat 05:1x, evidence 1): Leap's NW GPU check (scripts/nwcnt_check.py = nw_check.py with a
# counters switch) on S4's ref2 (ddbc0621, this run's frozen copy), counters ON then OFF: every output equality (warm == off,
# FORCE == reference, cold == reference, layer 47) must hold both ways, i.e. the counters touch no output.
rm -f data/s1b/nwcnt_check.txt
for on in 1 0; do
  NWCNT_ON=$on REF2_PATH="$FZ/Leap_scripts/ref2.py" QSC_COND="BigMoE S1b quiet window" BIGMOE_EXT_BUILD=build_qom \
    perl -e 'alarm shift; exec @ARGV' 600 "$PY" -I scripts/nwcnt_check.py >> data/s1b/nwcnt_check.log 2>&1
  say "nwcnt_check counters $on rc=$?"
done
"$PY" -I - "$CC" "$(date +%F)" "${S1B_FANLOG:-$HOME/Library/Logs/fanwatch.log}" >> "$LOG" 2>&1 <<'EOF'
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
for ln in open("data/s1b/c45.log"):
    m = re.match(r"(\d\d:\d\d:\d\d) (run|done) (\S+)", ln)
    if m:
        arm_t.setdefault(m.group(3), {})[m.group(2)] = D.datetime.fromisoformat(f"{day} {m.group(1)}")
    m = re.search(r"done (\S+) rc=\d+:.*lifetime peak ([\d.]+) GB", ln)
    if m:
        peak[m.group(1)] = float(m.group(2))
hit = {t for t, v in arm_t.items() if "run" in v and "done" in v and any(a < v["done"] and b > v["run"] for a, b in win)}
P = {"B": ("s1bb1", "s1bb2"), "NW": ("s1bn1", "s1bn2")}
runs, toks = {}, {}
for t in sum(P.values(), ()) + ("s1bnd",):
    try:
        R = {r["key"]: r for r in json.load(open(f"data/r57/{t}_{C}_runs.json"))}
        for c in json.load(open(f"data/r57/{t}_{C}_checks.json")):
            r = R.get(c["key"])
            if r is None:
                continue
            dec = r["n"] / r["tps"]                                   # s, the decode (tps = n / dec)
            kept, c0 = c["kept"], c.get("c0") or []
            wl = sum(c["ck"]) + sum(c["drf"][:len(c["ck"])])           # ms inside c0..c2, all checks
            r["fp"] = sum(kept[2:]) / (dec - c0[2] / 1e3) if len(c0) > 2 and dec > c0[2] / 1e3 else None
            r["out"] = (1e3 * dec - wl) / max(len(c["ck"]), 1)        # ms per check outside c0..c2
            r["kpc"] = S.mean(kept[2:]) if len(kept) > 2 else None
        runs[t] = R
        toks[t] = {k: list(v.values())[0][0] for k, v in json.load(open(f"data/r57/{t}_{C}_tokens.json")).items()}
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
L = [f"S1b A/B (NW v3, counters off), C {C} (data/s1b): B / NW / NW / B; freeze windows "
     f"{[(a.strftime('%H:%M:%S'), b.strftime('%H:%M:%S')) for a, b in win]}; passes dropped for a freeze: {sorted(hit) or 'none'}"]
for b, ps in P.items():
    pas = lambda f: ", ".join(f"{t} {g([f(r) for k, r in runs[t].items() if ok(t, k) and f(r)]):.1f}" for t in ps)
    L.append(f"{b:3s} FULL PERIOD {g(list(pm(ps, lambda r: r.get('fp')).values())):6.1f} tok/s (passes {pas(lambda r: r.get('fp'))}); "
             f"whole decode {g(list(pm(ps, lambda r: r['tps']).values())):6.1f} ({pas(lambda r: r['tps'])}); "
             f"tpsw {g(list(pm(ps, lambda r: r['tpsw']).values())):6.1f} ({pas(lambda r: r['tpsw'])})")
    L.append(f"    outside c0..c2 {S.mean(pm(ps, lambda r: r.get('out')).values() or [float('nan')]):.2f} ms/check; median cycle "
             f"{S.median([r['cyc'] for t in ps for k, r in runs[t].items() if ok(t, k) and isinstance(r.get('cyc'), (int, float))] or [float('nan')]):.1f} ms; "
             f"kept per check {g(list(pm(ps, lambda r: r.get('kpc')).values())):.2f}; tokens per task "
             f"{S.median(pm(ps, lambda r: r['n']).values() or [float('nan')]):.0f}; lifetime peak "
             f"{max([peak.get(f'{t}_{C}', 0.0) for t in ps]) or float('nan'):.1f} GB; nwk (calls, late calls) "
             f"{[sum((r.get('nwk') or [0, 0])[i] for t in ps for r in runs[t].values()) for i in (0, 1)]}")
r_, n_ = ratio(pm(P["NW"], lambda r: r.get("fp")), pm(P["B"], lambda r: r.get("fp")))
rw, _ = ratio(pm(P["NW"], lambda r: r["tps"]), pm(P["B"], lambda r: r["tps"]))
rt, _ = ratio(pm(P["NW"], lambda r: r["tpsw"]), pm(P["B"], lambda r: r["tpsw"]))
pkn = max([peak.get(f"{t}_{C}", 0.0) for t in P["NW"]]) or float("nan")
v = "MISSING" if n_ == 0 else "KILL" if r_ < 1.03 or pkn > 45.0 else "PASS" if r_ >= 1.08 else "BETWEEN"
L.append(f"Verdict S1b NW v3 counters off (pre-registered, NewIdeas Sat 04:5x): {v}: FULL PERIOD {100 * (r_ - 1):+.1f} % vs B over "
         f"{n_}, NW lifetime peak {pkn:.1f} GB (PASS >= +8 % AND peak <= 45.0 GB; KILL < +3 % or peak > 45.0 GB). Beside: "
         f"whole decode {100 * (rw - 1):+.1f} %, tpsw {100 * (rt - 1):+.1f} %. "
         + ("-> NW stays in v1 with R57_NWCNT=0 (EvalSuite: does S4's verdict carry?)" if v == "PASS" else "-> v1 = stackb + S2 + KEEP"))
same = lambda a, b: sum(1 for k in toks[a] if k in toks[b] and toks[a][k] == toks[b][k])
L.append(f"tokens identical per prompt (information): NW vs B {[same('s1bb1', 's1bn1'), same('s1bb2', 's1bn2')]} of 12; floor B1 vs B2 "
         f"{same('s1bb1', 's1bb2')}")
def lg(tags, i):                                         # NW counters per check (checks >= 2), median
    xs = []
    for t in tags:
        try:
            for e in json.load(open(f"data/r57/{t}_{C}_checks.json")):
                if not e.get("paused"):
                    xs += [x[i] for x in (e.get("nw") or [])[2:] if x and len(x) > i]
        except (OSError, ValueError):
            pass
    return S.median(xs) if xs else float("nan")
rd = runs.get("s1bnd") or {}
L.append(f"NW, counters read after the decode clock (s1bnd, information, for EvalSuite's carry-over question): "
         f"{'MISSING' if not rd else ''}full period {g([r.get('fp') for r in rd.values()]):.1f}, whole decode "
         f"{g([r['tps'] for r in rd.values()]):.1f} tok/s over {len(rd)}; per check (medians, checks >= 2) late groups "
         f"{lg(('s1bnd',), 0):.1f} vs S1 (read per check) {lg(('s1n1', 's1n2'), 0):.1f}, late uses {lg(('s1bnd',), 1):.1f} vs "
         f"{lg(('s1n1', 's1n2'), 1):.1f}, optional dropped now {lg(('s1bnd',), 6):.1f} vs {lg(('s1n1', 's1n2'), 6):.1f}; lifetime peak "
         f"{peak.get(f's1bnd_{C}', float('nan')):.1f} GB")
open("data/s1b/s1b_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? (data/s1b/s1b_report.txt)"
{ echo "NW GPU check, counters ON vs OFF (scripts/nwcnt_check.py on S4's ref2; information, evidence for EvalSuite):"
  grep -E '^NEVER-WAIT check (Mon|Tue|Wed|Thu|Fri|Sat|Sun)|^NEVER-WAIT check:' data/s1b/nwcnt_check.txt 2>/dev/null | sed 's/^/  /' \
    || echo "  MISSING (data/s1b/nwcnt_check.log)"; } >> data/s1b/s1b_report.txt
touch data/s1b/DONE
say "S1b A/B end"
