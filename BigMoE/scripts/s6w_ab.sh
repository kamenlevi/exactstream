#!/bin/bash
# Sat 7 Nov 09:0x (the orchestrator 09:0x; Leap FCP 038bfa1 = FIRSTCHK's fix): S6W, the wired-limit engine A/B on v1'.
#   B = v1' (final_env dd7ad0f: stackb + S2 + KEEP; QOM sets humaneval for these prompts)
#   W = B + R57_WIRED=1 (mx.set_wired_limit(max_recommended_working_set_size) once after model load; Leap: check 0 345 -> 90 ms,
#       peak 43.0 both; exact env bitwise 43/43 checks)
# C 256, the 12 HumanEval of S1 / S1b, palindrome b1 / w1 / w2 / b2, one process per arm.
# LINE (BigMoE + NewIdeas' call, written here before arming): pair-mean geo WHOLE DECODE tok/s (tps = n / decode wall; check 0
#   is in it, the full period leaves it out by design), W vs B: PASS >= +8 % AND full period >= -2 % AND W lifetime peak
#   <= 45.0 GB -> W joins v1' for S5 (and the EXACT default, Leap's bitwise proof); KILL < +3 % or W peak > 45.0 GB;
#   else BETWEEN. Beside: full period, tpsw, check 0 wall, median cycle, kept per check, tokens identical per prompt W vs B
#   next to the floor B1 vs B2 (QOM's outputs differ run to run; the carry-over proof is Leap's exact-env bitwise check).
# VTIER: its buffers are ext newBuffer allocations marked volatile (bigmoe_ext.cpp:2011), not MLX's, so MLX's residency set
#   does not wire them. The peak clause guards the 45 GB line.
# usage: NOTAFTER=hh:mm NOWAIT45=1 GAP0=0 GAP=0 MARGIN=4 bash scripts/turn45.sh 33 20 "..." /bin/bash scripts/s6w_ab.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "${S6_ROOT:-$HOME/BigMoE}" || exit 1
mkdir -p data/s6w data/r57/snap
LOG="$PWD/data/s6w/s6w.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
PY=${S6_PY:-$HOME/BigMoE/.venv/bin/python}
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
A=${S6_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2: C 256 does not fit, not run"; exit 0; }
[ -n "$S6_DRYRUN" ] || sleep 20
A=${S6_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2 at the 20 s look-again: not run"; exit 0; }
CC=256
FZ="$PWD/data/r57/snap/frozen_s6w"; rm -rf "$FZ"; mkdir -p "$FZ"
SD=data/r57/snap/s6w; mkdir -p "$SD"
cp scripts/r57_screen_fast.py "$SD/r57_screen_s6w.py" && cp scripts/relax_delta.py scripts/relax_span.py scripts/inert_span.py \
  "$SD/" && cp scripts/c45_runner.sh data/r57/snap/c45_runner_s6w.sh || exit 2
[ -n "$S6_DRYRUN" ] || { cp -R "$HOME/Leap/scripts" "$FZ/Leap_scripts" && cp -R "$HOME/Leap/third_party" "$FZ/Leap_third_party" \
  && cp -R "$HOME/TrainLab/sticky" "$FZ/TrainLab_sticky" && cp -R "$HOME/Breakthrough/keepset" "$FZ/Breakthrough_keepset" || exit 2; }
find "$FZ" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
sed -i '' -e "s#\"~/Leap/scripts\"#\"$FZ/Leap_scripts\"#g" -e "s#\"~/Leap/third_party\"#\"$FZ/Leap_third_party\"#g" \
  -e "s#\"~/TrainLab/sticky\"#\"$FZ/TrainLab_sticky\"#g" -e "s#\"/Users/kamen/Breakthrough/keepset\"#\"$FZ/Breakthrough_keepset\"#g" \
  "$SD/r57_screen_s6w.py"
RUN="/bin/bash data/r57/snap/c45_runner_s6w.sh"
export C45_OUT="$PWD/data/s6w" C45_SWAP0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}') C45_ALARM=1200 \
  MG=4 C45_DEADLINE=$(date -v+45M +%H%M) C45_SCREEN="$SD/r57_screen_s6w.py" C45_REFTOK=
K="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79"
K="$K,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,humaneval/HumanEval/56,humaneval/HumanEval/141"
K="$K,humaneval/HumanEval/101,humaneval/HumanEval/63"
B=$(grep -v '^#' data/stack/final_env.txt | grep . | tail -1 | sed 's/,R57_QOM_SETS=humaneval,mbpp$/,R57_QOM_SETS=humaneval/')
echo "$B" | grep -q 'NEVERWAIT' && { say "final_env.txt carries NW: not v1', not run"; exit 2; }
echo "$B" | tr ',' '\n' | grep -vq '=' && { say "v1' env still has a comma inside a value: not run"; exit 2; }
W="$B,R57_WIRED=1"
[ -n "$S6_DRYRUN" ] && { echo "dry run: $(grep -c frozen_s6w "$SD/r57_screen_s6w.py") screen paths; B $(echo "$B" | tr ',' '\n' | grep -c =) K=V, W $(echo "$W" | tr ',' '\n' | grep -c =)"; exit 0; }
for t in s6b1 s6w1 s6w2 s6b2; do rm -f data/r57/${t}_${CC}_* data/s6w/${t}_${CC}.txt; done
rm -f data/s6w/DONE
say "S6W A/B start (pid $$, NO-SWAP $A GB, C $CC); screen md5 $(md5 -q scripts/r57_screen_fast.py)"
for a in "s6b1:$B" "s6w1:$W" "s6w2:$W" "s6b2:$B"; do
  R57_KEYS=$K C45_ARMS="${a%%:*}_$CC:$CC:${a#*:}" $RUN
  say "${a%%:*} rc=$?"
done
"$PY" -I - "$CC" "$(date +%F)" "${S6_FANLOG:-$HOME/Library/Logs/fanwatch.log}" >> "$LOG" 2>&1 <<'EOF'
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
for ln in open("data/s6w/c45.log"):
    m = re.match(r"(\d\d:\d\d:\d\d) (run|done) (\S+)", ln)
    if m:
        arm_t.setdefault(m.group(3), {})[m.group(2)] = D.datetime.fromisoformat(f"{day} {m.group(1)}")
    m = re.search(r"done (\S+) rc=\d+:.*lifetime peak ([\d.]+) GB", ln)
    if m:
        peak[m.group(1)] = float(m.group(2))
hit = {t for t, v in arm_t.items() if "run" in v and "done" in v and any(a < v["done"] and b > v["run"] for a, b in win)}
P = {"B": ("s6b1", "s6b2"), "W": ("s6w1", "s6w2")}
runs, toks = {}, {}
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
wired = sum(1 for t in P["W"] for ln in open(f"data/s6w/{t}_{C}.txt", errors="ignore") if "R57_WIRED: mx.set_wired_limit" in ln) \
    if all(__import__("os").path.exists(f"data/s6w/{t}_{C}.txt") for t in P["W"]) else 0
L = [f"S6W A/B (wired limit), C {C} (data/s6w): B / W / W / B; freeze windows "
     f"{[(a.strftime('%H:%M:%S'), b.strftime('%H:%M:%S')) for a, b in win]}; passes dropped for a freeze: {sorted(hit) or 'none'}; "
     f"W passes that logged the wired limit: {wired} / 2"]
for b, ps in P.items():
    pas = lambda f: ", ".join(f"{t} {g([f(r) for k, r in runs[t].items() if ok(t, k) and f(r)]):.1f}" for t in ps)
    L.append(f"{b}: WHOLE DECODE {g(list(pm(ps, lambda r: r['tps']).values())):6.1f} tok/s ({pas(lambda r: r['tps'])}); full period "
             f"{g(list(pm(ps, lambda r: r.get('fp')).values())):6.1f} ({pas(lambda r: r.get('fp'))}); tpsw "
             f"{g(list(pm(ps, lambda r: r['tpsw']).values())):6.1f}")
    L.append(f"    check 0 {S.median(pm(ps, lambda r: r.get('ck0')).values() or [float('nan')]):.0f} ms (median); median cycle "
             f"{S.median([r['cyc'] for t in ps for k, r in runs[t].items() if ok(t, k) and isinstance(r.get('cyc'), (int, float))] or [float('nan')]):.1f} ms; "
             f"kept per check {g(list(pm(ps, lambda r: r.get('kpc')).values())):.2f}; tokens per task "
             f"{S.median(pm(ps, lambda r: r['n']).values() or [float('nan')]):.0f}; lifetime peak "
             f"{max([peak.get(f'{t}_{C}', 0.0) for t in ps]) or float('nan'):.1f} GB")
rw, n_ = ratio(pm(P["W"], lambda r: r["tps"]), pm(P["B"], lambda r: r["tps"]))
rf, _ = ratio(pm(P["W"], lambda r: r.get("fp")), pm(P["B"], lambda r: r.get("fp")))
rt, _ = ratio(pm(P["W"], lambda r: r["tpsw"]), pm(P["B"], lambda r: r["tpsw"]))
pkw = max([peak.get(f"{t}_{C}", 0.0) for t in P["W"]]) or float("nan")
v = ("MISSING" if n_ == 0 or wired < 2 else "KILL" if rw < 1.03 or pkw > 45.0 else
     "PASS" if rw >= 1.08 and rf >= 0.98 else "BETWEEN")
L.append(f"Verdict S6W wired limit (pre-registered, BigMoE + NewIdeas' call): {v}: WHOLE DECODE {100 * (rw - 1):+.1f} % vs B over {n_}; "
         f"full period {100 * (rf - 1):+.1f} %; W lifetime peak {pkw:.1f} GB (PASS >= +8 % AND full period >= -2 % AND peak <= 45.0 GB; "
         f"KILL < +3 % or peak > 45.0 GB). tpsw {100 * (rt - 1):+.1f} %. "
         + ("-> W joins v1' for S5 (R57_WIRED=1)" if v == "PASS" else "-> v1' stays as is"))
same = lambda a, b: sum(1 for k in toks[a] if k in toks[b] and toks[a][k] == toks[b][k])
L.append(f"tokens identical per prompt: W vs B {[same('s6b1', 's6w1'), same('s6b2', 's6w2')]} of 12; floor B1 vs B2 "
         f"{same('s6b1', 's6b2')}; W1 vs W2 {same('s6w1', 's6w2')} (QOM varies run to run; carry-over proof = Leap's exact bitwise check)")
open("data/s6w/s6w_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? (data/s6w/s6w_report.txt)"
touch data/s6w/DONE
say "S6W A/B end"
