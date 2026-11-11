#!/bin/bash
# Sat 7 Nov (NewIdeas QUEUE 8dae435 "Tonight is unchanged: S5 on v1' + wired, CERT from Sun 01:30"; EvalSuite 08:xx: write the frozen
# config to data/stack/cert_env.txt and freeze the 17 live Leap / TrainLab files into its snap): S5, the number for Kamen.
# 1. FREEZE into data/r57/snap/frozen_s5 + data/r57/snap/s5/, then rewrite EVERY copied .py's code paths to the frozen copies
#    (~/Leap/scripts, ~/Leap/third_party, ~/TrainLab/sticky, "~/TrainLab" -> TrainLab_root (its *.py: mtp125), Breakthrough's keepset).
#    S4's snapshot rewrote only the screen, so modules that insert ~/Leap/scripts themselves (ref2, ooo1cq, tierglu_nax6n, ...)
#    loaded 17 files live. Data paths (~/Leap/data, ...) stay live. Any code path left -> stop. The snapshot is committed.
# 2. Arms, C 256, quiet, one process each (R57_QOM_SETS=humaneval,mbpp and R57_MODMAN=1 in the environment):
#    12 HumanEval: s5q1 / s5v1 / s5v2 / s5q2, then MBPP12 (GOAL's 12): s5mq1 / s5mv1 / s5mv2 / s5mq2
#    q = qall (the FAST QOM base, Friday's reference), v = v1' + wired (final_env dd7ad0f + R57_WIRED=1; S4 PASS + S6W PASS)
# 3. REPORT (data/s5/s5_report.txt): per set, whole decode (tps) and full period (checks >= 2 over all their wall time) as the
#    headline pair, tpsw beside (information), check 0, median cycle, kept per check, tokens per task, lifetime peak per arm,
#    v vs q ratios; the module manifests (live outside modules must be 0).
# 4. CLEAN (all 8 arms, 0 live modules, v's lifetime peak <= 45.0 GB) -> data/stack/s5_env.txt + cert_env.txt (final_env's
#    format: 'Sat 256', 'MEM 47 GB', 'screen data/r57/snap/s5/r57_screen_s5.py'), committed. Not clean -> nothing written.
# usage: NOTAFTER=hh:mm NOWAIT45=1 GAP0=0 GAP=0 MARGIN=4 bash scripts/turn45.sh 33 30 "..." /bin/bash scripts/s5_final.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "${S5_ROOT:-$HOME/BigMoE}" || exit 1
mkdir -p data/s5 data/r57/snap
LOG="$PWD/data/s5/s5.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
PY=${S5_PY:-$HOME/BigMoE/.venv/bin/python}
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
A=${S5_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2: C 256 does not fit, not run"; exit 0; }
[ -n "$S5_DRYRUN" ] || sleep 20
A=${S5_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2 at the 20 s look-again: not run"; exit 0; }
CC=256
FZ="$PWD/data/r57/snap/frozen_s5"; rm -rf "$FZ"; mkdir -p "$FZ/TrainLab_root"
SD=data/r57/snap/s5; mkdir -p "$SD"
cp scripts/r57_screen_fast.py "$SD/r57_screen_s5.py" && cp scripts/relax_delta.py scripts/relax_span.py scripts/inert_span.py \
  "$SD/" && cp scripts/c45_runner.sh data/r57/snap/c45_runner_s5.sh || exit 2
cp -R "$HOME/Leap/scripts" "$FZ/Leap_scripts" && cp -R "$HOME/Leap/third_party" "$FZ/Leap_third_party" \
  && cp -R "$HOME/TrainLab/sticky" "$FZ/TrainLab_sticky" && cp "$HOME"/TrainLab/*.py "$FZ/TrainLab_root/" \
  && cp -R "$HOME/Breakthrough/keepset" "$FZ/Breakthrough_keepset" || exit 2
find "$FZ" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
RW=(-e "s#~/Leap/scripts#$FZ/Leap_scripts#g" -e "s#/Users/kamen/Leap/scripts#$FZ/Leap_scripts#g"
    -e "s#~/Leap/third_party#$FZ/Leap_third_party#g" -e "s#/Users/kamen/Leap/third_party#$FZ/Leap_third_party#g"
    -e "s#~/TrainLab/sticky#$FZ/TrainLab_sticky#g" -e "s#/Users/kamen/TrainLab/sticky#$FZ/TrainLab_sticky#g"
    -e "s#\"~/TrainLab\"#\"$FZ/TrainLab_root\"#g" -e "s#'~/TrainLab'#'$FZ/TrainLab_root'#g"
    -e "s#~/Breakthrough/keepset#$FZ/Breakthrough_keepset#g" -e "s#/Users/kamen/Breakthrough/keepset#$FZ/Breakthrough_keepset#g")
find "$FZ" "$SD" -name '*.py' -exec sed -i '' "${RW[@]}" {} +
LEFT=$(grep -rnE --include='*.py' "(~|/Users/kamen)/Leap/(scripts|third_party)|(~|/Users/kamen)/TrainLab/sticky|[\"']~/TrainLab[\"']|(~|/Users/kamen)/Breakthrough/keepset" "$FZ" "$SD" | wc -l | tr -d ' ')
NP=$(grep -rl --include='*.py' "frozen_s5" "$FZ" "$SD" | wc -l | tr -d ' ')
[ "$LEFT" = 0 ] || { say "FREEZE: $LEFT live code paths left in the snapshot: stop"; exit 2; }
say "FREEZE: $NP files point into frozen_s5, 0 live code paths left"
QA="BIGMOE_EXT_BUILD=build_qom,R57_REFTOK=,R57_FAST=QOM,R57_D17=1,R57_D7B=code,R57_HC8=1,REF2_HC8_FAST=1,R57_DNR=1,R57_HCB=1"
QA="$QA,REF2_SMALL_NATIVE=1,R57_D7BNAX=1,REF2_OOO_1C=1"
V=$(grep -v '^#' data/stack/final_env.txt | grep . | tail -1 | sed 's/,R57_QOM_SETS=humaneval,mbpp$//')
echo "$V" | grep -q 'NEVERWAIT\|R57_QOM_SETS' && { say "final_env.txt is not v1' (NW or a second QOM_SETS): stop"; exit 2; }
echo "$V" | tr ',' '\n' | grep -vq '=' && { say "v1' env has a comma inside a value: stop"; exit 2; }
V="$V,R57_WIRED=1"
[ -n "$S5_DRYRUN" ] && { echo "dry run: $NP files -> frozen_s5, $LEFT live left; qall $(echo "$QA" | tr ',' '\n' | grep -c =) K=V, v1'w $(echo "$V" | tr ',' '\n' | grep -c =) K=V"; exit 0; }
git add data/r57/snap/frozen_s5 ':!data/r57/snap/frozen_s5/**/__pycache__/**' "$SD" ':!data/r57/snap/s5/__pycache__/**' \
> (a quote from the project owner, kept private)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011DvdV6eUB4NgBbVkyVFPrT" || { say "snapshot commit failed: stop"; exit 3; }
SNAP=$(git rev-parse --short HEAD)
RUN="/bin/bash data/r57/snap/c45_runner_s5.sh"
export C45_OUT="$PWD/data/s5" C45_SWAP0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}') C45_ALARM=1500 \
  MG=4 C45_DEADLINE=$(date -v+75M +%H%M) C45_SCREEN="$SD/r57_screen_s5.py" C45_REFTOK= R57_QOM_SETS=humaneval,mbpp R57_MODMAN=1
K="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79"
K="$K,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,humaneval/HumanEval/56,humaneval/HumanEval/141"
K="$K,humaneval/HumanEval/101,humaneval/HumanEval/63"
KM="mbpp/16,mbpp/100,mbpp/146,mbpp/161,mbpp/183,mbpp/208,mbpp/224,mbpp/246,mbpp/268,mbpp/296,mbpp/305,mbpp/319"   # GOAL's 12
for t in s5pf s5q1 s5v1 s5v2 s5q2 s5mq1 s5mv1 s5mv2 s5mq2; do rm -f data/r57/${t}_${CC}_* data/s5/${t}_${CC}.txt; done
rm -f data/s5/DONE data/s5/CLEAN
say "S5 start (pid $$, NO-SWAP $A GB, C $CC, snapshot $SNAP); screen md5 $(md5 -q "$SD/r57_screen_s5.py")"
# pre-flight: the frozen snapshot runs (2 HumanEval, CAP 32, v1' + wired) and loads nothing live; else no S5 (no cert_env)
R57_KEYS="humaneval/HumanEval/68,humaneval/HumanEval/16" C45_CAP=32 C45_ARMS="s5pf_$CC:$CC:$V" $RUN
n=$(grep -o '"key"' "data/r57/s5pf_${CC}_runs.json" 2>/dev/null | wc -l | tr -d ' ')
bad=$(grep -cE 'Traceback|AssertionError|Error:' "data/s5/s5pf_${CC}.txt" 2>/dev/null)
lv=$("$PY" -I -c 'import json, sys; print(sum(1 for _, f in json.load(open(sys.argv[1])) if "/BigMoE/data/r57/snap/" not in f))' \
  "data/r57/s5pf_${CC}_modules.json" 2>/dev/null)
[ "${n:-0}" = 2 ] && [ "${bad:-1}" = 0 ] && [ "${lv:-1}" = 0 ] || {
  say "PRE-FLIGHT NOT CLEAN (ran ${n:-0} / 2, error lines ${bad:-none}, live modules ${lv:-none}): no S5, no cert_env"; touch data/s5/DONE; exit 0; }
say "PRE-FLIGHT CLEAN (2 / 2, 0 live modules): the S5 arms run"
for a in "s5q1:$QA:$K" "s5v1:$V:$K" "s5v2:$V:$K" "s5q2:$QA:$K" "s5mq1:$QA:$KM" "s5mv1:$V:$KM" "s5mv2:$V:$KM" "s5mq2:$QA:$KM"; do
  t=${a%%:*}; r=${a#*:}; e=${r%:*}; k=${r##*:}
  R57_KEYS=$k C45_ARMS="${t}_$CC:$CC:$e" $RUN
  say "$t rc=$?"
done
"$PY" -I - "$CC" "$(date +%F)" "${S5_FANLOG:-$HOME/Library/Logs/fanwatch.log}" "$SNAP" >> "$LOG" 2>&1 <<'EOF'
import datetime as D, json, math, os, re, statistics as S, sys
C, day, fanlog, snap = sys.argv[1:5]
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
for ln in open("data/s5/c45.log"):
    m = re.match(r"(\d\d:\d\d:\d\d) (run|done) (\S+)", ln)
    if m:
        arm_t.setdefault(m.group(3), {})[m.group(2)] = D.datetime.fromisoformat(f"{day} {m.group(1)}")
    m = re.search(r"done (\S+) rc=\d+:.*lifetime peak ([\d.]+) GB", ln)
    if m:
        peak[m.group(1)] = float(m.group(2))
hit = {t for t, v in arm_t.items() if "run" in v and "done" in v and any(a < v["done"] and b > v["run"] for a, b in win)}
SETS = {"12 HumanEval": {"qall": ("s5q1", "s5q2"), "v1' + wired": ("s5v1", "s5v2")},
        "MBPP12": {"qall": ("s5mq1", "s5mq2"), "v1' + wired": ("s5mv1", "s5mv2")}}
runs, toks, live, frz, miss = {}, {}, [], 0, []
for t in [x for s in SETS.values() for ps in s.values() for x in ps]:
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
        mods = json.load(open(f"data/r57/{t}_{C}_modules.json"))
        live += [f for _, f in mods if "/BigMoE/data/r57/snap/" not in f]
        frz = max(frz, sum(1 for _, f in mods if "/frozen_s5/" in f or "/snap/s5/" in f))
    except (OSError, ValueError, KeyError) as e:
        miss.append(t); runs[t], toks[t] = {}, {}
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
L = [f"S5, C {C} (data/s5; snapshot {snap}; quiet): qall / v1' + wired / v1' + wired / qall on 12 HumanEval, then MBPP12; "
     f"freeze windows {[(a.strftime('%H:%M:%S'), b.strftime('%H:%M:%S')) for a, b in win]}; passes dropped for a freeze: "
     f"{sorted(hit) or 'none'}; arms missing: {miss or 'none'}",
     f"Modules: {frz} loaded from the s5 snapshot per process; LIVE outside modules (Leap / TrainLab / Breakthrough): {len(set(live))}"
     + (f" {sorted(set(live))[:6]}" if live else "")]
vpk = 0.0
for sname, P in SETS.items():
    L.append(f"{sname}:")
    for b, ps in P.items():
        pas = lambda f: ", ".join(f"{g([f(r) for k, r in runs[t].items() if ok(t, k) and f(r)]):.1f}" for t in ps)
        pk = max([peak.get(f"{t}_{C}", 0.0) for t in ps]) or float("nan")
        if b != "qall":
            vpk = max(vpk, pk)
        L.append(f"  {b:13s} WHOLE DECODE {g(list(pm(ps, lambda r: r['tps']).values())):6.1f} tok/s ({pas(lambda r: r['tps'])}); FULL PERIOD "
                 f"{g(list(pm(ps, lambda r: r.get('fp')).values())):6.1f} ({pas(lambda r: r.get('fp'))}); tpsw "
                 f"{g(list(pm(ps, lambda r: r['tpsw']).values())):6.1f}; check 0 {S.median(pm(ps, lambda r: r.get('ck0')).values() or [float('nan')]):.0f} ms; "
                 f"cycle {S.median([r['cyc'] for t in ps for k, r in runs[t].items() if ok(t, k) and isinstance(r.get('cyc'), (int, float))] or [float('nan')]):.1f} ms; "
                 f"kept {g(list(pm(ps, lambda r: r.get('kpc')).values())):.2f}; tokens/task {S.median(pm(ps, lambda r: r['n']).values() or [float('nan')]):.0f}; "
                 f"lifetime peak {pk:.1f} GB")
    q, v = P["qall"], P["v1' + wired"]
    rw, n_ = ratio(pm(v, lambda r: r["tps"]), pm(q, lambda r: r["tps"]))
    rf, _ = ratio(pm(v, lambda r: r.get("fp")), pm(q, lambda r: r.get("fp")))
    same = lambda a, b: sum(1 for k in toks[a] if k in toks[b] and toks[a][k] == toks[b][k])
    L.append(f"  v1' + wired vs qall over {n_}: whole decode {100 * (rw - 1):+.1f} %, full period {100 * (rf - 1):+.1f} %; tokens identical "
             f"v1 vs v2 {same(v[0], v[1])} / {len(toks[v[0]])}, q1 vs q2 {same(q[0], q[1])} / {len(toks[q[0]])}")
clean = not miss and not live and frz > 0 and vpk <= 45.0
L.append(f"CLEAN: {clean} (all 8 arms, 0 live modules, v1' + wired lifetime peak {vpk:.1f} GB <= 45.0)")
open("data/s5/s5_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
if clean:
    open("data/s5/CLEAN", "w").write("\n".join(x for x in L if "v1' + wired " in x and "WHOLE DECODE" in x) + "\n")
EOF
say "report rc=$? (data/s5/s5_report.txt)"
if [ -f data/s5/CLEAN ]; then
  HE=$(grep -m1 -E "v1' \+ wired +WHOLE DECODE" data/s5/s5_report.txt | grep -oE 'WHOLE DECODE +[0-9.]+|FULL PERIOD +[0-9.]+' | tr -s ' ' | paste -sd',' -)
  for f in s5_env cert_env; do
    printf '%s\n' "# CERT config (BigMoE s5_final.sh, snapshot $SNAP): Sat 256, MEM 47 GB, screen data/r57/snap/s5/r57_screen_s5.py; v1' + wired (stackb_s2 + KEEP + R57_WIRED); S5 12 HumanEval: $HE tok/s. One comma-separated env line:" \
      "$V,R57_QOM_SETS=humaneval,mbpp" > data/stack/$f.txt
  done
  git add data/stack/s5_env.txt data/stack/cert_env.txt && say "cert_env.txt + s5_env.txt written (snapshot $SNAP)"
else
  say "S5 NOT CLEAN: cert_env.txt not written"
fi
git add data/s5 data/r57/s5q1_${CC}_* data/r57/s5q2_${CC}_* data/r57/s5v1_${CC}_* data/r57/s5v2_${CC}_* data/r57/s5mq1_${CC}_* \
  data/r57/s5mq2_${CC}_* data/r57/s5mv1_${CC}_* data/r57/s5mv2_${CC}_* 2>/dev/null
> (a quote from the project owner, kept private)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011DvdV6eUB4NgBbVkyVFPrT"
touch data/s5/DONE
say "S5 end"
