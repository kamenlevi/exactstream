#!/bin/bash
# Sat 7 Nov 17:3x (NewIdeas TABLE TONIGHT, slot 2; QUEUE c8dfaac): S5' = S5's config + the Sunday exact-safe bundle (seams a-e + DQ1;
# p4s PASS, bits 43/43, so S4 / CERT carry). New deep-frozen snapshot (Leap's ref2 >= 1f85c231), committed first.
#   B = v1' + wired (cert_env, S5's config); X = B + BIGMOE_PAUSE_EVERY_MS=50,R57_S0CARRY=1,REF2_SB_FIRST=1,REF2_PLE_LATE=1,REF2_COMMIT_SUB=9,R57_DQ1=1
# C 256, one process per arm, B / X / X / B on S5's 12 HumanEval, then MBPP12 (R57_QOM_SETS=humaneval,mbpp in the environment).
# REPORT (data/s5p/s5p_report.txt): per set, X = S5' (the table's 45 GB lossy number): WHOLE DECODE (headline), full period, end to end
#   (with prefill), check 0, cycle, kept, tokens per task, lifetime peak (<= 45.0 GB); X vs B; live outside modules (0).
# Writes data/stack/s5p_env.txt (final_env's format) when all 8 arms ran, 0 live modules and X's peak <= 45.0 GB.
# usage: NOTAFTER=hh:mm NOWAIT45=1 GAP0=0 GAP=0 MARGIN=4 bash scripts/turn45.sh 33 30 "..." /bin/bash scripts/s5p.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "${S5P_ROOT:-$HOME/BigMoE}" || exit 1
D=data/s5p; mkdir -p $D
LOG="$PWD/$D/s5p.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
PY=${S5P_PY:-$HOME/BigMoE/.venv/bin/python}
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
A=${S5P_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2: C 256 does not fit, not run"; exit 0; }
[ -n "$S5P_DRYRUN" ] || sleep 20
A=${S5P_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2 at the 20 s look-again: not run"; exit 0; }
FR=$(bash scripts/deep_freeze.sh s5p) || { say "freeze failed: $FR"; exit 2; }
SD=data/r57/snap/s5p
B=$(grep -v '^#' data/stack/cert_env.txt | grep . | tail -1 | sed 's/,R57_QOM_SETS=humaneval,mbpp$//')
echo "$B" | tr ',' '\n' | grep -qx 'R57_WIRED=1' || { say "cert_env is not v1' + wired: not run"; exit 2; }
BUN="BIGMOE_PAUSE_EVERY_MS=50,R57_S0CARRY=1,REF2_SB_FIRST=1,REF2_PLE_LATE=1,REF2_COMMIT_SUB=9,R57_DQ1=1"
X="$B,$BUN"
[ -n "$S5P_DRYRUN" ] && { echo "dry run: $FR; B $(echo "$B" | tr ',' '\n' | grep -c =) K=V, X $(echo "$X" | tr ',' '\n' | grep -c =) K=V"; exit 0; }
git add data/r57/snap/frozen_s5p ':!data/r57/snap/frozen_s5p/**/__pycache__/**' "$SD" ":!$SD/__pycache__/**" data/r57/snap/c45_runner_s5p.sh \
> (a quote from the project owner, kept private)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011DvdV6eUB4NgBbVkyVFPrT" || { say "snapshot commit failed: not run"; exit 3; }
SNAP=$(git rev-parse --short HEAD)
RUN="/bin/bash data/r57/snap/c45_runner_s5p.sh"
export C45_OUT="$PWD/$D" C45_SWAP0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}') C45_ALARM=1500 \
  MG=4 C45_DEADLINE=$(date -v+75M +%H%M) C45_SCREEN="$SD/r57_screen_s5p.py" C45_REFTOK= R57_QOM_SETS=humaneval,mbpp R57_MODMAN=1
K="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79"
K="$K,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,humaneval/HumanEval/56,humaneval/HumanEval/141"
K="$K,humaneval/HumanEval/101,humaneval/HumanEval/63"
KM="mbpp/16,mbpp/100,mbpp/146,mbpp/161,mbpp/183,mbpp/208,mbpp/224,mbpp/246,mbpp/268,mbpp/296,mbpp/305,mbpp/319"
for t in spb1 spx1 spx2 spb2 spmb1 spmx1 spmx2 spmb2; do rm -f data/r57/${t}_256_* $D/${t}_256.txt; done
rm -f $D/DONE
say "S5' start (pid $$, NO-SWAP $A GB, C 256, snapshot $SNAP)"
for a in "spb1:$B:$K" "spx1:$X:$K" "spx2:$X:$K" "spb2:$B:$K" "spmb1:$B:$KM" "spmx1:$X:$KM" "spmx2:$X:$KM" "spmb2:$B:$KM"; do
  t=${a%%:*}; r=${a#*:}; e=${r%:*}; k=${r##*:}
  R57_KEYS=$k C45_ARMS="${t}_256:256:$e" $RUN
  say "$t rc=$?"
done
"$PY" -I - "$(date +%F)" "${S5P_FANLOG:-$HOME/Library/Logs/fanwatch.log}" "$SNAP" >> "$LOG" 2>&1 <<'EOF'
import datetime as D_, json, math, re, statistics as S, sys
day, fanlog, snap = sys.argv[1:4]
g = lambda xs: (lambda ys: math.exp(sum(map(math.log, ys)) / len(ys)) if ys else float("nan"))([x for x in xs if x and x > 0])
win, t0 = [], None
try:
    for ln in open(fanlog):
        m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (froze|cool)", ln)
        if m and m.group(1).startswith(day):
            t = D_.datetime.fromisoformat(m.group(1))
            if m.group(2) == "froze":
                t0 = t
            elif t0 is not None:
                win.append((t0, t)); t0 = None
except OSError:
    pass
arm_t, peak = {}, {}
for ln in open("data/s5p/c45.log"):
    m = re.match(r"(\d\d:\d\d:\d\d) (run|done) (\S+)", ln)
    if m:
        arm_t.setdefault(m.group(3), {})[m.group(2)] = D_.datetime.fromisoformat(f"{day} {m.group(1)}")
    m = re.search(r"done (\S+) rc=\d+:.*lifetime peak ([\d.]+) GB", ln)
    if m:
        peak[m.group(1)] = float(m.group(2))
hit = {t for t, v in arm_t.items() if "run" in v and "done" in v and any(a < v["done"] and b > v["run"] for a, b in win)}
SETS = {"12 HumanEval": {"B": ("spb1", "spb2"), "X": ("spx1", "spx2")}, "MBPP12": {"B": ("spmb1", "spmb2"), "X": ("spmx1", "spmx2")}}
runs, live, miss = {}, 0, []
for t in [x for s in SETS.values() for ps in s.values() for x in ps]:
    try:
        R = {r["key"]: r for r in json.load(open(f"data/r57/{t}_256_runs.json"))}
        for c in json.load(open(f"data/r57/{t}_256_checks.json")):
            r = R.get(c["key"])
            if r is None:
                continue
            dec, c0 = r["n"] / r["tps"], c.get("c0") or []
            r["fp"] = sum(c["kept"][2:]) / (dec - c0[2] / 1e3) if len(c0) > 2 and dec > c0[2] / 1e3 else None
            r["e2e"] = r["n"] / (dec + r["pfs"]) if r.get("pfs") is not None else None
            r["ck0"] = c["bld"][0] + c["wt"][0] if c["bld"] else None
            r["kpc"] = S.mean(c["kept"][2:]) if len(c["kept"]) > 2 else None
        runs[t] = R
        live += sum(1 for _, f in json.load(open(f"data/r57/{t}_256_modules.json")) if "/BigMoE/data/r57/snap/" not in f)
    except (OSError, ValueError, KeyError):
        miss.append(t); runs[t] = {}
ok = lambda t, k: k in runs[t] and "PAUSED" not in runs[t][k].get("bad", "") and f"{t}_256" not in hit
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
L = [f"S5' (v1' + wired + the Sunday bundle), C 256 (data/s5p; snapshot {snap}; quiet): B / X / X / B on 12 HumanEval, then MBPP12; "
     f"freeze windows {[(a.strftime('%H:%M:%S'), b.strftime('%H:%M:%S')) for a, b in win]}; dropped for a freeze: {sorted(hit) or 'none'}; "
     f"arms missing: {miss or 'none'}; live outside modules: {live}"]
xpk = 0.0
for sname, P in SETS.items():
    L.append(f"{sname}:")
    for b, ps in P.items():
        pas = lambda f: ", ".join(f"{g([f(r) for k, r in runs[t].items() if ok(t, k) and f(r)]):.1f}" for t in ps)
        pk = max([peak.get(f"{t}_256", 0.0) for t in ps]) or float("nan")
        if b == "X":
            xpk = max(xpk, pk)
        L.append(f"  {b} WHOLE DECODE {g(list(pm(ps, lambda r: r['tps']).values())):6.1f} tok/s ({pas(lambda r: r['tps'])}); FULL PERIOD "
                 f"{g(list(pm(ps, lambda r: r.get('fp')).values())):6.1f}; END TO END {g(list(pm(ps, lambda r: r.get('e2e')).values())):6.1f}; "
                 f"check 0 {S.median(pm(ps, lambda r: r.get('ck0')).values() or [float('nan')]):.0f} ms; cycle "
                 f"{S.median([r['cyc'] for t in ps for k, r in runs[t].items() if ok(t, k) and isinstance(r.get('cyc'), (int, float))] or [float('nan')]):.1f} ms; "
                 f"kept {g(list(pm(ps, lambda r: r.get('kpc')).values())):.2f}; tokens/task {S.median(pm(ps, lambda r: r['n']).values() or [float('nan')]):.0f}; "
                 f"lifetime peak {pk:.1f} GB")
    rw, n_ = ratio(pm(P["X"], lambda r: r["tps"]), pm(P["B"], lambda r: r["tps"]))
    cyc = {b: S.median(pm(ps, lambda r: r.get("cyc")).values() or [float("nan")]) for b, ps in P.items()}
    L.append(f"  X vs B over {n_}: whole decode {100 * (rw - 1):+.1f} %, median cycle {100 * (cyc['X'] / cyc['B'] - 1):+.2f} %")
clean = not miss and not live and xpk <= 45.0
L.append(f"CLEAN: {clean} (all 8 arms, 0 live modules, X lifetime peak {xpk:.1f} GB <= 45.0)")
open("data/s5p/s5p_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
if clean:
    open("data/s5p/CLEAN", "w").write("ok\n")
EOF
say "report rc=$? ($D/s5p_report.txt)"
if [ -f $D/CLEAN ]; then
  HE=$(grep -m1 -E "^  X WHOLE DECODE" $D/s5p_report.txt | grep -oE 'WHOLE DECODE +[0-9.]+|FULL PERIOD +[0-9.]+' | tr -s ' ' | paste -sd',' -)
  printf '%s\n' "# S5' config (BigMoE s5p.sh, snapshot $SNAP): Sat 256, MEM 47 GB, screen data/r57/snap/s5p/r57_screen_s5p.py; v1' + wired + the Sunday exact-safe bundle (seams a-e + DQ1); S5' 12 HumanEval: $HE tok/s. One comma-separated env line:" \
    "$X,R57_QOM_SETS=humaneval,mbpp" > data/stack/s5p_env.txt
  git add data/stack/s5p_env.txt
fi
git add $D data/r57/spb1_256_* data/r57/spb2_256_* data/r57/spx1_256_* data/r57/spx2_256_* data/r57/spmb1_256_* data/r57/spmb2_256_* \
  data/r57/spmx1_256_* data/r57/spmx2_256_* 2>/dev/null
> (a quote from the project owner, kept private)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011DvdV6eUB4NgBbVkyVFPrT"
touch $D/DONE
say "S5' end"
