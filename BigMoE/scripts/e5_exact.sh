#!/bin/bash
# Sat 7 Nov 17:2x (the orchestrator 17:1x; Kamen: what would we publish now): E5, the 45 GB EXACT number at C 256.
#   E  = the EXACT default (R57_PRESET=45 defaults on the converted store + R57_NAX=1 = NAX6-N; the preset switch to NAX6-N for
#        exact runs is in the screen since this commit's parent) + REF2_KEEP_CACHES=1 + R57_WIRED=1
#   EB = E + the Sunday exact-safe bundle (p4s PASS: BIGMOE_PAUSE_EVERY_MS=50, R57_S0CARRY=1, REF2_SB_FIRST=1, REF2_PLE_LATE=1,
#        REF2_COMMIT_SUB=9, R57_DQ1=1; Leap's ref2 >= 1f85c231)
# Workload: S5's 12 HumanEval, then MBPP12, C 256, one process per arm, E / EB / EB / E per set, a new deep-frozen snapshot (committed).
# EXACT PROOF (the publish condition): every run's tokens == NAX6-N's one-token AR rows (EvalSuite results/ref2tnax for HumanEval,
#   results/ref2tnaxn.r2 for MBPP; first min(n, len) tokens; prompts matched by length), on every prompt of every pass of E and EB
#   (24 prompts x 4 passes >= the 3 asked). Any difference -> NOT EXACT, the number is not published.
# REPORT (data/e5/e5_report.txt): per set and arm, WHOLE DECODE (headline), full period, end to end with prefill (n / (prefill + decode)),
#   check 0, median cycle, kept per check, tokens per task, lifetime peak; EB vs E; live outside modules (must be 0).
# usage: NOTAFTER=hh:mm NOWAIT45=1 GAP0=0 GAP=0 MARGIN=4 bash scripts/turn45.sh 33 30 "..." /bin/bash scripts/e5_exact.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "${E5_ROOT:-$HOME/BigMoE}" || exit 1
D=data/e5; mkdir -p $D
LOG="$PWD/$D/e5.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
PY=${E5_PY:-$HOME/BigMoE/.venv/bin/python}
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
A=${E5_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2: C 256 does not fit, not run"; exit 0; }
[ -n "$E5_DRYRUN" ] || sleep 20
A=${E5_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2 at the 20 s look-again: not run"; exit 0; }
K="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79"
K="$K,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,humaneval/HumanEval/56,humaneval/HumanEval/141"
K="$K,humaneval/HumanEval/101,humaneval/HumanEval/63"
KM="mbpp/16,mbpp/100,mbpp/146,mbpp/161,mbpp/183,mbpp/208,mbpp/224,mbpp/246,mbpp/268,mbpp/296,mbpp/305,mbpp/319"
"$PY" -I - "$K,$KM" "$D/ar_ref_tokens.json" <<'EOF' || { say "AR reference file: failed"; exit 2; }
import json, os, sys
keys, out = sys.argv[1].split(","), sys.argv[2]
ref = {}
for f in ("~/EvalSuite/results/ref2tnax/decode.jsonl", "~/EvalSuite/results/ref2tnaxn.r2/decode.jsonl"):
    for ln in open(os.path.expanduser(f)):
        d = json.loads(ln)
        if d["key"] in keys and d["key"] not in ref:
            ref[d["key"]] = {"gate+t": [d["completion_ids"]], "n_prompt": d.get("n_prompt"), "finish": d.get("finish"), "src": f}
miss = [k for k in keys if k not in ref]
assert not miss, f"no NAX6-N AR row for {miss}"
json.dump(ref, open(out, "w"))
print(f"AR reference: {len(ref)} keys")
EOF
FR=$(bash scripts/deep_freeze.sh e5) || { say "freeze failed: $FR"; exit 2; }
SD=data/r57/snap/e5
E="R57_NAX=1,REF2_KEEP_CACHES=1,R57_WIRED=1"
EB="$E,BIGMOE_PAUSE_EVERY_MS=50,R57_S0CARRY=1,REF2_SB_FIRST=1,REF2_PLE_LATE=1,REF2_COMMIT_SUB=9,R57_DQ1=1"
[ -n "$E5_DRYRUN" ] && { echo "dry run: $FR; E $(echo "$E" | tr ',' '\n' | grep -c =) K=V, EB $(echo "$EB" | tr ',' '\n' | grep -c =) K=V; AR file $D/ar_ref_tokens.json"; exit 0; }
git add "data/r57/snap/frozen_e5" ":!data/r57/snap/frozen_e5/**/__pycache__/**" "$SD" ":!$SD/__pycache__/**" data/r57/snap/c45_runner_e5.sh \
> (a quote from the project owner, kept private)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011DvdV6eUB4NgBbVkyVFPrT" || { say "snapshot commit failed: not run"; exit 3; }
SNAP=$(git rev-parse --short HEAD)
RUN="/bin/bash data/r57/snap/c45_runner_e5.sh"
export C45_OUT="$PWD/$D" C45_SWAP0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}') C45_ALARM=2400 \
  MG=4 C45_DEADLINE=$(date -v+90M +%H%M) C45_SCREEN="$SD/r57_screen_e5.py" R57_MODMAN=1 R57_REFTOK="$PWD/$D/ar_ref_tokens.json:gate+t"
for t in e5h1 e5hb1 e5hb2 e5h2 e5m1 e5mb1 e5mb2 e5m2; do rm -f data/r57/${t}_256_* $D/${t}_256.txt; done
rm -f $D/DONE
say "E5 start (pid $$, NO-SWAP $A GB, C 256, snapshot $SNAP)"
for a in "e5h1:$E:$K" "e5hb1:$EB:$K" "e5hb2:$EB:$K" "e5h2:$E:$K" "e5m1:$E:$KM" "e5mb1:$EB:$KM" "e5mb2:$EB:$KM" "e5m2:$E:$KM"; do
  t=${a%%:*}; r=${a#*:}; e=${r%:*}; k=${r##*:}
  R57_KEYS=$k C45_ARMS="${t}_256:256:$e" $RUN
  say "$t rc=$?"
done
"$PY" -I - "$(date +%F)" "${E5_FANLOG:-$HOME/Library/Logs/fanwatch.log}" "$SNAP" >> "$LOG" 2>&1 <<'EOF'
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
for ln in open("data/e5/c45.log"):
    m = re.match(r"(\d\d:\d\d:\d\d) (run|done) (\S+)", ln)
    if m:
        arm_t.setdefault(m.group(3), {})[m.group(2)] = D_.datetime.fromisoformat(f"{day} {m.group(1)}")
    m = re.search(r"done (\S+) rc=\d+:.*lifetime peak ([\d.]+) GB", ln)
    if m:
        peak[m.group(1)] = float(m.group(2))
hit = {t for t, v in arm_t.items() if "run" in v and "done" in v and any(a < v["done"] and b > v["run"] for a, b in win)}
AR = json.load(open("data/e5/ar_ref_tokens.json"))
SETS = {"12 HumanEval": {"E": ("e5h1", "e5h2"), "EB": ("e5hb1", "e5hb2")}, "MBPP12": {"E": ("e5m1", "e5m2"), "EB": ("e5mb1", "e5mb2")}}
runs, toks, live, miss = {}, {}, 0, []
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
        toks[t] = {k: list(v.values())[0][0] for k, v in json.load(open(f"data/r57/{t}_256_tokens.json")).items()}
        live += sum(1 for _, f in json.load(open(f"data/r57/{t}_256_modules.json")) if "/BigMoE/data/r57/snap/" not in f)
    except (OSError, ValueError, KeyError) as e:
        miss.append(t); runs[t], toks[t] = {}, {}
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
eq_n = tot = 0; bad = []; pmis = []
for t in toks:
    for k, out in toks[t].items():
        ref = AR[k]["gate+t"][0]; m = min(len(out), len(ref)); tot += 1
        if out[:m] == ref[:m] and m > 0:
            eq_n += 1
        else:
            bad.append(f"{t}:{k.split('/')[-1]}@{next(i for i in range(m) if out[i] != ref[i]) if m and out[:m] != ref[:m] else 'len0'}")
        P = runs[t].get(k, {}).get("P")
        if P is not None and AR[k].get("n_prompt") is not None and P != AR[k]["n_prompt"]:
            pmis.append(f"{k} P {P} vs AR {AR[k]['n_prompt']}")
exact = tot > 0 and eq_n == tot and not miss and not pmis
L = [f"E5, the 45 GB EXACT number, C 256 (data/e5; snapshot {snap}; quiet): E / EB / EB / E on 12 HumanEval, then MBPP12; freeze "
     f"windows {[(a.strftime('%H:%M:%S'), b.strftime('%H:%M:%S')) for a, b in win]}; dropped for a freeze: {sorted(hit) or 'none'}; "
     f"arms missing: {miss or 'none'}; live outside modules: {live}",
     f"EXACT PROOF vs NAX6-N's one-token AR (EvalSuite ref2tnax / ref2tnaxn.r2): {'PASS' if exact else 'FAIL'}: {eq_n} / {tot} runs token-identical "
     f"over min(n, AR length); prompt-length mismatches {pmis or 'none'}" + (f"; DIFFER {bad[:8]}" if bad else "")]
for sname, P in SETS.items():
    L.append(f"{sname}:")
    for b, ps in P.items():
        pas = lambda f: ", ".join(f"{g([f(r) for k, r in runs[t].items() if ok(t, k) and f(r)]):.1f}" for t in ps)
        L.append(f"  {b:3s} WHOLE DECODE {g(list(pm(ps, lambda r: r['tps']).values())):6.1f} tok/s ({pas(lambda r: r['tps'])}); FULL PERIOD "
                 f"{g(list(pm(ps, lambda r: r.get('fp')).values())):6.1f}; END TO END {g(list(pm(ps, lambda r: r.get('e2e')).values())):6.1f} "
                 f"(prefill {S.median(pm(ps, lambda r: r.get('pfs')).values() or [float('nan')]):.2f} s median); check 0 "
                 f"{S.median(pm(ps, lambda r: r.get('ck0')).values() or [float('nan')]):.0f} ms; cycle "
                 f"{S.median([r['cyc'] for t in ps for k, r in runs[t].items() if ok(t, k) and isinstance(r.get('cyc'), (int, float))] or [float('nan')]):.1f} ms; "
                 f"kept {g(list(pm(ps, lambda r: r.get('kpc')).values())):.2f}; tokens/task {S.median(pm(ps, lambda r: r['n']).values() or [float('nan')]):.0f}; "
                 f"lifetime peak {max([peak.get(f'{t}_256', 0.0) for t in ps]) or float('nan'):.1f} GB")
    rw, n_ = ratio(pm(P["EB"], lambda r: r["tps"]), pm(P["E"], lambda r: r["tps"]))
    cyc = {b: S.median(pm(ps, lambda r: r.get("cyc")).values() or [float("nan")]) for b, ps in P.items()}
    L.append(f"  EB vs E over {n_}: whole decode {100 * (rw - 1):+.1f} %, median cycle {100 * (cyc['EB'] / cyc['E'] - 1):+.2f} %")
open("data/e5/e5_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? ($D/e5_report.txt)"
git add $D data/r57/e5h1_256_* data/r57/e5h2_256_* data/r57/e5hb1_256_* data/r57/e5hb2_256_* data/r57/e5m1_256_* data/r57/e5m2_256_* \
  data/r57/e5mb1_256_* data/r57/e5mb2_256_* 2>/dev/null
> (a quote from the project owner, kept private)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011DvdV6eUB4NgBbVkyVFPrT"
touch $D/DONE
say "E5 end"
