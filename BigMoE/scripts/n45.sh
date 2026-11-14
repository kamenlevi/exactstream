#!/bin/bash
# Sat 7 Nov 22:0x (the orchestrator 22:0x: the paper's missing 45 GB measurements, before CERT 01:30): N45, on S5''s committed snapshot
# (data/r57/snap/s5p, ref2 1f85c231, the same code E5 ran), C 256, one process per arm, quiet.
#   EB = the EXACT default (R57_PRESET=45 + NAX6-N) + KEEP + wired + the Sunday bundle (E5's EB arm)
#   L  = S5''s lossy config (v1' + wired + the bundle; QOM on humaneval / mbpp only, so chat / math run v1''s exact-acceptance profile)
# 1. CHAT + MATH SPEED: 12 IFEval, then 12 math (the first 12 of each in NAX6-N's AR file, EvalSuite results/ref2tnax), CAP 256,
#    EB / L / L / EB per set. WHOLE DECODE (headline), full period, END TO END, lifetime peak; every run vs NAX6-N's one-token AR
#    (EB: the exact proof; L: information).
# 2. PREFILL, EXACT: agent/edit10 (1565 tokens), agent/write13 (4730), agent/write11 (7932); CAP 16; EB x 2 passes: prefill wall (TTFT,
#    the screen's pfs: prefill + cache / drafter setup + the first token) and prefill tok/s = P / pfs.
# 3. PREFILL, LOSSY: the same 3 prompts, L x 2 passes.
# usage: NOTAFTER=hh:mm NOWAIT45=1 GAP0=0 GAP=0 MARGIN=4 bash scripts/turn45.sh 33 60 "..." /bin/bash scripts/n45.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "${N45_ROOT:-$HOME/BigMoE}" || exit 1
D=data/n45; mkdir -p $D
LOG="$PWD/$D/n45.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
PY=${N45_PY:-$HOME/BigMoE/.venv/bin/python}
SD=data/r57/snap/s5p
git diff --quiet HEAD -- "$SD" data/r57/snap/frozen_s5p data/r57/snap/c45_runner_s5p.sh 2>/dev/null || { say "the S5' snapshot differs from its commit: not run"; exit 2; }
SNAP=$(git log -1 --format=%h -- "$SD/r57_screen_s5p.py")
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
A=${N45_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2: not run"; exit 0; }
[ -n "$N45_DRYRUN" ] || sleep 20
A=${N45_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2 at the 20 s look-again: not run"; exit 0; }
BUN="BIGMOE_PAUSE_EVERY_MS=50,R57_S0CARRY=1,REF2_SB_FIRST=1,REF2_PLE_LATE=1,REF2_COMMIT_SUB=9,R57_DQ1=1"
EB="R57_NAX=1,REF2_KEEP_CACHES=1,R57_WIRED=1,$BUN"
L=$(grep -v '^#' data/stack/s5p_env.txt | grep . | tail -1 | sed 's/,R57_QOM_SETS=humaneval,mbpp$//')
echo "$L" | tr ',' '\n' | grep -qx 'REF2_COMMIT_SUB=9' || { say "s5p_env is not S5' with the bundle: not run"; exit 2; }
"$PY" -I - "$D/ar_ref_tokens.json" "$D/keys.txt" <<'EOF' || { say "AR reference file: failed"; exit 2; }
import json, os, sys
rows = [json.loads(l) for l in open(os.path.expanduser("~/EvalSuite/results/ref2tnax/decode.jsonl"))]
ki = [d["key"] for d in rows if d["set"] == "ifeval"][:12]; km = [d["key"] for d in rows if d["set"] == "math"][:12]
by = {d["key"]: d for d in rows}
json.dump({k: {"gate+t": [by[k]["completion_ids"]], "n_prompt": by[k].get("n_prompt")} for k in ki + km}, open(sys.argv[1], "w"))
open(sys.argv[2], "w").write(",".join(ki) + "\n" + ",".join(km) + "\n")
print(f"AR reference: {len(ki)} IFEval + {len(km)} math")
EOF
KI=$(sed -n 1p $D/keys.txt); KMA=$(sed -n 2p $D/keys.txt)
KP="agent/edit10,agent/write13,agent/write11"
[ -n "$N45_DRYRUN" ] && { echo "dry run: snapshot $SNAP; EB $(echo "$EB" | tr ',' '\n' | grep -c =) K=V, L $(echo "$L" | tr ',' '\n' | grep -c =) K=V; IFEval $(echo "$KI" | tr ',' '\n' | grep -c .), math $(echo "$KMA" | tr ',' '\n' | grep -c .)"; exit 0; }
RUN="/bin/bash data/r57/snap/c45_runner_s5p.sh"
export C45_OUT="$PWD/$D" C45_SWAP0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}') C45_ALARM=2400 \
  MG=4 C45_DEADLINE=$(date -v+170M +%H%M) C45_SCREEN="$SD/r57_screen_s5p.py" R57_QOM_SETS=humaneval,mbpp R57_MODMAN=1
rm -f $D/DONE
say "N45 start (pid $$, NO-SWAP $A GB, C 256, snapshot $SNAP)"
for a in "nie1:$EB:$KI" "nil1:$L:$KI" "nil2:$L:$KI" "nie2:$EB:$KI" "nme1:$EB:$KMA" "nml1:$L:$KMA" "nml2:$L:$KMA" "nme2:$EB:$KMA"; do
  t=${a%%:*}; r=${a#*:}; e=${r%:*}; k=${r##*:}; rm -f data/r57/${t}_256_* $D/${t}_256.txt
  R57_KEYS=$k R57_REFTOK="$PWD/$D/ar_ref_tokens.json:gate+t" C45_ARMS="${t}_256:256:$e" $RUN
  say "$t rc=$?"
done
for a in "npe1:$EB" "npe2:$EB" "npl1:$L" "npl2:$L"; do
  t=${a%%:*}; e=${a#*:}; rm -f data/r57/${t}_256_* $D/${t}_256.txt
  R57_KEYS=$KP C45_CAP=16 C45_REFTOK= C45_ARMS="${t}_256:256:$e" $RUN
  say "$t rc=$?"
done
"$PY" -I - "$(date +%F)" "${N45_FANLOG:-$HOME/Library/Logs/fanwatch.log}" "$SNAP" >> "$LOG" 2>&1 <<'EOF'
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
for ln in open("data/n45/c45.log"):
    m = re.match(r"(\d\d:\d\d:\d\d) (run|done) (\S+)", ln)
    if m:
        arm_t.setdefault(m.group(3), {})[m.group(2)] = D_.datetime.fromisoformat(f"{day} {m.group(1)}")
    m = re.search(r"done (\S+) rc=\d+:.*lifetime peak ([\d.]+) GB", ln)
    if m:
        peak[m.group(1)] = float(m.group(2))
hit = {t for t, v in arm_t.items() if "run" in v and "done" in v and any(a < v["done"] and b > v["run"] for a, b in win)}
AR = json.load(open("data/n45/ar_ref_tokens.json"))
runs, toks, live, miss = {}, {}, 0, []
ALL = ("nie1", "nil1", "nil2", "nie2", "nme1", "nml1", "nml2", "nme2", "npe1", "npe2", "npl1", "npl2")
for t in ALL:
    try:
        R = {r["key"]: r for r in json.load(open(f"data/r57/{t}_256_runs.json"))}
        for c in json.load(open(f"data/r57/{t}_256_checks.json")):
            r = R.get(c["key"])
            if r is None:
                continue
            dec, c0 = r["n"] / r["tps"], c.get("c0") or []
            r["fp"] = sum(c["kept"][2:]) / (dec - c0[2] / 1e3) if len(c0) > 2 and dec > c0[2] / 1e3 else None
            r["e2e"] = r["n"] / (dec + r["pfs"]) if r.get("pfs") is not None else None
        runs[t] = R
        toks[t] = {k: list(v.values())[0][0] for k, v in json.load(open(f"data/r57/{t}_256_tokens.json")).items()}
        live += sum(1 for _, f in json.load(open(f"data/r57/{t}_256_modules.json")) if "/BigMoE/data/r57/snap/" not in f)
    except (OSError, ValueError, KeyError):
        miss.append(t); runs[t], toks[t] = {}, {}
ok = lambda t, k: k in runs[t] and "PAUSED" not in runs[t][k].get("bad", "") and f"{t}_256" not in hit
def pm(ps, f):
    out = {}
    for k in set().union(*[runs[t] for t in ps]):
        v = [f(runs[t][k]) for t in ps if ok(t, k) and f(runs[t][k]) is not None]
        if v:
            out[k] = sum(v) / len(v)
    return out
def areq(ts):
    n = e = 0
    for t in ts:
        for k, o in toks[t].items():
            r = AR.get(k, {}).get("gate+t", [None])[0]
            if r is None:
                continue
            m = min(len(o), len(r)); n += 1; e += m > 0 and o[:m] == r[:m]
    return e, n
L = [f"N45 (data/n45; S5' snapshot {snap}; C 256; quiet): freeze windows {[(a.strftime('%H:%M:%S'), b.strftime('%H:%M:%S')) for a, b in win]}; "
     f"dropped for a freeze: {sorted(hit) or 'none'}; arms missing: {miss or 'none'}; live outside modules: {live}",
     "1. CHAT + MATH SPEED (CAP 256; EB = exact default + KEEP + wired + bundle; L = S5''s lossy config, chat / math at v1''s exact acceptance):"]
for sname, P in (("12 IFEval", {"EB": ("nie1", "nie2"), "L": ("nil1", "nil2")}), ("12 math", {"EB": ("nme1", "nme2"), "L": ("nml1", "nml2")})):
    for b, ps in P.items():
        e, n = areq(ps)
        pas = ", ".join(f"{g([r['tps'] for k, r in runs[t].items() if ok(t, k)]):.1f}" for t in ps)
        L.append(f"  {sname:9s} {b:2s}: WHOLE DECODE {g(list(pm(ps, lambda r: r['tps']).values())):6.1f} tok/s ({pas}); FULL PERIOD "
                 f"{g(list(pm(ps, lambda r: r.get('fp')).values())):6.1f}; END TO END {g(list(pm(ps, lambda r: r.get('e2e')).values())):6.1f}; "
                 f"tokens/task {S.median(pm(ps, lambda r: r['n']).values() or [float('nan')]):.0f}; lifetime peak "
                 f"{max([peak.get(f'{t}_256', 0.0) for t in ps]) or float('nan'):.1f} GB; == NAX6-N AR {e} / {n}"
                 + (" (EXACT PROOF " + ("PASS" if n and e == n else "FAIL") + ")" if b == "EB" else " (information)"))
L.append("2-3. PREFILL (CAP 16; pfs = prefill + cache / drafter setup + the first token): per prompt P tokens -> TTFT s (passes) -> prefill tok/s")
for b, ps in (("EXACT EB", ("npe1", "npe2")), ("LOSSY L", ("npl1", "npl2"))):
    for k in ("agent/edit10", "agent/write13", "agent/write11"):
        v = [(runs[t][k]["P"], runs[t][k].get("pfs")) for t in ps if k in runs[t] and runs[t][k].get("pfs")]
        if v:
            P_ = v[0][0]; ts = [x[1] for x in v]
            L.append(f"  {b:8s} {k:14s}: P {P_:5d}; TTFT {S.mean(ts):6.2f} s ({', '.join(f'{x:.2f}' for x in ts)}); prefill {P_ / S.mean(ts):6.0f} tok/s; "
                     f"lifetime peak {max([peak.get(f'{t}_256', 0.0) for t in ps]) or float('nan'):.1f} GB")
        else:
            L.append(f"  {b:8s} {k:14s}: MISSING")
open("data/n45/n45_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? ($D/n45_report.txt)"
git add $D $(for t in nie1 nil1 nil2 nie2 nme1 nml1 nml2 nme2 npe1 npe2 npl1 npl2; do echo data/r57/${t}_256_*; done) 2>/dev/null
> (a quote from the project owner, kept private)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011DvdV6eUB4NgBbVkyVFPrT"
touch $D/DONE
say "N45 end"
