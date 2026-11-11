#!/bin/bash
# Sat 7 Nov 15:xx (NewIdeas PLAN 220, its exact-safe line): the engine A/B for an exact-safe lever, reusable.
# usage: XS_NAME=<short tag, e.g. p1a> XS_KEYS="K=V,K=V" [XS_WHY="..."] bash scripts/xsafe_ab.sh   (or via turn45)
# 0. A NEW deep-frozen snapshot (scripts/deep_freeze.sh $XS_NAME: Leap's levers live in ~/Leap/scripts), committed first.
# 1. EXACT bits (the NAX6-N exact env as S2c / GDNB: R57_PRESET=45 defaults + R57_NAX=1, no R57_FAST; C 224; HumanEval/68,
#    HumanEval/16, ifeval/1220; CAP 64; R57_LOGHASH=1): e0 = off, e1 = + XS_KEYS. BITS = the verify-logits hash on every check of
#    the 3 prompts AND tokens 3 / 3.
# 2. Engine A/B at C 256, S1's 12 HumanEval, one process per arm, B / X / X / B: B = cert_env's v1' + wired, X = B + XS_KEYS.
# LINE (NewIdeas' exact-safe line, fixed here before any arming): PASS = median cycle (median of the prompts' median cycles,
#   pair-mean) at least 1 % lower AND BITS; KILL = cycle >= 0.5 % worse; else BETWEEN. Beside: ms per check, whole decode, full
#   period, check 0, kept per check, peaks, and ref2's counters that X engaged (COUNTS > 0 in X, 0 in B), per arm.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "${XS_ROOT:-$HOME/BigMoE}" || exit 1
N=${XS_NAME:?XS_NAME}; XK=${XS_KEYS:?XS_KEYS}
echo "$XK" | tr ',' '\n' | grep -vq '=' && { echo "XS_KEYS has a token without '='"; exit 2; }
D=data/xs_$N; mkdir -p "$D"
LOG="$PWD/$D/xs.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
PY=${XS_PY:-$HOME/BigMoE/.venv/bin/python}
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
A=${XS_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2: C 256 does not fit, not run"; exit 0; }
[ -n "$XS_DRYRUN" ] || sleep 20
A=${XS_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2 at the 20 s look-again: not run"; exit 0; }
FR=$(bash scripts/deep_freeze.sh "$N") || { say "freeze failed: $FR"; exit 2; }
SD=data/r57/snap/$N
B=$(grep -v '^#' data/stack/cert_env.txt | grep . | tail -1 | sed 's/,R57_QOM_SETS=humaneval,mbpp$//')
echo "$B" | tr ',' '\n' | grep -qx 'R57_WIRED=1' || { say "cert_env is not v1' + wired: not run"; exit 2; }
X="$B,$XK"
EX="R57_NAX=1,R57_REFTOK=,R57_LOGHASH=1"
[ -n "$XS_DRYRUN" ] && { echo "dry run $N: $FR; B $(echo "$B" | tr ',' '\n' | grep -c =) K=V, X $(echo "$X" | tr ',' '\n' | grep -c =) K=V; keys $XK"; exit 0; }
git add "data/r57/snap/frozen_$N" ":!data/r57/snap/frozen_$N/**/__pycache__/**" "$SD" ":!$SD/__pycache__/**" "data/r57/snap/c45_runner_$N.sh" \
> (a quote from the project owner, kept private)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011DvdV6eUB4NgBbVkyVFPrT" || { say "snapshot commit failed: not run"; exit 3; }
SNAP=$(git rev-parse --short HEAD)
RUN="/bin/bash data/r57/snap/c45_runner_$N.sh"
export C45_OUT="$PWD/$D" C45_SWAP0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}') C45_ALARM=1200 \
  MG=4 C45_DEADLINE=$(date -v+50M +%H%M) C45_SCREEN="$SD/r57_screen_$N.py" C45_REFTOK= R57_QOM_SETS=humaneval,mbpp R57_MODMAN=1
K="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79"
K="$K,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,humaneval/HumanEval/56,humaneval/HumanEval/141"
K="$K,humaneval/HumanEval/101,humaneval/HumanEval/63"
for t in e0 e1; do rm -f data/r57/${N}${t}_224_* $D/${N}${t}_224.txt; done
for t in b1 x1 x2 b2; do rm -f data/r57/${N}${t}_256_* $D/${N}${t}_256.txt; done
rm -f $D/DONE
say "xsafe $N start (pid $$, NO-SWAP $A GB, snapshot $SNAP): X = B + $XK ${XS_WHY:+($XS_WHY)}"
for a in "e0:$EX" "e1:$EX,$XK"; do
  R57_KEYS="humaneval/HumanEval/68,humaneval/HumanEval/16,ifeval/1220" C45_CAP=64 C45_ARMS="$N${a%%:*}_224:224:${a#*:}" $RUN
  say "$N${a%%:*} rc=$?"
done
for a in "b1:$B" "x1:$X" "x2:$X" "b2:$B"; do
  R57_KEYS=$K C45_ARMS="$N${a%%:*}_256:256:${a#*:}" $RUN
  say "$N${a%%:*} rc=$?"
done
"$PY" -I - "$N" "$D" "$(date +%F)" "${XS_FANLOG:-$HOME/Library/Logs/fanwatch.log}" "$XK" "$SNAP" >> "$LOG" 2>&1 <<'EOF'
import datetime as D_, json, math, re, statistics as S, sys
N, Dd, day, fanlog, XK, snap = sys.argv[1:7]
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
for ln in open(f"{Dd}/c45.log"):
    m = re.match(r"(\d\d:\d\d:\d\d) (run|done) (\S+)", ln)
    if m:
        arm_t.setdefault(m.group(3), {})[m.group(2)] = D_.datetime.fromisoformat(f"{day} {m.group(1)}")
    m = re.search(r"done (\S+) rc=\d+:.*lifetime peak ([\d.]+) GB", ln)
    if m:
        peak[m.group(1)] = float(m.group(2))
hit = {t for t, v in arm_t.items() if "run" in v and "done" in v and any(a < v["done"] and b > v["run"] for a, b in win)}
def load(t, c):
    try:
        R = {r["key"]: r for r in json.load(open(f"data/r57/{t}_{c}_runs.json"))}
        tk = {k: list(v.values())[0][0] for k, v in json.load(open(f"data/r57/{t}_{c}_tokens.json")).items()}
        return R, tk
    except (OSError, ValueError, KeyError):
        return {}, {}
def counts(t, c):
    try:
        return json.load(open(f"data/r57/{t}_{c}_counts.json"))
    except (OSError, ValueError):
        return {}
L = [f"xsafe {N} (snapshot {snap}): X = B + {XK}; freeze windows {[(a.strftime('%H:%M:%S'), b.strftime('%H:%M:%S')) for a, b in win]}; "
     f"passes dropped for a freeze: {sorted(hit) or 'none'}"]
(r0, k0), (r1, k1) = load(f"{N}e0", 224), load(f"{N}e1", 224)
ks = sorted(set(k0) & set(k1))
teq = sum(1 for k in ks if k0[k] == k1[k])
hs = [(a == b) for k in ks for a, b in zip(r0[k].get("lh") or [], r1[k].get("lh") or [])]
hn = sum(len(r0[k].get("lh") or []) for k in ks)
bits = len(ks) == 3 and teq == 3 and bool(hs) and all(hs) and len(hs) == hn
c0e, c1e = counts(f"{N}e0", 224), counts(f"{N}e1", 224)
eng_e = {k: v for k, v in c1e.items() if v and not c0e.get(k)}
L.append(f"BITS (exact env, C 224, CAP 64, e1 vs e0): {'PASS' if bits else 'FAIL'}: tokens equal {teq} / {len(ks)}; verify-logits hash "
         f"equal {sum(hs)} / {len(hs)} checks (e0 has {hn}); engaged in e1 only: {eng_e or 'none'}; peaks {peak.get(f'{N}e0_224')} / {peak.get(f'{N}e1_224')} GB")
P = {"B": (f"{N}b1", f"{N}b2"), "X": (f"{N}x1", f"{N}x2")}
runs = {}
for t in sum(P.values(), ()):
    R, _ = load(t, 256)
    try:
        for c in json.load(open(f"data/r57/{t}_256_checks.json")):
            r = R.get(c["key"])
            if r is None:
                continue
            dec, c0 = r["n"] / r["tps"], c.get("c0") or []
            r["fp"] = sum(c["kept"][2:]) / (dec - c0[2] / 1e3) if len(c0) > 2 and dec > c0[2] / 1e3 else None
            r["ck0"] = c["bld"][0] + c["wt"][0] if c["bld"] else None
            r["kpc"] = S.mean(c["kept"][2:]) if len(c["kept"]) > 2 else None
    except (OSError, ValueError):
        pass
    runs[t] = R
ok = lambda t, k: k in runs[t] and "PAUSED" not in runs[t][k].get("bad", "") and f"{t}_256" not in hit
def pm(ps, f):
    out = {}
    for k in set().union(*[runs[t] for t in ps]):
        v = [f(runs[t][k]) for t in ps if ok(t, k) and f(runs[t][k]) is not None]
        if v:
            out[k] = sum(v) / len(v)
    return out
def ratio(a, b):
    ks_ = [k for k in a if k in b]
    return (g([a[k] / b[k] for k in ks_]), len(ks_)) if ks_ else (float("nan"), 0)
cyc = {b: S.median(pm(ps, lambda r: r.get("cyc")).values() or [float("nan")]) for b, ps in P.items()}
cb = {k: sum(counts(t, 256).get(k, 0) for t in P["B"]) for k in set().union(*[counts(t, 256) for t in sum(P.values(), ())])}
cx = {k: sum(counts(t, 256).get(k, 0) for t in P["X"]) for k in cb}
eng = {k: v for k, v in cx.items() if v and not cb.get(k)}
for b, ps in P.items():
    L.append(f"{b}: median cycle {cyc[b]:.2f} ms; WHOLE DECODE {g(list(pm(ps, lambda r: r['tps']).values())):6.1f}; full period "
             f"{g(list(pm(ps, lambda r: r.get('fp')).values())):6.1f}; check 0 {S.median(pm(ps, lambda r: r.get('ck0')).values() or [float('nan')]):.0f} ms; "
             f"kept {g(list(pm(ps, lambda r: r.get('kpc')).values())):.2f}; peak {max([peak.get(f'{t}_256', 0.0) for t in ps]) or float('nan'):.1f} GB")
rc = cyc["X"] / cyc["B"]
rw, n_ = ratio(pm(P["X"], lambda r: r["tps"]), pm(P["B"], lambda r: r["tps"]))
rf, _ = ratio(pm(P["X"], lambda r: r.get("fp")), pm(P["B"], lambda r: r.get("fp")))
v = "MISSING" if n_ == 0 or math.isnan(rc) else "KILL" if rc >= 1.005 else "PASS" if rc <= 0.99 and bits else "BETWEEN"
L.append(f"Verdict xsafe {N} (pre-registered, NewIdeas' exact-safe line): {v}: median cycle {100 * (rc - 1):+.2f} % ({cyc['X'] - cyc['B']:+.2f} ms "
         f"per check) vs B, BITS {'PASS' if bits else 'FAIL'} (PASS = cycle <= -1 % AND BITS; KILL = cycle >= +0.5 %). Beside: whole decode "
         f"{100 * (rw - 1):+.1f} %, full period {100 * (rf - 1):+.1f} % over {n_}; engaged in X only (ref2 COUNTS, both passes): {eng or 'none'}")
open(f"{Dd}/xs_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? ($D/xs_report.txt)"
git add "$D" data/r57/${N}e0_224_* data/r57/${N}e1_224_* data/r57/${N}b1_256_* data/r57/${N}b2_256_* data/r57/${N}x1_256_* \
  data/r57/${N}x2_256_* 2>/dev/null
> (a quote from the project owner, kept private)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011DvdV6eUB4NgBbVkyVFPrT"
touch $D/DONE
say "xsafe $N end"
