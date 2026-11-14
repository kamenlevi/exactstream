#!/bin/bash
# Sat 7 Nov 14:4x (NewIdeas 15:3x-labelled msg, PLAN 220 #2; Leap TABLE_350 c69e024: v1' has REF2_GDN_LAZY=1 but no R57_GDNB, so
# the lazy GDN commit is inert): GDNB, the exact-safe engine A/B of R57_GDNB=1 (ref2 gdn_body, one-dispatch GDN body; GDNB-2
# bitwise PASS Wed; gdnbody / ref2 changed since, so the bitwise check is re-run here). On S5's committed snapshot (d8eb2e4).
# 0. EXACT bitwise (NAX6-N exact env as S2c: R57_PRESET=45 defaults + R57_NAX=1, no R57_FAST; C 224; HumanEval/68, HumanEval/16,
#    ifeval/1220; CAP 64; R57_LOGHASH=1): gx0 = REF2_GDN_LAZY=1; gx1 = gx0 + R57_GDNB=1 (G's form); gx2 = R57_GDNB=1 without the lazy
#    commit (G''s form). BITS(G) = gx1 == gx0 and BITS(G') = gx2 == gx0: the verify-logits hash on every check of the 3 prompts AND
#    tokens 3 / 3.
# 1. Engine A/B at C 256 on S1's 12 HumanEval, one process per arm, palindrome gb1 / gg1 / gp1 / gp2 / gg2 / gb2:
#    B = v1' + wired (cert_env's line); G = B + R57_GDNB=1; G' = G - REF2_GDN_LAZY=1 (Leap's suggestion; NewIdeas: run it if G lands
#    worse than -0.7 ms per check, run here at once to save a quiet slot)
# LINE (NewIdeas' exact-safe line, before arming), per arm X in {G, G'} vs B: PASS = median cycle (median of the prompts' median
#   cycles, pair-mean) at least 1 % lower AND BITS(X); KILL if the cycle is >= 0.5 % worse; else BETWEEN. Beside: delta ms per
#   check, whole decode, full period, check 0, kept per check, peaks. A PASS joins the Sunday-night exact-safe bundle (no new gate).
# usage: AFTER='[ -f data/t16he/DONE ]' NOTAFTER=hh:mm NOWAIT45=1 GAP0=0 GAP=0 MARGIN=4 bash scripts/turn45.sh 33 25 "..." /bin/bash scripts/gdnb_ab.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "${GB_ROOT:-$HOME/BigMoE}" || exit 1
mkdir -p data/gdnb
LOG="$PWD/data/gdnb/gdnb.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
PY=${GB_PY:-$HOME/BigMoE/.venv/bin/python}
SD=data/r57/snap/s5
git diff --quiet HEAD -- "$SD" data/r57/snap/frozen_s5 data/r57/snap/c45_runner_s5.sh 2>/dev/null || { say "the S5 snapshot differs from its commit: not run"; exit 2; }
SNAP=$(git log -1 --format=%h -- "$SD/r57_screen_s5.py")
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
A=${GB_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2: C 256 does not fit, not run"; exit 0; }
[ -n "$GB_DRYRUN" ] || sleep 20
A=${GB_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2 at the 20 s look-again: not run"; exit 0; }
RUN="/bin/bash data/r57/snap/c45_runner_s5.sh"
export C45_OUT="$PWD/data/gdnb" C45_SWAP0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}') C45_ALARM=1200 \
  MG=4 C45_DEADLINE=$(date -v+50M +%H%M) C45_SCREEN="$SD/r57_screen_s5.py" C45_REFTOK= R57_QOM_SETS=humaneval,mbpp R57_MODMAN=1
K="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79"
K="$K,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,humaneval/HumanEval/56,humaneval/HumanEval/141"
K="$K,humaneval/HumanEval/101,humaneval/HumanEval/63"
B=$(grep -v '^#' data/stack/cert_env.txt | grep . | tail -1 | sed 's/,R57_QOM_SETS=humaneval,mbpp$//')
echo "$B" | tr ',' '\n' | grep -qx 'R57_WIRED=1' && echo "$B" | tr ',' '\n' | grep -qx 'REF2_GDN_LAZY=1' || { say "cert_env is not v1' + wired with the lazy commit: not run"; exit 2; }
echo "$B" | tr ',' '\n' | grep -q '^R57_GDNB=' && { say "cert_env already carries R57_GDNB: not run"; exit 2; }
G="$B,R57_GDNB=1"
GP=$(echo "$G" | tr ',' '\n' | grep -vx 'REF2_GDN_LAZY=1' | paste -sd, -)
EX="R57_NAX=1,R57_REFTOK=,R57_LOGHASH=1"
[ -n "$GB_DRYRUN" ] && { echo "dry run: snapshot $SNAP; B $(echo "$B" | tr ',' '\n' | grep -c =), G $(echo "$G" | tr ',' '\n' | grep -c =), G' $(echo "$GP" | tr ',' '\n' | grep -c =) K=V"; exit 0; }
for t in gx0 gx1 gx2; do rm -f data/r57/${t}_224_* data/gdnb/${t}_224.txt; done
for t in gb1 gg1 gp1 gp2 gg2 gb2; do rm -f data/r57/${t}_256_* data/gdnb/${t}_256.txt; done
rm -f data/gdnb/DONE
say "GDNB A/B start (pid $$, NO-SWAP $A GB, snapshot $SNAP)"
for a in "gx0:$EX,REF2_GDN_LAZY=1" "gx1:$EX,REF2_GDN_LAZY=1,R57_GDNB=1" "gx2:$EX,R57_GDNB=1"; do
  R57_KEYS="humaneval/HumanEval/68,humaneval/HumanEval/16,ifeval/1220" C45_CAP=64 C45_ARMS="${a%%:*}_224:224:${a#*:}" $RUN
  say "${a%%:*} rc=$?"
done
for a in "gb1:$B" "gg1:$G" "gp1:$GP" "gp2:$GP" "gg2:$G" "gb2:$B"; do
  R57_KEYS=$K C45_ARMS="${a%%:*}_256:256:${a#*:}" $RUN
  say "${a%%:*} rc=$?"
done
"$PY" -I - "$(date +%F)" "${GB_FANLOG:-$HOME/Library/Logs/fanwatch.log}" >> "$LOG" 2>&1 <<'EOF'
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
for ln in open("data/gdnb/c45.log"):
    m = re.match(r"(\d\d:\d\d:\d\d) (run|done) (\S+)", ln)
    if m:
        arm_t.setdefault(m.group(3), {})[m.group(2)] = D.datetime.fromisoformat(f"{day} {m.group(1)}")
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
L = [f"GDNB A/B (R57_GDNB=1 on v1' + wired), data/gdnb; freeze windows {[(a.strftime('%H:%M:%S'), b.strftime('%H:%M:%S')) for a, b in win]}; "
     f"passes dropped for a freeze: {sorted(hit) or 'none'}"]
(r0, k0) = load("gx0", 224)
bits = {}
for name, t in (("G", "gx1"), ("G'", "gx2")):
    r1, k1 = load(t, 224)
    ks = sorted(set(k0) & set(k1))
    teq = sum(1 for k in ks if k0[k] == k1[k])
    hs = [(a == b) for k in ks for a, b in zip(r0[k].get("lh") or [], r1[k].get("lh") or [])]
    hn = sum(len(r0[k].get("lh") or []) for k in ks)
    bits[name] = len(ks) == 3 and teq == 3 and hs and all(hs) and len(hs) == hn
    L.append(f"BITS({name}) (exact env, C 224, CAP 64, {t} vs gx0): {'PASS' if bits[name] else 'FAIL'}: tokens equal {teq} / {len(ks)}; "
             f"verify-logits hash equal {sum(hs)} / {len(hs)} checks (gx0 has {hn}); peaks {peak.get('gx0_224')} / {peak.get(t + '_224')} GB")
P = {"B": ("gb1", "gb2"), "G": ("gg1", "gg2"), "G'": ("gp1", "gp2")}
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
    ks = [k for k in a if k in b]
    return (g([a[k] / b[k] for k in ks]), len(ks)) if ks else (float("nan"), 0)
cyc = {b: S.median(pm(ps, lambda r: r.get("cyc")).values() or [float("nan")]) for b, ps in P.items()}
for b, ps in P.items():
    L.append(f"{b:2s}: median cycle {cyc[b]:.2f} ms; WHOLE DECODE {g(list(pm(ps, lambda r: r['tps']).values())):6.1f}; full period "
             f"{g(list(pm(ps, lambda r: r.get('fp')).values())):6.1f}; check 0 {S.median(pm(ps, lambda r: r.get('ck0')).values() or [float('nan')]):.0f} ms; "
             f"kept {g(list(pm(ps, lambda r: r.get('kpc')).values())):.2f}; peak {max([peak.get(f'{t}_256', 0.0) for t in ps]) or float('nan'):.1f} GB")
for x in ("G", "G'"):
    rc = cyc[x] / cyc["B"]
    rw, n_ = ratio(pm(P[x], lambda r: r["tps"]), pm(P["B"], lambda r: r["tps"]))
    rf, _ = ratio(pm(P[x], lambda r: r.get("fp")), pm(P["B"], lambda r: r.get("fp")))
    v = ("MISSING" if n_ == 0 or math.isnan(rc) else "KILL" if rc >= 1.005 else "PASS" if rc <= 0.99 and bits[x] else "BETWEEN")
    L.append(f"Verdict GDNB {x} (pre-registered, NewIdeas' exact-safe line): {v}: median cycle {100 * (rc - 1):+.2f} % "
             f"({cyc[x] - cyc['B']:+.2f} ms per check) vs B, BITS {'PASS' if bits[x] else 'FAIL'} (PASS = cycle <= -1 % AND BITS; "
             f"KILL = cycle >= +0.5 %). Beside: whole decode {100 * (rw - 1):+.1f} %, full period {100 * (rf - 1):+.1f} % over {n_}")
open("data/gdnb/gdnb_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? (data/gdnb/gdnb_report.txt)"
git add data/gdnb data/r57/gx0_224_* data/r57/gx1_224_* data/r57/gx2_224_* data/r57/gb1_256_* data/r57/gb2_256_* data/r57/gg1_256_* \
  data/r57/gg2_256_* data/r57/gp1_256_* data/r57/gp2_256_* 2>/dev/null
> (a quote from the project owner, kept private)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011DvdV6eUB4NgBbVkyVFPrT"
touch data/gdnb/DONE
say "GDNB A/B end"
