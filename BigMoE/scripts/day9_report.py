> (a quote from the project owner, kept private)
files (data/r57/<tag>_{runs,checks,tokens,warmsplit}.json) of the DAY9 arms and prints one block per group with its line:
- B2 (condition 9): C 300 / 320 vs C 256 (tonight's c1d_256, else C50's c45l_256 text): steady tok/s, lifetime peak, SSD MB
  and misses per kept token, every token == REF2-T
- chat switch: (a) every token == D-23d's (the REF2-T loop), (b) the switched process's lifetime peak <= 45 GB with both
  drafters resident, (c) chat tok/s B vs A at C 256 (time-weighted, geo-mean, per language) beside D-23d's C 192
- C1 (condition 4): NAX6 vs the default: steady tok/s, check ms, regular-check per-row slope (check ms ~ rows + misses),
  determinism (NAX vs NAX again); the warm split's per-row GPU slope default vs NAX (kill: < 1.2x on the row term)
- warm split (NewIdeas' Monday-night 1): warm ms per row, intercept, cold 16 - warm 16 (the SSD's exposed share)
- P on L (Monday-night 2): c45p_256 vs c1d_256: misses + MB per kept token, steady tok/s, tokens
- A6 (condition 7): per arm and prompt, the first cycle's excess over the median of checks 2-8, split SSD / draft / other;
  rep's copy 1 vs copy 2; pre-loop costs; line: the control's split names a cause >= 30 % (else KILL)
- SSD wait (the per-row lead's step 3): demand SSD ms per check regressed on reads (misses) vs on MB, every logged check
usage: .venv/bin/python scripts/day9_report.py [SUFFIX=256] [OUT=data/day9/day9_report.txt]"""
import glob, json, math, os, re, sys
import numpy as np

R = os.path.expanduser("~/BigMoE/data/r57")
SUF = sys.argv[1] if len(sys.argv) > 1 else "256"
OUTF = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/BigMoE/data/day9/day9_report.txt")
L = []
def say(s=""):
    print(s); L.append(s)
def J(tag, kind):
    p = os.path.join(R, f"{tag}_{kind}.json")
    try:
        return json.load(open(p))
    except Exception:
        return None
med = lambda v: float(np.median(v)) if len(v) else float("nan")
def runs(tag):
    return [r for r in (J(tag, "runs") or []) if not r["bad"] and not r["cut"]]
def peak(tag):
    return max((r["mem"][6] for r in (J(tag, "runs") or [])), default=float("nan"))
def agg(tag):
    rs = runs(tag)
    if not rs:
        return None
    n = sum(r["n"] for r in rs)
    return dict(n=len(rs), tpsw=med([r["tpsw"] for r in rs]), tps=med([r["tps"] for r in rs]), ck=med([r["ck"] for r in rs]),
                miss=med([r["miss"] for r in rs]), mb=sum(r["mb"] * r["n"] for r in rs) / n,
                mpk=sum(r["mpk"] * r["n"] for r in rs) / n, peak=peak(tag))
def ref(name):
    return json.load(open(os.path.join(R, name)))
REF45, REFCH = ref("s45t_B_tokens.json"), ref("chat23_ref_tokens.json")
def tok_check(tag, refd):
    """(prompts equal, prompts compared, keys that differ) of every run's tokens vs the reference prefix"""
    T, eq, nn, bad = J(tag, "tokens") or {}, 0, 0, []
    for k, d in T.items():
        want = refd.get(k, {}).get("gate+t", [None])[0]
        for outs in d.values():
            for o in outs:
                nn += 1
                if want is not None and o[:min(len(o), len(want))] == want[:min(len(o), len(want))]:
                    eq += 1
                else:
                    bad.append(k)
    return eq, nn, sorted(set(bad))
def row(tag, a):
    return (f"{tag:10s} steady {a['tpsw']:5.1f} tok/s, whole {a['tps']:5.1f}, check {a['ck']:6.1f} ms, misses/check {a['miss']:5.0f}, "
            f"per kept token {a['mb']:5.1f} MB / {a['mpk']:5.1f} misses, lifetime peak {a['peak']:.1f} GB ({a['n']} runs)")
def slope(tags):
    """regular checks (from the 3rd, not warm-split samples): check ms ~ a + b * rows + c * misses -> (b, c, n)"""
    X, y = [], []
    for t in tags:
        for d in J(t, "checks") or []:
            if d.get("paused"):
                continue
            ws = d.get("ws") or [0] * len(d["ck"])
            for i in range(2, len(d["ck"])):
                if not ws[i]:
                    X.append([1.0, d["rows"][i], d["miss"][i]]); y.append(d["ck"][i])
    if len(y) < 10:
        return None
    b = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)[0]
    return float(b[1]), float(b[2]), len(y)
def warm(tag):
    S = J(tag, "warmsplit") or []
    sl, ic, ex, wm = [], [], [], []
    for d in S:
        for s in d["samples"]:
            x, y = np.array([w[0] for w in s["warm"]], float), np.array([w[1] for w in s["warm"]], float)
            b, a = np.polyfit(x, y, 1)
            sl.append(b); ic.append(a); wm.append(sum(w[2] for w in s["warm"]))
            ex.append(s["cold"][1] - next(w[1] for w in s["warm"] if w[0] == s["cold"][0]))
    return dict(n=len(sl), slope=med(sl), ic=med(ic), ex=med(ex), wmiss=med(wm)) if sl else None

say(f"DAY9 report (BigMoE), C {SUF}; data {R}")
# ---------------------------------------------------------------- B2: C 300 / 320
say("\n== B2 (condition 9): C 300 / 320 with per-layer sizes vs C 256")
base = agg(f"c1d_{SUF}")
if base:
    say("  " + row(f"c1d_{SUF}", base))
for C in ("300", "320"):
    a = agg(f"c45l_{C}")
    if not a:
        say(f"  c45l_{C}: not run (skipped by its NO-SWAP gate, or not reached)"); continue
    e = tok_check(f"c45l_{C}", REF45)
    say("  " + row(f"c45l_{C}", a) + f"; tokens == REF2-T {e[0]} / {e[1]}"
        + (f"; vs C {SUF}: steady {100 * (a['tpsw'] / base['tpsw'] - 1):+.1f} %, MB per kept token "
           f"{100 * (a['mb'] / base['mb'] - 1):+.1f} %" if base else ""))
# ---------------------------------------------------------------- the chat switch
> (a quote from the project owner, kept private)
A, B = {r["key"]: r for r in runs(f"chA_{SUF}")}, {r["key"]: r for r in runs(f"chB_{SUF}")}
if A and B:
    ea, eb = tok_check(f"chA_{SUF}", REFCH), tok_check(f"chB_{SUF}", REFCH)
    same = all((J(f"chA_{SUF}", "tokens") or {}).get(k) == (J(f"chB_{SUF}", "tokens") or {}).get(k) for k in A)
    say(f"  (a) every token == D-23d's: A {ea[0]} / {ea[1]}, B {eb[0]} / {eb[1]}{' differ: ' + str(ea[2] + eb[2]) if ea[2] or eb[2] else ''}; "
        f"A == B on every prompt: {same}")
    pB = peak(f"chB_{SUF}")
    say(f"  (b) lifetime peak: A {peak(f'chA_{SUF}'):.1f} GB, B {pB:.1f} GB (both drafters resident) -> "
        f"{'PASS' if pB <= 45.0 else 'FAIL'} (<= 45 GB)")
    chat = [k for k in A if k in B and B[k]["type"] == "chat"]
    if chat:
        n = sum(A[k]["n"] for k in chat)
        tA, tB = sum(A[k]["n"] / A[k]["tps"] for k in chat), sum(B[k]["n"] / B[k]["tps"] for k in chat)
        geo = math.exp(np.mean([math.log(B[k]["tps"] / A[k]["tps"]) for k in chat]))
        D23 = {}
        for l in open(os.path.expanduser("~/TrainLab/data/d23c_runs.jsonl")):
            d = json.loads(l)
            if d["C"] == 192 and d["head"] == "b5" and d.get("pre") == "45+ooo" and not (d["cut"] or d["paused"] or d["invalid"]):
                D23.setdefault(d["key"], {})[d["arm"]] = d
        dk = [k for k in chat if {"dfl", "mtp15e"} <= set(D23.get(k, {}))]
        g192 = (sum(D23[k]["dfl"]["n"] / D23[k]["dfl"]["tok_s"] for k in dk)
                / sum(D23[k]["mtp15e"]["n"] / D23[k]["mtp15e"]["tok_s"] for k in dk) - 1) if dk else float("nan")
        say(f"  (c) chat ({len(chat)} prompts, {n} tokens): A {n / tA:.1f} -> B {n / tB:.1f} tok/s = {100 * (tA / tB - 1):+.1f} % "
            f"time-weighted, geo-mean {geo:.3f}, B faster on {sum(B[k]['tps'] > A[k]['tps'] for k in chat)} of {len(chat)}; "
            f"kept per check {np.mean([A[k]['kept'] for k in chat]):.2f} -> {np.mean([B[k]['kept'] for k in chat]):.2f}; "
            f"D-23d at C 192 on the same prompts: {100 * g192:+.1f} %")
        langs = {}
        for k in chat:
            langs.setdefault(re.sub(r"\d+$", "", k.split("/")[1]) if k.startswith("multi/") else "ifeval(en)", []).append(k)
        say("      per language: " + ", ".join(f"{g} {100 * (sum(A[k]['n'] / A[k]['tps'] for k in ks) / sum(B[k]['n'] / B[k]['tps'] for k in ks) - 1):+.0f} %"
                                          for g, ks in sorted(langs.items())))
        say("      MTP history build (in B's clock): median " + f"{1e3 * med([B[k]['hist'] for k in chat]):.0f} ms per prompt")
    ag = [k for k in A if k in B and B[k]["type"] != "chat"]
    if ag:
        say(f"  non-chat prompts (DFlash2 in both): " + ", ".join(f"{k} [{B[k]['type']}] {A[k]['tps']:.1f} -> {B[k]['tps']:.1f}" for k in ag))
else:
    say("  not run")
# ---------------------------------------------------------------- C1 + warm split + P on L
say("\n== C1 (condition 4): NAX6 live (ref2 tier_nax_ooo) vs the default; warm split; P on L")
for t in (f"c1d_{SUF}", f"c1k_{SUF}", f"c1n_{SUF}", f"c1n_{SUF}r", f"ws_{SUF}", f"wsn_{SUF}", f"c45p_{SUF}"):
    a = agg(t)
    say("  " + (row(t, a) if a else f"{t}: not run"))
k_ = agg(f"c1k_{SUF}")
if k_ and agg(f"c1d_{SUF}"):
    e = tok_check(f"c1k_{SUF}", REF45)
    d0 = agg(f"c1d_{SUF}")
    say(f"  K-83 vs the default: steady {100 * (k_['tpsw'] / d0['tpsw'] - 1):+.1f} %, check {100 * (k_['ck'] / d0['ck'] - 1):+.1f} %, "
        f"draft {med([r['drf'] for r in runs(f'c1d_{SUF}')]):.1f} -> {med([r['drf'] for r in runs(f'c1k_{SUF}')]):.1f} ms; tokens == "
        f"REF2-T {e[0]} / {e[1]} -> {'ADOPT (exact, not slower)' if e[0] == e[1] and k_['tpsw'] >= d0['tpsw'] * 0.99 else 'hold'}")
if agg(f"c1d_{SUF}"):
    e = tok_check(f"c1d_{SUF}", REF45)
    say(f"  default tokens == REF2-T: {e[0]} / {e[1]}")
T1, T2 = J(f"c1n_{SUF}", "tokens"), J(f"c1n_{SUF}r", "tokens")
if T1 and T2:
    det = all(T1[k] == T2.get(k) for k in T1)
    e = tok_check(f"c1n_{SUF}", REF45)
    say(f"  NAX determinism (process 1 == process 2 on every prompt): {det}; NAX tokens == REF2-T on {e[0]} / {e[1]} prompts "
        f"(a new reference: differences expected)")
d_, n_ = agg(f"c1d_{SUF}"), agg(f"c1n_{SUF}")
if d_ and n_:
    say(f"  NAX vs default: steady {100 * (n_['tpsw'] / d_['tpsw'] - 1):+.1f} %, check {100 * (n_['ck'] / d_['ck'] - 1):+.1f} %")
s_d, s_n = slope([f"c1d_{SUF}"]), slope([f"c1n_{SUF}", f"c1n_{SUF}r"])
if s_d and s_n:
    say(f"  regular-check per-row slope (check ms ~ rows + misses): default {s_d[0]:.2f} ms/row ({s_d[1]:.3f} ms/miss, {s_d[2]} checks), "
        f"NAX {s_n[0]:.2f} ms/row ({s_n[1]:.3f} ms/miss, {s_n[2]} checks)")
w_d, w_n = warm(f"ws_{SUF}"), warm(f"wsn_{SUF}")
for nm, w in (("default", w_d), ("NAX", w_n)):
    if w:
        say(f"  warm split {nm}: {w['n']} samples; warm slope {w['slope']:.2f} ms per row, intercept {w['ic']:.1f} ms; cold 16 - warm 16 "
            f"{w['ex']:.1f} ms; misses in the warm passes {w['wmiss']:.0f}")
if w_d and w_n:
    x = w_d["slope"] / w_n["slope"]
    say(f"  C1 verdict (kill: < 1.2x on the row term): warm per-row {w_d['slope']:.2f} -> {w_n['slope']:.2f} ms = {x:.2f}x -> "
        f"{'KILL' if x < 1.2 else 'PASS'}")
p_ = agg(f"c45p_{SUF}")
if p_ and d_:
    e = tok_check(f"c45p_{SUF}", REF45)
    say(f"  P on L vs L: misses per kept token {d_['mpk']:.1f} -> {p_['mpk']:.1f} ({100 * (p_['mpk'] / d_['mpk'] - 1):+.1f} %), MB "
        f"{100 * (p_['mb'] / d_['mb'] - 1):+.1f} %, steady {100 * (p_['tpsw'] / d_['tpsw'] - 1):+.1f} %; tokens == REF2-T {e[0]} / {e[1]}")
# ---------------------------------------------------------------- A6
def a6(sfx):
    tags = [t for t in ("a6c", "a6r", "a6k", "a6w", "a6p") if J(f"{t}_{sfx}", "checks")]
    if not tags:
        return
    say(f"\n== A6 (condition 7) at C {sfx}: first cycle after a long prefill, excess over the median of checks 2-8")
    ctl_share, E2E = [], {}                              # E2E (the orchestrator 11-02 16:3x): prefill end -> the 1st commit,
    for t in tags:                                       # pre-paid work (pools / dwin / dummy, run after the prefill) counted
        C, Rr = J(f"{t}_{sfx}", "checks"), J(f"{t}_{sfx}", "runs") or []
        e = tok_check(f"{t}_{sfx}", REFCH)
        say(f"  {t}_{sfx} (tokens == reference {e[0]} / {e[1]}):")
        for i, d in enumerate(C):
            if len(d["ck"]) < 4:
                continue
            b = slice(1, 8)
            md = {k: med(d[k][b]) for k in ("drf", "ck", "io", "bld", "miss")}
            ex = (d["drf"][0] - md["drf"]) + (d["ck"][0] - md["ck"])
            ssd, dr = d["io"][0] - md["io"], d["drf"][0] - md["drf"]
            oth = ex - ssd - dr
            sh = [x / ex if ex > 0 else float("nan") for x in (ssd, dr, oth)]
            if t == "a6c":
                ctl_share.append(max(sh))
            pre = Rr[i]["a6"] if i < len(Rr) else {}
            rep = ""
            if "rep1" in pre:
                c1 = pre["rep1"][0] + pre["rep1"][1]
                c2 = d["bld"][0] + d["wt"][0]
                rep = (f"; rep: copy 1 {c1:.0f} ms ({pre['rep1'][2]} misses, SSD {pre['rep1'][3]:.0f}) vs copy 2 {c2:.0f} ms "
                       f"({d['miss'][0]:.0f} misses) -> one-time {c1 - c2 - (pre['rep1'][3] - d['io'][0]):.0f} ms beyond SSD")
            say(f"    {d['key']}: excess {ex:.0f} ms = SSD {ssd:.0f} ({100 * sh[0]:.0f} %) + draft {dr:.0f} ({100 * sh[1]:.0f} %) + other "
                f"{oth:.0f} ({100 * sh[2]:.0f} %); 1st check {d['ck'][0]:.0f} vs {md['ck']:.0f} ms (build {d['bld'][0]:.0f} vs "
                f"{md['bld']:.0f}), misses {d['miss'][0]:.0f} vs {md['miss']:.0f}; pre-loop "
                f"{ {k: v for k, v in pre.items() if k != 'rep1'} }{rep}{'; PAUSED (timings void)' if d.get('paused') else ''}")
            pms = sum(v for k, v in pre.items() if k.endswith("_ms"))
            if t != "a6r":                               # rep's copy 1 sits outside drf / ck: no end-to-end figure
                say(f"      end to end: pre-loop {pms:.0f} + 1st cycle {d['drf'][0] + d['ck'][0]:.0f} ms (draft + check + "
                    f"commit); excess incl. pre-paid {pms + ex:.0f} ms")
                E2E.setdefault(t, {})[i] = float("nan") if d.get("paused") else pms + ex   # by prompt index (skips, pauses)
    if ctl_share:
        v = "PASS (the split names a cause >= 30 %)" if max(ctl_share) >= 0.30 else "KILL (nothing >= 30 %)"
        say(f"  A6 line on the control: largest share per prompt {[round(100 * s) for s in ctl_share]} % -> {v}")
    b, fm = E2E.get("a6c", {}), lambda xs: "[" + ", ".join(f"{x:.0f}" for x in xs) + "]"
    for t, v in E2E.items():
        ii = [i for i in v if i in b]
        vs = "" if t == "a6c" or not ii else (f" vs a6c {fm(b[i] for i in ii)} ms ("
                                               f"{', '.join(f'{100 * (v[i] / b[i] - 1):+.0f}' for i in ii)} %)")
        say(f"  end to end {t}: excess incl. pre-paid {fm(v.values())} ms{vs}; lifetime peak {peak(f'{t}_{sfx}'):.1f} GB")
    w_ = [(t, max(x for x in v.values() if x == x)) for t, v in E2E.items() if any(x == x for x in v.values())]
    if w_:
        t, w = min(w_, key=lambda x: x[1])
        say(f"  condition 7 (start-up <= 2-5 ms, end to end): best arm {t}, worst prompt {w:.0f} ms -> "
            f"{'met' if w <= 5 else 'not met'}")
a6(SUF)
if SUF != "160":
    a6("160")
# ---------------------------------------------------------------- SSD wait regression (lead step 3)
X = []
for f in sorted(glob.glob(os.path.join(R, "*_checks.json"))):
    for d in json.load(open(f)):
        if d.get("paused") or "mb" not in d:
            continue
        for i in range(len(d["ck"])):
            if d["miss"][i] > 0:
                X.append((d["io"][i], d["miss"][i], d["mb"][i]))
if len(X) > 50:
    X = np.array(X)
    def fit(cols):
        A_ = np.column_stack([np.ones(len(X))] + [X[:, c] for c in cols])
        b, res, *_ = np.linalg.lstsq(A_, X[:, 0], rcond=None)
        r2 = 1 - float(((X[:, 0] - A_ @ b) ** 2).sum()) / float(((X[:, 0] - X[:, 0].mean()) ** 2).sum())
        return b, r2
    (b1, r1), (b2, r2), (b3, r3) = fit([1]), fit([2]), fit([1, 2])
    say(f"\n== SSD wait per check (lead step 3), {len(X)} checks with misses: demand SSD ms ~ reads: {b1[1]:.3f} ms per miss "
        f"(R^2 {r1:.3f}); ~ MB: {b2[1]:.3f} ms per MB = {1 / b2[1] if b2[1] > 0 else float('nan'):.1f} GB/s (R^2 {r2:.3f}); both: "
        f"{b3[1]:.3f} ms/miss + {b3[2]:.3f} ms/MB (R^2 {r3:.3f}) -> {'reads (latency-bound: merging reads pays)' if r1 > r2 + 0.01 else 'bytes (bandwidth-bound)' if r2 > r1 + 0.01 else 'no separation (reads and MB collinear)'}")
# ---------------------------------------------------------------- W18 (NewIdeas 11-02): SSD rate per check, in night order
W18 = []
for f in sorted(glob.glob(os.path.join(R, "*_checks.json")), key=os.path.getmtime):
    win, rd_ = [], []
    for d in json.load(open(f)):
        if d.get("paused") or "mb" not in d:
            continue
        for i in range(len(d["ck"])):
            if d["mb"][i] > 1 and d["ck"][i] > 0:
                win.append(d["mb"][i] / d["ck"][i])     # MB per ms = GB/s over the check window
                if d["io"][i] > 0:
                    rd_.append(d["mb"][i] / d["io"][i])   # GB/s while the loader reads (demand SSD busy)
    if win:
        W18.append(f"{os.path.basename(f)[:-12]} {np.median(win):.2f} / {np.median(rd_) if rd_ else float('nan'):.1f}")
if W18:
    say(f"\n== W18: SSD GB/s per check, median over the window / while reading, in run order (heat throttling = a falling "
        f"second figure): " + "; ".join(W18))
os.makedirs(os.path.dirname(OUTF), exist_ok=True)
open(OUTF, "w").write("\n".join(L) + "\n")
