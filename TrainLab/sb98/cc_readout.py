"""WEEK98 S7 ACCSET read-out (MTP Head; REPORT 'WEEK98 S7 ACCSET'): kept/32 under the exact walk and under the close-call
walk at delta 0.5 / 1 / 2, per run, on MBPP-50 and HE-30 (evalx's first 50 / 30 windows, sorted).
Table: the teacher-forced chain from every anchor at every depth, built as Sandbox98's core.chain_table builds it
(read-only use of core.chain / load_W / lm_logits; the argmax is the target's own 4-bit lm_head), plus per cell
dlp = lp_target(head's #1) - lp_target(top-1) from the window's stored top-32 (-inf when the #1 is outside it).
Close-call walk: a wrong cell is accepted when dlp >= -delta (a teacher-forced upper bound, as Breakthrough's
closecall_walk.py). The walk is core.kept_walk's (from row max(0, gen - 1 - a0), up to 31 depths, +1 per check).
Per walk: kept/32, changed tokens (accepted cells that are not the true token) per 1K kept, walk-strict 0-14.
Pairs every run with the first (the control): pooled kept/32 difference, window bootstrap 95 % CI, two-sided sign test
over windows. JUDGE='run=delta,...' applies the pre-registered screen line to those runs.
Self-checks: the first run's table equals core.chain_table cell for cell on each set's first window; its exact kept/32
prints beside evalx's stored value when REF='mbpp=..,he=..' is given. SMOKE=n limits each set to n windows.
Usage: cd ~/Sandbox98 && PY ~/TrainLab/sb98/cc_readout.py OUT.json CTL RUN..."""
import os, sys, glob, json, math
sys.path.insert(0, os.path.expanduser("~/Sandbox98"))
import numpy as np
import mlx.core as mx
from sb import core as C

C.limit_memory(4.5)                      # Sandbox98's caps (06:06 incident), plus clear_cache per window
OUT, RUNS = sys.argv[1], sys.argv[2:]
DELTAS = [0.0, 0.5, 1.0, 2.0]
NS = int(os.environ.get("SMOKE", "0"))
SETS = (("mbpp", NS or 50), ("he", NS or 30))
REF = dict(kv.split("=") for kv in os.environ.get("REF", "").split(",") if "=" in kv)
JUDGE = {k: float(v) for k, v in (kv.split("=") for kv in os.environ.get("JUDGE", "").split(",") if "=" in kv)}
PAIRS = [tuple(x.split(":")) for x in os.environ.get("PAIRS", "").split(",") if ":" in x]   # extra pairs 'run:ctl' -> CMP keys 'run@ctl|set|delta'; JUDGE may name them as 'run@ctl=delta'


def files(d, n):
    return sorted(f for f in glob.glob(os.path.join(C.ROOT, "data", d, "*.npz")) if not f.endswith(".taps.npz"))[:n]


def table(D, f, Dp=31):
    """core.chain_table's rt [n_anchors, Dp] (1 right, 0 wrong, -1 no target) plus dlp per cell"""
    z = np.load(f)
    ids = np.asarray(z["ids"], np.int32); gen = int(z["gen"])
    tk = np.asarray(z["tk"], np.int64); lp = np.asarray(z["lp"], np.float32)
    W = C.load_W(z)
    a0 = max(0, gen - 2); S = len(ids); n = S - 1 - a0
    rt = np.full((n, Dp), -1, np.int8); dl = np.full((n, Dp), np.nan, np.float32)
    C.REPL.update(R=None, K=1, fn=None)
    C.CUR["f"], C.CUR["train"], C.CUR["x"] = f, False, None
    hs = []

    def on_depth(j, h, anc):
        m = min(h.shape[1], S - 1 - j - a0)
        if m > 0:
            hs.append((j, m, D.norm(h[0, :m]).astype(mx.bfloat16)))
    C.chain(D, W, mx.array(ids), Dp, on_depth, a0)
    Hs = mx.concatenate([x for _, _, x in hs], 0)
    am = []
    for c0 in range(0, Hs.shape[0], 2048):
        am.append(mx.argmax(C.lm_logits(Hs[c0:c0 + 2048]), -1))
        mx.eval(am[-1])
    am = np.array(mx.concatenate(am)).astype(np.int64)
    o = 0
    for j, m, _ in hs:
        a = am[o:o + m]; o += m
        rt[:m, j - 1] = (a == ids[a0 + j + 1:a0 + j + 1 + m]).astype(np.int8)
        rows = np.arange(a0 + j, a0 + j + m)                      # lp row r scores ids[r + 1]
        eq = tk[rows] == a[:, None]
        dl[:m, j - 1] = np.where(eq.any(1), lp[rows, eq.argmax(1)] - lp[rows, 0], -np.inf)
    del hs, Hs, W
    mx.clear_cache()
    return rt, dl, gen, a0


def walk(rt, dl, r0, delta):
    """core.kept_walk on the close-call-accepted table: kept per check, changed tokens, walk-strict counts 0-14"""
    acc = rt.copy()
    if delta > 0:
        acc[(rt == 0) & (dl >= -delta)] = 1
    sub, ex = acc[r0:], rt[r0:]
    kept, changed, strict = [], 0, np.zeros((2, 15))
    p = 0
    while p < sub.shape[0]:
        k = 0
        while k < min(31, sub.shape[1]) and sub[p, k] == 1:
            changed += int(ex[p, k] == 0)
            k += 1
        for j in range(min(15, k + 1, sub.shape[1])):
            if sub[p, j] >= 0:
                strict[1, j] += 1; strict[0, j] += int(sub[p, j] == 1)
        kept.append(k + 1); p += k + 1
    return kept, changed, strict


def sign_p(npos, nneg):
    n, k = npos + nneg, min(npos, nneg)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


R = {}
for ri, run in enumerate(RUNS):
    D = C.load_drafter(run)
    for st, nw in SETS:
        per = {d: [] for d in DELTAS}                              # per window: (kept sum, checks, changed)
        strict = {d: np.zeros((2, 15)) for d in DELTAS}
        for wi, f in enumerate(files(st, nw)):
            rt, dl, gen, a0 = table(D, f)
            if ri == 0 and wi == 0:
                ref = C.chain_table(D, f, 31)[0]
                mx.clear_cache()
                ok = ref.shape == rt.shape and bool((ref == rt).all())
                print(f"SELF-CHECK table == core.chain_table on {os.path.basename(f)}: {'PASS' if ok else 'FAIL'}", flush=True)
                if not ok:
                    sys.exit(2)
            r0 = max(0, gen - 1 - a0)
            for d in DELTAS:
                kp, ch, s = walk(rt, dl, r0, d)
                per[d].append((sum(kp), len(kp), ch)); strict[d] += s
        res = {}
        for d in DELTAS:
            P = np.array(per[d], float)
            ws = strict[d][0] / np.maximum(strict[d][1], 1)
            res[f"{d:g}"] = {"kept_per_32": float(P[:, 0].sum() / P[:, 1].sum()),
                             "changed_per_1k": float(1000 * P[:, 2].sum() / P[:, 0].sum()),
                             "walk_strict_0_14": ws.round(4).tolist(), "walk_strict_mean": float(ws.mean()),
                             "per_window": P.tolist()}
        R[f"{run}|{st}"] = res
        refv = REF.get(st) if ri == 0 else None
        print(f"{run} {st}: exact kept/32 {res['0']['kept_per_32']:.4f}" + (f" (evalx {refv})" if refv else "")
              + f", walk-strict 0-14 {100 * res['0']['walk_strict_mean']:.2f} %; close-call: "
              + "; ".join(f"d{d:g} {res[f'{d:g}']['kept_per_32']:.3f} ({res[f'{d:g}']['changed_per_1k']:.1f} changed/1K, "
                          f"strict {100 * res[f'{d:g}']['walk_strict_mean']:.2f} %)" for d in DELTAS[1:]), flush=True)
    del D
    mx.clear_cache()

rng = np.random.default_rng(1115)
CMP, VER = {}, {}
ctl = RUNS[0]


def pair(run, base, label):
    for st, _ in SETS:
        for d in DELTAS:
            A = np.array(R[f"{base}|{st}"][f"{d:g}"]["per_window"]); B = np.array(R[f"{run}|{st}"][f"{d:g}"]["per_window"])
            ka, kb = A[:, 0].sum() / A[:, 1].sum(), B[:, 0].sum() / B[:, 1].sum()
            idx = rng.integers(0, len(A), (5000, len(A)))
            bs = B[idx, 0].sum(1) / B[idx, 1].sum(1) - A[idx, 0].sum(1) / A[idx, 1].sum(1)
            w = B[:, 0] / B[:, 1] - A[:, 0] / A[:, 1]
            npos, nneg = int((w > 0).sum()), int((w < 0).sum())
            c = CMP[f"{label}|{st}|{d:g}"] = {"diff": float(kb - ka), "rel": float(kb / ka - 1),
                                              "ci95": np.percentile(bs, [2.5, 97.5]).tolist(),
                                              "p_sign": sign_p(npos, nneg), "n_pos": npos, "n_neg": nneg}
            print(f"  {label} {st} delta {d:g}: {ka:.3f} -> {kb:.3f} ({c['diff']:+.3f}, {100 * c['rel']:+.1f} %) "
                  f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] p {c['p_sign']:.3g} ({npos}+/{nneg}-); changed/1K "
                  f"{R[f'{base}|{st}'][f'{d:g}']['changed_per_1k']:.1f} -> {R[f'{run}|{st}'][f'{d:g}']['changed_per_1k']:.1f}")


print(f"\npairs vs the control {ctl} (kept/32 diff [window bootstrap 95 % CI], sign test p over windows):")
for run in RUNS[1:]:
    pair(run, ctl, run)
for run, base in PAIRS:
    print(f"\nextra pair {run} vs {base}:")
    pair(run, base, f"{run}@{base}")
for run, d in JUDGE.items():                                       # the pre-registered screen line
    mb, he = CMP[f"{run}|mbpp|{d:g}"], CMP[f"{run}|he|{d:g}"]       # run may be 'arm@comparator' (an extra pair)
    if mb["diff"] >= 0.10 and mb["p_sign"] < 0.05 and he["diff"] >= 0:
        call = "PROMISING"
    elif mb["diff"] <= -0.10 and mb["p_sign"] < 0.05:
        call = "HURTS"
    else:
        call = "NO EFFECT"
    VER[run] = {"call": call, "delta": d, "mbpp_diff": mb["diff"], "mbpp_p": mb["p_sign"], "he_diff": he["diff"],
                "exact_mbpp_diff": CMP[f"{run}|mbpp|0"]["diff"], "exact_he_diff": CMP[f"{run}|he|0"]["diff"]}
    print(f"VERDICT {run} (delta-{d:g} walk vs {run.split('@')[1] if '@' in run else ctl}): {call}: MBPP {mb['diff']:+.3f} (p {mb['p_sign']:.3g}), HE "
          f"{he['diff']:+.3f}; exact kept/32 MBPP {VER[run]['exact_mbpp_diff']:+.3f}, HE {VER[run]['exact_he_diff']:+.3f}",
          flush=True)
json.dump({"runs": RUNS, "res": R, "cmp": CMP, "verdict": VER}, open(OUT, "w"), indent=1)
