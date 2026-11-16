"""CERTIFY98 amendment 8's lines (docs/PREREG_certify98_2026-11-15.md, amendment 8; NewIdeas' input
~/NewIdeas/research/week98/AMENDMENT_INPUT_1116.md at 64d8a29, approved by the orchestrator Sun 21:5x-22:1x). CPU only,
numpy only. Inputs: BigMoE's TOPK dump of one run on one set (scripts/dd_topk.py: certify98's layout plus fi, row, files
and the target's token y), its TREE dump (scripts/dd_tree.py), and an auditor's teacher-forced rows (MoonShots' layout:
fname, row, top1, margin2b, tk0 per answer row).
  .venv/bin/python tools/certify98_lines.py lab <topk.npz> [--label L]                  # LAB: true state, not deployable
  .venv/bin/python tools/certify98_lines.py c1 <topk.npz> <tree.npz> [--label L]        # band-0 coverage, tree-16 / 32
  .venv/bin/python tools/certify98_lines.py c2 <topk.npz> <rows.npz> [--label L]        # t30b + the frozen 2B audit
  .venv/bin/python tools/certify98_lines.py treek <HE+ topk> <HE+ tree> <MBPP topk> <MBPP tree> --c32 X --r3216 Y
  .venv/bin/python tools/certify98_lines.py reproduce       # the dev numbers amendment 8 quotes (NewIdeas, MoonShots)
  .venv/bin/python tools/certify98_lines.py power [reps]    # a window bootstrap of the dev sets at the fresh window counts
Bounds are certify98's: the ratio estimate minus t(K-1, 0.95) x the window-clustered CR1 SE (one-sided 95 %); a ratio of
two walks (tree-K) uses certify98_tree.ratio (the delta method on log r); C2's kept/32 difference uses the same
window-clustered influences. Walk-strict counts are targets only (amendment 8). Each line command ends with one line:
`CERTIFY98 <LINE> <label>: PASS | BETWEEN | FAIL (...)`; exit 0, or 2 on bad input."""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify98 as C                                    # noqa: E402
import certify98_tree as T                               # noqa: E402

H = os.path.expanduser("~")
CE = f"{H}/BigMoE/data/deepdraft/ceiling"
MS = f"{H}/MoonShots/week98"
BANDS = C.BANDS                     # 0 / 1-2 / 3-6 / 7-10 / 11-14 / 15-22 / 23-30
KAMEN = 97.8                        # Kamen's target (11-15 21:4x): C1's and C2's point bar, and every LAB point's quote
LINE = 97.0                         # HE_LINE: every lower bound
LAB_PT = 97.5                       # LAB: every band's point
TAU = np.array([4.0] + [2.0] * 2 + [1.0] * 4 + [0.5] * 24)    # the frozen audit rule's margin bar by depth, nats
P_HEAD = 2                          # C2's headline: audit passes per check (P = 1 is reported)
C2_BANDS = ((15, 22), (23, 30))     # C2's PRIMARY bands
NET = T.NET                         # tree-K's net margin, as amendment 6a


def ok(st, pt):
    return st["acc"] >= pt and st["lcb"] >= LINE


def fmt(st):
    return f"{st['acc']:.2f} [LB {st['lcb']:.2f}]"


# ---- LAB: the head's guess with the 125B's true state at every depth (section 1) ------------------------------------

def lab_windows(topk):
    """lab[a, d] = rt[a + d, 0] wherever the cell (a, d) exists: at depth d from anchor a the head reads the trunk at
    row a + d, so its guess IS the depth-0 guess of anchor a + d (NewIdeas' lab_truestate.py)"""
    out = []
    for rt, _ in C.windows(topk):
        n = rt.shape[0]; t = np.full_like(rt, -1)
        for d in range(C.DEPTHS):
            src = np.arange(n - 1) + d
            m = (src < n - 1) & (rt[:-1, d] >= 0)
            t[:-1, d][m] = rt[src[m], 0]
        out.append((t, None))
    return out


def lab_counts(topk):
    """per window and band: right cells and cells (MARGINAL: every cell), [K, 7] each"""
    LW = lab_windows(topk)
    y = np.array([[int((t[:, lo:hi + 1] == 1).sum()) for lo, hi in BANDS] for t, _ in LW], float)
    x = np.array([[int((t[:, lo:hi + 1] >= 0).sum()) for lo, hi in BANDS] for t, _ in LW], float)
    return y, x


def stat(y, x):
    """certify98's ratio, CR1 SE and one-sided 95 % lower bound over the windows with cells"""
    k = x > 0
    p, se = C.cr1(y[k], x[k]); K = int(k.sum())
    return {"acc": 100 * p, "se": 100 * se, "lcb": 100 * (p - C.t_one(K - 1) * se), "K": K, "rows": int(x.sum())}


def lab(topk, label):
    y, x = lab_counts(topk)
    S = [stat(y[:, b], x[:, b]) for b in range(len(BANDS))]
    print(f"{label}: LAB (true state; NOT deployable), MARGINAL accuracy per band over every cell")
    for (lo, hi), s in zip(BANDS, S):
        print(f"  {lo}-{hi}: {fmt(s)}  ({s['acc'] - KAMEN:+.2f} pt vs {KAMEN}; {s['rows']} cells, K {s['K']})")
    low = [f"{lo}-{hi}" for (lo, hi), s in zip(BANDS, S) if not ok(s, LAB_PT)]
    pts = " / ".join(f"{s['acc']:.2f}" for s in S)
    v = f"FAIL (bands {', '.join(low)})" if low else "PASS"
    print(f"CERTIFY98 LAB {label}: {v} (every band's point >= {LAB_PT} and lower bound >= {LINE}; points {pts})")
    return S


# ---- C1: walk-strict coverage at band 0, tree-16 and tree-32 (section 2) --------------------------------------------

def c1_counts(topk, tree):
    """per tree N: per-window coverage (num, den) under the tree-N walk, targets only"""
    W, bad, ns, k, nodes = T.load(topk, tree)
    return {N: T.coverage_w(W, lambda X, N=N: X["acc"][N], N - 1, True) for N in (16, 32) if N in ns}, bad, k


def c1(topk, tree, label):
    per, bad, k = c1_counts(topk, tree)
    for b in bad:
        print(f"  CHECK FAILED: {b}")
    out = {}
    for N, P in per.items():
        S = {f"{lo}-{hi}": C.strict_stats(P, lo, hi) for lo, hi in BANDS if hi < N - 1}
        out[N] = S
        print(f"{label}: tree-{N} walk-strict coverage (targets only): "
              + "  ".join(f"{b} {fmt(s)}" for b, s in S.items()))
        v = "VOID (a check failed)" if bad else "PASS" if ok(S["0-0"], KAMEN) else "FAIL"
        print(f"CERTIFY98 C1 {label} tree-{N}: {v} (band 0 {fmt(S['0-0'])}; bars: point >= {KAMEN}, LB >= {LINE}; "
              f"tree k {k})")
    return out


# ---- C2: t30b + the frozen 2B audit (section 3) ---------------------------------------------------------------------

def mats(topk, vals):
    """per window [n0 + 1, 31] float matrices of a per-cell array, in certify98.windows' layout (nan: no cell)"""
    d = np.load(topk, allow_pickle=True)
    dep = d["depth"].astype(int)
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0)); en = np.append(st[1:], len(dep))
    out = []
    for s, t in zip(st, en):
        dw = dep[s:t]; M = np.full((int((dw == 0).sum()) + 1, C.DEPTHS), np.nan)
        for j in range(C.DEPTHS):
            k = dw == j
            M[:int(k.sum()), j] = vals[s:t][k]
        out.append(M)
    return out


def audit_join(topk, rows, against="y"):
    """per cell: the auditor is right and its margin, joined on (file, row) as MoonShots' join_heplus.py; every cell
    must find its row. against "y" (registered): right = its top-1 equals the dump's target y, the label of the head's
    own cells. "tk0": MoonShots' dev convention, its top-1 equals the rows' 125B top-1. On the runtime path tk0 == y on
    every row (validity); the rows where they differ are counted: (right, margin, cells, rows)."""
    d = np.load(topk, allow_pickle=True); R = np.load(rows, allow_pickle=True)
    files = [os.path.basename(str(f)) for f in d["files"]]
    key = {(os.path.basename(str(n)), int(r)): i for i, (n, r) in enumerate(zip(R["fname"], R["row"]))}
    idx = np.array([key.get((files[int(f)], int(r)), -1) for f, r in zip(d["fi"], d["row"])])
    if (idx < 0).any():
        raise SystemExit(f"audit join: {int((idx < 0).sum())} of {len(idx)} cells find no auditor row")
    tk0, y = R["tk0"][idx].astype(np.int64), d["y"].astype(np.int64)
    diff = tk0 != y
    nrows = len(set(zip(d["fi"][diff].tolist(), d["row"][diff].tolist())))
    lab = y if against == "y" else tk0
    return (R["top1"][idx].astype(np.int64) == lab).astype(float), R["margin2b"][idx].astype(float), int(diff.sum()), nrows


def audit_walk(rt, comb, P, cap=31):
    """MoonShots' walk (week98/walkstrict_bands.py): at each check start, depth by depth, the cell's outcome is the
    audited one until P repairs are used (a head miss the auditor repairs uses one), then the head's. A cell with no
    target ends the check uncounted. Returns (sum kept, checks, num [31], den [31]); P = 0 is the head's own walk."""
    num, den = np.zeros(C.DEPTHS), np.zeros(C.DEPTHS)
    p, ks, nc = C.R0, 0, 0
    while p < rt.shape[0]:
        k = used = 0
        while k < cap:
            hv, cv = rt[p, k], comb[p, k]
            if hv == -1:
                break
            if used >= P:
                good = hv == 1
            elif cv == 1:
                good = True; used += hv == 0
            else:
                good = False
            den[k] += 1; num[k] += good
            if not good:
                break
            k += 1
        ks += k + 1; nc += 1; p += k + 1
    return ks, nc, num, den


def c2_walks(topk, rows, passes=(0, 1, 2), against="y"):
    """per P: per-window (sum kept, checks, num, den), and the tk0 != y count (cells, rows). The audited cell (MoonShots'
    frozen rule): where the auditor's margin >= tau(depth), its outcome replaces the head's (a repair if it is right, a
    break if it is wrong). Per cell this equals 'the first confident disagreement replaces the draft': where the head
    is right, a confident auditor disagrees only when it is wrong; where the head is wrong, the cell is right only if a
    confident auditor is right."""
    right, mg, nc, nr = audit_join(topk, rows, against)
    A, G = mats(topk, right), mats(topk, mg)
    out = {P: [] for P in passes}
    for (rt, _), a, g in zip(C.windows(topk), A, G):
        conf = ~np.isnan(g) & (np.nan_to_num(g, nan=-1.0) >= TAU[None, :])
        comb = np.where(rt < 0, -1, np.where(conf, np.nan_to_num(a), rt)).astype(np.int8)
        for P in passes:
            out[P].append(audit_walk(rt, comb, P))
    return out, (nc, nr)


def paired(A, B):
    """kept per check of walk A minus walk B over the same windows ([K, 2] rows of (sum kept, checks)): (d, se, lcb)"""
    ka, na, kb, nb = A[:, 0], A[:, 1], B[:, 0], B[:, 1]
    ra, rb = ka.sum() / na.sum(), kb.sum() / nb.sum()
    inf = (ka - ra * na) / na.sum() - (kb - rb * nb) / nb.sum()
    K = len(ka); se = float(np.sqrt(K / (K - 1) * (inf ** 2).sum()))
    return float(ra - rb), se, float(ra - rb - C.t_one(K - 1) * se)


def c2(topk, rows, label, against="y", dev=False):
    """the C2 read-out; on fresh data (dev False) a row whose tk0 != the dump's y VOIDs it (the runtime-path validity)"""
    Wk, (nc, nr) = c2_walks(topk, rows, against=against)
    head = np.array([w[:2] for w in Wk[0]], float)
    if abs(head[:, 0].sum() / head[:, 1].sum() - C.summary(C.windows(topk))["kept_per_32"]["mean"]) > 1e-9:
        print("  CHECK FAILED: the P = 0 walk != certify98's kept/32"); return None
    if nr:
        print(f"  {'NOTE (dev answers, not the runtime path)' if dev else 'CHECK FAILED'}: {nr} rows ({nc} cells) where "
              f"the auditor rows' 125B top-1 (tk0) != the dump's target y; the auditor is scored against "
              f"{'y' if against == 'y' else 'tk0'}")
        if not dev:
            print(f"CERTIFY98 C2 {label}: VOID (tk0 != y on {nr} rows)"); return None
    res = {}
    for P in (P_HEAD, 1):
        per = [(w[2], w[3]) for w in Wk[P]]
        S = {f"{lo}-{hi}": C.strict_stats(per, lo, hi) for lo, hi in BANDS}
        Kp = np.array([w[:2] for w in Wk[P]], float)
        d, se, lcb = paired(Kp, head); r0 = head[:, 0].sum() / head[:, 1].sum()
        res[P] = (S, d, lcb)
        print(f"{label}: t30b + 2B audit, P = {P}: kept/32 {Kp[:, 0].sum() / Kp[:, 1].sum():.3f} vs t30b {r0:.3f}, "
              f"difference {d:+.3f} ({100 * d / r0:+.1f} %; paired LB {lcb:+.3f})")
        print("  walk-strict (targets only): " + "  ".join(f"{b} {fmt(s)}" for b, s in S.items()))
        prim = [f"{lo}-{hi}" for lo, hi in C2_BANDS if not ok(S[f"{lo}-{hi}"], KAMEN)]
        tag = "" if P == P_HEAD else " (reported)"
        v = "PASS" if not prim and lcb > 0 else "FAIL (" + "; ".join(
            ([f"bands {', '.join(prim)}"] if prim else []) + ([f"kept/32 LB {lcb:+.3f} <= 0"] if lcb <= 0 else [])) + ")"
        print(f"CERTIFY98 C2 {label} P={P}{tag}: {v} (PRIMARY 15-22 {fmt(S['15-22'])}, 23-30 {fmt(S['23-30'])}: point >= "
              f"{KAMEN}, LB >= {LINE}; CO-PRIMARY kept/32 paired LB {lcb:+.3f} > 0)")
    return res


# ---- the tree-K line: T8 at K* = 32 (section 4) ---------------------------------------------------------------------

def tlabel(r, lcb, bar):
    """amendment 6's three labels: FAIL (point short of NET x bar), PASS (bound >= bar too) or BETWEEN"""
    return "FAIL" if r < NET * bar else "PASS" if lcb >= bar else "BETWEEN"


def treek_stats(topk, tree):
    R = T.score(topk, tree)
    if 32 not in R["ns"]:
        R["bad"].append(f"the dump has {R['nodes']} nodes, so no tree-32")
        return R, None, None
    if R["k"] != T.K_LINE:
        R["bad"].append(f"the line reads a top-{T.K_LINE} tree (T8); this dump has k = {R['k']}")
    a = T.ratio(R["_t32"], R["_c32"]); b = R["tree32_vs_chain16"]
    return R, (a[0], a[2]), (b["r"], b["lcb95"])


def treek(argv):
    if len(argv) < 4 or "--c32" not in argv or "--r3216" not in argv:
        print(__doc__); return 2
    c32 = float(argv[argv.index("--c32") + 1]); r3216 = float(argv[argv.index("--r3216") + 1])
    bars = {"T32/32": c32, "T32/16": r3216 * c32}
    labs, parts, bad = {}, {}, False
    for name, tk, tr in (("HE+", argv[0], argv[1]), ("MBPP", argv[2], argv[3])):
        R, t3232, t3216 = treek_stats(tk, tr)
        for b in R["bad"]:
            print(f"  CHECK FAILED ({name}): {b}")
        bad |= bool(R["bad"])
        if t3232 is None:
            continue
        for lin, (r, lcb) in (("T32/32", t3232), ("T32/16", t3216)):
            labs[(lin, name)] = tlabel(r, lcb, bars[lin])
            parts[(lin, name)] = f"{name} {r:.3f} [LB {lcb:.3f}]"
            print(f"{name}: {lin} {r:.3f} [LB {lcb:.3f}] vs bar {bars[lin]:.4f} (point bar {NET * bars[lin]:.4f}): "
                  f"{labs[(lin, name)]}{'' if (lin, name) != ('T32/16', 'MBPP') else ' (information)'}")
    if bad:
        print("CERTIFY98 TREE-K: VOID (a check failed)"); return 0
    g = [labs[("T32/32", "HE+")], labs[("T32/32", "MBPP")]]
    o = "PASS" if g == ["PASS", "PASS"] else "FAIL" if "FAIL" in g else "BETWEEN"
    print(f"CERTIFY98 TREE-K T32/32: {o} ({parts[('T32/32', 'HE+')]}; {parts[('T32/32', 'MBPP')]}; bars: LB >= c32 "
          f"{c32}, point >= {NET * c32:.4f})")
    print(f"CERTIFY98 TREE-K T32/16 HE+: {labs[('T32/16', 'HE+')]} ({parts[('T32/16', 'HE+')]}; bars: LB >= R32/R16 x "
          f"c32 = {bars['T32/16']:.4f}, point >= {NET * bars['T32/16']:.4f}; MBPP information: {parts[('T32/16', 'MBPP')]})")
    return 0


# ---- reproduce: the dev numbers the amendment quotes ----------------------------------------------------------------

DEV = {"HE+": ("heplus24", "heplus"), "MBPP": ("mbppbank", "mbpp")}
C1_PUB = {("HE+", 32): 98.47, ("MBPP", 32): 98.61, ("HE+", 16): 98.58, ("MBPP", 16): 97.79}   # 151d997; NewIdeas
TK_PUB = {"HE+": ((1.181, 1.126), (1.531, 1.470)), "MBPP": ((1.304, 1.283), (1.371, 1.351))}  # 151d997


def reproduce():
    good = True
    L = json.load(open(f"{H}/NewIdeas/research/week98/lab_truestate.json"))
    M = json.load(open(f"{MS}/walkstrict_bands.json"))
    for s, (tag, rtag) in DEV.items():
        tk, tr = f"{CE}/topk_t30b_{tag}.npz", f"{CE}/tree_t30b_{tag}.npz"
        print(f"== {s} dev ({tag})")
        S = lab(tk, f"{s} dev")
        pub = L["he" if s == "HE+" else "mbpp"]["LAB true state"]["marginal"]
        dl = max(abs(round(x["acc"], 2) - p) for x, p in zip(S, pub)); good &= dl <= 0.011
        print(f"  LAB vs NewIdeas' lab_truestate.json: max |diff| {dl:.3f} pt -> {'PASS' if dl <= 0.011 else 'FAIL'}")
        out = c1(tk, tr, f"{s} dev")
        for N in (16, 32):
            v = out[N]["0-0"]["acc"]; dd = abs(round(v, 2) - C1_PUB[(s, N)]); good &= dd <= 0.011
            print(f"  C1 tree-{N} band 0 {v:.2f} vs {C1_PUB[(s, N)]} -> {'PASS' if dd <= 0.011 else 'FAIL'}")
        print(f"  -- C2 as registered (the auditor scored against the dump's y):")
        c2(tk, f"{MS}/rows2b_{rtag}.npz", f"{s} dev", "y", dev=True)
        print(f"  -- C2 in MoonShots' dev convention (against tk0), for the reproduction:")
        c2(tk, f"{MS}/rows2b_{rtag}.npz", f"{s} dev (tk0)", "tk0", dev=True)
        Wk = c2_walks(tk, f"{MS}/rows2b_{rtag}.npz", against="tk0")[0]
        for P, key in ((0, "head"), (1, "2B P=1"), (2, "2B P=2")):
            m = M[f"{rtag} {key}"]; per = [(w[2], w[3]) for w in Wk[P]]
            kk = np.array([w[:2] for w in Wk[P]], float); kept = kk[:, 0].sum() / kk[:, 1].sum()
            diffs = [abs(kept - m["kept32"])]
            for lo, hi in BANDS:
                b = f"{lo}-{hi}" if hi > lo else f"{lo}"; st = C.strict_stats(per, lo, hi)
                diffs.append(abs(round(st["acc"], 2) - m["bands"][b]["pt"]))
                good &= st["rows"] == m["bands"][b]["n"]
            mx = max(diffs); good &= mx <= 0.011
            print(f"  C2 {key} vs MoonShots' walkstrict_bands.json: kept/32 {kept:.3f} vs {m['kept32']}; max |diff| "
                  f"{mx:.4f} -> {'PASS' if mx <= 0.011 else 'FAIL'}")
        R, a, b = treek_stats(tk, tr)
        (pa, pal), (pb, pbl) = TK_PUB[s]
        dd = max(abs(round(a[0], 3) - pa), abs(round(a[1], 3) - pal), abs(round(b[0], 3) - pb), abs(round(b[1], 3) - pbl))
        good &= dd <= 0.0011 and not R["bad"]
        print(f"  tree-K: T32/32 {a[0]:.3f} [LB {a[1]:.3f}], T32/16 {b[0]:.3f} [LB {b[1]:.3f}] vs 151d997 -> "
              f"{'PASS' if dd <= 0.0011 and not R['bad'] else 'FAIL'}")
    print(f"CERTIFY98 LINES-REPRODUCE: {'PASS' if good else 'FAIL'}")
    return 0 if good else 1


# ---- power: a window bootstrap of the dev sets at the fresh window counts --------------------------------------------

def b_ratio(y, x, idx, shift=0.0):
    """per draw: the point and CR1 lower bound (%) of sum y / sum x over the drawn windows; shift (pt) moves the truth"""
    Y, X = y[idx], x[idx]; K = idx.shape[1]
    p = Y.sum(1) / X.sum(1)
    se = np.sqrt(K / (K - 1) * ((Y - p[:, None] * X) ** 2).sum(1)) / X.sum(1)
    p = p + shift / 100
    return 100 * p, 100 * (p - C.t_one(K - 1) * se)


def b_tree(t, c, idx):
    """per draw: tree / chain kept per check and its lower bound (certify98_tree.ratio, vectorized)"""
    a, b, e, f = t[:, 0][idx], t[:, 1][idx], c[:, 0][idx], c[:, 1][idx]; K = idx.shape[1]
    r = (a.sum(1) / b.sum(1)) / (e.sum(1) / f.sum(1))
    inf = a / a.sum(1)[:, None] - b / b.sum(1)[:, None] - e / e.sum(1)[:, None] + f / f.sum(1)[:, None]
    return r, r * np.exp(-C.t_one(K - 1) * np.sqrt(K / (K - 1) * (inf ** 2).sum(1)))


def b_paired(A, B, idx):
    ka, na, kb, nb = A[:, 0][idx], A[:, 1][idx], B[:, 0][idx], B[:, 1][idx]; K = idx.shape[1]
    ra, rb = ka.sum(1) / na.sum(1), kb.sum(1) / nb.sum(1)
    inf = (ka - ra[:, None] * na) / na.sum(1)[:, None] - (kb - rb[:, None] * nb) / nb.sum(1)[:, None]
    return ra - rb - C.t_one(K - 1) * np.sqrt(K / (K - 1) * (inf ** 2).sum(1))


def power(reps=2000):
    rng = np.random.default_rng(1116)
    print(f"power: {reps} window draws with replacement from the dev windows at the fresh counts (HE+ 140, its strict 80; "
          f"MBPP 300); every bound as registered (CR1, t(K-1), one-sided 95 %); targets only")
    tk = {s: f"{CE}/topk_t30b_{t}.npz" for s, (t, _) in DEV.items()}
    tr = {s: f"{CE}/tree_t30b_{t}.npz" for s, (t, _) in DEV.items()}
    Kf = {"HE+": (140, 80), "MBPP": (300,)}
    # LAB, HE+ gated
    y, x = lab_counts(tk["HE+"])
    for K in Kf["HE+"]:
        idx = rng.integers(0, len(y), (reps, K))
        pts, lbs = zip(*[b_ratio(y[:, b], x[:, b], idx) for b in range(len(BANDS))])
        pts, lbs = np.array(pts), np.array(lbs)
        p_line = ((pts >= LAB_PT) & (lbs >= LINE)).all(0).mean(); p98 = (pts >= KAMEN).all(0).mean()
        print(f" LAB HE+ K {K}: P(line: every point >= {LAB_PT} AND LB >= {LINE}) {p_line:.2f}; P(every point >= {KAMEN}) "
              f"{p98:.2f}; per band P(point >= {LAB_PT} AND LB >= {LINE}) "
              + " ".join(f"{((pts[b] >= LAB_PT) & (lbs[b] >= LINE)).mean():.2f}" for b in range(len(BANDS))))
    # C1, HE+ gated (MBPP reported)
    for s in ("HE+", "MBPP"):
        per = c1_counts(tk[s], tr[s])[0]
        for N in (16, 32):
            n = np.array([q[0][0] for q in per[N]], float); d = np.array([q[1][0] for q in per[N]], float)
            for K in Kf[s]:
                pt, lb = b_ratio(n, d, rng.integers(0, len(n), (reps, K)))
                print(f" C1 {s} tree-{N} K {K}: P(band 0 point >= {KAMEN} AND LB >= {LINE}) "
                      f"{((pt >= KAMEN) & (lb >= LINE)).mean():.2f}{'' if s == 'HE+' else ' (reported)'}")
    # C2, HE+ gated: PRIMARY, CO-PRIMARY and both; the truth shifted for the deep bands (registered: against y)
    Wk = c2_walks(tk["HE+"], f"{MS}/rows2b_heplus.npz")[0]
    head = np.array([w[:2] for w in Wk[0]], float)
    for P in (P_HEAD, 1):
        Kp = np.array([w[:2] for w in Wk[P]], float)
        nd = [(np.array([w[2][lo:hi + 1].sum() for w in Wk[P]]), np.array([w[3][lo:hi + 1].sum() for w in Wk[P]]))
              for lo, hi in C2_BANDS]
        dev = [100 * n.sum() / d.sum() for n, d in nd]
        for K in Kf["HE+"]:
            for truth in (None, 98.0, 97.9, 97.8):
                idx = rng.integers(0, len(head), (reps, K))
                prim = np.ones(reps, bool)
                for (n, d), dv in zip(nd, dev):
                    pt, lb = b_ratio(n, d, idx, 0.0 if truth is None else truth - dv)
                    prim &= (pt >= KAMEN) & (lb >= LINE)
                co = b_paired(Kp, head, idx) > 0
                lab_ = f"dev truth {dev[0]:.2f} / {dev[1]:.2f}" if truth is None else f"both deep bands' truth {truth}"
                print(f" C2 HE+ P={P} K {K}, {lab_}: P(PRIMARY) {prim.mean():.2f}; P(CO-PRIMARY) {co.mean():.2f}; "
                      f"P(both) {(prim & co).mean():.2f}")
    # tree-K: T32/32 on HE+ AND MBPP; T32/16 on HE+ (bars pinned Tue by amendment 8a; these are the priors)
    T32 = {}
    for s in ("HE+", "MBPP"):
        R = T.score(tk[s], tr[s])
        for K in Kf[s]:
            idx = rng.integers(0, R["windows"], (reps, K))
            T32[(s, K)] = (b_tree(R["_t32"], R["_c32"], idx), b_tree(R["_t32"], R["_c16"], idx))
    for c32 in (1.03, 1.04, 1.05):
        (r1, l1), _ = T32[("HE+", 140)]; (r2, l2), _ = T32[("MBPP", 300)]
        p1 = (r1 >= NET * c32) & (l1 >= c32); p2 = (r2 >= NET * c32) & (l2 >= c32)
        print(f" T32/32 c32 {c32}: P(PASS) HE+ K 140 {p1.mean():.2f}, MBPP K 300 {p2.mean():.2f}, both {(p1 & p2).mean():.2f}")
    for r3216 in (1.40, 1.43, 1.445, 1.480):
        for c32 in (1.03, 1.04, 1.05):
            bar = r3216 * c32; _, (r, l) = T32[("HE+", 140)]
            print(f" T32/16 HE+ K 140 at R32/R16 {r3216} x c32 {c32} = bar {bar:.3f} (point {NET * bar:.3f}): P(PASS) "
                  f"{((r >= NET * bar) & (l >= bar)).mean():.2f}")
    return 0


def main(argv):
    if not argv:
        print(__doc__); return 2
    cmd, a = argv[0], argv[1:]
    lab_ = a[a.index("--label") + 1] if "--label" in a else (os.path.basename(a[0]) if a else "")
    if cmd == "reproduce":
        return reproduce()
    if cmd == "power":
        return power(int(a[0]) if a else 2000)
    if cmd == "treek":
        return treek(a)
    if cmd == "lab" and len(a) >= 1:
        lab(a[0], lab_); return 0
    if cmd == "c1" and len(a) >= 2:
        c1(a[0], a[1], lab_); return 0
    if cmd == "c2" and len(a) >= 2:
        c2(a[0], a[1], lab_); return 0
    print(__doc__); return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
