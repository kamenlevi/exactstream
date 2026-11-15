"""WEEK98 S9c REAL EAGLE-2 TREES on the 125B head (MTP Head; REPORT 'WEEK98 S9c' + amendment 1), CPU only.
Data: BigMoE's tree_t30b_{heplus24,mbppbank}.npz (scripts/dd_tree.py 109fe0b): at EVERY anchor, the best-first EAGLE-2
tree of t30b itself (32 nodes, top-8 children, path probability; nodes under a wrong sibling are real head outputs),
with on_true = the node's token is y under a true-path parent. With BigMoE's TOPK / cells (S9b's layout) for chain-N
and S9b's off-path proxy on the same anchors.
Budget: an N-row check = N - 1 nodes = the first N - 1 pops; accepted = max{d + 1 : on_true, node < N - 1} else 0;
kept = accepted + 1; the walk = certify98.kept_walk's (R0 = 1). Exact mode only.
Usage: BigMoE venv python -I -B tree_real125.py OUT.json"""
import os, sys, json, heapq
import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/EvalSuite/tools"))
import certify98 as C98                                     # noqa: E402

H = os.path.expanduser("~")
CE = f"{H}/BigMoE/data/deepdraft/ceiling"
SETS = {"mbpp": ("mbppbank", 6.92), "he": ("heplus24", 13.77)}
DEP, K = 31, 8
C16 = (1.006, 1.021)                                         # Leap's tree-verify overhead on a ~62 ms 16-row check
R32S = (1.625, 1.3)
rng = np.random.default_rng(1115)
lines = []


def say(s=""):
    print(s, flush=True); lines.append(s)


def load_cells(tag, published):
    """S9b's layout (certify98.windows): per window rt, R (y's rank in the head's order; bf16 ties -> 1), Q (top-8 p),
    and the window's depth-0 anchors s (= row - 1) in layout-row order"""
    cz, tz = np.load(f"{CE}/cells_t30b_{tag}.npz"), np.load(f"{CE}/topk_t30b_{tag}.npz")
    for k in ("depth", "fi", "row", "right"):
        assert (cz[k] == tz[k]).all(), f"{tag}: {k} differs between cells and topk"
    dep, right, yr = tz["depth"].astype(int), tz["right"].astype(int), tz["y_rank"].astype(int)
    yr = np.where((yr == 0) & (right == 0), 1, yr)
    tp, row, fi = tz["top_p"].astype(np.float32), tz["row"].astype(np.int64), tz["fi"].astype(np.int64)
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0)); en = np.append(st[1:], len(dep))
    ref = C98.windows(f"{CE}/cells_t30b_{tag}.npz")
    W, k32 = [], []
    for wi, (s, t) in enumerate(zip(st, en)):
        dw = dep[s:t]; n0 = int((dw == 0).sum())
        rt = np.full((n0 + 1, DEP), -1, np.int8); R = np.full((n0 + 1, DEP), -1, np.int16)
        Q = np.zeros((n0 + 1, DEP, K), np.float32)
        for j in range(DEP):
            m = dw == j; c = int(m.sum())
            rt[:c, j] = right[s:t][m]; R[:c, j] = yr[s:t][m]; Q[:c, j] = tp[s:t][m]
        assert (rt == ref[wi][0]).all(), f"{tag}: window {wi} layout != certify98"
        R[rt < 0] = -1
        k32 += C98.kept_walk(rt, C98.R0, 31)
        an = row[s:t][dw == 0] - 1
        W.append({"rt": rt, "R": R, "Q": Q, "fi": int(fi[s]), "an": an, "n": n0 + 1})
    say(f"SELF-CHECK {tag}: cells layout == certify98, chain-32 {np.mean(k32):.2f} vs {published} -> "
        f"{'PASS' if abs(np.mean(k32) - published) < 0.01 else 'FAIL'}")
    return W


def walk_acc(acc, n, r0=C98.R0):
    p = r0; ks = nc = 0; starts = []
    while p < n:
        a = int(acc[p]); starts.append(p); ks += a + 1; nc += 1; p += a + 1
    return ks, nc, starts


def chain_acc(X, N):
    R = X["R"]
    k = np.where((R != 0).any(1), (R != 0).argmax(1), DEP)
    return np.minimum(k, N - 1)


def proxy_acc(X, N, gamma=1):
    """S9b's EAGLE-2 with the off-path proxy, per anchor (for the bias)"""
    R, Q, n = X["R"], X["Q"], X["n"]; out = np.zeros(n, np.int64)
    for p in range(n):
        hp = [(-float(Q[p, 0, r]), 0, bool(R[p, 0] == r)) for r in range(K) if Q[p, 0, r] > 0]
        heapq.heapify(hp); acc = cnt = 0
        while hp and cnt < N - 1:
            ns, j, on = heapq.heappop(hp); cnt += 1
            if on:
                acc = j + 1
            if j + 1 >= DEP:
                continue
            for r in range(K):
                q = float(Q[p, j + 1, r])
                if q > 0:
                    heapq.heappush(hp, (ns * (q if on else q ** gamma), j + 1, bool(on and R[p, j + 1] == r)))
        out[p] = acc
    return out


def kpc(x):
    return x[:, 0].sum() / x[:, 1].sum()


def ratio_ci(a, b):
    idx = rng.integers(0, len(a), (2000, len(a)))
    r = (a[idx, 0].sum(1) / a[idx, 1].sum(1)) / (b[idx, 0].sum(1) / b[idx, 1].sum(1))
    return float(kpc(a) / kpc(b)), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


RES = {}
for s, (tag, pub) in SETS.items():
    W = load_cells(tag, pub)
    tz = np.load(f"{CE}/tree_t30b_{tag}.npz"); kz = np.load(f"{CE}/topk_t30b_{tag}.npz")
    nfi, nan_, nnode, ndep, ontrue, pp = (tz[k] for k in ("fi", "an", "node", "depth", "on_true", "path_p"))
    afi, aan, aws = tz["a_fi"], tz["a_an"], tz["a_walk_start"].astype(bool)
    byw = {X["fi"]: X for X in W}
    # checks: anchors, walk starts, the roots' first pop vs TOPK's depth-0 top-1
    bad_an = 0
    for X in W:
        got = np.sort(aan[afi == X["fi"]]); bad_an += int(not np.array_equal(got, np.sort(X["an"])))
    acc = {N: {X["fi"]: np.zeros(X["n"], np.int64) for X in W} for N in (16, 32)}
    rowof = {X["fi"]: {int(a): i for i, a in enumerate(X["an"])} for X in W}
    sel = ontrue.astype(bool)
    for f_, a_, nd, d_ in zip(nfi[sel], nan_[sel], nnode[sel], ndep[sel]):
        r = rowof[int(f_)][int(a_)]
        for N in (16, 32):
            if nd < N - 1:
                acc[N][int(f_)][r] = max(acc[N][int(f_)][r], int(d_) + 1)
    p0 = {}
    for X in W:
        m = (kz["fi"] == X["fi"]) & (kz["depth"] == 0)
        p0.update({(X["fi"], int(rw) - 1): float(p) for rw, p in zip(kz["row"][m], kz["top_p"][m, 0])})
    z0 = nnode == 0
    d0 = np.array([abs(float(p) - p0[(int(f_), int(a_))]) for f_, a_, p in zip(nfi[z0], nan_[z0], pp[z0])])
    ws_bad = 0
    for X in W:
        _, _, st32 = walk_acc(chain_acc(X, 32), X["n"])
        flags = np.zeros(X["n"], bool); m = afi == X["fi"]
        for a_, w_ in zip(aan[m], aws[m]):
            flags[rowof[X["fi"]][int(a_)]] = w_
        ws_bad += int(not np.array_equal(np.flatnonzero(flags), np.array([p for p in st32 if p < X["n"] - 1])))
    say(f"SELF-CHECK {tag} tree dump: anchors == cells' depth-0 anchors in {len(W) - bad_an} / {len(W)} windows; "
        f"a_walk_start == the chain-32 walk's starts in {len(W) - ws_bad} / {len(W)}; first pop path_p vs TOPK depth-0 "
        f"top-1: max |diff| {d0.max():.2e} over {len(d0)} roots")
    row = {}
    c16 = np.array([walk_acc(chain_acc(X, 16), X["n"])[:2] for X in W])
    c32 = np.array([walk_acc(chain_acc(X, 32), X["n"])[:2] for X in W])
    row["chain16"], row["chain32"] = float(kpc(c16)), float(kpc(c32))
    for N in (16, 32):
        t = np.array([walk_acc(acc[N][X["fi"]], X["n"])[:2] for X in W])
        r, lo, hi = ratio_ci(t, c16)
        px = np.array([walk_acc(proxy_acc(X, N), X["n"])[:2] for X in W])
        # paired at the chain-32 walk's starts (the pre-registered paired form)
        pr_t = pr_c = 0.0; npr = 0
        for X in W:
            _, _, st32 = walk_acc(chain_acc(X, 32), X["n"])
            ca = chain_acc(X, 16)
            for p in st32:
                pr_t += acc[N][X["fi"]][p] + 1; pr_c += ca[p] + 1; npr += 1
        row[f"real{N}"] = {"kpc": float(kpc(t)), "vs_chain16": r, "ci95": [lo, hi], "proxy_kpc": float(kpc(px)),
                           "proxy_bias": float(kpc(t) / kpc(px)), "paired_at_starts_vs_chain16": pr_t / pr_c, "n_starts": npr}
        say(f"  {s} REAL EAGLE-2-{N}: {kpc(t):.3f} = x{r:.3f} chain-16 [{lo:.3f}, {hi:.3f}]; S9b proxy {kpc(px):.3f} "
            f"(real / proxy {kpc(t) / kpc(px):.3f}); paired at the {npr} chain-32 walk starts x{pr_t / pr_c:.3f}")
    say(f"  {s}: chain-16 {kpc(c16):.3f}, chain-32 {kpc(c32):.3f} (x{kpc(c32) / kpc(c16):.3f})")
    RES[s] = row

say("\nVERDICTS (pre-registered, EXACT): NET SAME-COST WIN = real EAGLE-2-16 / chain-16 >= 1.03 x 1.021 (= 1.052) with "
    "the CI above 1.021, both sets; 32 ROWS PAY = real EAGLE-2-32 / chain-16 >= R32 (c32 = 1.0: no 32-node overhead "
    "given), both sets")
a, b = RES["mbpp"], RES["he"]
win = all(x["real16"]["vs_chain16"] >= 1.03 * C16[1] and x["real16"]["ci95"][0] > C16[1] for x in (a, b))
say(f"  real EAGLE-2-16 x{a['real16']['vs_chain16']:.3f} / x{b['real16']['vs_chain16']:.3f} -> "
    f"{'NET SAME-COST WIN' if win else 'no'}")
for R32 in R32S:
    pay = all(x["real32"]["vs_chain16"] >= R32 for x in (a, b))
    say(f"  real EAGLE-2-32 x{a['real32']['vs_chain16']:.3f} / x{b['real32']['vs_chain16']:.3f} vs R32 {R32} -> "
        f"{'PAYS' if pay else 'no'}")
say("\nTHROUGHPUT vs chain-16 (kept/check / cost; the 16-node tree at Leap's c16 1.006 .. 1.021; 32 rows at R32, c32 1.0)")
for s, x in RES.items():
    for R32 in R32S:
        opts = {"chain-16": 1.0, "tree-16": (x["real16"]["vs_chain16"] / C16[1], x["real16"]["vs_chain16"] / C16[0]),
                "chain-32": x["chain32"] / x["chain16"] / R32, "tree-32": x["real32"]["vs_chain16"] / R32}
        best = max(opts, key=lambda k: opts[k] if not isinstance(opts[k], tuple) else opts[k][0])
        say(f"  {s} R32 {R32}: chain-16 x1.000, tree-16 x{opts['tree-16'][0]:.3f}..{opts['tree-16'][1]:.3f}, chain-32 "
            f"x{opts['chain-32']:.3f}, tree-32 x{opts['tree-32']:.3f} -> best {best}")
OUT = sys.argv[1]
json.dump(RES, open(OUT, "w"), indent=1)
open(OUT.replace(".json", ".txt"), "w").write("\n".join(lines) + "\n")
