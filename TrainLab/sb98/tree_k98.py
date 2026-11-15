"""WEEK98 K98 (MTP Head; REPORT 'WEEK98 K98'), CPU only: the smallest exact best-first EAGLE-2 tree (t30b) whose
walk-strict COVERAGE is >= 98 % in every band to depth 30.
K = rows = K - 1 nodes. K <= 32: BigMoE's REAL tree dump (first K - 1 pops); every K: S9b's proxy (on-path children
exact from TOPK, off-path children = the true path's top-8 at the same depth ^ gamma). Walk = certify98's rule (R0 = 1).
Coverage at depth j: among the walk's checks whose true path is in the tree through j - 1 (and has a target at j), the
share whose depth-j true token is in the tree. Top-8 ceiling: the share of reached positions with y in the head's top-8.
Usage: BigMoE venv python -I -B tree_k98.py OUT.json"""
import os, sys, json, heapq
import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/EvalSuite/tools"))
import certify98 as C98                                     # noqa: E402

H = os.path.expanduser("~")
CE = f"{H}/BigMoE/data/deepdraft/ceiling"
SETS = {"mbpp": ("mbppbank", 6.92), "he": ("heplus24", 13.77)}
DEP, K8 = 31, 8
KS = (16, 32, 48, 64, 96, 128)
BANDS = [(0, 0), (1, 2), (3, 6), (7, 14), (15, 22), (23, 30)]
rng = np.random.default_rng(1115)
lines = []


def say(s=""):
    print(s, flush=True); lines.append(s)


def load(tag, published):
    cz, tz = np.load(f"{CE}/cells_t30b_{tag}.npz"), np.load(f"{CE}/topk_t30b_{tag}.npz")
    for k in ("depth", "fi", "row", "right"):
        assert (cz[k] == tz[k]).all()
    dep, right, yr = tz["depth"].astype(int), tz["right"].astype(int), tz["y_rank"].astype(int)
    yr = np.where((yr == 0) & (right == 0), 1, yr)
    tp, row, fi = tz["top_p"].astype(np.float32), tz["row"].astype(np.int64), tz["fi"].astype(np.int64)
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0)); en = np.append(st[1:], len(dep))
    ref = C98.windows(f"{CE}/cells_t30b_{tag}.npz")
    W, k32 = [], []
    for wi, (s, t) in enumerate(zip(st, en)):
        dw = dep[s:t]; n0 = int((dw == 0).sum())
        rt = np.full((n0 + 1, DEP), -1, np.int8); R = np.full((n0 + 1, DEP), -1, np.int16)
        Q = np.zeros((n0 + 1, DEP, K8), np.float32)
        for j in range(DEP):
            m = dw == j; c = int(m.sum())
            rt[:c, j] = right[s:t][m]; R[:c, j] = yr[s:t][m]; Q[:c, j] = tp[s:t][m]
        assert (rt == ref[wi][0]).all()
        R[rt < 0] = -1
        k32 += C98.kept_walk(rt, C98.R0, 31)
        W.append({"R": R, "Q": Q, "fi": int(fi[s]), "an": row[s:t][dw == 0] - 1, "n": n0 + 1})
    assert abs(np.mean(k32) - published) < 0.01, f"{tag}: chain-32 {np.mean(k32):.2f} != {published}"
    # the REAL trees: accepted per anchor for K <= 32 (max d + 1 over on_true nodes among the first K - 1 pops)
    tr = np.load(f"{CE}/tree_t30b_{tag}.npz")
    rowof = {X["fi"]: {int(a): i for i, a in enumerate(X["an"])} for X in W}
    for X in W:
        X["real"] = {Kr: np.zeros(X["n"], np.int64) for Kr in KS if Kr <= 32}
    byfi = {X["fi"]: X for X in W}
    sel = tr["on_true"].astype(bool)
    for f_, a_, nd, d_ in zip(tr["fi"][sel], tr["an"][sel], tr["node"][sel], tr["depth"][sel]):
        X = byfi[int(f_)]; r = rowof[int(f_)][int(a_)]
        for Kr in X["real"]:
            if nd < Kr - 1 and d_ + 1 > X["real"][Kr][r]:
                X["real"][Kr][r] = int(d_) + 1
    say(f"SELF-CHECK {tag}: layout == certify98, chain-32 {np.mean(k32):.2f} == {published}; real trees loaded")
    return W


def proxy_check(Q, R, N, gamma):
    """S9b's EAGLE-2 (N - 1 pops): -> (accepted, proxied pops)"""
    hp = [(-float(Q[0, r]), 0, bool(R[0] == r), False) for r in range(K8) if Q[0, r] > 0]
    heapq.heapify(hp); acc = cnt = prox = 0
    while hp and cnt < N - 1:
        ns, j, on, px = heapq.heappop(hp); cnt += 1; prox += px
        if on:
            acc = j + 1
        if j + 1 >= DEP:
            continue
        for r in range(K8):
            q = float(Q[j + 1, r])
            if q > 0:
                heapq.heappush(hp, (ns * (q if on else q ** gamma), j + 1, bool(on and R[j + 1] == r), not on))
    return acc, prox


def run(W, N, kind, gamma=1):
    """the tree's walk: per window [kept sum, checks], coverage counts [2, 31] (covered, reached), ceiling counts, proxied
    share"""
    out, cov, ceil = [], np.zeros((len(W), 2, DEP)), np.zeros((len(W), DEP))
    prox = tot = 0
    for wi, X in enumerate(W):
        R, Q, n = X["R"], X["Q"], X["n"]
        p = C98.R0; ks = nc = 0
        while p < n:
            if kind == "real":
                a = int(X["real"][N][p])
            else:
                a, px = proxy_check(Q[p], R[p], N, gamma); prox += px; tot += N - 1
            for j in range(a):
                cov[wi, 0, j] += 1; cov[wi, 1, j] += 1; ceil[wi, j] += 1
            if a < DEP and R[p, a] >= 0:
                cov[wi, 1, a] += 1; ceil[wi, a] += int(R[p, a] < K8)
            ks += a + 1; nc += 1; p += a + 1
        out.append((ks, nc))
    return np.array(out), cov, ceil, (prox / tot if tot else 0.0)


def band_stats(cov, ceil):
    res = []
    for lo, hi in BANDS:
        c = cov[:, 0, lo:hi + 1].sum(1); r = cov[:, 1, lo:hi + 1].sum(1); cl = ceil[:, lo:hi + 1].sum(1)
        pt = c.sum() / max(r.sum(), 1)
        idx = rng.integers(0, len(c), (2000, len(c)))
        bs = c[idx].sum(1) / np.maximum(r[idx].sum(1), 1)
        res.append({"band": f"{lo}-{hi}" if hi > lo else f"{lo}", "cov": float(pt), "lb": float(np.percentile(bs, 2.5)),
                    "ub": float(np.percentile(bs, 97.5)), "n": int(r.sum()), "ceil8": float(cl.sum() / max(r.sum(), 1))})
    return res


RES = {}
DATA = {s: load(tag, pub) for s, (tag, pub) in SETS.items()}
for s, W in DATA.items():
    for N in KS:
        for kind, g in ([("real", 1)] if N <= 32 else []) + [("proxy", 1)] + ([("proxy", 0.5), ("proxy", 2)] if N >= 48 else []):
            kp, cov, ceil, pshare = run(W, N, kind, g)
            bs = band_stats(cov, ceil)
            key = f"{s}|K{N}|{kind}{'' if g == 1 else g}"
            RES[key] = {"kept": float(kp[:, 0].sum() / kp[:, 1].sum()), "bands": bs, "proxied_share": pshare}
            ok = all(b["cov"] >= 0.98 and b["lb"] >= 0.97 for b in bs)
            RES[key]["meets"] = ok
            say(f"{s} K{N:>3} {kind:5}{'' if g == 1 else ' g' + str(g):5} kept {RES[key]['kept']:6.3f} | " +
                " ".join(f"{b['band']}: {100 * b['cov']:.1f} [{100 * b['lb']:.1f}] (c8 {100 * b['ceil8']:.1f})" for b in bs) +
                f" | proxied {100 * pshare:.0f} % | {'MEETS' if ok else 'no'}")
say("\nK98 (smallest K meeting coverage >= 98 % with LB >= 97 % in every band, both sets; real where K <= 32, else the "
    "central proxy):")
k98 = None
for N in KS:
    kind = "real" if N <= 32 else "proxy"
    if all(RES[f"{s}|K{N}|{kind}"]["meets"] for s in SETS):
        k98 = N
        break
say(f"  K98 = {k98 if k98 else 'none <= 128'}")
for N in KS:
    kind = "real" if N <= 32 else "proxy"
    fails = {s: [b["band"] for b in RES[f"{s}|K{N}|{kind}"]["bands"] if not (b["cov"] >= 0.98 and b["lb"] >= 0.97)]
             for s in SETS}
    say(f"  K{N} ({kind}): failing bands MBPP {fails['mbpp'] or '-'} / HE+ {fails['he'] or '-'}")
OUT = sys.argv[1]
json.dump({"K98": k98, "res": RES}, open(OUT, "w"), indent=1)
open(OUT.replace(".json", ".txt"), "w").write("\n".join(lines) + "\n")


# ---- EXPLORATORY (added after the run, not judged): per-depth BEAM trees. At every depth keep the b best nodes by path
# probability among the kept nodes' top-8 children (proxy as above for off-path nodes), so the tree holds b nodes at each
# of the 31 depths: K = 1 + 31 b rows. Coverage targets each depth instead of the total expected acceptance.
def beam_check(Q, R, b):
    cands = [(float(Q[0, r]), bool(R[0] == r)) for r in range(K8) if Q[0, r] > 0]
    beams = sorted(cands, key=lambda x: -x[0])[:b]
    acc = 0
    for j in range(DEP):
        if not any(on for _, on in beams):
            break
        acc = j + 1
        if j + 1 >= DEP:
            break
        cands = [(sc * float(Q[j + 1, r]), bool(on and R[j + 1] == r)) for sc, on in beams for r in range(K8)
                 if Q[j + 1, r] > 0]
        beams = sorted(cands, key=lambda x: -x[0])[:b]
    return acc


def run_beam(W, b):
    out, cov, ceil = [], np.zeros((len(W), 2, DEP)), np.zeros((len(W), DEP))
    for wi, X in enumerate(W):
        R, Q, n = X["R"], X["Q"], X["n"]; p = C98.R0; ks = nc = 0
        while p < n:
            a = beam_check(Q[p], R[p], b)
            for j in range(a):
                cov[wi, 0, j] += 1; cov[wi, 1, j] += 1; ceil[wi, j] += 1
            if a < DEP and R[p, a] >= 0:
                cov[wi, 1, a] += 1; ceil[wi, a] += int(R[p, a] < K8)
            ks += a + 1; nc += 1; p += a + 1
        out.append((ks, nc))
    return np.array(out), cov, ceil


say("\nEXPLORATORY (added after the run, not judged): per-depth BEAM trees, b nodes per depth, K = 1 + 31 b, proxy")
for s, W in DATA.items():
    for b in (2, 3, 4, 6, 8):
        kp, cov, ceil = run_beam(W, b)
        bs = band_stats(cov, ceil)
        ok = all(x["cov"] >= 0.98 and x["lb"] >= 0.97 for x in bs)
        RES[f"{s}|beam{b}"] = {"K": 1 + 31 * b, "kept": float(kp[:, 0].sum() / kp[:, 1].sum()), "bands": bs, "meets": ok}
        say(f"{s} beam b{b} K{1 + 31 * b:>4} kept {RES[f'{s}|beam{b}']['kept']:6.3f} | " +
            " ".join(f"{x['band']}: {100 * x['cov']:.1f} [{100 * x['lb']:.1f}]" for x in bs) + f" | {'MEETS' if ok else 'no'}")
json.dump({"K98": k98, "res": RES}, open(OUT, "w"), indent=1)
open(OUT.replace(".json", ".txt"), "w").write("\n".join(lines) + "\n")
