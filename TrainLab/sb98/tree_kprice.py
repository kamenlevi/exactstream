"""WEEK98 S11 KPRICE (MTP Head; REPORT 'WEEK98 S11 KPRICE'), CPU only: kept per check vs K and the best-first shape for
t30b's EAGLE-2 trees on the 125B (lab numbers: teacher-forced along the true path, under the walk; not real decoding).
K = rows = K - 1 nodes. REAL to K 32 (BigMoE's tree dump: the first K - 1 pops); proxy at every K over a gamma grid
(S9b's: on-path children exact from TOPK, off-path children = the true path's top-8 at the same depth ^ gamma).
Calibrated gamma = the grid value whose K 16 + 32 kept best matches the real trees on both sets.
Shape at each K over the walk's check starts: mean nodes per depth, max depth (mean / p10 / p50 / p90), the share of
nodes off the #1 spine (any non-#1 step on their path), widths at depths 0-3.
Usage: BigMoE venv python -I -B tree_kprice.py OUT.json"""
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
GAMMAS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
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
    assert abs(np.mean(k32) - published) < 0.01
    # real trees: per anchor, accepted and the depth histogram / off-spine count of the first K - 1 pops
    tr = np.load(f"{CE}/tree_t30b_{tag}.npz")
    rowof = {X["fi"]: {int(a): i for i, a in enumerate(X["an"])} for X in W}
    byfi = {X["fi"]: X for X in W}
    for X in W:
        X["racc"] = {Kr: np.zeros(X["n"], np.int64) for Kr in (16, 32)}
        X["rhist"] = {Kr: np.zeros((X["n"], DEP), np.int16) for Kr in (16, 32)}
        X["roff"] = {Kr: np.zeros(X["n"], np.int16) for Kr in (16, 32)}
    nfi, nan_, nnode, ndep, ont, par = (tr[k] for k in ("fi", "an", "node", "depth", "on_true", "parent"))
    tops = tr["top_ids"]; tok = tr["token"]
    # off-spine: a node is on the #1 spine if every node on its path is its parent's top-1 child
    order = np.lexsort((nnode, nan_, nfi))
    spine = {}
    for i in order:
        key = (int(nfi[i]), int(nan_[i]))
        p = int(par[i])
        if p < 0:
            # a root child: on the spine if its token is the root's top-1 (the first pop of the anchor is node 0)
            sp = int(nnode[i]) == 0
        else:
            pk = key + (p,)
            psp, ptop = spine.get(pk, (False, -1))
            sp = psp and int(tok[i]) == ptop
        spine[key + (int(nnode[i]),)] = (sp, int(tops[i, 0]))
        X = byfi[key[0]]; r = rowof[key[0]][key[1]]
        for Kr in (16, 32):
            if nnode[i] < Kr - 1:
                X["rhist"][Kr][r, int(ndep[i])] += 1
                X["roff"][Kr][r] += int(not sp)
                if ont[i] and ndep[i] + 1 > X["racc"][Kr][r]:
                    X["racc"][Kr][r] = int(ndep[i]) + 1
    say(f"SELF-CHECK {tag}: layout == certify98, chain-32 {np.mean(k32):.2f} == {published}; real trees loaded")
    return W


def proxy_check(Q, R, N, gamma):
    """-> accepted, depth histogram of the N - 1 pops, off-spine pops"""
    hp = [(-float(Q[0, r]), 0, bool(R[0] == r), r == 0) for r in range(K8) if Q[0, r] > 0]
    heapq.heapify(hp); acc = cnt = off = 0; hist = np.zeros(DEP, np.int16)
    while hp and cnt < N - 1:
        ns, j, on, sp = heapq.heappop(hp); cnt += 1; hist[j] += 1; off += int(not sp)
        if on:
            acc = j + 1
        if j + 1 >= DEP:
            continue
        for r in range(K8):
            q = float(Q[j + 1, r])
            if q > 0:
                heapq.heappush(hp, (ns * (q if on else q ** gamma), j + 1, bool(on and R[j + 1] == r), sp and r == 0))
    return acc, hist, off


def run(W, N, kind, gamma=1.0):
    ks = nc = 0; hists, offs = [], []
    for X in W:
        p = C98.R0
        while p < X["n"]:
            if kind == "real":
                a = int(X["racc"][N][p]); h = X["rhist"][N][p]; o = int(X["roff"][N][p])
            else:
                a, h, o = proxy_check(X["Q"][p], X["R"][p], N, gamma)
            hists.append(h); offs.append(o)
            ks += a + 1; nc += 1; p += a + 1
    H_ = np.array(hists, float)
    maxd = np.array([np.flatnonzero(h).max() + 1 if h.any() else 0 for h in H_])
    return {"kept": ks / nc, "checks": nc, "nodes_per_depth": H_.mean(0).round(3).tolist(),
            "max_depth": {"mean": float(maxd.mean()), "p10": float(np.percentile(maxd, 10)),
                          "p50": float(np.percentile(maxd, 50)), "p90": float(np.percentile(maxd, 90))},
            "off_spine_share": float(np.sum(offs) / max(H_.sum(), 1)), "width_d0_3": H_.mean(0)[:4].round(2).tolist()}


RES = {}
DATA = {s: load(tag, pub) for s, (tag, pub) in SETS.items()}
for s, W in DATA.items():
    for N in KS:
        if N <= 32:
            RES[f"{s}|K{N}|real"] = run(W, N, "real")
        for g in GAMMAS:
            RES[f"{s}|K{N}|g{g}"] = run(W, N, "proxy", g)
# calibrated gamma: the smallest summed |log(proxy / real)| over K 16 + 32 on both sets
err = {g: sum(abs(np.log(RES[f"{s}|K{N}|g{g}"]["kept"] / RES[f"{s}|K{N}|real"]["kept"])) for s in SETS for N in (16, 32))
       for g in GAMMAS}
gc = min(err, key=err.get)
say("calibration (sum |log proxy / real| at K 16 + 32, both sets): " + ", ".join(f"g{g} {e:.4f}" for g, e in err.items())
    + f" -> calibrated gamma {gc}")
say("\nKEPT PER CHECK vs K (lab, exact, under the walk); real <= 32; proxy central = calibrated gamma; band = gamma 0.5 .. 2")
for s in SETS:
    say(f"  {s}: " + " | ".join(
        f"K{N} " + (f"real {RES[f'{s}|K{N}|real']['kept']:.2f}, " if N <= 32 else "")
        + f"proxy {RES[f'{s}|K{N}|g{gc}']['kept']:.2f} [{RES[f'{s}|K{N}|g0.5']['kept']:.2f}..{RES[f'{s}|K{N}|g2.0']['kept']:.2f}]"
        for N in KS))
say("\nSHAPE at each K (over the walk's check starts; real <= 32, else the calibrated proxy):")
for s in SETS:
    for N in KS:
        x = RES[f"{s}|K{N}|real"] if N <= 32 else RES[f"{s}|K{N}|g{gc}"]
        npd = x["nodes_per_depth"]; last = max(i for i, v in enumerate(npd) if v > 0.005) if any(v > 0.005 for v in npd) else 0
        say(f"  {s} K{N}: max depth mean {x['max_depth']['mean']:.1f} (p10 {x['max_depth']['p10']:.0f} / p50 "
            f"{x['max_depth']['p50']:.0f} / p90 {x['max_depth']['p90']:.0f}); off the #1 spine {100 * x['off_spine_share']:.0f} % "
            f"of nodes; widths d0-3 {x['width_d0_3']}; nodes per depth " + " ".join(f"{v:.1f}" for v in npd[:last + 1]))
OUT = sys.argv[1]
json.dump({"calibrated_gamma": gc, "calibration_err": err, "res": RES}, open(OUT, "w"), indent=1)
open(OUT.replace(".json", ".txt"), "w").write("\n".join(lines) + "\n")
