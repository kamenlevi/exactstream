"""WEEK98 S9 TREE-DYN pricing (MTP Head; REPORT 'WEEK98 S9 TREE-DYN'), CPU only.
A tree = a main chain of D nodes + branches: at main depth d the head's #2 (w = 2) or #2 and #3 (w = 3), each sibling
with L nodes (itself + L - 1 sub-chain nodes); rows = 1 + D + B (w - 1) L <= N. Teacher-forced and exact along the true
path: a check dies at the main chain's first miss k; a branch at k whose sibling is accepted carries the check on along
the sub-chain's cells (the table's own cells: that sub-chain feeds the true token). kept = accepted + 1; the walk moves on
by kept (certify98.kept_walk's rule; chain-N = cap N - 1).
Policies: STATIC (branches at main depths 0..B-1), DYN (s30 only: the earliest B main depths whose head confidence p1 or
log(p1/p2) is under tau; the decision at the first miss uses only depths at or before it), ORACLE (one branch with every
spare row, exactly at the first miss when a sibling is accepted; best D in-set). STATIC / DYN parameters are chosen on
the OTHER set of the same model (MBPP <-> HE) and read on this one.
Sets: the 125B = BigMoE's cells_t30b_mbppbank.npz + cells_t30b_heplus24.npz (read-only, certify98's window layout,
imported read-only); s30 = data/sb98/tree_s30_{mbpp,he}.npz (sb98/tree_cells.py).
Modes: exact; delta 1 (#1 accepted when right or its target gap >= -1; on the 125B a sibling counts only when it is the
true token = a lower bound; s30 also reports 'd1full', siblings accepted when IN).
Usage: python3 -I tree_price.py OUT.json  (the BigMoE venv's python: numpy)"""
import os, sys, json, itertools
import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/EvalSuite/tools"))
import certify98 as C98                                     # noqa: E402  (windows, kept_walk, R0, DEPTHS; read-only)

H = os.path.expanduser("~")
OUT = sys.argv[1]
DEP = 31
R32 = 1.625
rng = np.random.default_rng(1115)
lines = []


def say(s=""):
    print(s, flush=True); lines.append(s)


def first_miss(A1):
    """FM[p, j0] = the first depth >= j0 whose #1 is not accepted (31 if none)"""
    n = A1.shape[0]; FM = np.full((n, DEP + 1), DEP, np.int16)
    for j in range(DEP - 1, -1, -1):
        FM[:, j] = np.where(A1[:, j] == 1, FM[:, j + 1], j)
    return FM


def prep(A1, S2, S3, CF=None):
    """per anchor, everything a policy needs at the main chain's first miss k (policy-independent)"""
    FM = first_miss(A1); n = A1.shape[0]; ar = np.arange(n)
    k = FM[:, 0].astype(np.int64); kk = np.minimum(k, DEP - 1)
    R = np.where(A1 == 1, 0, np.where(A1 < 0, -1, np.where(S2, 1, np.where(S3, 2, 3)))).astype(np.int8)
    return {"n": n, "ar": ar, "k": k, "kk": kk, "tgt": A1[ar, kk] >= 0, "S2k": S2[ar, kk], "S3k": S3[ar, kk], "R": R,
            "run0": FM[ar, np.minimum(kk + 1, DEP)].astype(np.int64) - (kk + 1), "CF": CF, "cache": {}}


def load125(path, published):
    d = np.load(path)
    dep, right, rank, dlp = d["depth"].astype(int), d["right"].astype(int), d["rank"].astype(int), d["dlp"].astype(float)
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0)); en = np.append(st[1:], len(dep))
    ref = C98.windows(path)
    W = {"exact": [], "d1": []}
    k32 = []
    for wi, (s, t) in enumerate(zip(st, en)):
        dw = dep[s:t]; n0 = int((dw == 0).sum())
        rt = np.full((n0 + 1, DEP), -1, np.int8); rk = np.full((n0 + 1, DEP), -1, np.int16)
        dl = np.full((n0 + 1, DEP), np.nan)
        for j in range(DEP):
            k = dw == j; c = int(k.sum())
            rt[:c, j] = right[s:t][k]; rk[:c, j] = rank[s:t][k]; dl[:c, j] = dlp[s:t][k]
        assert rt.shape == ref[wi][0].shape and (rt == ref[wi][0]).all(), f"{path}: window {wi} layout != certify98"
        k32 += C98.kept_walk(rt, C98.R0, 31)
        a1d = np.where((rt == 0) & (dl >= -1.0), 1, rt).astype(np.int8)
        W["exact"].append(prep(rt, rk == 1, rk == 2))
        W["d1"].append(prep(a1d, rk == 1, rk == 2))
    say(f"SELF-CHECK {os.path.basename(path)}: certify98 layout PASS; chain-32 kept/check {np.mean(k32):.2f} vs published "
        f"{published} -> {'PASS' if abs(np.mean(k32) - published) < 0.01 else 'FAIL'}")
    return W, C98.R0


def loads30(path, ref32):
    z = np.load(path)
    W = {"exact": [], "d1": [], "d1full": []}
    for i in range(len(z["files"])):
        rt, r3, hlp, inn, dlp = (z[f"{i}_{k}"] for k in ("right", "rank3", "hlp", "inn", "dlp"))
        hlp = hlp.astype(np.float32); dlp = dlp.astype(np.float32)
        CF = np.stack([np.exp(hlp[..., 0]), hlp[..., 0] - hlp[..., 1]], -1)      # p1, log(p1 / p2)
        a1d = np.where((rt == 0) & (dlp >= -1.0), 1, rt).astype(np.int8)
        mm = int(((a1d == 1) != (inn[..., 0] == 1)).sum())                   # fp16 rounding at the 1-nat edge only
        if mm:
            say(f"  note {os.path.basename(path)} window {i}: #1 IN vs right-or-gap differ on {mm} cells (fp16 dlp)")
        Q = np.nan_to_num(np.exp(hlp), nan=0.0)                               # the head's #1..#3 probabilities
        for mode, X in (("exact", prep(rt, r3 == 1, r3 == 2, CF)), ("d1", prep(a1d, r3 == 1, r3 == 2, CF)),
                        ("d1full", prep(a1d, inn[..., 1] == 1, inn[..., 2] == 1, CF))):
            X["Q"] = Q; W[mode].append(X)
    r0 = int(z["gen"][0] - 1 - z["a0"][0])
    assert all(int(g - 1 - a) == r0 for g, a in zip(z["gen"], z["a0"])) and r0 == 1
    k = walk(W["exact"], r0, 32, 31, 0, 0, 2, "static")
    say(f"SELF-CHECK {os.path.basename(path)}: chain-32 exact kept/check {k[:, 0].sum() / k[:, 1].sum():.3f} vs S7's "
        f"read-out {ref32} -> {'PASS' if abs(k[:, 0].sum() / k[:, 1].sum() - ref32) < 0.001 else 'FAIL'}")
    return W, r0


def walk(Ws, r0, N, D, L, B, w, pol, sig=0, tau=None):
    """per window [kept sum, checks] for one tree policy (vectorized per anchor, then the walk's jumps)"""
    out = np.zeros((len(Ws), 2), np.int64)
    for wi, X in enumerate(Ws):
        k, kk = X["k"], X["kk"]
        if pol == "static":
            br = k < B
        elif pol == "oracle":
            br = np.ones(X["n"], bool)
        else:
            key = (sig, tau)
            if key not in X["cache"]:
                low = X["CF"][..., sig] < tau                              # NaN (no target) -> False
                before = np.concatenate([np.zeros((X["n"], 1), np.int64), np.cumsum(low, 1)], 1)
                X["cache"][key] = (low[X["ar"], kk], before[X["ar"], kk])
            lowk, befk = X["cache"][key]
            br = lowk & (befk < B)
        ok = X["S2k"] | (X["S3k"] & (w == 3 or pol == "oracle"))
        acc = np.where(k >= D, D, np.where(X["tgt"] & br & ok, kk + 1 + np.minimum(X["run0"], L - 1), k)).tolist()
        n = X["n"]; p = r0; ks = nc = 0
        while p < n:
            a = acc[p]; ks += a + 1; nc += 1; p += a + 1
        out[wi] = ks, nc
    return out


def kpc(x):
    return x[:, 0].sum() / x[:, 1].sum()


def grid(N, dyn=False):
    for w in (2, 3):
        for B in ((1, 2, 3, 4, 6, 8) if dyn else (1, 2, 3, 4, 5, 6, 8)):
            for L in ((2, 3, 4, 6, 8, 12) if dyn else (1, 2, 3, 4, 5, 6, 8, 10, 12, 15)):
                D = N - 1 - B * (w - 1) * L
                if D >= max(B, 1):
                    if dyn:
                        for sig, taus in ((0, (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99)), (1, (0.5, 1, 1.5, 2, 3, 4, 6))):
                            for tau in taus:
                                yield dict(D=D, L=L, B=B, w=w, sig=sig, tau=tau)
                    else:
                        yield dict(D=D, L=L, B=B, w=w)


def best(Ws, r0, N, pol):
    """the best parameters on these windows (kept/check)"""
    bk, bp = -1, None
    for P in grid(N, pol == "dyn"):
        k = kpc(walk(Ws, r0, N, P["D"], P["L"], P["B"], P["w"], pol, P.get("sig", 0), P.get("tau")))
        if k > bk:
            bk, bp = k, P
    return bp, bk


def oracle_best(Ws, r0, N):
    """the best FIXED main length D (D = N - 1: no branch) with one branch of every spare row at the first miss"""
    bk, bD = -1, None
    for D in range(1, N):
        k = kpc(walk(Ws, r0, N, D, N - 1 - D, 1, 2, "oracle"))
        if k > bk:
            bk, bD = k, D
    return bD, bk


def oracle_adaptive(Ws, r0, N):
    """per check: if a sibling is right at the first miss k, main chain 0..k plus one branch of the N - 2 - k spare
    nodes at k; else the plain chain (the ceiling for single-level #2 / #3 trees: a tree must hold #1 up to the branch)"""
    out = np.zeros((len(Ws), 2), np.int64)
    for wi, X in enumerate(Ws):
        k, kk = X["k"], X["kk"]
        L = (N - 1) - (k + 1)
        ok = X["tgt"] & (X["S2k"] | X["S3k"]) & (k < N - 1) & (L >= 1)
        acc = np.where(ok, kk + 1 + np.minimum(X["run0"], np.maximum(L, 1) - 1), np.minimum(k, N - 1)).tolist()
        n = X["n"]; p = r0; ks = nc = 0
        while p < n:
            a = acc[p]; ks += a + 1; nc += 1; p += a + 1
        out[wi] = ks, nc
    return out


def oracle_multi(Ws, r0, N):
    """ADDED AFTER THE RUN (not pre-registered): the ceiling for ANY #2 / #3 tree. Per check, a perfect tree rescues
    every miss on the true path whose true token is the head's #2 (+1 row: the wrong #1 must be in the tree) or #3
    (+2 rows: #1 and #2), while 1 + accepted + extra rows <= N"""
    out = np.zeros((len(Ws), 2), np.int64)
    for wi, X in enumerate(Ws):
        R, n = X["R"], X["n"]; acc = np.zeros(n, np.int64)
        for p in range(r0, n):
            J = m = 0
            for j in range(DEP):
                r = int(R[p, j])
                if r < 0 or r > 2 or J + 1 + m + r > N - 1:
                    break
                J += 1; m += r
            acc[p] = J
        acc = acc.tolist(); p = r0; ks = nc = 0
        while p < n:
            a = acc[p]; ks += a + 1; nc += 1; p += a + 1
        out[wi] = ks, nc
    return out


def eagle2(Ws, r0, N):
    """ADDED AFTER THE RUN (exploratory, s30 only): an EAGLE-2-style dynamic tree. Per check, best-first selection of
    the N - 1 rank-sequence nodes with the highest path score (the product of the head's probabilities of the chosen
    slots, top-3 per node). Proxy: an off-path node at depth i uses the true path's top-3 probabilities at depth i
    (the cells hold only the true path's). Accepted = the longest prefix of the true path's rank sequence selected."""
    import heapq
    out = np.zeros((len(Ws), 2), np.int64)
    for wi, X in enumerate(Ws):
        R, Q, n = X["R"], X["Q"], X["n"]; p = r0; ks = nc = 0
        while p < n:
            q = Q[p]; sel = set()
            hp = [(-float(q[0, r]), (r,)) for r in range(3) if q[0, r] > 0]
            heapq.heapify(hp)
            while hp and len(sel) < N - 1:
                ns, t = heapq.heappop(hp); sel.add(t); j = len(t)
                if j < DEP:
                    for r in range(3):
                        if q[j, r] > 0:
                            heapq.heappush(hp, (ns * float(q[j, r]), t + (r,)))
            t = (); a = 0
            for j in range(DEP):
                r = int(R[p, j])
                if r < 0 or r > 2 or t + (r,) not in sel:
                    break
                t = t + (r,); a += 1
            ks += a + 1; nc += 1; p += a + 1
        out[wi] = ks, nc
    return out


def seq_tree(Wo, r0o, N):
    """Sequoia-style optimal static tree: the N - 1 most frequent true-path rank prefixes (slots 0 / 1 / 2) at the
    check starts of the OTHER set's chain-N walk (prefix-closed)"""
    from collections import Counter
    cnt = Counter()
    for X in Wo:
        R, n = X["R"], X["n"]
        acc = np.minimum(X["k"], N - 1).tolist(); p = r0o
        while p < n:
            t = ()
            for j in range(DEP):
                r = int(R[p, j])
                if r < 0 or r > 2:
                    break
                t = t + (r,); cnt[t] += 1
            p += acc[p] + 1
    T = set()
    for t, _ in sorted(cnt.items(), key=lambda x: (-x[1], len(x[0]))):
        if len(T) >= N - 1:
            break
        if len(t) == 1 or t[:-1] in T:
            T.add(t)
    return T


def walk_seq(Ws, r0, T):
    out = np.zeros((len(Ws), 2), np.int64)
    for wi, X in enumerate(Ws):
        R, n = X["R"], X["n"]; p = r0; ks = nc = 0
        while p < n:
            t = (); a = 0
            for j in range(DEP):
                r = int(R[p, j])
                if r < 0 or r > 2 or t + (r,) not in T:
                    break
                t = t + (r,); a += 1
            ks += a + 1; nc += 1; p += a + 1
        out[wi] = ks, nc
    return out


def ratio_ci(a, b):
    """paired window bootstrap of kpc(a) / kpc(b)"""
    idx = rng.integers(0, len(a), (2000, len(a)))
    r = (a[idx, 0].sum(1) / a[idx, 1].sum(1)) / (b[idx, 0].sum(1) / b[idx, 1].sum(1))
    return kpc(a) / kpc(b), np.percentile(r, 2.5), np.percentile(r, 97.5)


RES = {}
MODELS = {
    "125B": {"sets": {"mbpp": (f"{H}/BigMoE/data/deepdraft/ceiling/cells_t30b_mbppbank.npz", 6.92),
                      "he": (f"{H}/BigMoE/data/deepdraft/ceiling/cells_t30b_heplus24.npz", 13.77)},
             "modes": ("exact", "d1"), "pols": ("static",)},
    "s30": {"sets": {"mbpp": (f"{H}/TrainLab/data/sb98/tree_s30_mbpp.npz", 4.319),
                     "he": (f"{H}/TrainLab/data/sb98/tree_s30_he.npz", 6.239)},
            "modes": ("exact", "d1", "d1full"), "pols": ("static", "dyn")}}
for mname, M in MODELS.items():
    data = {}
    for sname, (path, ref) in M["sets"].items():
        data[sname] = load125(path, ref) if mname == "125B" else loads30(path, ref)
    for mode in M["modes"]:
        for sname in M["sets"]:
            other = "he" if sname == "mbpp" else "mbpp"
            Ws, r0 = data[sname][0][mode], data[sname][1]
            Wo, r0o = data[other][0][mode], data[other][1]
            c16 = walk(Ws, r0, 16, 15, 0, 0, 2, "static"); c32 = walk(Ws, r0, 32, 31, 0, 0, 2, "static")
            row = {"chain16": kpc(c16), "chain32": kpc(c32), "chain32_vs_16": kpc(c32) / kpc(c16)}
            for N in (16, 32):
                for pol in M["pols"]:
                    P, _ = best(Wo, r0o, N, pol)                  # chosen on the other set
                    t = walk(Ws, r0, N, P["D"], P["L"], P["B"], P["w"], pol, P.get("sig", 0), P.get("tau"))
                    r, lo, hi = ratio_ci(t, c16)
                    Pin, kin = best(Ws, r0, N, pol)               # in-set best, for reference
                    row[f"{pol}{N}"] = {"params": P, "kpc": kpc(t), "vs_chain16": r, "ci95": [lo, hi], "inset_best": kin,
                                        "inset_params": Pin}
                D, ko = oracle_best(Ws, r0, N)
                row[f"oracleD{N}"] = {"D": D, "kpc": ko, "vs_chain16": ko / kpc(c16)}
                if "Q" in Ws[0]:
                    e2 = eagle2(Ws, r0, N)
                    r_, lo_, hi_ = ratio_ci(e2, c16)
                    row[f"eagle2{N}"] = {"kpc": kpc(e2), "vs_chain16": r_, "ci95": [lo_, hi_]}
                om = oracle_multi(Ws, r0, N)
                row[f"oraclemulti{N}"] = {"kpc": kpc(om), "vs_chain16": kpc(om) / kpc(c16)}
                oa = oracle_adaptive(Ws, r0, N)
                row[f"oracle{N}"] = {"kpc": kpc(oa), "vs_chain16": kpc(oa) / kpc(c16)}
                T = seq_tree(Wo, r0o, N)
                ts = walk_seq(Ws, r0, T)
                r, lo, hi = ratio_ci(ts, c16)
                row[f"seq{N}"] = {"kpc": kpc(ts), "vs_chain16": r, "ci95": [lo, hi], "nodes": len(T),
                                  "branch_nodes": sum(1 for t in T if any(x > 0 for x in t)),
                                  "max_depth": max(len(t) for t in T)}
            RES[f"{mname}|{sname}|{mode}"] = row
            say(f"\n== {mname} {sname} {mode}: chain-16 {row['chain16']:.3f}, chain-32 {row['chain32']:.3f} "
                f"(x{row['chain32_vs_16']:.3f})")
            for N in (16, 32):
                for pol in M["pols"]:
                    e = row[f"{pol}{N}"]
                    say(f"  {pol}-{N}: {e['kpc']:.3f} = x{e['vs_chain16']:.3f} chain-16 [{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}]"
                        f" with {e['params']} (from {other}); in-set best {e['inset_best']:.3f}")
                e = row[f"seq{N}"]
                say(f"  seq-{N} (optimal static, from {other}): {e['kpc']:.3f} = x{e['vs_chain16']:.3f} chain-16 "
                    f"[{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}] ({e['nodes']} nodes, {e['branch_nodes']} off the #1 path, "
                    f"depth {e['max_depth']})")
                if f"eagle2{N}" in row:
                    e = row[f"eagle2{N}"]
                    say(f"  eagle2-{N} (added after the run; off-path proxy): {e['kpc']:.3f} = x{e['vs_chain16']:.3f} chain-16 "
                        f"[{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}]")
                o, od = row[f"oracle{N}"], row[f"oracleD{N}"]
                say(f"  oracle-{N} (per check): {o['kpc']:.3f} = x{o['vs_chain16']:.3f} chain-16; fixed-D oracle "
                    f"{od['kpc']:.3f} (D {od['D']}); multi-level oracle (added after the run) "
                    f"{row[f'oraclemulti{N}']['kpc']:.3f} = x{row[f'oraclemulti{N}']['vs_chain16']:.3f}")

say("\nVERDICTS (pre-registered: SAME-COST WIN tree-16 >= 1.03 x chain-16 with CI > 1 on both sets; 32 ROWS PAY "
    "tree-32 / chain-16 >= 1.625 on both sets)")
for mname, M in MODELS.items():
    for mode in M["modes"]:
        for pol in M["pols"] + ("seq", "oracle"):
            w16 = all((RES[f"{mname}|{s}|{mode}"][f"{pol}16"]["vs_chain16"] >= 1.03 and
                       (pol == "oracle" or RES[f"{mname}|{s}|{mode}"][f"{pol}16"]["ci95"][0] > 1)) for s in M["sets"])
            p32 = all(RES[f"{mname}|{s}|{mode}"][f"{pol}32"]["vs_chain16"] >= R32 for s in M["sets"])
            v16 = " / ".join(f"{RES[f'{mname}|{s}|{mode}'][f'{pol}16']['vs_chain16']:.3f}" for s in M["sets"])
            v32 = " / ".join(f"{RES[f'{mname}|{s}|{mode}'][f'{pol}32']['vs_chain16']:.3f}" for s in M["sets"])
            say(f"  {mname} {mode} {pol}: tree-16 x{v16} -> {'WIN' if w16 else 'no'}; tree-32 x{v32} -> "
                f"{'PAYS' if p32 else 'no'} (MBPP / HE)")
# the 125B DYN estimate from s30's efficiency e = (DYN - STATIC) / (ORACLE - STATIC)
say("\n125B DYN estimate (reported, not judged): STATIC + e x (ORACLE - STATIC), e from s30 (same set, mode d1 <- d1)")
for mode in ("exact", "d1"):
    for s in ("mbpp", "he"):
        for N in (16, 32):
            a = RES[f"s30|{s}|{mode}"]; b = RES[f"125B|{s}|{mode}"]
            den = a[f"oracle{N}"]["kpc"] - a[f"static{N}"]["kpc"]
            e = (a[f"dyn{N}"]["kpc"] - a[f"static{N}"]["kpc"]) / den if den > 1e-9 else 0.0
            est = b[f"static{N}"]["kpc"] + e * (b[f"oracle{N}"]["kpc"] - b[f"static{N}"]["kpc"])
            RES[f"125B|{s}|{mode}"][f"dyn{N}_est"] = {"e": e, "kpc": est, "vs_chain16": est / b["chain16"]}
            say(f"  {mode} {s} {N} rows: e {e:.2f} -> ~{est:.3f} = x{est / b['chain16']:.3f} chain-16")
say("\n125B EAGLE-2 estimate (added after the run, not judged): chain-16 x (1 + f x (multi-oracle / chain-16 - 1)), "
    "f = s30's (eagle2 - chain-16) / (multi-oracle - chain-16), same set and mode")
for mode in ("exact", "d1"):
    for s_ in ("mbpp", "he"):
        for N in (16, 32):
            a = RES[f"s30|{s_}|{mode}"]; b = RES[f"125B|{s_}|{mode}"]
            f = (a[f"eagle2{N}"]["vs_chain16"] - 1) / (a[f"oraclemulti{N}"]["vs_chain16"] - 1)
            est = 1 + f * (b[f"oraclemulti{N}"]["vs_chain16"] - 1)
            RES[f"125B|{s_}|{mode}"][f"eagle2{N}_est"] = {"f": f, "vs_chain16": est}
            say(f"  {mode} {s_} {N} rows: f {f:.2f} -> ~x{est:.3f} chain-16 (multi-oracle x{b[f'oraclemulti{N}']['vs_chain16']:.3f})")
json.dump(RES, open(OUT, "w"), indent=1, default=float)
open(OUT.replace(".json", ".txt"), "w").write("\n".join(lines) + "\n")
