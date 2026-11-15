"""WEEK98 S9b EAGLE-2 ON THE 125B (MTP Head; REPORT 'WEEK98 S9b' + amendment 1), CPU only.
Data: BigMoE 9f49e03, topk_t30b_{heplus24,mbppbank}.npz: per cell (cells_t30b_<set>.npz order) the head's top-8 ids /
probabilities at the on-true-path node (teacher-forced), y, y_rank (y's rank in the head's order), right, walk_start;
cells' dlp for delta-1 #1 acceptance. Layout = certify98.windows (imported read-only), walk from R0 = 1.
Policies (kept/check; N rows = 1 + nodes):
  chain-N; EAGLE-2-N (best-first by head path probability, up to 8 children per node; nodes whose parent is on the true
  path are exact; descendants of a wrong sibling are proxied by the true path's top-8 at the same depth to the power
  gamma: 1 central, 0.5 pessimistic, 2 optimistic, 'worst' = a no-decay #1 chain per wrong sibling); DYN single-level
  (S9's, parameters from the other set); STATIC, SEQ; the per-check single-level oracle and the multi-level oracle
  (rescue at rank r costs r rows for the wrong siblings above y).
Modes: exact (judged); d1 (#1 accepted when right or dlp >= -1; a sibling only when it is y: a lower bound).
Usage: BigMoE venv python -I -B tree_eagle125.py OUT.json"""
import os, sys, json, heapq
from collections import Counter
import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/EvalSuite/tools"))
import certify98 as C98                                     # noqa: E402

H = os.path.expanduser("~")
CE = f"{H}/BigMoE/data/deepdraft/ceiling"
SETS = {"mbpp": ("mbppbank", 6.92), "he": ("heplus24", 13.77)}
OUT = sys.argv[1]
DEP, K = 31, 8
rng = np.random.default_rng(1115)
lines = []


def say(s=""):
    print(s, flush=True); lines.append(s)


def load(tag, published):
    cz, tz = np.load(f"{CE}/cells_t30b_{tag}.npz"), np.load(f"{CE}/topk_t30b_{tag}.npz")
    for k in ("depth", "fi", "row", "right"):
        assert (cz[k] == tz[k]).all(), f"{tag}: {k} differs between cells and topk"
    dep = tz["depth"].astype(int); right = tz["right"].astype(int); yr = tz["y_rank"].astype(int)
    ties = (yr == 0) & (right == 0)
    yr = np.where(ties, 1, yr)                               # bf16 ties: y = #2 at equal p (BigMoE)
    tp = tz["top_p"].astype(np.float32); dlp = cz["dlp"].astype(float); ws = tz["walk_start"]
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0)); en = np.append(st[1:], len(dep))
    ref = C98.windows(f"{CE}/cells_t30b_{tag}.npz")
    W, k32, ws_ok = [], [], 0
    for wi, (s, t) in enumerate(zip(st, en)):
        dw = dep[s:t]; n0 = int((dw == 0).sum())
        rt = np.full((n0 + 1, DEP), -1, np.int8); R = np.full((n0 + 1, DEP), -1, np.int16)
        Q = np.zeros((n0 + 1, DEP, K), np.float32); dl = np.full((n0 + 1, DEP), np.nan)
        wst = np.zeros(n0 + 1, bool)
        for j in range(DEP):
            m = dw == j; c = int(m.sum())
            rt[:c, j] = right[s:t][m]; R[:c, j] = yr[s:t][m]; Q[:c, j] = tp[s:t][m]; dl[:c, j] = dlp[s:t][m]
            if j == 0:
                wst[:c] = ws[s:t][m]
        assert rt.shape == ref[wi][0].shape and (rt == ref[wi][0]).all(), f"{tag}: window {wi} layout != certify98"
        kw = C98.kept_walk(rt, C98.R0, 31); k32 += kw
        starts = np.cumsum([C98.R0] + kw[:-1])
        ws_ok += int(np.array_equal(np.flatnonzero(wst), starts[starts < n0]))   # the layout's last row has no depth 0
        W.append({"rt": rt, "R": R, "Q": Q, "dl": dl})
    say(f"SELF-CHECK {tag}: layout == certify98, cells == topk (depth / fi / row / right); chain-32 {np.mean(k32):.2f} "
        f"vs published {published} -> {'PASS' if abs(np.mean(k32) - published) < 0.01 else 'FAIL'}; walk_start == "
        f"the chain-32 walk's starts in {ws_ok} / {len(W)} windows; bf16 ties {int(ties.sum())} -> y = #2")
    return W


def mode_view(X, mode):
    """the true path's rank sequence for a mode: 0 = #1 accepted, r = y is the head's r-th (a sibling), -1 none"""
    R = X["R"].astype(np.int16).copy()
    if mode == "d1":
        R[(X["rt"] == 0) & (X["dl"] >= -1.0)] = 0
    R[X["rt"] < 0] = -1
    return R


def walk_acc(acc, n, r0=C98.R0):
    p = r0; ks = nc = 0
    while p < n:
        a = acc[p]; ks += a + 1; nc += 1; p += a + 1
    return ks, nc


def kpc(x):
    return x[:, 0].sum() / x[:, 1].sum()


def ratio_ci(a, b):
    idx = rng.integers(0, len(a), (2000, len(a)))
    r = (a[idx, 0].sum(1) / a[idx, 1].sum(1)) / (b[idx, 0].sum(1) / b[idx, 1].sum(1))
    return float(kpc(a) / kpc(b)), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def chain(W, mode, N):
    out = []
    for X in W:
        R = mode_view(X, mode); n = R.shape[0]
        k = np.where((R != 0).any(1), (R != 0).argmax(1), DEP)
        out.append(walk_acc(np.minimum(k, N - 1).tolist(), n))
    return np.array(out)


def eagle2_check(Q, R, N, gamma):
    """one check: Q [31, 8] the head's top-8 at the on-path nodes, R [31] the true path's ranks; -> accepted"""
    hp = []
    for r in range(K):
        if Q[0, r] > 0:
            hp.append((-float(Q[0, r]), 0, bool(R[0] == r)))
    heapq.heapify(hp)
    acc = cnt = 0
    while hp and cnt < N - 1:
        ns, j, on = heapq.heappop(hp); cnt += 1
        if on:
            acc = j + 1
        if j + 1 >= DEP:
            continue
        if on:
            for r in range(K):
                q = float(Q[j + 1, r])
                if q > 0:
                    heapq.heappush(hp, (ns * q, j + 1, bool(R[j + 1] == r)))
        elif gamma == "worst":
            heapq.heappush(hp, (ns, j + 1, False))
        else:
            for r in range(K):
                q = float(Q[j + 1, r])
                if q > 0:
                    heapq.heappush(hp, (ns * q ** gamma, j + 1, False))
    return acc


def eagle2(W, mode, N, gamma):
    out = []
    for X in W:
        R = mode_view(X, mode); Q = X["Q"]; n = R.shape[0]
        p = C98.R0; ks = nc = 0
        while p < n:
            a = eagle2_check(Q[p], R[p], N, gamma); ks += a + 1; nc += 1; p += a + 1
        out.append((ks, nc))
    return np.array(out)


def single(W, mode, N, D, L, B, w, pol, sig=0, tau=None):
    """S9's single-level policies: STATIC (branches at depths 0..B-1), DYN (earliest B depths with confidence < tau),
    ORACLE (per check: main 0..k + one branch of every spare node at the first miss)"""
    out = []
    for X in W:
        R = mode_view(X, mode); n = R.shape[0]; ar = np.arange(n)
        FM = np.full((n, DEP + 1), DEP, np.int16)
        for j in range(DEP - 1, -1, -1):
            FM[:, j] = np.where(R[:, j] == 0, FM[:, j + 1], j)
        k = FM[:, 0].astype(np.int64); kk = np.minimum(k, DEP - 1)
        rk = R[ar, kk]; tgt = rk >= 0
        sib = (rk == 1) | ((rk == 2) & (w == 3 or pol == "oracle"))
        run0 = FM[ar, np.minimum(kk + 1, DEP)].astype(np.int64) - (kk + 1)
        if pol == "oracle":
            Lp = (N - 1) - (k + 1)
            ok = tgt & sib & (k < N - 1) & (Lp >= 1)
            acc = np.where(ok, kk + 1 + np.minimum(run0, np.maximum(Lp, 1) - 1), np.minimum(k, N - 1))
        else:
            if pol == "static":
                br = k < B
            else:
                Qp = X["Q"]
                cf = Qp[..., 0] if sig == 0 else np.log(np.maximum(Qp[..., 0], 1e-12) / np.maximum(Qp[..., 1], 1e-12))
                low = (cf < tau) & (R >= 0)
                before = np.concatenate([np.zeros((n, 1), np.int64), np.cumsum(low, 1)], 1)
                br = low[ar, kk] & (before[ar, kk] < B)
            acc = np.where(k >= D, D, np.where(tgt & br & sib, kk + 1 + np.minimum(run0, L - 1), k))
        out.append(walk_acc(acc.tolist(), n))
    return np.array(out)


def grid(N, dyn):
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


def best_single(W, mode, N, pol):
    bk, bp = -1, None
    for P in grid(N, pol == "dyn"):
        k = kpc(single(W, mode, N, P["D"], P["L"], P["B"], P["w"], pol, P.get("sig", 0), P.get("tau")))
        if k > bk:
            bk, bp = k, P
    return bp


def oracle_multi(W, mode, N):
    out = []
    for X in W:
        R = mode_view(X, mode); n = R.shape[0]; acc = [0] * n
        for p in range(C98.R0, n):
            J = m = 0
            for j in range(DEP):
                r = int(R[p, j])
                if r < 0 or r >= K or J + 1 + m + r > N - 1:
                    break
                J += 1; m += r
            acc[p] = J
        out.append(walk_acc(acc, n))
    return np.array(out)


def seq_tree(Wo, mode, N):
    cnt = Counter()
    for X in Wo:
        R = mode_view(X, mode); n = R.shape[0]
        k = np.where((R != 0).any(1), (R != 0).argmax(1), DEP)
        acc = np.minimum(k, N - 1).tolist(); p = C98.R0
        while p < n:
            t = ()
            for j in range(DEP):
                r = int(R[p, j])
                if r < 0 or r >= K:
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


def seq_walk(W, mode, T):
    out = []
    for X in W:
        R = mode_view(X, mode); n = R.shape[0]; p = C98.R0; ks = nc = 0
        while p < n:
            t = (); a = 0
            for j in range(DEP):
                r = int(R[p, j])
                if r < 0 or t + (r,) not in T:
                    break
                t = t + (r,); a += 1
            ks += a + 1; nc += 1; p += a + 1
        out.append((ks, nc))
    return np.array(out)


DATA = {s: load(tag, pub) for s, (tag, pub) in SETS.items()}
for s, (tag, pub) in SETS.items():
    v = kpc(chain(DATA[s], "exact", 32))
    say(f"SELF-CHECK {tag}: this script's chain-32 (from y_rank) {v:.2f} vs published {pub} -> "
        f"{'PASS' if abs(v - pub) < 0.01 else 'FAIL'}")
RES = {}
for mode in ("exact", "d1"):
    for s in SETS:
        o = "he" if s == "mbpp" else "mbpp"
        W, Wo = DATA[s], DATA[o]
        c16, c32 = chain(W, mode, 16), chain(W, mode, 32)
        row = {"chain16": float(kpc(c16)), "chain32": float(kpc(c32))}
        say(f"\n== 125B {s} {mode}: chain-16 {kpc(c16):.3f}, chain-32 {kpc(c32):.3f} (x{kpc(c32) / kpc(c16):.3f})")
        for N in (16, 32):
            for g in (1, 0.5, 2, "worst"):
                e = eagle2(W, mode, N, g); r, lo, hi = ratio_ci(e, c16)
                row[f"eagle2_{N}_{g}"] = {"kpc": float(kpc(e)), "vs_chain16": r, "ci95": [lo, hi]}
            ec = row[f"eagle2_{N}_1"]
            say(f"  EAGLE-2-{N}: {ec['kpc']:.3f} = x{ec['vs_chain16']:.3f} chain-16 [{ec['ci95'][0]:.3f}, {ec['ci95'][1]:.3f}]"
                f"; proxy band x{row[f'eagle2_{N}_0.5']['vs_chain16']:.3f} (pess.) .. x{row[f'eagle2_{N}_2']['vs_chain16']:.3f}"
                f" (opt.); worst case x{row[f'eagle2_{N}_worst']['vs_chain16']:.3f}")
            for pol in ("static", "dyn"):
                P = best_single(Wo, mode, N, pol)
                t = single(W, mode, N, P["D"], P["L"], P["B"], P["w"], pol, P.get("sig", 0), P.get("tau"))
                r, lo, hi = ratio_ci(t, c16)
                row[f"{pol}{N}"] = {"kpc": float(kpc(t)), "vs_chain16": r, "ci95": [lo, hi], "params": P}
                say(f"  {pol}-{N}: {kpc(t):.3f} = x{r:.3f} [{lo:.3f}, {hi:.3f}] with {P} (from {o})")
            T = seq_tree(Wo, mode, N); t = seq_walk(W, mode, T); r, lo, hi = ratio_ci(t, c16)
            row[f"seq{N}"] = {"kpc": float(kpc(t)), "vs_chain16": r, "ci95": [lo, hi],
                              "off_path_nodes": sum(1 for x in T if any(v > 0 for v in x))}
            oa = single(W, mode, N, 0, 0, 0, 3, "oracle"); om = oracle_multi(W, mode, N)
            row[f"oracle{N}"] = float(kpc(oa) / kpc(c16)); row[f"oraclemulti{N}"] = float(kpc(om) / kpc(c16))
            say(f"  seq-{N}: x{r:.3f} ({row[f'seq{N}']['off_path_nodes']} off-path nodes); per-check single-level oracle "
                f"x{row[f'oracle{N}']:.3f}; multi-level oracle x{row[f'oraclemulti{N}']:.3f}")
        RES[f"{s}|{mode}"] = row

say("\nVERDICTS (pre-registered, EXACT judged; d1 reported): SAME-COST WIN = EAGLE-2-16 >= 1.03 x chain-16 with CI > 1 on "
    "both sets; 32 ROWS PAY = EAGLE-2-32 / chain-16 >= R32 on both sets")
for mode in ("exact", "d1"):
    a, b = RES[f"mbpp|{mode}"], RES[f"he|{mode}"]
    win = all(x["eagle2_16_1"]["vs_chain16"] >= 1.03 and x["eagle2_16_1"]["ci95"][0] > 1 for x in (a, b))
    say(f"  {mode}: EAGLE-2-16 x{a['eagle2_16_1']['vs_chain16']:.3f} / x{b['eagle2_16_1']['vs_chain16']:.3f} -> "
        f"{'SAME-COST WIN' if win else 'no'}" + (" (reported only)" if mode == "d1" else ""))
    for R32 in (1.625, 1.3):
        pay = all(x["eagle2_32_1"]["vs_chain16"] >= R32 for x in (a, b))
        say(f"  {mode}: EAGLE-2-32 x{a['eagle2_32_1']['vs_chain16']:.3f} / x{b['eagle2_32_1']['vs_chain16']:.3f} vs R32 "
            f"{R32} -> {'PAYS' if pay else 'no'}")
say("\nTHROUGHPUT vs chain-16 (kept/check / cost; cost 16 rows = 1, 32 rows = R32; the 16-node tree verify assumed = a "
    "16-row chain, Leap prices it)")
for mode in ("exact", "d1"):
    for s in SETS:
        x = RES[f"{s}|{mode}"]
        for R32 in (1.625, 1.3):
            opts = {"chain-16": 1.0, "EAGLE-2-16": x["eagle2_16_1"]["vs_chain16"],
                    "chain-32": x["chain32"] / x["chain16"] / R32, "EAGLE-2-32": x["eagle2_32_1"]["vs_chain16"] / R32}
            bst = max(opts, key=opts.get)
            say(f"  {mode} {s} R32 {R32}: " + ", ".join(f"{k} x{v:.3f}" for k, v in opts.items()) + f" -> best {bst}")
        say(f"  {mode} {s}: break-even 16-node tree verify = {x['eagle2_16_1']['vs_chain16']:.3f} x a 16-row chain's")
json.dump(RES, open(OUT, "w"), indent=1, default=str)
open(OUT.replace(".json", ".txt"), "w").write("\n".join(lines) + "\n")
