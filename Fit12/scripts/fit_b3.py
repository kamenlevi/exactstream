"""Fit12 B3 (2026-10-25; NewIdeas' bet B3, Sun 04:12): data-aware 2-bit expert codes (trellis) vs MSB-3's OUTPUT error, on
Q2's 40 experts (layers 2 / 12 / 24 / 36 / 46, 8 each, seed 0) and captured MoE inputs (~/Fit12/data/q2/x_L*.npy; all
tokens). Q2 killed GPTQ-2 g64 (3.65x MSB-3), Q3 KLT + water-filling (7.7x at 2 b); never tested here: a QTIP-class code.
  TR-k   data-free (Breakthrough's D2 recipe): RHT (random signs + block Hadamard on the input dim) + per-row RMS (bf16) +
         bitshift trellis (L = 12, k bits / weight, Gaussian-quantile LUT, exact Viterbi) on 16 x 16 tiles.
  TRL-k  the same + BlockLDLQ (QTIP / QuIP#): 16-column blocks quantized jointly, error E = (W_B - Q_B) U_BB^-1 fed to the
         later columns (W_rest -= E U_B,rest; U = upper Cholesky of (H' + 1 % damp)^-1, as fit_q2.gptq), H' = R^T H R.
  RD     ideal Gaussian code: reverse water-filling of the H-weighted coefficient variances on H's eigenbasis (column-only
         and row x column variance models), no side information: a floor, not a code.
Rates: k + 12/256 (initial state per tile) + 16/K (row scale) = 2.05-2.07 / 3.05-3.07 b. Metric = Q2's: tr(dW H dW^T) /
tr(W4 H W4^T), mean of the per-projection ratios vs MSB-3 (recomputed; Q2: gate 0.0165, up 0.0219, down 0.0260).
Kill line (NewIdeas B3, pre-registered in PLAN.md): TRL-2 > 1.5x MSB-3 -> B3 killed.
Usage: python fit_b3.py [test | rd | oos | b10a]   (CPU; B3_THREADS Viterbi workers, VECLIB_MAXIMUM_THREADS for BLAS)"""
import ctypes, json, os, sys, time
from statistics import NormalDist
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_q as F
import fit_q2 as Q2

D = os.path.expanduser("~/Fit12/data/q2")
LAYERS, NE, L, TT, KS, RATES = (2, 12, 24, 36, 46), 8, 12, 16, (2, 3), (2.0, 2.5, 3.0)
NTHR = int(os.environ.get("B3_THREADS", "6"))
_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fit_trellis.dylib"))
_lib.viterbi_tiles_n.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int] * 4 + [ctypes.c_void_p, ctypes.c_int]
_q = np.array([NormalDist().inv_cdf((i + 0.5) / (1 << L)) for i in range(1 << L)], np.float32)
LUT = np.ascontiguousarray(_q[np.random.default_rng(0).permutation(1 << L)])     # = Breakthrough's make_lut(L, 0)
_x3 = (np.uint64(89226354) * np.arange(1 << L, dtype=np.uint64) + np.uint64(64248484)) & np.uint64(0xFFFFFFFF)
_m3 = ((_x3 & np.uint64(0x8FFF8FFF)) ^ np.uint64(0x3B603B60)).astype(np.uint32)
_c3 = (_m3 & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32) + (_m3 >> 16).astype(np.uint16).view(np.float16).astype(np.float32)
LUT3 = np.ascontiguousarray(_c3 / np.sqrt((_c3 * _c3).mean()), np.float32)   # QTIP 3INST as Leap's kernel computes it
# (inst3(s) of ~/Leap/scripts/b10_trellis.py; no offset, the row scale absorbs 1 / RMS, so the format stays decodable)


def vit(t, k, lut=None):
    """t [n, 256] -> its trellis reconstruction (exact-MSE Viterbi, C); lut = the 4,096 state values (default D2's LUT)"""
    x = np.ascontiguousarray(t, np.float32)
    out = np.empty_like(x)
    lut = LUT if lut is None else lut
    _lib.viterbi_tiles_n(x.ctypes.data, out.ctypes.data, x.shape[0], x.shape[1], L, k, lut.ctypes.data, NTHR)
    return out.astype(np.float64)


def tr(W, k, sc):
    """data-free: 16 x 16 tiles of W / row scale"""
    o, K = W.shape
    t = (W / sc[:, None]).reshape(o // TT, TT, K // TT, TT).transpose(0, 2, 1, 3).reshape(-1, TT * TT)
    return vit(t, k).reshape(o // TT, K // TT, TT, TT).transpose(0, 2, 1, 3).reshape(o, K) * sc[:, None]


def trl(W, U, k, sc, lut=None):
    """BlockLDLQ: 16-column blocks (the same 16 x 16 tiles), error feedback through U (upper, H^-1 = U^T U)"""
    W = W.copy()
    o, K = W.shape
    Q = np.empty_like(W)
    for i1 in range(0, K, TT):
        i2 = i1 + TT
        WB = W[:, i1:i2]
        QB = vit((WB / sc[:, None]).reshape(o // TT, TT * TT), k, lut).reshape(o, TT) * sc[:, None]
        Q[:, i1:i2] = QB
        if i2 < K:
            E = np.linalg.solve(U[i1:i2, i1:i2].T, (WB - QB).T).T
            W[:, i2:] -= E @ U[i1:i2, i2:]
    return Q


def rot(K, seed):
    """R = diag(random signs) x block-diagonal Hadamard (orthogonal); W' = W R, H' = R^T H R"""
    Hb, R = F.HB[K].astype(np.float64), np.zeros((K, K))
    B = Hb.shape[0]
    for j in range(K // B):
        R[j * B:(j + 1) * B, j * B:(j + 1) * B] = Hb
    return np.random.default_rng(seed).choice(np.array([-1.0, 1.0]), K)[:, None] * R


def rd(W, eig):
    """ideal-Gaussian-code floor at RATES: [[col model], [row x col model]] relative weighted errors"""
    lam, V = eig
    C2 = (W @ V) ** 2
    s2 = C2.mean(0) + 1e-300
    out = []
    for rc in (False, True):
        r2 = (C2 / s2).mean(1) if rc else np.ones(W.shape[0])
        v = ((np.maximum(lam, 0.0) * s2)[None, :] * r2[:, None]).ravel()
        lv, tot, res = np.log(v + 1e-300), v.sum(), []
        for R in RATES:
            lo, hi = lv.min() - 2 * np.log(2) * (R + 4), lv.max() + 1.0     # theta can sit below every variance
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if np.maximum(0.0, lv - mid).mean() / (2 * np.log(2)) > R:
                    lo = mid
                else:
                    hi = mid
            res.append(float(np.minimum(np.exp(hi), v).sum() / tot))
        out.append(res)
    return out


def test():
    g = np.random.default_rng(1).standard_normal((64, 4096))
    one = np.ones(64)
    for k in KS:
        t1 = time.time()
        q = tr(g, k, one)
        print(f"TR-{k} iid N(0,1): NMSE {((q - g) ** 2).mean() / (g * g).mean():.4f} (Breakthrough, L = 12 on real 35B experts: "
              f"k=2 0.071, k=3 0.0191; RD {2.0 ** (-2 * k):.4f}); {time.time() - t1:.1f} s for {g.size} weights", flush=True)
    g2 = g[:, :512]
    U, _ = Q2.prep(np.eye(512))
    print("TRL(H = I) == TR (no feedback):", bool(np.array_equal(trl(g2, U, 2, one), tr(g2, 2, one))), flush=True)
    print("RD floor at H = I (col model, 2.0 / 2.5 / 3.0 b; theory 0.0625 / 0.0312 / 0.0156):",
          [round(x, 4) for x in rd(g2, np.linalg.eigh(np.eye(512)))[0]], flush=True)


def main(rd_only=False):
    t0 = time.time()
    rng = np.random.default_rng(0)
    res = {}

    def add(nm, p, v):
        res.setdefault(nm, {q: [] for q in F.PROJ})[p].append(v)
    for Ly in LAYERS:
        X = np.load(os.path.join(D, f"x_L{Ly}.npy")).astype(np.float64)
        Hx = X.T @ X
        eig_x = np.linalg.eigh(Hx)
        Rx = rot(2560, Ly)
        Hxr = Rx.T @ Hx @ Rx
        Ux, dead = Q2.prep(Hxr)
        assert not dead.any()
        es = sorted(rng.choice(512, NE, replace=False).tolist())
        for e in es:
            rec = F.read(Ly, e)
            W4, W3 = {}, {}
            for p in F.PROJ:
                r = rec[p]
                s3, b3, bb = F.bf16(r["s3u"]), F.bf16(r["b3u"]), F.bf16(r["bu"])
                W4[p] = (np.repeat(s3 / 2, 64, 1) * (2 * r["m"] + r["l"]) + np.repeat(bb, 64, 1)).astype(np.float64)
                W3[p] = (np.repeat(s3, 64, 1) * r["m"] + np.repeat(b3, 64, 1)).astype(np.float64)
            a = X @ W4["gate"].T
            h = (a / (1 + np.exp(-a))) * (X @ W4["up"].T)
            Hh = h.T @ h
            Rh = rot(640, 1000 * Ly + e)
            Hhr = Rh.T @ Hh @ Rh
            Uh, dead = Q2.prep(Hhr)
            assert not dead.any()
            eig_h = np.linalg.eigh(Hh)
            for p in F.PROJ:
                H, Hr, R, U, eg = (Hh, Hhr, Rh, Uh, eig_h) if p == "down" else (Hx, Hxr, Rx, Ux, eig_x)
                add("MSB-3 (3.5 b)", p, Q2.err(W3[p], W4[p], H))
                Wr = W4[p] @ R
                sc = F.rbf16(np.sqrt((Wr * Wr).mean(1)).astype(np.float32)).astype(np.float64)
                for k in (() if rd_only else KS):
                    add(f"TR-{k} data-free ({k}.06 b)", p, Q2.err(tr(Wr, k, sc), Wr, Hr))
                    add(f"TRL-{k} BlockLDLQ ({k}.06 b)", p, Q2.err(trl(Wr, U, k, sc), Wr, Hr))
                for nm, vals in zip(("col", "row x col"), rd(W4[p], eg)):
                    for Rt, v in zip(RATES, vals):
                        add(f"RD floor {nm} ({Rt} b)", p, v)
        print(f"[b3] layer {Ly} done ({time.time() - t0:.0f} s)", flush=True)
    base = {p: np.mean(res["MSB-3 (3.5 b)"][p]) for p in F.PROJ}
    rel = lambda nm, ps=F.PROJ: np.mean([np.mean(res[nm][p]) / base[p] for p in ps])
    print("format | gate | up | down | mean vs MSB-3", flush=True)
    for nm, v in res.items():
        m = {p: np.mean(v[p]) for p in F.PROJ}
        print(f"{nm} | {m['gate']:.4f} | {m['up']:.4f} | {m['down']:.4f} | {rel(nm):.2f}x", flush=True)
    t2, t3 = "TRL-2 BlockLDLQ (2.06 b)", "TRL-3 BlockLDLQ (3.06 b)"
    if rd_only:
        return
    mix = np.mean([np.mean(res[t2][p]) / base[p] for p in ("gate", "up")] + [np.mean(res[t3]["down"]) / base["down"]])
    print(f"TRL-2 gate / up + TRL-3 down (2.39 b) | mean vs MSB-3 {mix:.2f}x", flush=True)
    r2, r3 = rel(t2), rel(t3)
    s = np.log(r2 / r3)
    print(f"TRL bits at MSB-3 parity ~{2.06 + np.log(r2) / s:.2f} b, at 1.5x ~{2.06 + np.log(r2 / 1.5) / s:.2f} b "
          f"(log-linear between k = 2 and 3); {time.time() - t0:.0f} s", flush=True)


def main_oos():
    """B3b: out-of-sample check of TRL (data-aware codes are fitted to H): LDLQ's H from the FIRST half of the captured
    tokens, every error judged on the SECOND half (same 40 experts; MSB-3 and the data-free TR on the same test half)."""
    t0 = time.time()
    rng = np.random.default_rng(0)
    res = {}

    def add(nm, p, v):
        res.setdefault(nm, {q: [] for q in F.PROJ})[p].append(v)
    silu = lambda a: a / (1 + np.exp(-a))
    for Ly in LAYERS:
        X = np.load(os.path.join(D, f"x_L{Ly}.npy"))
        _, first = np.unique(np.ascontiguousarray(X).view(np.dtype((np.void, X.shape[1] * 4))).ravel(), return_index=True)
        n0, X = X.shape[0], X[np.sort(first)].astype(np.float64)      # repeated rows (one token seen twice) kept once
        n2 = X.shape[0] // 2
        print(f"[b3b] layer {Ly}: {n0} captured rows, {X.shape[0]} unique; fit on the first {n2}", flush=True)
        X1, X2 = X[:n2], X[n2:]
        Rx = rot(2560, Ly)
        Hx1r, Hx2r = Rx.T @ (X1.T @ X1) @ Rx, Rx.T @ (X2.T @ X2) @ Rx
        Hx2 = X2.T @ X2
        Ux1, dead = Q2.prep(Hx1r)
        assert not dead.any()
        es = sorted(rng.choice(512, NE, replace=False).tolist())
        for e in es:
            rec = F.read(Ly, e)
            W4, W3 = {}, {}
            for p in F.PROJ:
                r = rec[p]
                s3, b3, bb = F.bf16(r["s3u"]), F.bf16(r["b3u"]), F.bf16(r["bu"])
                W4[p] = (np.repeat(s3 / 2, 64, 1) * (2 * r["m"] + r["l"]) + np.repeat(bb, 64, 1)).astype(np.float64)
                W3[p] = (np.repeat(s3, 64, 1) * r["m"] + np.repeat(b3, 64, 1)).astype(np.float64)
            h1 = silu(X1 @ W4["gate"].T) * (X1 @ W4["up"].T)
            h2 = silu(X2 @ W4["gate"].T) * (X2 @ W4["up"].T)
            Rh = rot(640, 1000 * Ly + e)
            Hh1r, Hh2 = Rh.T @ (h1.T @ h1) @ Rh, h2.T @ h2
            Hh2r = Rh.T @ Hh2 @ Rh
            Uh1, dead = Q2.prep(Hh1r)
            assert not dead.any()
            for p in F.PROJ:
                H1r, H2, H2r, R, U1 = (Hh1r, Hh2, Hh2r, Rh, Uh1) if p == "down" else (Hx1r, Hx2, Hx2r, Rx, Ux1)
                add("MSB-3 (3.5 b)", p, Q2.err(W3[p], W4[p], H2))
                Wr = W4[p] @ R
                sc = F.rbf16(np.sqrt((Wr * Wr).mean(1)).astype(np.float32)).astype(np.float64)
                add("TR-2 data-free (2.06 b)", p, Q2.err(tr(Wr, 2, sc), Wr, H2r))
                q2 = trl(Wr, U1, 2, sc)
                add("TRL-2 fit half 1, judged half 1 (in-sample)", p, Q2.err(q2, Wr, H1r))
                add("TRL-2 fit half 1, judged half 2 (OUT-OF-SAMPLE)", p, Q2.err(q2, Wr, H2r))
                if p == "down":
                    add("TRL-3 fit half 1, judged half 2 (OUT-OF-SAMPLE)", p, Q2.err(trl(Wr, U1, 3, sc), Wr, H2r))
        print(f"[b3b] layer {Ly} done ({time.time() - t0:.0f} s)", flush=True)
    base = {p: np.mean(res["MSB-3 (3.5 b)"][p]) for p in F.PROJ}
    print("format (errors on the test half unless marked) | gate | up | down | mean vs MSB-3", flush=True)
    for nm, v in res.items():
        m = {p: np.mean(v[p]) if v[p] else float("nan") for p in F.PROJ}
        ps = [p for p in F.PROJ if v[p]]
        print(f"{nm} | {m['gate']:.4f} | {m['up']:.4f} | {m['down']:.4f} | "
              f"{np.mean([m[p] / base[p] for p in ps]):.2f}x" + ("" if len(ps) == 3 else " (down only)"), flush=True)
    t2, t3 = "TRL-2 fit half 1, judged half 2 (OUT-OF-SAMPLE)", "TRL-3 fit half 1, judged half 2 (OUT-OF-SAMPLE)"
    mix = np.mean([np.mean(res[t2][p]) / base[p] for p in ("gate", "up")] + [np.mean(res[t3]["down"]) / base["down"]])
    print(f"OUT-OF-SAMPLE TRL-2 gate / up + TRL-3 down (2.39 b) | mean vs MSB-3 {mix:.2f}x; {time.time() - t0:.0f} s", flush=True)


CAPD = os.path.expanduser("~/Fit12/data/b10cap")
LAYERS9 = (2, 7, 12, 18, 24, 30, 36, 41, 46)


def dedupe(X, ids=None):
    """drop repeated rows (one token seen twice by the capture), keeping order; the routing ids follow the rows"""
    X = np.ascontiguousarray(X)
    _, first = np.unique(X.view(np.dtype((np.void, X.shape[1] * X.dtype.itemsize))).ravel(), return_index=True)
    k = np.sort(first)
    return X[k].astype(np.float64), (None if ids is None else ids[k])


def main_b10a(ne=4):
    """B10a (NewIdeas B10 part (a)): held-out experts (seed 2; none of Q2's 40), held-out TEXT (fit on captures A0..A7 = ~32K tokens
    of local notes / Swift / Python, judged on capture B = the 4-bit reference model's own prompts + completions), 9 layers; judged
    on all of B's tokens and on the tokens ROUTED to each expert (B's top-10 ids). Codes: MSB-3; TRL-2 base; gate/up-2 +
    down-3; base + trellis residual (k_r = 1 / 2, LDLQ, own row scale) for rank 0-1 uses, scored against E4 = W4's own
    rounding error (s4^2 / 12 per weight, H-weighted; MSB-3 must come out at ~3 E4). Encode seconds per expert logged."""
    t0 = time.time()
    q2 = np.random.default_rng(0)
    q2sel = {Ly: set(q2.choice(512, NE, replace=False).tolist()) for Ly in LAYERS}
    rng = np.random.default_rng(2)
    res, tim = {}, {}
    silu = lambda a: a / (1 + np.exp(-a))

    def add(nm, p, v):
        res.setdefault(nm, {q: [] for q in F.PROJ})[p].append(v)

    def e4(s3u, H):                                   # W4's rounding error (s4 = s3 / 2 per 64-input group), H-weighted
        s4 = np.repeat(F.bf16(s3u) / 2, 64, 1).astype(np.float64)
        return float(((s4 * s4 / 12.0) * np.diag(H)[None, :]).sum())
    def report(fh):
        for tn in ("all", "routed"):
            if f"[{tn}] MSB-3 (3.5 b)" not in res:
                continue
            base = {p: np.mean(res[f"[{tn}] MSB-3 (3.5 b)"][p]) for p in F.PROJ}
            print(f"judged on B, {tn} tokens | gate | up | down | mean (vs MSB-3 for errors)", file=fh, flush=True)
            for nm, v in res.items():
                if not nm.startswith(f"[{tn}]"):
                    continue
                m = {p: (np.mean(v[p]) if v[p] else float("nan")) for p in F.PROJ}
                ps = [p for p in F.PROJ if v[p]]
                rel = "E4" in nm
                mean = np.mean([m[p] for p in ps]) if rel else np.mean([m[p] / base[p] for p in ps])
                print(f"{nm} | {m['gate']:.4f} | {m['up']:.4f} | {m['down']:.4f} | {mean:.3f}{'' if rel else 'x'}", file=fh, flush=True)
            t2 = res[f"[{tn}] TRL-2 base (2.06 b)"]
            mix = np.mean([np.mean(t2[p]) / base[p] for p in ("gate", "up")] + [np.mean(res[f"[{tn}] TRL-3 down (3.07 b)"]["down"]) / base["down"]])
            print(f"[{tn}] TRL-2 gate / up + TRL-3 down (2.40 b) | mean vs MSB-3 {mix:.2f}x", file=fh, flush=True)
        r = res["routed rows per expert"]["gate"]
        print(f"routed judge rows per expert: mean {np.mean(r):.0f}, min {min(r):.0f}; experts with < 8 (no routed test): "
              f"{sum(x < 8 for x in r)} of {len(r)}", file=fh, flush=True)
        for k, v in tim.items():
            print(f"encode {k}: {np.mean(v):.2f} s per matrix (median {np.median(v):.2f}; B3_THREADS={NTHR}, quiet mode: heat "
                  f"pauses can land inside one); per expert (3 matrices) {3 * np.median(v):.1f} s", file=fh, flush=True)
    parts, done = sorted(d for d in os.listdir(CAPD) if d.startswith("A")), []
    for Ly in LAYERS9:
        XAp = [dedupe(np.load(os.path.join(CAPD, d, f"x_L{Ly}.npy")))[0] for d in parts]
        XA, XA2 = np.concatenate(XAp), np.concatenate(XAp[:2])          # all ~32K fit tokens / the first ~8K (trend)
        XB, idB = dedupe(np.load(os.path.join(CAPD, "B", f"x_L{Ly}.npy")), np.load(os.path.join(CAPD, "B", f"ids_L{Ly}.npy")))
        Rx = rot(2560, Ly)
        Ux, dead = Q2.prep(Rx.T @ (XA.T @ XA) @ Rx)
        Ux2, dead2 = Q2.prep(Rx.T @ (XA2.T @ XA2) @ Rx)
        assert not dead.any() and not dead2.any()
        HB = XB.T @ XB
        pool = [e for e in range(512) if e not in q2sel.get(Ly, ())]
        es = sorted(rng.choice(pool, ne, replace=False).tolist())
        print(f"[b10a] layer {Ly}: fit rows {XA.shape[0]} ({len(parts)} parts; trend fit {XA2.shape[0]}), judge rows "
              f"{XB.shape[0]}, experts {es}", flush=True)
        for e in es:
            rec = F.read(Ly, e)
            W4, W3 = {}, {}
            for p in F.PROJ:
                r = rec[p]
                s3, b3, bb = F.bf16(r["s3u"]), F.bf16(r["b3u"]), F.bf16(r["bu"])
                W4[p] = (np.repeat(s3 / 2, 64, 1) * (2 * r["m"] + r["l"]) + np.repeat(bb, 64, 1)).astype(np.float64)
                W3[p] = (np.repeat(s3, 64, 1) * r["m"] + np.repeat(b3, 64, 1)).astype(np.float64)
            hA = silu(XA @ W4["gate"].T) * (XA @ W4["up"].T)
            hA2 = hA[:XA2.shape[0]]
            Rh = rot(640, 1000 * Ly + e)
            Uh, dead = Q2.prep(Rh.T @ (hA.T @ hA) @ Rh)
            Uh2, dead2 = Q2.prep(Rh.T @ (hA2.T @ hA2) @ Rh)
            assert not dead.any() and not dead2.any()
            mine = (idB == e).any(1)                                   # judge rows routed to this expert
            XBe = XB[mine]
            hB, hBe = (silu(Z @ W4["gate"].T) * (Z @ W4["up"].T) for Z in (XB, XBe))
            tests = {"all": {"x": HB, "h": hB.T @ hB}}
            if mine.sum() >= 8:                                        # B is 4,096 tokens: ~80 routed rows per expert
                tests["routed"] = {"x": XBe.T @ XBe, "h": hBe.T @ hBe}
            add("routed rows per expert", "gate", float(mine.sum()))
            for p in F.PROJ:
                R, U, U2, key = (Rh, Uh, Uh2, "h") if p == "down" else (Rx, Ux, Ux2, "x")
                Wr = W4[p] @ R
                sc = F.rbf16(np.sqrt((Wr * Wr).mean(1)).astype(np.float32)).astype(np.float64)
                t1 = time.time(); qb = trl(Wr, U, 2, sc); tim.setdefault("TRL-2 base", []).append(time.time() - t1)
                qb8 = trl(Wr, U2, 2, sc)                                   # fit on the first ~8K tokens only (trend)
                q3 = trl(Wr, U, 3, sc) if p == "down" else None
                r0 = Wr - qb
                sr = F.rbf16(np.sqrt((r0 * r0).mean(1)).astype(np.float32)).astype(np.float64)
                full = {}
                for kr in (1, 2):
                    t1 = time.time(); full[kr] = qb + trl(r0, U, kr, sr)
                    tim.setdefault(f"residual k={kr}", []).append(time.time() - t1)
                for tn, T in tests.items():
                    H = T[key]
                    Hr = R.T @ H @ R
                    den = float(((W4[p] @ H) * W4[p]).sum())
                    E4 = e4(rec[p]["s3u"], H) / den
                    add(f"[{tn}] MSB-3 (3.5 b)", p, Q2.err(W3[p], W4[p], H))
                    add(f"[{tn}] MSB-3 / E4 (sanity ~3)", p, Q2.err(W3[p], W4[p], H) / E4)
                    add(f"[{tn}] TRL-2 base (2.06 b)", p, Q2.err(qb, Wr, Hr))
                    add(f"[{tn}] TRL-2 base fit on ~8K tokens only", p, Q2.err(qb8, Wr, Hr))
                    if q3 is not None:
                        add(f"[{tn}] TRL-3 down (3.07 b)", p, Q2.err(q3, Wr, Hr))
                    for kr in (1, 2):
                        add(f"[{tn}] base + residual k={kr}: error / E4 (kill > 0.2)", p, Q2.err(full[kr], Wr, Hr) / E4)
        print(f"[b10a] layer {Ly} done ({time.time() - t0:.0f} s)", flush=True)
        done.append(Ly)
        with open(os.path.expanduser("~/Fit12/data/b10/b10a_partial.txt"), "w") as fh:   # kept if the lock runs out first
            print(f"partial: layers {done} ({time.time() - t0:.0f} s)", file=fh)
            report(fh)
    report(sys.stdout)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


B10R = os.environ.get("F12_B10R_DIR", os.path.expanduser("~/Fit12/data/b10rcap"))
SMOKE = os.environ.get("F12_B10R_SMOKE") == "1"        # 1 layer, 1 expert, on stand-in captures: a code check only
DAMPS = (0.01, 0.1, 0.3)
PEERS = os.path.expanduser("~/.claude-peers")


def step_aside():
    """RULES.md 'CPU jobs vs the GPU queue' (peer grants quiet only while no session uses >= 80 % CPU): between experts,
    wait while another session holds or waits for a quiet lock, or the heat PAUSE file exists"""
    me, t0, said = os.environ.get("CLAUDE_PID", "23445"), time.time(), False
    while True:
        busy = os.path.exists(os.path.expanduser("~/.local/share/quiet/PAUSE"))
        for f in os.listdir(os.path.join(PEERS, "queue")):
            if me not in f and time.time() - os.path.getmtime(os.path.join(PEERS, "queue", f)) < 3 * 3600:
                busy = True
        for f in os.listdir(os.path.join(PEERS, "locks")):
            try:
                d = json.load(open(os.path.join(PEERS, "locks", f)))
            except Exception:
                continue
            if me not in f and d.get("mode") == "quiet" and d.get("until", 0) > time.time():
                busy = True
        if not busy:
            break
        if not said:
            print(f"[b10r] {time.strftime('%H:%M:%S')} stepping aside (quiet lock held / queued, or heat pause)", flush=True)
            said = True
        time.sleep(20)
    if said:
        print(f"[b10r] {time.strftime('%H:%M:%S')} resumed after {time.time() - t0:.0f} s", flush=True)


def main_b10r(ne=4):
    """B10-r (NewIdeas' final 2-bit test, ~/NewIdeas/QUEUE.md; pre-registered in PLAN.md 'B10-r'): is B10a's held-out
    failure domain shift? FIT = R0..R7 (32,768 tokens of a broad mix: math, code, instructions, multilingual, Bulgarian
    wiki, MMLU-Pro, Kamen's agent sessions; fit_b10r_text.py), VALIDATION = R8..R9 (8,192 tokens, its own agent sessions),
    judges = B (as B10a: all + routed tokens) and U (English literary prose, absent from the calibration). The same 36
    experts as B10a (seed 2). H damping 0.01 / 0.1 / 0.3 of mean diag: TRL-2 (D2's LUT, as B10a) at all three; the
    DECISIVE TRL-2 is the one damping with the lowest mean validation error (never chosen on B or U); per matrix the
    validation-best damping also feeds TRL-2 and TRL-3 with the 3INST table (Leap's decodable code). Errors = Q2's metric,
    aggregated as B10a (per projection mean error / mean MSB-3 error, then the mean of the three)."""
    t0 = time.time()
    q2 = np.random.default_rng(0)
    q2sel = {Ly: set(q2.choice(512, NE, replace=False).tolist()) for Ly in LAYERS}
    rng = np.random.default_rng(2)
    res, tim = {}, {}
    silu = lambda a: a / (1 + np.exp(-a))
    c3 = float(_c3.mean() / np.sqrt((_c3 * _c3).mean()))
    print(f"[b10r] 3INST table: mean / RMS {c3:+.4f} over the 4,096 states (kept: the kernel computes it as is)", flush=True)

    def add(nm, p, v):
        res.setdefault(nm, {q: [] for q in F.PROJ})[p].append(v)

    def e4(s3u, H):
        s4 = np.repeat(F.bf16(s3u) / 2, 64, 1).astype(np.float64)
        return float(((s4 * s4 / 12.0) * np.diag(H)[None, :]).sum())

    def ld(d, Ly, ids=False):
        X, I = dedupe(np.load(os.path.join(d, f"x_L{Ly}.npy")), np.load(os.path.join(d, f"ids_L{Ly}.npy")) if ids else None)
        return (X, I) if ids else X

    def agg(tn, arm, ps=F.PROJ):
        base = res[f"[{tn}] MSB-3 (3.5 b)"]
        return float(np.mean([np.mean(res[f"[{tn}] {arm}"][p]) / np.mean(base[p]) for p in ps]))

    def report(fh):
        tns = [tn for tn in ("val", "B all", "B routed", "U all", "U routed") if f"[{tn}] MSB-3 (3.5 b)" in res]
        for tn in tns:
            print(f"judged on {tn} | gate | up | down | mean (errors vs MSB-3)", file=fh)
            base = {p: np.mean(res[f"[{tn}] MSB-3 (3.5 b)"][p]) for p in F.PROJ}
            for nm, v in res.items():
                if nm.startswith(f"[{tn}]"):
                    m = {p: np.mean(v[p]) for p in F.PROJ}
                    rel = "E4" in nm
                    mean = np.mean([m[p] for p in F.PROJ]) if rel else np.mean([m[p] / base[p] for p in F.PROJ])
                    print(f"{nm} | {m['gate']:.4f} | {m['up']:.4f} | {m['down']:.4f} | {mean:.3f}{'' if rel else 'x'}", file=fh)
        ds = {d: agg("val", f"TRL-2 LUT d={d}") for d in DAMPS}
        dstar = min(DAMPS, key=ds.get)
        print("validation (R8..R9) TRL-2 vs MSB-3 by damping: " + ", ".join(f"d={d} {v:.3f}x" for d, v in ds.items())
              + f" -> chosen d* = {dstar}", file=fh)
        arm = f"TRL-2 LUT d={dstar}"
        line = {"B all": 1.5, "B routed": 1.5, "U all": 1.8}
        for tn in ("B all", "B routed", "U all"):
            if f"[{tn}] {arm}" in res:
                v = agg(tn, arm)
                print(f"DECISIVE TRL-2 (2.06 b, d* {dstar}) on {tn}: {v:.3f}x MSB-3 (kill > {line[tn]}x) -> "
                      f"{'KILL' if v > line[tn] else 'pass'}", file=fh)
        for tn in tns:
            if tn == "val":
                continue
            t2 = {p: agg(tn, "TRL-2 3INST (val-best d per matrix)", (p,)) for p in F.PROJ}
            t3 = {p: agg(tn, "TRL-3 3INST (val-best d per matrix)", (p,)) for p in F.PROJ}
            mix = np.mean([agg(tn, arm, ("gate",)), agg(tn, arm, ("up",)), t3["down"]])
            print(f"[{tn}] TRL-2 3INST (decodable) gate {t2['gate']:.3f}x up {t2['up']:.3f}x down {t2['down']:.3f}x | "
                  f"TRL-3 3INST (~3.07 b) gate {t3['gate']:.3f}x up {t3['up']:.3f}x down {t3['down']:.3f}x "
                  f"(mean {np.mean(list(t3.values())):.3f}x) | gate / up TRL-2 d* + down TRL-3 (2.40 b) {mix:.3f}x", file=fh)
        dm = res["val-best damping per matrix"]
        print("val-best damping per matrix: " + ", ".join(f"{p} " + "/".join(str(sum(x == d for x in dm[p])) for d in DAMPS)
              for p in F.PROJ) + f" (counts of {DAMPS})", file=fh)
        for k in ("B", "U"):
            r = res[f"routed rows {k}"]["gate"]
            print(f"routed rows per expert in {k}: mean {np.mean(r):.0f}, min {min(r):.0f}, < 8 (no routed test) "
                  f"{sum(x < 8 for x in r)} of {len(r)}", file=fh)
        for k, v in tim.items():
            print(f"encode {k}: median {np.median(v):.2f} s per matrix (B3_THREADS={NTHR})", file=fh)
        fh.flush()

    for Ly in (LAYERS9[:1] if SMOKE else LAYERS9):
        XF = np.concatenate([ld(os.path.join(B10R, f"R{i}"), Ly) for i in range(8)])
        XV = np.concatenate([ld(os.path.join(B10R, f"R{i}"), Ly) for i in (8, 9)])
        XB, idB = ld(os.path.join(CAPD, "B"), Ly, True)
        XU, idU = ld(os.path.join(B10R, "U"), Ly, True)
        Rx = rot(2560, Ly)
        HFr = Rx.T @ (XF.T @ XF) @ Rx
        Ux = {}
        for d in DAMPS:
            Ux[d], dead = Q2.prep(HFr, d)
            assert not dead.any()
        HX = {tn: Z.T @ Z for tn, Z in (("val", XV), ("B all", XB), ("U all", XU))}
        HXr = {tn: Rx.T @ H @ Rx for tn, H in HX.items()}
        pool = [e for e in range(512) if e not in q2sel.get(Ly, ())]
        es = sorted(rng.choice(pool, ne, replace=False).tolist())[:1 if SMOKE else ne]
        print(f"[b10r] layer {Ly}: fit rows {XF.shape[0]}, val {XV.shape[0]}, B {XB.shape[0]}, U {XU.shape[0]}, experts {es}", flush=True)
        for e in es:
            step_aside()
            rec = F.read(Ly, e)
            W4, W3 = {}, {}
            for p in F.PROJ:
                r = rec[p]
                s3, b3, bb = F.bf16(r["s3u"]), F.bf16(r["b3u"]), F.bf16(r["bu"])
                W4[p] = (np.repeat(s3 / 2, 64, 1) * (2 * r["m"] + r["l"]) + np.repeat(bb, 64, 1)).astype(np.float64)
                W3[p] = (np.repeat(s3, 64, 1) * r["m"] + np.repeat(b3, 64, 1)).astype(np.float64)
            hs = lambda Z: silu(Z @ W4["gate"].T) * (Z @ W4["up"].T)
            hF = hs(XF)
            Rh = rot(640, 1000 * Ly + e)
            HhFr = Rh.T @ (hF.T @ hF) @ Rh
            Uh = {}
            for d in DAMPS:
                Uh[d], dead = Q2.prep(HhFr, d)
                assert not dead.any()
            tests = {"val": XV, "B all": XB, "U all": XU}
            mB, mU = (idB == e).any(1), (idU == e).any(1)
            add("routed rows B", "gate", float(mB.sum()))
            add("routed rows U", "gate", float(mU.sum()))
            if mB.sum() >= 8:
                tests["B routed"] = XB[mB]
            if mU.sum() >= 8:
                tests["U routed"] = XU[mU]
            Hs = {}
            for tn, Z in tests.items():
                hz = hs(Z)
                Hx = HX[tn] if tn in HX else Z.T @ Z
                Hs[tn] = {"x": (Hx, HXr[tn] if tn in HXr else Rx.T @ Hx @ Rx), "h": (hz.T @ hz, Rh.T @ (hz.T @ hz) @ Rh)}
            for p in F.PROJ:
                R, U, key = (Rh, Uh, "h") if p == "down" else (Rx, Ux, "x")
                Wr = W4[p] @ R
                sc = F.rbf16(np.sqrt((Wr * Wr).mean(1)).astype(np.float32)).astype(np.float64)
                q = {}
                for d in DAMPS:
                    t1 = time.time()
                    q[f"TRL-2 LUT d={d}"] = trl(Wr, U[d], 2, sc)
                    tim.setdefault("TRL-2 LUT", []).append(time.time() - t1)
                Hv, Hvr = Hs["val"][key]
                dm = min(DAMPS, key=lambda d: Q2.err(q[f"TRL-2 LUT d={d}"], Wr, Hvr))
                add("val-best damping per matrix", p, dm)
                t1 = time.time()
                q["TRL-2 3INST (val-best d per matrix)"] = trl(Wr, U[dm], 2, sc, LUT3)
                tim.setdefault("TRL-2 3INST", []).append(time.time() - t1)
                t1 = time.time()
                q["TRL-3 3INST (val-best d per matrix)"] = trl(Wr, U[dm], 3, sc, LUT3)
                tim.setdefault("TRL-3 3INST", []).append(time.time() - t1)
                q["TRL-2 LUT (val-best d per matrix)"] = q[f"TRL-2 LUT d={dm}"]
                for tn in tests:
                    Hj, Hjr = Hs[tn][key]
                    msb = Q2.err(W3[p], W4[p], Hj)
                    add(f"[{tn}] MSB-3 (3.5 b)", p, msb)
                    add(f"[{tn}] MSB-3 / E4 (sanity ~3)", p, msb / (e4(rec[p]["s3u"], Hj) / float(((W4[p] @ Hj) * W4[p]).sum())))
                    for nm, Q in q.items():
                        add(f"[{tn}] {nm}", p, Q2.err(Q, Wr, Hjr))
            print(f"[b10r] layer {Ly} expert {e} done ({time.time() - t0:.0f} s)", flush=True)
        with open(os.path.expanduser("~/Fit12/data/b10r/b10r_partial.txt" if not SMOKE else "~/Fit12/data/b10r/smoke_partial.txt"), "w") as fh:   # kept if the run stops early
            print(f"partial: layers through {Ly} ({time.time() - t0:.0f} s)", file=fh)
            report(fh)
    report(sys.stdout)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    test() if a == ["test"] else main_oos() if a == ["oos"] else main_b10a() if a == ["b10a"] else main_b10r() if a == ["b10r"] \
        else main(rd_only=a == ["rd"])
