#!/usr/bin/env python3
"""VERDICT-DEPTH on BigMoE's day capture (NewIdeas, Tue 17 Nov; the time is the commit's). A minimal-change copy of Lunatic's
~/FableGod/lunatic/floor_lens.py (number-checked 07:5x: it reproduced exactly), with ONLY the features and the lens fit
changed so it reads today's capture:
  features  the 4-stream mean of the capture's widened stream X at layers 2 / 4 / 8 / 16 / 24 / 32 / 40 / 44 (layer L = the
            output of layers[L-1]; its layer-16 mean == the bank's raw tap 15 bitwise, BigMoE's self-check), plus the
            CONTROLS of the original: 47 = the final layer's 4-stream mean through the same ridge, 48 = the exact mixer;
  fit       the ridge to the 125B's final mixed vector is CROSS-FITTED by file parity on the capture itself (rawtaps has
            none of these layers): lambda picked on half of the fitting parity's files, refit on the whole parity, then
            every file is scored with the other parity's map.
Everything after the features (top-8 logits, LOCATOR, DECIDER, FIX-r walks, pricing) is Lunatic's code unchanged, plus
Fable C's line in the LOCATOR: recall at <= 20 % false flags (PASS recall >= 0.8 at L16, with <= 20 % false flags).
Lunatic's LIVE line: a layer <= 24 gives decider phi >= 0.7 at psi <= 0.5 % (read from the DECIDER rows: fix % at false %).
usage: taskpolicy -b ~/BigMoE/.venv/bin/python -I -B floor_lens_capture.py [he|mbpp|both]
"""
import glob
import importlib.util
import os
import sys
import time
import numpy as np
H = os.path.expanduser("~")
spec = importlib.util.spec_from_file_location("ed_lens", f"{H}/NewIdeas/research/week98/ed_lens.py")
EL = importlib.util.module_from_spec(spec); spec.loader.exec_module(EL)
CE = f"{H}/BigMoE/data/deepdraft/ceiling"; BANK = f"{H}/datasets/deepdraft/bank"
HERE = os.path.dirname(os.path.abspath(__file__))
WHICH = sys.argv[1] if len(sys.argv) > 1 else "both"
OUT = os.path.join(HERE, f"floor_lens_capture_{WHICH}.out"); lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)
CAPL = (2, 4, 8, 16, 24, 32, 40, 44)
TAPS = CAPL + (47, 48); NT = len(TAPS); NFIT = NT - 1; D = 2560
LYR = CAPL + (48, 48)
LENS = EL.Lens(False)
t0 = time.time()

def xnorm(x):
    x = x.astype(np.float32)
    return np.concatenate([x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6), np.ones((len(x), 1), np.float32)], 1)

DEP = 31
F = {1: 480, 2: 406, 3: 365, 4: 337, 6: 300, 8: 276, 12: 241, 16: 218, 24: 185, 32: 163}
def f_of(R):
    ks = sorted(F)
    if R in F: return F[R]
    lo = max(k for k in ks if k <= R); hi = min(k for k in ks if k >= R)
    return F[lo] + (F[hi] - F[lo]) * (np.log(R) - np.log(lo)) / (np.log(hi) - np.log(lo))
U32 = 32 * f_of(32)
def check_ms(r=32):
    return max(7.6, 4.9 * r / 32) + 16.4 * (r * f_of(r)) / U32 + 1.2 * r / 32 + 1.5 + 3.0 * r / 31
def round_ms(Lyr, n):
    n = max(1, n)
    return (Lyr / 48) * (7.6 + 16.4 * (n * f_of(n)) / U32 + 1.2 * n / 32) + 0.5 + 0.3 + 3.0 * n / 31

SETS = [("heplus24", "tf_code_HumanEval_*.npz"), ("mbppbank", "dlog__mbpp*.npz")]
if WHICH == "he": SETS = SETS[:1]
if WHICH == "mbpp": SETS = SETS[1:]
for tag, pat in SETS:
    tz = np.load(f"{CE}/topk_t30b_{tag}.npz")
    files = [os.path.basename(str(f)) for f in tz["files"]]
    dep = tz["depth"].astype(int); fi = tz["fi"].astype(int); an = tz["row"].astype(int)
    y = tz["y"]; right = tz["right"].astype(np.int8); tid = tz["top_ids"]; tp = tz["top_p"].astype(np.float64)
    nc = len(dep)
    cz = np.load(f"{CE}/mid_{tag}.npz")
    X = cz["X"]; cfi = cz["fi"].astype(int); crow = cz["row"].astype(int); cly = cz["layer"].astype(int)   # X read ONCE
    tfiles = [os.path.basename(str(f)) for f in np.load(f"{CE}/tree_t30b_{tag}_n127k8.npz", allow_pickle=True)["files"]]
    cidx = {fn: k for k, fn in enumerate(tfiles)}
    bank = {os.path.basename(f): f for f in glob.glob(f"{BANK}/{pat}")}
    say(f"== {tag}: capture X {X.shape} {X.dtype}, layers {sorted(set(cly.tolist()))}; {len(files)} topk files")
    # ---- per-file features at the cells' verifying rows: [rows, NFIT, D] (capture layers + the final mean), Y, wd ----
    FE, YY, WD, URW, CELL, RINV = {}, {}, {}, {}, {}, {}
    missing = 0
    for k, fn in enumerate(files):
        z = np.load(bank[fn]); wd = z["wd"]
        cells = np.flatnonzero(fi == k); vr = an[cells]
        ok = vr < len(wd); cells = cells[ok]; vr = vr[ok]
        urow, rinv = np.unique(vr, return_inverse=True)
        fe = np.zeros((len(urow), NFIT, D), np.float32)
        ck = cidx[fn]; sel = cfi == ck
        for t, L in enumerate(CAPL):
            m = sel & (cly == L)
            ix = dict(zip(crow[m].tolist(), np.flatnonzero(m).tolist()))
            got = np.array([ix.get(int(r), -1) for r in urow])
            missing += int((got < 0).sum())
            g = got >= 0
            fe[g, t] = X[got[g]].astype(np.float32).reshape(-1, 4, D).mean(1)
        fe[:, NFIT - 1] = wd[urow].astype(np.float32).reshape(-1, 4, D).mean(1)              # control 47: the final mean
        FE[k] = fe; YY[k] = LENS.mixed(wd[urow]).astype(np.float32); WD[k] = wd[urow]
        URW[k] = urow; CELL[k] = cells; RINV[k] = rinv
    say(f"   rows lacking a capture state: {missing} (zero-filled)")
    # ---- cross-fitted ridge per tap: fit on one parity (lambda by its own half), score the other parity ----
    B = {}
    for par in (0, 1):
        fit_k = [k for k in FE if k % 2 == par]
        tr_k = [k for k in fit_k if (k // 2) % 2 == 0]; va_k = [k for k in fit_k if (k // 2) % 2 == 1]
        for t in range(NFIT):
            def acc(ks):
                XX = np.zeros((D + 1, D + 1), np.float64); XY = np.zeros((D + 1, D), np.float64)
                for k in ks:
                    Xn = xnorm(FE[k][:, t]); XX += Xn.T @ Xn; XY += Xn.T @ YY[k]
                return XX, XY
            XXt, XYt = acc(tr_k); sc = np.trace(XXt) / (D + 1); best = None
            for lam in (1e-4, 1e-3, 1e-2, 1e-1):
                Bt = np.linalg.solve(XXt + lam * sc * np.eye(D + 1), XYt).astype(np.float32)
                err = np.mean([float(((xnorm(FE[k][:, t]) @ Bt - YY[k]) ** 2).mean()) for k in va_k]) if va_k else 0.0
                if best is None or err < best[0]: best = (err, lam)
            XXa, XYa = acc(fit_k); sca = np.trace(XXa) / (D + 1)
            B[(par, t)] = np.linalg.solve(XXa + best[1] * sca * np.eye(D + 1), XYa).astype(np.float32)
            if par == 0:
                say(f"   tap after layer {TAPS[t]:2d}: lambda {best[1]:g} (val MSE {best[0]:.4f}, fit parity 0)")
    say(f"   (fit {time.time() - t0:.0f} s)")
    # ---- lens logits on the head's top-8 at every cell (each file scored by the OTHER parity's map) ----
    S = np.full((NT, nc, 8), np.nan, np.float32); tkok = np.zeros(nc, bool)
    for k in FE:
        cells = CELL[k]; rinv = RINV[k]
        z = np.load(bank[files[k]]); tk = z["tk"]
        tkok[cells] = tk[an[cells], 0] == y[cells]
        uid, iinv = np.unique(tid[cells], return_inverse=True); iinv = iinv.reshape(-1, 8)
        Wv = LENS.rows(uid)
        for t in range(NT):
            m = (xnorm(FE[k][:, t]) @ B[(1 - k % 2, t)]) if t < NFIT else LENS.mixed(WD[k])
            LU = m @ Wv.T
            S[t, cells] = LU[rinv[:, None], iinv]
    say(f"   {nc} cells; 125B argmax at the verifying row == cell y on {100 * tkok.mean():.2f} %")
    del X, FE
    # ---------------- unchanged from floor_lens.py from here (Fable C's recall@20 % added) ----------------
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0)); en = np.append(st[1:], len(dep))
    WIN = []
    for s, t_ in zip(st, en):
        dw = dep[s:t_]; n0 = int((dw == 0).sum())
        C = np.full((n0 + 1, DEP), -1, np.int64)
        for j in range(DEP):
            idx = s + np.flatnonzero(dw == j); C[:len(idx), j] = idx
        WIN.append(C)
    miss_cells, pre_cells, lab_ok = [], [], []
    for C in WIN:
        n = C.shape[0]; p = 1
        while p < n:
            a = 0
            while a < 31 and C[p, a] >= 0 and right[C[p, a]] == 1:
                pre_cells.append(C[p, a]); a += 1
            if a < 31 and C[p, a] >= 0:
                miss_cells.append(C[p, a]); lab_ok.append(p + a < n and C[p + a, 0] >= 0 and right[C[p + a, 0]] == 1)
            p += a + 1
    miss_cells = np.array(miss_cells); pre_cells = np.array(pre_cells)
    say(f"   plain chain-32 walk: {len(miss_cells)} first misses; LAB head right there {100 * np.mean(lab_ok):.1f} %; "
        f"y in head top-8 there {100 * np.mean((tid[miss_cells] == y[miss_cells, None]).any(1)):.1f} %")
    def auc_rec(sc):
        a = sc[miss_cells]; b = sc[pre_cells]
        allv = np.concatenate([a, b]); rk = np.argsort(np.argsort(allv)) + 1.0
        auc = (rk[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
        thr2 = np.quantile(b, 0.98); thr20 = np.quantile(b, 0.80)
        return auc, float((a > thr2).mean()), float((a > thr20).mean())
    lp_ = np.log(np.maximum(tp, 1e-9))
    a0, r0, q0 = auc_rec(lp_[:, 1] - lp_[:, 0])
    say(f"   LOCATOR (first-miss rows vs right rows): head margin alone AUC {a0:.3f}, recall@2% {100 * r0:4.1f} %, recall@20% {100 * q0:4.1f} %")
    for t in range(NT):
        Zl = np.nan_to_num(S[t], nan=-1e9); gl = np.max(Zl[:, 1:], 1) - Zl[:, 0]
        Zc = Zl + lp_; gc = np.max(Zc[:, 1:], 1) - Zc[:, 0]
        a1, r1, q1 = auc_rec(gl); a2, r2, q2 = auc_rec(gc)
        say(f"     layer {TAPS[t]:2d}: lens alone AUC {a1:.3f} (recall@2% {100 * r1:4.1f} %, @20% {100 * q1:4.1f} %) | "
            f"lens + head AUC {a2:.3f} (recall@2% {100 * r2:4.1f} %, @20% {100 * q2:4.1f} %)")
    for t in range(NT):
        for beta in (0.0, 1.0):
            Z = S[t] + beta * np.log(np.maximum(tp, 1e-9))
            best = np.nanargmax(np.nan_to_num(Z, nan=-1e9), 1); gap = Z[np.arange(nc), best] - Z[:, 0]
            pick_y = tid[np.arange(nc), best] == y
            mf = miss_cells; pr = pre_cells
            row = []
            for tau in (0.0, 1.0, 2.0, 4.0):
                ov = (best != 0) & (gap >= tau)
                fixed = (ov[mf] & pick_y[mf]).mean(); false = ov[pr].mean()
                row.append(f"tau {tau:g}: fix {100 * fixed:4.1f} % / false {100 * false:5.2f} %")
            say(f"   layer {TAPS[t]:2d} beta {beta:g}: lens-on-top8 = y at first misses {100 * pick_y[mf].mean():5.1f} % | " + " | ".join(row))
    say("   FIX-r with the real decider (no re-anchor = exact from the cells; re-anchor = upper bound), priced at the bound:")
    base_tok = 0; base_n = 0
    for C in WIN:
        n = C.shape[0]; p = 1
        while p < n:
            a = 0
            while a < 31 and C[p, a] >= 0 and right[C[p, a]] == 1: a += 1
            base_tok += a + 1; base_n += 1; p += a + 1
    say(f"     plain chain-32: kept {base_tok / base_n:6.2f}  tok/s@bound {1000 * base_tok / (base_n * check_ms()):4.0f}")
    for t in range(NT):
        for beta in (0.0, 1.0):
            Z = S[t] + beta * np.log(np.maximum(tp, 1e-9))
            best = np.nanargmax(np.nan_to_num(Z, nan=-1e9), 1); gap = Z[np.arange(nc), best] - Z[:, 0]
            pick_y = tid[np.arange(nc), best] == y
            for tau in (1.0, 2.0, 4.0):
                ov = (best != 0) & (gap >= tau)
                res = []
                for reanchor in (False, True):
                    for r in (1, 2, 3):
                        tok = 0; nchk = 0; extra = 0.0; nfix = 0
                        for C in WIN:
                            n = C.shape[0]; p = 1
                            while p < n:
                                q = p; k = 0; acc = 0; used = 0
                                while acc < 31 and q < n and k < 31 and C[q, k] >= 0:
                                    c = C[q, k]
                                    if used < r and ov[c]:
                                        used += 1; extra += round_ms(LYR[t], 31 - acc)
                                        if not pick_y[c]: break
                                        nfix += 1; acc += 1
                                        if reanchor: q = q + k + 1; k = 0
                                        else: k += 1
                                        continue
                                    if right[c] == 1: acc += 1; k += 1; continue
                                    break
                                tok += acc + 1; nchk += 1; p += acc + 1
                        res.append((tok / nchk, 1000 * tok / (nchk * check_ms() + extra)))
                say(f"     layer {TAPS[t]:2d} (L={LYR[t]}) beta {beta:g} tau {tau:g}: no re-anchor r1/2/3 kept "
                    + " ".join(f"{k_:5.2f}({s_:3.0f})" for k_, s_ in res[:3]) + " | re-anchor "
                    + " ".join(f"{k_:5.2f}({s_:3.0f})" for k_, s_ in res[3:]))
    say(f"   ({time.time() - t0:.0f} s)")
    say()
open(OUT, "w").write("\n".join(lines) + "\n")
