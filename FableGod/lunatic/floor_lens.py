"""BREAK98 information floor, screen 2 (Lunatic, Tue 17 Nov 08:0x). CPU only, read-only, <= ~3 GB.

floor_fix.out says the chain needs the 125B's DECISION at 1-3 rows per block (perfect FIX-2 25.68 HE+ beats the
true state at all 31 rows, 20.77), and a decision is cheap only if a SHALLOW batched pass over the whole draft can make
it. The bank already stores the 125B's mid-layer taps (raw = the mean of the 4 HC streams after zero-based layers
5 / 15 / 25 / 35 / 45, [n, 5 x 2560]; TrainLab/sticky/taps_extract.py format) along the true text, so the decider can be
measured today. Two CONTROLS: "tap 47" = the FINAL layer's 4-stream mean through the same ridge (is the mean format the
bottleneck?), "tap 48" = the exact final mixer (the 125B itself; the ceiling of the top-8 decider).
  TUNED LENS per tap: ridge from the RMS-normalised tap (+ bias) to the 125B's final mixed vector (NewIdeas' ed_lens.Lens
  .mixed on the bank's wd), fitted on 60 independent code windows (datasets/deepdraft/rawtaps, magicoder), alpha
  chosen on 20 more; then the 125B's own lm_head. Scored on the HE+ / MBPP banks (never fitted on them).
  DECIDER (non-oracle): at each drafted row, re-rank the head's own top-8 (topk cells) by lens logit (+ beta * log
  head p). It OVERRIDES the draft's #1 when another candidate leads by >= tau nats.
  FIX-r WALK with the real decider: scan the draft row by row (the batched partial pass sees every row); the first
  override is the round's fix. A wrong override ends the block (the final check rejects it and gives the bonus), so
  false overrides cost kept but never quality. Variants: no re-anchor (the drifted chain continues, teacher-forced
  cells: exactly simulable) and re-anchor (the true-state chain: an upper bound, the head must read the mapped state).
Row convention (checked on the bank): a topk cell (fi, row, depth d) stores its own verifying row (tk[row, 0] == y
at every depth; y = ids[row + 1]).
usage: nice -n 10 ~/BigMoE/.venv/bin/python3 -I floor_lens.py"""
import glob
import importlib.util
import os
import sys
import time
import numpy as np
H = os.path.expanduser("~")
spec = importlib.util.spec_from_file_location("ed_lens", f"{H}/NewIdeas/research/week98/ed_lens.py")
EL = importlib.util.module_from_spec(spec); spec.loader.exec_module(EL)      # read-only import (sets the CPU device)
CE = f"{H}/BigMoE/data/deepdraft/ceiling"; BANK = f"{H}/datasets/deepdraft/bank"; RT = f"{H}/datasets/deepdraft/rawtaps"
OUT = f"{H}/FableGod/lunatic/floor_lens.out"; lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)
TAPS = (5, 15, 25, 35, 45, 47, 48); NT = 7; NFIT = 6; NR = 5; D = 2560      # 47 = CONTROL: the final layer's 4-stream MEAN; 48 = CONTROL: the exact final mixer
LYR = (6, 16, 26, 36, 46, 48, 48)
def feats(z):
    raw = z["raw"].reshape(-1, NR, D); wm = z["wd"].reshape(-1, 4, D).astype(np.float32).mean(1)
    return np.concatenate([raw.astype(np.float32), wm[:, None, :]], 1)
LENS = EL.Lens(False)
t0 = time.time()

def xnorm(x):
    x = x.astype(np.float32)
    return np.concatenate([x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6), np.ones((len(x), 1), np.float32)], 1)

# ---------- 1. fit the tuned lens on independent code windows ----------
rt = sorted(glob.glob(f"{RT}/*.npz"))
fit_f, val_f = rt[:60], rt[60:80]
XX = np.zeros((NFIT, D + 1, D + 1), np.float32); XY = np.zeros((NFIT, D + 1, D), np.float32)
for f in fit_f:
    z = np.load(f); raw = feats(z); Y = LENS.mixed(z["wd"]).astype(np.float32)
    for t in range(NFIT):
        X = xnorm(raw[:, t]); XX[t] += X.T @ X; XY[t] += X.T @ Y
val = [(feats(np.load(f)), LENS.mixed(np.load(f)["wd"])) for f in val_f]
B = []
for t in range(NFIT):
    sc = np.trace(XX[t]) / (D + 1); best = None
    for lam in (1e-4, 1e-3, 1e-2, 1e-1):
        Bt = np.linalg.solve(XX[t].astype(np.float64) + lam * sc * np.eye(D + 1), XY[t].astype(np.float64)).astype(np.float32)
        # validation: top-1 agreement of the lens with the final mixed vector's top-1 on 600 sampled rows (fast)
        err = sum(float(((xnorm(r[:, t]) @ Bt - y) ** 2).mean()) for r, y in val) / len(val)
        if best is None or err < best[0]: best = (err, lam, Bt)
    B.append(best[2]); say(f"tap after layer {TAPS[t]:2d}: ridge lambda {best[1]:g}, val MSE {best[0]:.4f}")
del XX, XY, val
say(f"   (fit {time.time()-t0:.0f} s)")

# ---------- 2. lens logits on the head's top-8 at every cell of the banks ----------
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

for tag, pat in (("heplus24", "tf_code_HumanEval_*.npz"), ("mbppbank", "dlog__mbpp*.npz")):
    tz = np.load(f"{CE}/topk_t30b_{tag}.npz")
    files = list(tz["files"]); dep = tz["depth"].astype(int); fi = tz["fi"].astype(int); an = tz["row"].astype(int)
    y = tz["y"]; right = tz["right"].astype(np.int8); tid = tz["top_ids"]; tp = tz["top_p"].astype(np.float64)
    nc = len(dep); S = np.full((NT, nc, 8), np.nan, np.float32); tkok = np.zeros(nc, bool)
    bank = {os.path.basename(f): f for f in glob.glob(f"{BANK}/{pat}")}
    for k, fn in enumerate(files):
        z = np.load(bank[fn]); raw = feats(z); tk = z["tk"]
        cells = np.flatnonzero(fi == k); vr = an[cells]                         # the cell row IS its verifying row (tk[row, 0] == y at every depth; checked)
        ok = vr < len(raw); cells = cells[ok]; vr = vr[ok]
        tkok[cells] = tk[vr, 0] == y[cells]
        urow, rinv = np.unique(vr, return_inverse=True)
        uid, iinv = np.unique(tid[cells], return_inverse=True); iinv = iinv.reshape(-1, 8)
        Wv = LENS.rows(uid)                                            # [|uid|, 2560]
        for t in range(NT):
            m = (xnorm(raw[urow, t]) @ B[t]) if t < NFIT else LENS.mixed(z["wd"][urow])   # predicted / exact mixed
            LU = m @ Wv.T                                              # [rows, |uid|]
            S[t, cells] = LU[rinv[:, None], iinv]
    say(f"== {tag}: {nc} cells; 125B argmax at the verifying row == cell y on {100*tkok.mean():.2f} % (near-tie flips)")
    # windows of anchors x depth -> cell index (as floor_fix / drift_forksplit)
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0)); en = np.append(st[1:], len(dep))
    WIN = []
    for s, t_ in zip(st, en):
        dw = dep[s:t_]; n0 = int((dw == 0).sum())
        C = np.full((n0 + 1, DEP), -1, np.int64)
        for j in range(DEP):
            idx = s + np.flatnonzero(dw == j); C[:len(idx), j] = idx
        WIN.append(C)
    # decision-row diagnostics: the plain chain's first misses and the rows before them
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
    say(f"   plain chain-32 walk: {len(miss_cells)} first misses; LAB head right there {100*np.mean(lab_ok):.1f} %; "
        f"y in head top-8 there {100*np.mean((tid[miss_cells] == y[miss_cells, None]).any(1)):.1f} %")
    # LOCATOR: does the tap flag the first miss (no token needed) better than the head's own margin? AUC + recall
    def auc_rec(sc):
        a = sc[miss_cells]; b = sc[pre_cells]
        allv = np.concatenate([a, b]); rk = np.argsort(np.argsort(allv)) + 1.0
        auc = (rk[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
        thr = np.quantile(b, 0.98); return auc, float((a > thr).mean())
    lp_ = np.log(np.maximum(tp, 1e-9))
    a0, r0 = auc_rec(lp_[:, 1] - lp_[:, 0])
    say(f"   LOCATOR (first-miss rows vs right rows of the same walks): head margin alone AUC {a0:.3f}, recall at 2 % false {100*r0:4.1f} %")
    for t in range(NT):
        Zl = np.nan_to_num(S[t], nan=-1e9); gl = np.max(Zl[:, 1:], 1) - Zl[:, 0]
        Zc = Zl + lp_; gc = np.max(Zc[:, 1:], 1) - Zc[:, 0]
        a1, r1 = auc_rec(gl); a2, r2 = auc_rec(gc)
        say(f"     tap {TAPS[t]:2d}: lens alone AUC {a1:.3f} (recall@2% {100*r1:4.1f} %) | lens + head AUC {a2:.3f} (recall@2% {100*r2:4.1f} %)")
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
                row.append(f"tau {tau:g}: fix {100*fixed:4.1f} % / false {100*false:5.2f} %")
            say(f"   tap {TAPS[t]:2d} beta {beta:g}: lens-on-top8 = y at first misses {100*pick_y[mf].mean():5.1f} % | " + " | ".join(row))
    # FIX-r walks with the real decider (best tap / beta / tau searched on this set: information, see note)
    say("   FIX-r with the real decider (no re-anchor = exact from the cells; re-anchor = upper bound), priced at the bound:")
    base_tok = 0; base_n = 0
    for C in WIN:
        n = C.shape[0]; p = 1
        while p < n:
            a = 0
            while a < 31 and C[p, a] >= 0 and right[C[p, a]] == 1: a += 1
            base_tok += a + 1; base_n += 1; p += a + 1
    say(f"     plain chain-32: kept {base_tok/base_n:6.2f}  tok/s@bound {1000*base_tok/(base_n*check_ms()):4.0f}")
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
                say(f"     tap {TAPS[t]:2d} (L={LYR[t]}) beta {beta:g} tau {tau:g}: no re-anchor r1/2/3 kept "
                    + " ".join(f"{k_:5.2f}({s_:3.0f})" for k_, s_ in res[:3]) + " | re-anchor "
                    + " ".join(f"{k_:5.2f}({s_:3.0f})" for k_, s_ in res[3:]))
    say(f"   ({time.time()-t0:.0f} s)")
    say()
open(OUT, "w").write("\n".join(lines) + "\n")
