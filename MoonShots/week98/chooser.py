"""WEEK98 result E (MoonShots, Sun 15 Nov 2026): the LEARNED CHOOSER screen, exactly as pre-registered in
LEVERS_A_MOONSHOTS.md before this file was written. Draft-time features only (depth bins, the 2B's margin at the row and the
previous row, the 2B's agreement with the drafted token at the previous 1-2 rows, the 2B's top-1 token class, depth x
margin); NOT the cells' m / rank / dlp (the 125B's log-probs = the verifier's answer) and not head-side features (not in the
dumps). L2 logistic regression on the decision cells (exactly one of head / 2B right; y = the 2B is right), override where
P > 0.5. Train: MBPP's first 50 files (sorted); test: MBPP's other 50 and all 24 HE+ windows (split by problem).
LINE: ALIVE if HE+ 23-30 >= 96.0 and MBPP(test) 23-30 >= 88.0 with depth 0 within 0.3 pt of the head on both; else the
frozen rule stands. Inputs read-only: BigMoE's CEILING cells (MBPP) and MTP Head's (HE+), rows2b_*.npz, the 2B tokenizer.
usage: ~/BigMoE/.venv/bin/python -I chooser.py"""
import glob, json, os
import numpy as np

H = os.path.expanduser
OUT = H("~/MoonShots/week98")
BANDS = [(0, 0), (1, 2), (3, 6), (7, 10), (11, 14), (15, 22), (23, 30)]
SCHED = {0: 4.0, 1: 2.0, 2: 2.0, **{j: 1.0 for j in range(3, 7)}, **{j: 0.5 for j in range(7, 31)}}
TOKJ = sorted(glob.glob(H("~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-2B-MLX-4bit/snapshots/*/tokenizer.json")))[0]


def mbpp_cells():
    """BigMoE's CEILING cells, aligned to (file, row) by the stored margin (as join_mbpp.py)"""
    C = np.load(H("~/BigMoE/data/deepdraft/ceiling/cells_t30b_mbpp100.npz"))
    files = sorted(glob.glob(H("~/TrainLab/data/dlog/d1102mbpp/dlog__mbpp*.npz")))[:100]
    depth, right, mcell = C["depth"], C["right"].astype(bool), C["m"]
    starts = np.r_[0, np.flatnonzero(np.diff(depth) < 0) + 1, len(depth)]
    fi = np.empty(len(depth), np.int64); rw = np.empty(len(depth), np.int64)
    for k, f in enumerate(files):
        z = np.load(f); lp = np.asarray(z["lp"], np.float32); L, gen = len(lp), int(z["gen"]); a0 = max(0, gen - 2)
        for j in range(31):
            seg = np.arange(starts[k], starts[k + 1]); seg = seg[depth[seg] == j]
            rows = np.arange(a0 + j + 1, L); rows = rows[rows >= gen - 1]; cm = lp[rows, 0] - lp[rows, 1]; p = 0
            for c in seg:
                while cm[p] != mcell[c]:
                    p += 1
                fi[c], rw[c] = k, rows[p]; p += 1
    return dict(depth=depth, right=right, fi=fi, row=rw, an=rw - depth - 1, files=[os.path.basename(f) for f in files])


def heplus_cells():
    C = np.load(H("~/BigMoE/data/deepdraft/ceiling/cells_t30b_heplus.npz"), allow_pickle=True)
    return dict(depth=C["depth"], right=C["right"].astype(bool), fi=C["fi"], row=C["row"], an=C["an"],
                files=[os.path.basename(str(f)) for f in C["files"]])


def attach_2b(c, rows2b, cls_of):
    R = np.load(rows2b)
    key = {(os.path.basename(str(n)), int(r)): (bool(t == k), float(mg), int(t))
           for n, r, t, k, mg in zip(R["fname"], R["row"], R["top1"], R["tk0"], R["margin2b"])}
    def get(f, r, i):
        v = key.get((c["files"][int(f)], int(r)))
        return v[i] if v is not None else None
    n = len(c["depth"])
    c["r2b"] = np.array([get(f, r, 0) for f, r in zip(c["fi"], c["row"])], bool)
    c["m2b"] = np.array([get(f, r, 1) for f, r in zip(c["fi"], c["row"])], np.float32)
    cls = np.array([cls_of(get(f, r, 2)) for f, r in zip(c["fi"], c["row"])], np.int64)
    prev = []
    for d in (1, 2):                                       # the 2B at the previous rows (a missing row counts as unknown)
        vals = [key.get((c["files"][int(f)], int(r) - d)) for f, r in zip(c["fi"], c["row"])]
        prev.append((np.array([v[0] if v else 0.5 for v in vals], np.float32),
                     np.array([v[1] if v else 0.0 for v in vals], np.float32)))
    c["cls"], c["agree1"], c["m2b_prev"], c["agree2"] = cls, prev[0][0], prev[0][1], prev[1][0]
    return c


def features(c):
    d = c["depth"]
    B = np.stack([(d >= a) & (d <= b) for a, b in BANDS], 1).astype(np.float32)
    m = np.minimum(c["m2b"], 12.0)[:, None] / 4.0
    mp = np.minimum(c["m2b_prev"], 12.0)[:, None] / 4.0
    K = np.eye(6, dtype=np.float32)[c["cls"]]
    return np.concatenate([B, m, B * m, mp, c["agree1"][:, None], c["agree2"][:, None], K], 1)


def fit_logreg(X, y, lam=1e-2, iters=50):
    Xb = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1).astype(np.float64)
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):                                 # Newton / IRLS with an L2 penalty
        p = 1 / (1 + np.exp(-Xb @ w)); g = Xb.T @ (p - y) + lam * w
        Hm = (Xb * (p * (1 - p))[:, None]).T @ Xb + lam * np.eye(len(w))
        step = np.linalg.solve(Hm, g); w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def predict(w, X):
    Xb = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1)
    return 1 / (1 + np.exp(-(Xb @ w)))


def bands(c, ok, sel):
    d = c["depth"][sel]; o = ok[sel]
    return {(f"{a}-{b}" if a != b else str(a)): round(100 * float(o[(d >= a) & (d <= b)].mean()), 2) for a, b in BANDS}


def walk(c, sel, ok_head, ok_comb, P):
    fi, an, dp = c["fi"][sel], c["an"][sel], c["depth"][sel]
    h_, c_ = ok_head[sel], ok_comb[sel]
    o = np.lexsort((dp, an, fi)); fi, an, h_, c_ = fi[o], an[o], h_[o], c_[o]
    st = np.r_[0, np.flatnonzero((np.diff(fi) != 0) | (np.diff(an) != 0)) + 1, len(fi)]
    acc = []
    for s0, s1 in zip(st[:-1], st[1:]):
        n, used = 0, 0
        for j in range(s0, s1):
            if P is not None and used >= P:
                if h_[j]:
                    n += 1; continue
                break
            if h_[j] and c_[j]:
                n += 1
            elif not h_[j] and c_[j]:
                used += 1; n += 1
            else:
                break
        acc.append(n)
    return round(float(np.mean(acc)), 3)


def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKJ)
    cache = {}
    def cls_of(t):
        if t is None:
            return 5
        if t not in cache:
            s = tok.decode([int(t)])
            cache[t] = (0 if "\n" in s else 1 if s.strip() == "" else 4 if s.strip().isdigit()
                        else 3 if any(ch.isalpha() or ch == "_" for ch in s) else 2 if s.strip() else 5)
        return cache[t]
    M = attach_2b(mbpp_cells(), H("~/MoonShots/week98/rows2b_mbpp.npz"), cls_of)
    E = attach_2b(heplus_cells(), H("~/MoonShots/week98/rows2b_heplus.npz"), cls_of)
    tr = M["fi"] < 50; te_m = ~tr; te_h = np.ones(len(E["depth"]), bool)
    XM, XE = features(M), features(E)
    dec = tr & (M["right"] ^ M["r2b"])
    w = fit_logreg(XM[dec], M["r2b"][dec].astype(np.float64))
    print(f"trained on {int(dec.sum())} decision cells of MBPP files 0-49")
    res = {"features": XM.shape[1], "train_decision_cells": int(dec.sum())}
    for name, c, X, sel in (("mbpp_test", M, XM, te_m), ("heplus", E, XE, te_h)):
        tau = np.array([SCHED[int(d)] for d in c["depth"]], np.float32)
        frozen = np.where(c["m2b"] >= tau, c["r2b"], c["right"])
        ch = np.where(predict(w, X) > 0.5, c["r2b"], c["right"])
        oracle = c["right"] | c["r2b"]
        r = {}
        for nm_, ok in (("head", c["right"]), ("frozen", frozen), ("chooser", ch), ("oracle", oracle)):
            r[nm_] = dict(bands=bands(c, ok, sel), acc_P1=walk(c, sel, c["right"], ok, 1 if nm_ != "head" else 0),
                          acc_P2=walk(c, sel, c["right"], ok, 2 if nm_ != "head" else 0))
        h23, f23, c23, o23 = (r[k]["bands"]["23-30"] for k in ("head", "frozen", "chooser", "oracle"))
        r["share_of_bound_23_30"] = dict(frozen=round((f23 - h23) / max(o23 - h23, 1e-9), 3), chooser=round((c23 - h23) / max(o23 - h23, 1e-9), 3))
        res[name] = r
        print(f"== {name}")
        for k in ("head", "frozen", "chooser", "oracle"):
            print(f"  {k:8s} {r[k]['bands']}  accepted P1 {r[k]['acc_P1']} P2 {r[k]['acc_P2']}")
        print("  share of the bound at 23-30:", r["share_of_bound_23_30"])
    hm, he = res["mbpp_test"], res["heplus"]
    alive = (he["chooser"]["bands"]["23-30"] >= 96.0 and hm["chooser"]["bands"]["23-30"] >= 88.0
             and he["chooser"]["bands"]["0"] >= he["head"]["bands"]["0"] - 0.3 and hm["chooser"]["bands"]["0"] >= hm["head"]["bands"]["0"] - 0.3)
    res["line"] = "ALIVE" if alive else "the frozen rule stands"
    print("LINE:", res["line"])
    json.dump(res, open(os.path.join(OUT, "chooser.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
