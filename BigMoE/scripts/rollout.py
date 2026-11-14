#!/usr/bin/env python3
"""ROLLOUT (docs/PREREG_DEEPDRAFT_ROLLOUT_2026-11-14.md): a linear state-transition model of the trunk's widened stream,
z_{t+1} = A z_t + B e_{t+1} + c in the top-k PCA space of wd, with e = the 125B's input embedding of the token at t+1.
  rollout.py emb              the 125B's input embeddings of every token in the bank -> ckpt/bank_emb.npz (MLX, ~1 min)
  rollout.py fit [--k 512]    PCA + ridge + diagnostics (numpy) -> ckpt/rollout_k<k>.npz + data/deepdraft/bank/rollout_fit_k<k>.json
The diagnostics on the 40 val files, with the direction error = bank_proxy.py's RMS-normalized error:
  - the PCA ceiling;
  - the one-step teacher-forced error;
  - the rollout's error by depth from a fresh anchor, fed the true tokens;
  - the stale anchor's error by depth.
"""
import argparse, glob, json, os, sys, time
import numpy as np

H = os.path.expanduser
BANK = H("~/datasets/deepdraft/bank"); CK = H("~/datasets/deepdraft/ckpt")
SPLIT = H("~/BigMoE/data/deepdraft/bank/split.json"); OUTD = H("~/BigMoE/data/deepdraft/bank")
DEPTHS = (1, 2, 4, 8, 16, 30)


def ne(P, Y):
    pn = P / np.sqrt((P ** 2).mean(-1, keepdims=True) + 1e-12); yn = Y / np.sqrt((Y ** 2).mean(-1, keepdims=True) + 1e-12)
    return float(np.sqrt(((pn - yn) ** 2).sum() / (yn ** 2).sum()))


def cmd_emb(a):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import dd_ttt as DD
    import mlx.core as mx
    rt, lm, head = DD.M.load(0.03)
    ids = set()
    for f in glob.glob(os.path.join(BANK, "*.npz")):
        ids.update(np.load(f)["ids"].tolist())
    u = np.array(sorted(ids), np.int32)
    E = np.concatenate([np.array(lm.model.embed_tokens(mx.array(u[i:i + 4096])).astype(mx.float32)) for i in range(0, len(u), 4096)])
    np.savez(os.path.join(CK, "bank_emb.npz"), ids=u, emb=E.astype(np.float16))
    print(f"[ro] embeddings: {len(u)} distinct bank tokens x {E.shape[1]} -> {CK}/bank_emb.npz", flush=True)


def cmd_fit(a):
    t0 = time.time()
    s = json.load(open(SPLIT)); em = np.load(os.path.join(CK, "bank_emb.npz")); eid, E = em["ids"], em["emb"].astype(np.float32)
    e_of = lambda tok: E[np.searchsorted(eid, tok)]
    files = {nm: np.load(os.path.join(BANK, nm)) for nm in s["train"] + s["val"]}
    # 1. PCA of wd from every 4th answer row of the train files
    S = np.concatenate([np.asarray(files[nm]["wd"][int(files[nm]["gen"]) - 1::4], np.float32) for nm in s["train"]])
    mu = S.mean(0); C = (S - mu).T @ (S - mu) / len(S)
    w, V = np.linalg.eigh(C); V = V[:, ::-1][:, :a.k].astype(np.float32); w = w[::-1]
    print(f"[ro] PCA from {len(S)} rows: top {a.k} components hold {100 * w[:a.k].sum() / w.sum():.1f} % of the variance ({time.time() - t0:.0f} s)", flush=True)
    del S, C
    proj = lambda X: (X - mu) @ V
    back = lambda Z: mu + Z @ V.T
    # 2. pairs (t, t+1) on answer rows, every 2nd t
    def pairs(names, step, keep_w):
        Zt, Et, Zn, Wn = [], [], [], []
        for nm in names:
            z = files[nm]; wd = np.asarray(z["wd"], np.float32); ids = np.asarray(z["ids"]); gen = int(z["gen"]); n = wd.shape[0]
            t = np.arange(gen - 1, n - 1)[::step]
            Zt.append(proj(wd[t])); Et.append(e_of(ids[t + 1])); Zn.append(proj(wd[t + 1]))
            if keep_w:
                Wn.append(wd[t + 1])
        return np.concatenate(Zt), np.concatenate(Et), np.concatenate(Zn), (np.concatenate(Wn) if keep_w else None)
    Zt, Et, Zn, _ = pairs(s["train"], 2, False)
    Zv, Ev, Znv, Wnv = pairs(s["val"], 1, True)
    X = np.concatenate([Zt, Et], 1); Xv = np.concatenate([Zv, Ev], 1)
    xm, ym = X.mean(0), Zn.mean(0); Xc = X - xm
    XtX = Xc.T.astype(np.float64) @ Xc; XtY = Xc.T.astype(np.float64) @ (Zn - ym); dg = float(np.mean(np.diag(XtX)))
    best = None
    for lf in (1e-4, 1e-3, 1e-2, 1e-1):
        Wr = np.linalg.solve(XtX + lf * dg * np.eye(X.shape[1]), XtY).astype(np.float32)
        e1 = ne(back((Xv - xm) @ Wr + ym), Wnv)
        print(f"[ro] lambda {lf:g}: one-step direction error {e1:.4f}", flush=True)
        if best is None or e1 < best[0]:
            best = (e1, lf, Wr)
    e1, lf, Wr = best
    k = a.k
    A, B, c = Wr[:k].T, Wr[k:].T, (ym - xm @ Wr).astype(np.float32)          # z' = A z + B e + c
    res = {"k": k, "lambda_factor": lf, "train_pairs": len(X), "val_pairs": len(Xv), "pca_ceiling": ne(back(Znv), Wnv), "one_step": e1}
    print(f"[ro] PCA ceiling {res['pca_ceiling']:.4f}; one-step {e1:.4f} (best lambda {lf:g}; {len(X)} train pairs)", flush=True)
    # 3. rollout vs stale by depth on val anchors (every 4th answer row)
    ro = {d: [[], []] for d in DEPTHS}; st = {d: [[], []] for d in DEPTHS}
    for nm in s["val"]:
        z = files[nm]; wd = np.asarray(z["wd"], np.float32); ids = np.asarray(z["ids"]); gen = int(z["gen"]); n = wd.shape[0]
        anc = np.arange(gen - 1, n - 31)[::4]
        if len(anc) == 0:
            continue
        Z = proj(wd[anc])
        for d in range(1, 31):
            Z = Z @ A.T + e_of(ids[anc + d]) @ B.T + c
            if d in DEPTHS:
                ro[d][0].append(back(Z)); ro[d][1].append(wd[anc + d]); st[d][0].append(wd[anc]); st[d][1].append(wd[anc + d])
    res["rollout_by_depth"] = {d: ne(np.concatenate(ro[d][0]), np.concatenate(ro[d][1])) for d in DEPTHS}
    res["stale_by_depth"] = {d: ne(np.concatenate(st[d][0]), np.concatenate(st[d][1])) for d in DEPTHS}
    for d in DEPTHS:
        print(f"[ro] depth {d:2d}: rollout direction error {res['rollout_by_depth'][d]:.4f}, stale anchor {res['stale_by_depth'][d]:.4f}", flush=True)
    np.savez(os.path.join(CK, f"rollout_k{k}.npz"), mu=mu.astype(np.float32), V=V, A=A, B=B, c=c)
    json.dump(res, open(os.path.join(OUTD, f"rollout_fit_k{k}.json"), "w"), indent=1)
    print(f"[ro] saved; {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["emb", "fit"]); ap.add_argument("--k", type=int, default=512)
    a = ap.parse_args(); {"emb": cmd_emb, "fit": cmd_fit}[a.cmd](a)
