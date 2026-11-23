"""TrainLab (10-22): zero-shot SOFT stickiness on the 125B traces, next to Chat1000's hard caps (scripts/
switchcap_hard125.py: same traces, error energy, unions and KLD law). Policy: the previous token's set S gets +delta on
its logits (x e^delta on its probabilities) before the top-10; members of S outside t's top-20 have unknown weight and
are dropped (mode "drop": conservative for the bonus) or get t's 20th weight as score and energy (mode "w20": the
most a dropped-out member can weigh; optimistic for stickiness, pessimistic for energy). The truth lies between. CPU only.
-> data/bonus125.txt"""
import glob, os
import numpy as np
fs = sorted(glob.glob("/Users/kamen/Breakthrough/data/expA/traces125/*_800_g.npz"))
NL, K = 48, 10
hard = {"bulgarian", "chat"}
out = []


def P(s):
    print(s, flush=True); out.append(s)


def law(name, e):                                        # Chat1000's fitted laws (switchcap_hard125.py)
    return 0.0045 + 29.9 * e ** 1.5 if name in hard else -0.0003 + 6.49 * e ** 1.5


P("text mode delta | switch rate | U(16) U(32) U(64) | error energy E | law KLD")
for f in fs:
    name = os.path.basename(f).split("_")[0]
    d = np.load(f)
    ids = np.stack([d[f"L{l}"] for l in range(NL)], 1).astype(np.int64)           # [T, NL, 20], by weight
    g = np.stack([d[f"G{l}"] for l in range(NL)], 1).astype(np.float64)
    w = g / g[:, :, :K].sum(-1, keepdims=True)
    T = ids.shape[0]
    for mode, delta in [("drop", x) for x in (0.0, 0.25, 0.5, 1.0, 2.0)] + [("w20", x) for x in (0.25, 0.5, 1.0, 2.0, 3.0)]:
        mult = np.exp(delta)
        sets = np.zeros((T, NL, K), np.int64); sw = 0; E = 0.0
        for l in range(NL):
            S = ids[0, l, :K].copy()
            for t in range(T):
                cand, wt = ids[t, l], w[t, l]
                if t > 0:
                    out_ = S[~np.isin(S, cand)] if mode == "w20" else S[:0]       # old members below t's top-20
                    c = np.concatenate([cand, out_])
                    cw = np.concatenate([wt, np.full(len(out_), wt[-1])])
                    score = cw * np.where(np.isin(c, S), mult, 1.0)
                    new = c[np.argsort(-score, kind="stable")[:K]]
                    sw += K - int(np.isin(new, S).sum())
                    S = new
                sel = np.isin(cand, S)
                n_out = K - int(sel.sum())                                         # selected below the top-20
                E += float((wt[:K][~sel[:K]] ** 2).sum() + (wt[K:][sel[K:]] ** 2).sum() + n_out * wt[-1] ** 2)
                sets[t, l] = S

        def U(B):
            return np.mean([len(np.unique(sets[b * B:(b + 1) * B, l])) for b in range(T // B) for l in range(NL)])
        e = E / (T * NL)
        P(f"{name:9s} {mode:4s} {delta:4.2f} | {sw / ((T - 1) * NL * K):.3f} | {U(16):5.1f} {U(32):5.1f} {U(64):5.1f} | {e:.4f} | "
          f"{law(name, e):.3f}")
open(os.path.expanduser("~/TrainLab/data/bonus125.txt"), "w").write("\n".join(out) + "\n")
