"""A1 low-temperature sweep: lossless speculative sampling at T ~ the numeric-noise scale (target p_T, drafter q_T')."""
import sys, os, numpy as np
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p4s/Leap_scripts"))
from a1_ceiling import overlap, tau

def soft(lp, T, k=20):
    x = lp[:, :k].astype(np.float32) / T
    x -= x[:, :1]
    p = np.exp(x); return p / p.sum(1, keepdims=True)

def run(name, fp, fq, Ts=(0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0), classes=("tf_ifeval", "tf_multi", "tf_code", "tf_gsm")):
    P, Q = np.load(fp), np.load(fq)
    kp = P["seq"].astype(np.int64) * 10**7 + P["pos"]; kq = Q["seq"].astype(np.int64) * 10**7 + Q["pos"]
    _, ip, iq = np.intersect1d(kp, kq, return_indices=True)
    idp, lpp, idq, lpq = P["ids"][ip][:, :32], P["lp"][ip][:, :32], Q["ids"][iq][:, :32], Q["lp"][iq][:, :32]
    seq, pos = P["seq"][ip], P["pos"][ip]
    cls = np.array([str(s) for s in P["sets"]])[seq]
    print(f"== {name}")
    rows = {"greedy": (idq[:, 0] == idp[:, 0]).astype(float)}
    for T in Ts:
        rows[f"T{T}"] = overlap(idp[:, :20], soft(lpp, T), idq[:, :20], soft(lpq, T))
    for c in classes:
        m = cls == c
        out = []
        for k, a in rows.items():
            tn, tp = 0, 0.0
            for s in np.unique(seq[m]):
                ms = np.where(seq == s)[0]; ms = ms[np.argsort(pos[ms])]
                tn += len(ms); tp += len(ms) / tau(a[ms], 15)
            out.append(f"{k} {a[m].mean():.4f}/{tn / tp:5.2f}")
        print(f"  {c:9s} " + "  ".join(out))

E = os.path.expanduser("~/EvalSuite/")
run("numeric twin", E + "cache/ref/tf_ref.npz", E + "results/ref/tfstep.npz")
run("AutoRound int4 (first 40k pos per class enough)", E + "cache/ref/tf_ref.npz", E + "results/ar4/tf.npz")
run("near-copy 10%", E + "results/ref/tfstep.npz", E + "results/w2f_r10/tfstep.npz")
run("near-copy 5%", E + "results/ref/tfstep.npz", E + "results/w2f_r05/tfstep.npz")
