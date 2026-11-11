"""A1 addendum: how much of the greedy cap is ties at the numeric-noise scale? Accept the drafter's top-1 if the target's
log-prob of it is within delta nats of the target's top-1 (non-exact). Gap scale reference = the numeric twin's flips."""
import sys, numpy as np
sys.path.insert(0, __import__("os").path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfb/Leap_scripts"))
from a1_ceiling import tau

def gaps(fp, fq):
    P, Q = np.load(fp), np.load(fq)
    kp = P["seq"].astype(np.int64) * 10**7 + P["pos"]; kq = Q["seq"].astype(np.int64) * 10**7 + Q["pos"]
    _, ip, iq = np.intersect1d(kp, kq, return_indices=True)
    idp, lpp = P["ids"][ip][:, :32], P["lp"][ip][:, :32].astype(np.float32)
    dq = Q["ids"][iq][:, 0]
    hit = idp == dq[:, None]
    lp_d = np.where(hit.any(1), (lpp * hit).sum(1), -99.0)      # target log-prob of the drafter's token (-99 if outside top-32)
    g = lpp[:, 0] - lp_d                                       # 0 when the drafter's token is the target's top-1
    cls = np.array([str(s) for s in P["sets"]])[P["seq"][ip]]
    return g, cls, P["seq"][ip], P["pos"][ip]

def report(name, fp, fq, deltas=(0.0, 0.01, 0.03, 0.1, 0.3)):
    g, cls, seq, pos = gaps(fp, fq)
    print(f"== {name}")
    for c in ("tf_ifeval", "tf_multi", "tf_code", "tf_gsm"):
        m = cls == c
        miss = m & (g > 0)
        q = np.percentile(g[miss], [25, 50, 75, 90]) if miss.any() else [np.nan] * 4
        row = []
        for d in deltas:
            a = (g <= d).astype(float)
            tot_n, tot_pass = 0, 0.0
            for s in np.unique(seq[m]):
                ms = np.where(seq == s)[0]; ms = ms[np.argsort(pos[ms])]
                t = tau(a[ms], 15); tot_n += len(ms); tot_pass += len(ms) / t
            row.append(f"d{d:<4} {a[m].mean():.4f}/{tot_n / tot_pass:5.2f}")
        print(f"  {c:9s} miss gaps p25/50/75/90 {q[0]:.3f} {q[1]:.3f} {q[2]:.3f} {q[3]:.3f} | per-token/tau16: " + "  ".join(row))

E = __import__("os").path.expanduser("~/EvalSuite/")
report("numeric twin (step vs prefill path)", E + "cache/ref/tf_ref.npz", E + "results/ref/tfstep.npz")
report("near-copy 10%", E + "results/ref/tfstep.npz", E + "results/w2f_r10/tfstep.npz")
report("AutoRound int4", E + "cache/ref/tf_ref.npz", E + "results/ar4/tf.npz")
