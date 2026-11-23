"""A1 ceiling screen (Leap, 2026-10-23): is the greedy drafter ceiling a greedy artifact?
Target p = the exact 125B, drafter q = a near-copy (or a numeric twin), both stored as sorted top-K log-probs at the
same teacher-forced positions (the 125B's own greedy continuations, Eval suite tf_gen). Per position:
  greedy acceptance  = 1[argmax q == argmax p]
  sampling acceptance = sum_x min(p'(x), q'(x)), p' / q' = temperature T, then top-k, then top-p (HF order), the
                        lossless speculative-sampling rule (Leviathan / Chen 2023), drafter sampled from its own q'.
tau(K) = tokens per verify pass with K drafts, blocks simulated along each sequence (expected value, DP), as Chat1000.
Usage: a1_ceiling.py PAIR...   PAIR = name:p_file:q_file (npz with seq,pos,ids,lp; aligned on (seq,pos))."""
import sys, numpy as np

CONFIGS = [("greedy", None), ("T1.0 k20 p0.95 (official thinking)", (1.0, 20, 0.95)),
           ("T0.7 k20 p0.80 (official instruct)", (0.7, 20, 0.80)), ("T1.0 full", (1.0, 32, 1.0)),
           ("T0.6 k20 p0.95", (0.6, 20, 0.95))]
KS = (3, 7, 15)


def load(path, keep):
    z = np.load(path)
    out = {k: z[k] for k in keep if k in z.files}
    return out, z["keys"], z["sets"]


def proc(lp, T, k, top_p):
    x = lp[:, :k].astype(np.float32) / T
    x -= x[:, :1]
    p = np.exp(x)
    if k >= lp.shape[1] and T == 1.0:     # full distribution: keep the true mass, tail outside top-K stays unassigned
        return p * np.exp(lp[:, :1].astype(np.float32))
    p /= p.sum(1, keepdims=True)
    if top_p < 1.0:
        cum = np.cumsum(p, 1)
        p = np.where(cum - p < top_p, p, 0.0)
        p /= p.sum(1, keepdims=True)
    return p


def overlap(ip, p, iq, q, chunk=20000):
    acc = np.empty(len(p), np.float64)
    for a in range(0, len(p), chunk):
        m = ip[a:a + chunk, :, None] == iq[a:a + chunk, None, :]
        qa = (m * q[a:a + chunk, None, :]).sum(2)          # q at p's ids (0 if absent from q's list)
        acc[a:a + chunk] = np.minimum(p[a:a + chunk], qa).sum(1)
    return acc


def tau(a, K):
    """expected tokens per pass with K drafts; a = per-position acceptance probabilities of one sequence"""
    n = len(a)
    E = np.zeros(n + K + 2)          # E[j] = expected passes to finish from block start j
    for j in range(n - 1, -1, -1):
        surv, e = 1.0, 0.0
        for L in range(0, K + 1):    # L drafts accepted, then the pass's own token
            if L < K and j + L < n:
                pr = surv * (1.0 - a[j + L])
                surv *= a[j + L]
            else:
                pr = surv
            nxt = j + L + 1
            e += pr * (E[nxt] if nxt < n else 0.0)
            if L >= K or j + L >= n:
                break
        E[j] = 1.0 + e
    return n / E[0]


def main():
    for spec in sys.argv[1:]:
        name, fp, fq = spec.split(":")
        P, keys, sets = load(fp, ("seq", "pos", "ids", "lp"))
        Q, _, _ = load(fq, ("seq", "pos", "ids", "lp"))
        kp = P["seq"].astype(np.int64) * 10**7 + P["pos"]
        kq = Q["seq"].astype(np.int64) * 10**7 + Q["pos"]
        common, ip_, iq_ = np.intersect1d(kp, kq, return_indices=True)
        idp, lpp = P["ids"][ip_][:, :32], P["lp"][ip_][:, :32]
        idq, lpq = Q["ids"][iq_][:, :32], Q["lp"][iq_][:, :32]
        seq = P["seq"][ip_]
        cls = np.array([str(s) for s in sets])[seq]
        p1 = np.exp(lpp[:, 0].astype(np.float32))
        print(f"== {name}: {len(common)} positions ({len(kp)} p, {len(kq)} q)")
        res = {}
        for cname, cfg in CONFIGS:
            if cfg is None:
                res[cname] = (idq[:, 0] == idp[:, 0]).astype(np.float64)
            else:
                T, k, tp = cfg
                res[cname] = overlap(idp[:, :k], proc(lpp, T, k, tp), idq[:, :k], proc(lpq, T, k, tp))
        for c in sorted(set(cls)):
            m = cls == c
            fork = m & (p1 < 0.7)
            print(f"  {c:10s} n {m.sum():6d}  forks(p1<0.7) {fork.sum() / m.sum():.3f}")
            for cname, _ in CONFIGS:
                a = res[cname]
                taus = []
                for K in KS:
                    tot_n, tot_pass = 0, 0.0
                    for s in np.unique(seq[m]):
                        ms = seq == s
                        order = np.argsort(P["pos"][ip_][ms])
                        t = tau(a[ms][order], K)
                        tot_n += ms.sum(); tot_pass += ms.sum() / t
                    taus.append(tot_n / tot_pass)
                print(f"    {cname:36s} per-token {a[m].mean():.4f}  forks {a[fork].mean() if fork.any() else float('nan'):.3f}"
                      f"  non-forks {a[m & ~fork].mean():.4f}  " + "  ".join(f"tau{K + 1} {t:5.2f}" for K, t in zip(KS, taus)))


if __name__ == "__main__":
    main()
