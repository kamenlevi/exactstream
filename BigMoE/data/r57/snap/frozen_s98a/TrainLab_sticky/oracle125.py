"""TrainLab (10-22): a LOWER BOUND on the error energy of ANY expert-selection policy (causal or not, trained or not)
whose union per 16-token block is U, on the 125B traces (Chat1000's energy: sum over each token's true top-10 experts left
out of its set of w^2, w = top-10-normalised gate weights). Ignoring the cost of fill-in experts (counted 0, as Chat1000
counts members below the top-20), the best block set is exact: the U experts with the largest sum_t w_te^2 over tokens
whose top-10 holds e. Compared with the hard caps at equal U(16). CPU only. -> data/oracle125.txt"""
import glob, os
import numpy as np
fs = sorted(glob.glob("/Users/kamen/Breakthrough/data/expA/traces125/*_800_g.npz"))
NL, K, B = 48, 10, 16
cap = {}
for l in open('/Users/kamen/Chat1000/data/switchcap_hard125.txt').read().splitlines()[1:]:
    p = l.split('|'); n, m = p[0].split(); cap.setdefault(n, []).append((int(m), float(p[2].split()[0]), float(p[3])))
out = []
for f in fs:
    name = os.path.basename(f).split("_")[0]
    d = np.load(f)
    ids = np.stack([d[f"L{l}"] for l in range(NL)], 1).astype(np.int64)[:, :, :K]
    g = np.stack([d[f"G{l}"] for l in range(NL)], 1).astype(np.float64)[:, :, :K]
    w2 = (g / g.sum(-1, keepdims=True)) ** 2
    T = ids.shape[0] // B * B
    res = {}
    for U in (10, 16, 22, 32, 40, 49, 57, 64, 73):
        E = 0.0
        for b in range(0, T, B):
            for l in range(NL):
                c = {}
                for t in range(b, b + B):
                    for e, x in zip(ids[t, l], w2[t, l]):
                        c[e] = c.get(e, 0.0) + x
                v = np.sort(np.fromiter(c.values(), float))[::-1]
                E += v[U:].sum()
        res[U] = E / (T * NL)
    for m, u, e in cap[name]:
        Us = sorted(res)
        lo = max(x for x in Us if x <= u); hi = min(x for x in Us if x >= u)
        eo = res[lo] if hi == lo else np.exp(np.log(res[lo]) + (u - lo) / (hi - lo) * (np.log(res[hi]) - np.log(res[lo])))
        s = f"{name:9s} cap m={m}: U(16) {u:5.1f} E {e:.4f} | block oracle at that union E >= {eo:.4f} -> cap / oracle {e / eo:.1f}x"
        print(s, flush=True); out.append(s)
    s = f"{name:9s} oracle E by union: " + " ".join(f"U{U} {res[U]:.4f}" for U in sorted(res))
    print(s, flush=True); out.append(s)
open(os.path.expanduser("~/TrainLab/data/oracle125.txt"), "w").write("\n".join(out) + "\n")
