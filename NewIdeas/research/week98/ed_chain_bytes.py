"""ED-CHAIN upper bound (NewIdeas, Mon 16 Nov 16:2x). CPU only, read-only.
Chain early death with a PERFECT flag at layer L: rows at or past `kept` (the first miss's correction row is the last
needed row) are dropped from layers > L. Expert bytes per 32-row answer block on the 24 code windows of the decision map,
under Breakthrough's tier rule (g1g2_map.build: I = sum_d s^d w^2, 4-bit if I > 2.775 lam, MSB-3 if > 0.8457 lam, else
drop) at the real pieces (2.784 / 2.064 MB). kept ~ geometric in the per-draft acceptance p, capped at 32.
LINE (written before the run): if even a PERFECT flag at L 16 saves < 8 % of C6's expert bytes at p 0.95, the chain
version is dead whatever the probe; >= 15 % keeps it worth Wed's probe arm.
usage: taskpolicy -b ~/BigMoE/.venv/bin/python -I -B ed_chain_bytes.py"""
import os, sys, json
import numpy as np
O = os.path.expanduser("~/FableGod/decision_map/out_code24/scores")
OUT = os.path.expanduser("~/NewIdeas/research/week98/ed_chain_bytes")
NL, ROWS, S_ = 48, 32, 0.98
B4, B3 = 2.784, 2.064
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)

def block_bytes(ids, w, rows, lam):
    """MB per layer for these rows (a prefix of a block), tiers on the prefix only"""
    rw = S_ ** np.arange(len(rows))
    out = np.zeros(NL)
    for l in range(NL):
        u, inv = np.unique(ids[rows, l].ravel(), return_inverse=True)
        imp = np.bincount(inv, weights=(w[rows, l] ** 2 * rw[:, None]).ravel(), minlength=len(u))
        n4 = int((imp > 2.775 * lam).sum()); n3 = int(((imp > 0.8457 * lam) & ~(imp > 2.775 * lam)).sum())
        out[l] = n4 * B4 + n3 * B3
    return out

def main():
    files = sorted(f for f in os.listdir(O) if f.endswith(".npz"))
    res = {}
    for lam in (1.6e-2, 1.8e-2):
        per_k = []          # [blocks, 33, NL] MB per layer for prefix k (k = 1..32)
        for f in files:
            z = np.load(os.path.join(O, f)); a = int(z["a"])
            ids, w = z["ids"].astype(np.int64), z["scores"].astype(np.float64)
            n = ids.shape[0]
            for b0 in range(a, n - ROWS + 1, ROWS):
                M = np.zeros((ROWS + 1, NL))
                for k in range(1, ROWS + 1):
                    M[k] = block_bytes(ids, w, np.arange(b0, b0 + k), lam)
                per_k.append(M)
        P = np.array(per_k)                                  # [B, 33, NL]
        full = P[:, ROWS, :].sum(1).mean()                   # MB per block, all 48 layers
        say(f"lambda {lam:.1e}: {len(P)} full blocks; C6-rule expert MB per 32-row check {full:.0f} "
            f"(x48 layers; {full / 1000:.2f} GB)")
        r = {"full_MB": float(full)}
        for p in (0.90, 0.95, 0.978, 0.988):
            pk = np.array([p ** (k - 1) * (1 - p) for k in range(1, ROWS)] + [p ** (ROWS - 1)])   # P(kept = k)
            ekept = float((pk * np.arange(1, ROWS + 1)).sum())
            for L in (8, 16, 24):
                ed = (P[:, ROWS, :L + 1].sum(1)[:, None] + P[:, 1:, L + 1:].sum(2)) @ pk   # [B]
                sv = 1 - ed.mean() / full
                r[f"p{p}_L{L}"] = float(sv)
                if L == 16 or p == 0.95:
                    say(f"  p {p:.3f} (E kept {ekept:5.2f}), perfect flag after layer {L + 1:2d}: expert bytes "
                        f"-{100 * sv:4.1f} %  ({ed.mean() / 1000:.2f} GB)")
        res[f"{lam:.1e}"] = r
    s95 = res["1.8e-02"]["p0.95_L16"]
    v = "KILL (dead whatever the probe)" if s95 < 0.08 else ("KEEP (worth Wed's arm)" if s95 >= 0.15 else "BETWEEN")
    say(f"\nLINE (C6, p 0.95, perfect flag at layer 16: < 8 % dead, >= 15 % keep): -{100 * s95:.1f} % -> {v}")
    json.dump(res, open(OUT + ".json", "w"), indent=1)
    open(OUT + ".txt", "w").write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
