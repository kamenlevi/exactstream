"""Fit12 PRN (Mon 26 Oct 2026; pre-registered in ~/Fit12/PLAN.md 'Rapid screens, batch 1'): static expert pruning
(REAP-style, bytes side only). Per layer, the experts with the least routed gate mass on the fit sets (sum of the top-10
gate weights; expA traces keep each token's top-20 ids L<n> and weights G<n>) are removed; the judged sets are re-routed
(top-10 = the first 10 unpruned ids of each token's top-20) and replayed through fit_v3replay.Tier (engine-faithful):
12 GB = C_m 51 / C_l 20, LRU, checks of 4 rows accepting 3; 24 GB = C_m 147 / C_l 46, LFU 0.99, checks 16 / 10. Bytes per
miss as the engine reads (records3). Also logged: the share of the original top-10 gate mass held by pruned experts.
Usage: ~/BigMoE/.venv/bin/python -B /Users/kamen/Fit12/scripts/fit_prn.py   (CPU, 1 core)"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_v3replay as V

NL, T0, MSB, LSB = V.NL, V.T0, V.MSB, V.LSB
CFGS = (("12 GB", 51, 20, 0.0, 4, 3), ("24 GB", 147, 46, 0.99, 16, 10))
FRACS = (0.0, 0.125, 0.25, 0.375, 0.5)


def reroute(ids, keep):
    """each row's first 10 kept ids, in rank order (fewer if fewer than 10 of its 20 are kept)"""
    return [[e for e in r if keep[e]][:10] for r in ids.tolist()]


def replay(rows, C_m, C_l, d, R, A):
    m, l, bm, bl = V.Tier(C_m, d), V.Tier(C_l, d), 0, 0
    for t in range(0, len(rows), A):
        blk = rows[t:t + R]
        a = m.call(list(dict.fromkeys(e for r in blk for e in r)))
        b = l.call(list(dict.fromkeys(e for r in blk for e in r[:2])))
        if t >= T0:
            bm, bl = bm + a, bl + b
    return bm * MSB + bl * LSB


def main():
    t0 = time.time()
    z = {s: np.load(os.path.join(V.TR, f"{s}_800_g.npz")) for s in V.SETS}
    ids = {s: [z[s][f"L{L}"].astype(np.int64) for L in range(NL)] for s in V.SETS}
    gw = {s: [z[s][f"G{L}"].astype(np.float64) for L in range(NL)] for s in V.SETS}
    T = ids[V.SETS[0]][0].shape[0]
    print(f"[prn] {T} tokens per set, top-{ids[V.SETS[0]][0].shape[1]} kept per token; G0 row 0 sum of 20 "
          f"{gw[V.SETS[0]][0][0].sum():.3f}, of top-10 {gw[V.SETS[0]][0][0][:10].sum():.3f}", flush=True)
    for fit, test in ((("bulgarian", "chat"), ("code", "math")), (("code", "math"), ("bulgarian", "chat"))):
        mass = [np.zeros(512) for _ in range(NL)]
        for s in fit:
            for L in range(NL):
                np.add.at(mass[L], ids[s][L][:, :10].reshape(-1), gw[s][L][:, :10].reshape(-1))
        base = {}
        for f in FRACS:
            keeps = []
            for L in range(NL):
                k = np.ones(512, bool)
                k[np.argsort(mass[L], kind="stable")[:int(round(f * 512))]] = False
                keeps.append(k)
            lost, short = [], 0
            for s in test:
                for L in range(NL):
                    g = gw[s][L][T0:, :10]
                    lost.append((g * ~keeps[L][ids[s][L][T0:, :10]]).sum() / g.sum())
            out = []
            for name, C_m, C_l, d, R, A in CFGS:
                per = []
                for s in test:
                    b, tok = 0, sum(min(A, T - t) for t in range(0, T, A) if t >= T0)
                    for L in range(NL):
                        rows = reroute(ids[s][L], keeps[L])
                        short += sum(len(r) < 10 for r in rows[T0:]) if name == "12 GB" else 0
                        b += replay(rows, C_m, C_l, d, R, A)
                    per.append(b / tok / 1e6)
                mb = float(np.mean(per))
                base.setdefault(name, mb)
                out.append(f"{name} {mb:.1f} MB/token ({(mb / base[name] - 1) * 100:+.1f} %)")
            print(f"[prn] fit {'+'.join(fit)} -> {'+'.join(test)} | pruned {f * 100:.1f} % | " + " | ".join(out)
                  + f" | top-10 gate mass on pruned experts {np.mean(lost) * 100:.2f} % | rows short of 10: {short}"
                  f" ({time.time() - t0:.0f} s)", flush=True)
    print(f"[prn] done ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
