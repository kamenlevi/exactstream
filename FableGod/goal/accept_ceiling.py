"""ACCEPT-CEILING (GOAL, Thu 5 Nov 2026): what could ANY lossy acceptance rule keep on code?
CPU only, numpy, python -I. Join of
  ~/TrainLab/data/d02b_walk_d23h_dfl_<dom>.npz   DFlash2 L1-20 drafts at every position (dtok / dcf own chain, ttok teacher-forced)
  ~/Breakthrough/data/rulec/rulec_<set>.json     the 125B's top-8 tokens + log-probs at every answer position (its own path)
At each draft miss: the 125B's p1 at that position, the draft token's rank and logit gap. Then the official loop
(0.3 cut) with "profile" acceptance: a miss the rule forgives is treated as a hit and the walk goes on along the
teacher-forced picks (M-41's convention; an upper bound, the real 125B is then conditioned on the deviation).
"""
import json
import math
import numpy as np

D = "/Users/kamen/TrainLab/data/"
R = "/Users/kamen/Breakthrough/data/rulec/"
SETS = [("code", "rulec_he12.json", "HumanEval+ (12 tasks)"), ("mbpp", "rulec_mbpp.json", "MBPP")]
CUT = 0.3
INF = 1e9


def rows_cut(cf, n, t=CUT):
    run = np.cumprod(cf[:n].astype(np.float64))
    bad = np.nonzero(run < t)[0]
    return max(int(bad[0]) if len(bad) else n, 1)


def main():
    for dom, rf, label in SETS:
        z = np.load(D + f"d02b_walk_d23h_dfl_{dom}.npz", allow_pickle=False)
        rc = json.load(open(R + rf))
        names = [str(x) for x in z["names"]]
        S = []
        agree = tot = 0
        for s, nm in enumerate(names):
            if nm not in rc:
                continue
            r = rc[nm]
            ids = z[f"{s}_ids"]; ch = z[f"{s}_checks"]
            a0 = int(r["a"])
            tk = r["tk"]; lp = r["lp"]
            # tk[i] = the 125B's top-8 for position a0 + 1 + i
            n = min(len(tk), len(ids) - a0 - 1)
            agree += sum(int(tk[i][0] == ids[a0 + 1 + i]) for i in range(n)); tot += n
            S.append((ids, ch, z[f"{s}_dtok"], z[f"{s}_dcf"], z[f"{s}_ttok"], a0, tk, lp, n))
        print(f"\n=== {label}: {len(S)} joined samples; the 125B's top-1 == the walk's token at {agree}/{tot} positions")

        def info(smp, pos, tok):
            """(p1, rank of tok in the 125B's top-8 or 99, logit gap to the top or INF)"""
            ids, ch, dt, cf, tt, a0, tk, lp, n = smp
            i = pos - a0 - 1
            if i < 0 or i >= n:
                return None
            p1 = math.exp(lp[i][0])
            row = tk[i]
            if tok in row:
                j = row.index(tok)
                return p1, j + 1, lp[i][0] - lp[i][j]
            return p1, 99, INF

        RULES = {
            "exact": lambda p1, rk, gap: False,
            "R1(0.5)": lambda p1, rk, gap: gap <= 0.5,
            "R1(1)": lambda p1, rk, gap: gap <= 1.0,
            "R1(2)": lambda p1, rk, gap: gap <= 2.0,
            "R1(3)": lambda p1, rk, gap: gap <= 3.0,
            "R2 (rank 2, p >= .25 p1)": lambda p1, rk, gap: rk == 2 and gap <= math.log(4.0),
            "any of the 125B's top-8": lambda p1, rk, gap: rk <= 8,
            "every miss where p1 < 0.9": lambda p1, rk, gap: p1 < 0.9,
            "every miss where p1 < 0.99": lambda p1, rk, gap: p1 < 0.99,
            "every miss where p1 >= 0.99 (a drafter with no sure-token errors)": lambda p1, rk, gap: p1 >= 0.99,
        }

        # ---- anatomy of the first misses of the official exact loop
        band = {}; n_miss = 0; rk2 = 0; in8 = 0; r1 = 0
        for smp in S:
            ids, ch, dt, cf, tt, a0, tk, lp, n = smp
            idx = {int(a): i for i, (k, m, a) in enumerate(ch)}
            a = max(int(ch[:, 2].min()), a0)
            while a in idx:
                i = idx[a]; m = int(ch[i, 1])
                if m <= 0:
                    break
                d = dt[i]; true = ids[a + 1:a + 1 + m]
                Rw = rows_cut(cf[i], m)
                miss = np.nonzero(d[:Rw] != true[:Rw])[0]
                k = int(miss[0]) if len(miss) else Rw
                if len(miss):
                    x = info(smp, a + 1 + k, int(d[k]))
                    if x is not None:
                        p1, rk, gap = x
                        b = ">=0.99" if p1 >= 0.99 else "0.9-0.99" if p1 >= 0.9 else "0.5-0.9" if p1 >= 0.5 else "<0.5"
                        y = band.setdefault(b, [0, 0, 0, 0])
                        y[0] += 1; y[1] += int(rk <= 8); y[2] += int(gap <= 0.5); y[3] += int(rk == 2)
                        n_miss += 1
                a += k + 1
        print(f"  first misses of the exact loop (0.3 cut): {n_miss}")
        for b in (">=0.99", "0.9-0.99", "0.5-0.9", "<0.5"):
            y = band.get(b, [0, 0, 0, 0])
            if y[0]:
                print(f"    the 125B's p1 {b:9s}: {100 * y[0] / n_miss:5.1f} % of first misses; draft token in its top-8 {100 * y[1] / y[0]:5.1f} %, "
                      f"its rank 2 {100 * y[3] / y[0]:5.1f} %, within 0.5 logits {100 * y[2] / y[0]:5.1f} %")

        # ---- the official loop under each rule (profile: forgiven misses count as hits, walk on teacher-forced picks)
        base = None
        for name, rule in RULES.items():
            kept = rows = checks = dev = 0
            for smp in S:
                ids, ch, dt, cf, tt, a0, tk, lp, n = smp
                idx = {int(a): i for i, (k, m, a) in enumerate(ch)}
                a = max(int(ch[:, 2].min()), a0)
                while a in idx:
                    i = idx[a]; m = int(ch[i, 1])
                    if m <= 0:
                        break
                    true = ids[a + 1:a + 1 + m]
                    Rw = rows_cut(cf[i], m)
                    t = tt[i]
                    k = Rw
                    for j in range(Rw):
                        if t[j] != true[j]:
                            x = info(smp, a + 1 + j, int(t[j]))
                            if x is not None and rule(*x):
                                dev += 1
                                continue
                            k = j
                            break
                    kept += k + 1; rows += Rw + 1; checks += 1
                    a += k + 1
            tps = kept / (checks * 62.0 + rows * 2.8) * 1000
            if base is None:
                base = (kept / checks, tps)
            print(f"  {name:66s} kept {kept / checks:6.2f} (x{kept / checks / base[0]:.3f})  tok/s x{tps / base[1]:.3f}  forgiven {100 * dev / kept:5.2f} % of kept tokens")


def study2():
    """Longer checks: (a) a lower confidence cut inside the 15-draft block; (b) B chained blocks per check, block b+1 drafted
    from the anchor at the end of block b (the stored every-position draft there: an IDEAL re-anchoring drafter, an upper bound).
    Running confidence carried across blocks. Cost: F + r * rows + (B_drafted - 1) * d."""
    for dom, rf, label in SETS:
        z = np.load(D + f"d02b_walk_d23h_dfl_{dom}.npz", allow_pickle=False)
        rc = json.load(open(R + rf))
        names = [str(x) for x in z["names"]]
        S = []
        for s, nm in enumerate(names):
            if nm not in rc:
                continue
            r = rc[nm]
            ids = z[f"{s}_ids"]; a0 = int(r["a"])
            n = min(len(r["tk"]), len(ids) - a0 - 1)
            S.append((ids, z[f"{s}_checks"], z[f"{s}_dtok"], z[f"{s}_dcf"], z[f"{s}_ttok"], a0, r["tk"], r["lp"], n))

        def info(smp, pos, tok):
            ids, ch, dt, cf, tt, a0, tk, lp, n = smp
            i = pos - a0 - 1
            if i < 0 or i >= n:
                return None
            p1 = math.exp(lp[i][0]); row = tk[i]
            if tok in row:
                j = row.index(tok)
                return p1, j + 1, lp[i][0] - lp[i][j]
            return p1, 99, INF

        RULES = {
            "exact": lambda p1, rk, gap: False,
            "R1(0.5)": lambda p1, rk, gap: gap <= 0.5,
            "R1(2)": lambda p1, rk, gap: gap <= 2.0,
            "forgive p1 < 0.9": lambda p1, rk, gap: p1 < 0.9,
            "forgive p1 < 0.99": lambda p1, rk, gap: p1 < 0.99,
            "forgive every miss in the 125B's top-8": lambda p1, rk, gap: rk <= 8,
        }

        def run(rule, t, B):
            kept = rows = checks = dev = blocks = 0
            for smp in S:
                ids, ch, dt, cf, tt, a0, tk, lp, n = smp
                idx = {int(a): i for i, (k, m, a) in enumerate(ch)}
                a = max(int(ch[:, 2].min()), a0)
                while a in idx:
                    ktot = 0; rtot = 1; conf = 1.0; nb = 0; stop = False
                    for b in range(B):
                        i = idx.get(a + 15 * b)
                        if i is None:
                            break
                        m = int(ch[i, 1])
                        if m <= 0:
                            break
                        nb += 1
                        true = ids[a + 15 * b + 1:a + 15 * b + 1 + m]
                        c = cf[i]; tkk = tt[i]
                        k = 0
                        for j in range(m):
                            conf *= float(c[j])
                            if conf < t and (b > 0 or j > 0):
                                stop = True
                                break
                            rtot += 1
                            if tkk[j] != true[j]:
                                x = info(smp, a + 15 * b + 1 + j, int(tkk[j]))
                                if x is not None and rule(*x):
                                    dev += 1
                                else:
                                    stop = True
                                    break
                            k += 1
                        ktot += k
                        if stop or m < 15:
                            break
                    if nb == 0:
                        break
                    kept += ktot + 1; rows += rtot; checks += 1; blocks += B
                    a += ktot + 1
            return kept, rows, checks, dev, blocks

        print(f"\n=== longer checks, {label}: kept per check / rows per check / tok/s vs today's (cut 0.3, 1 block, exact) under two cost models")
        base = run(RULES["exact"], 0.3, 1)
        def tps(x, F, r, d, B):
            kept, rows, checks, dev, blocks = x
            return kept / (checks * (F + d * (B - 1)) + rows * r) * 1000
        b1 = tps(base, 62.0, 2.8, 5.8, 1); b2 = tps(base, 30.0, 1.5, 3.0, 1)
        print(f"  base: kept {base[0] / base[2]:.2f}, rows {base[1] / base[2]:.2f}; {b1:.0f} tok/s at 62 + 2.8/row, {b2:.0f} at 30 + 1.5/row")
        for name, rule in RULES.items():
            for (t, B) in ((0.3, 1), (0.1, 1), (0.0, 1), (0.3, 2), (0.1, 2), (0.03, 2), (0.1, 4), (0.03, 4), (0.0, 4)):
                x = run(rule, t, B)
                print(f"  {name:38s} cut {t:<4} blocks {B}: kept {x[0] / x[2]:6.2f} rows {x[1] / x[2]:6.2f}  "
                      f"tok/s x{tps(x, 62.0, 2.8, 5.8, B) / b1:.3f} (today) x{tps(x, 30.0, 1.5, 3.0, B) / b2:.3f} (floor)  forgiven {100 * x[3] / x[0]:.2f} %")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "2":
        study2()
    else:
        main()
