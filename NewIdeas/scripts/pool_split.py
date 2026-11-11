"""POOLSPLIT (NewIdeas, Wed 4 Nov 2026 13:25; pre-registered QUEUE 13:25; CPU only, no GPU).
Under C's FAST stack (QOM cf 0 + 3-bit sure rows + the gated last row, engine convention) only unsure rows (last rows
with no next draft) read LSB parts, so the LSB pool's RAM may be worth more as base slots. At equal bytes per layer
(the default split's Cm * MR + Cl * LR), sweep Cl and give the rest to Cm; replay C's reads with ledger_top1's loop
(plain LRU per layer, prompt warm-up, misses counted before each block). 11-bit lines use the packed store's part sizes
2,064,384 / 720,896 B (M-27b; CORRECTED 13:4x: the 13:28 run shrank both records to x0.824, MoonShots' net -17.6 % at the
45 GB pools, which already counts the extra parts per GB: double counting; that output is pool_split_1328_wrong11bit.txt).
Keep = pool_quality's LRU kept share (MASS-weighted) of sure-row rank 2-9 pairs at each split's Cm (24 GB lines).
Writes ~/NewIdeas/data/ledger/pool_split.txt. Usage (BigMoE venv): python pool_split.py
"""
import os, sys, time
import multiprocessing as mp
from collections import OrderedDict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ten_x_ledger as T  # noqa: E402
import ledger_lastrow as LL  # noqa: E402
import pool_quality as PQ  # noqa: E402
P3 = (2162688, 770048)                                         # records3 MSB / LSB part bytes (= T.MR, T.LR)
P11 = (2064384, 720896)                                        # M-27b PACKED11 part bytes (packed45_256.json)
BUDGETS = {"24 GB": (147, 46, P3), "24 GB 11-bit": (147, 46, P11), "12 GB": (51, 32, P3), "12 GB 11-bit": (51, 32, P11)}
CLS = [0, 4, 8, 12, 16, 24, 32, 40, 46]


def split(budget, cl):
    cm0, cl0, (mr, lr) = BUDGETS[budget]
    B = cm0 * T.MR + cl0 * T.LR                                  # bytes per layer of today's split, unchanged by 11-bit
    if (mr, lr) == P3 and cl == cl0: return cm0, cl0
    return int((B - cl * lr) // mr), cl


def replay(task):                                                # ledger_top1.replay for C's rule (top-2, var None)
    dom, Cm, Cl = task
    bp = [OrderedDict() for _ in range(48)]; lp = [OrderedDict() for _ in range(48)]
    mb = ml = kept = 0
    for w in T.W[dom]:
        ids = w["ids"]
        for p in range(w["a"]):
            for l in range(48):
                for e in ids[p, l].tolist(): T.touch(bp[l], e, Cm)
                for e in ids[p, l, :2].tolist(): T.touch(lp[l], e, Cl)
        for rows, cfs, kp in LL.blocks_conv(w, "engine"):
            sure = [c >= 0.0 for c in cfs]
            for l in range(48):
                Ub, Ul = set(), set()
                for r, s_ in zip(rows, sure):
                    Ub.update(ids[r, l, :2 if s_ else 10].tolist())
                    if not s_: Ul.update(ids[r, l, :2].tolist())
                mb += sum(1 for e in Ub if e not in bp[l])
                for e in Ub: T.touch(bp[l], e, Cm)
                ml += sum(1 for e in Ul if e not in lp[l])
                for e in Ul: T.touch(lp[l], e, Cl)
            kept += kp
    return task, (mb, ml, kept)


def keep(task):
    dom, cm = task
    n, mem, res, wn, wmem, wres, ld, ms = PQ.run((dom, "lru", cm))[1]
    return task, 100 * (wmem + wres) / wn


def job(t):
    return replay(t[1:]) if t[0] == "b" else keep(t[1:])


def main():
    t0 = time.time(); T.W = T.windows()
    cells = {(b, cl): split(b, cl) for b in BUDGETS for cl in CLS if cl <= BUDGETS[b][1] or cl == 0}
    for b in BUDGETS:                                            # the 11-bit default split at equal RAM per pool
        cm0, cl0, (mr, lr) = BUDGETS[b]
        if (mr, lr) != P3: cells[(b, "dflt")] = (int(cm0 * T.MR // mr), int(cl0 * T.LR // lr))   # 154 / 49, 53 / 34
    runs = sorted({(d,) + v for d in T.W for v in cells.values()})
    kms = sorted({str(v[0]) for (b, _), v in cells.items() if b.startswith("24")} | {"147", "m"})
    tasks = [("b",) + r for r in runs] + [("k", d, c) for d in T.W for c in kms]
    with mp.get_context("fork").Pool(int(os.environ.get("PQ_PROCS", "3"))) as P:
        R = dict(P.map(job, tasks, chunksize=1))
    L = ["POOLSPLIT: C's stack (QOM cf 0 + 3-bit sure rows + gated last row, engine convention), SSD MB per kept token at "
         "equal bytes per layer, LSB slots (Cl) traded for base slots (Cm); keep = LRU kept share (MASS) of sure-row rank 2-9 "
         "pairs; gap = share of the keep gap 147 slots -> the 45 GB pool",
         "| budget | domain | Cm / Cl | MB per kept (x default) | of it LSB | keep | gap closed |", "|---|---|---|---|---|---|---|"]
    best = {}
    for b in BUDGETS:
        mr, lr = BUDGETS[b][2]
        for d in T.W:
            dkey = (b, "dflt") if (b, "dflt") in cells else (b, BUDGETS[b][1])
            def mbk(v):
                mb, ml, kp = R[(d,) + v]; return (mb * mr + ml * lr) / max(kp, 1) / 1e6, ml * lr / max(kp, 1) / 1e6
            ref, _ = mbk(cells[dkey])
            k147, km = R[(d, "147")], R[(d, "m")]
            rows = sorted({v for (bb, _), v in cells.items() if bb == b}, key=lambda v: -v[1])
            for v in rows:
                x, lsb = mbk(v)
                kk = R.get((d, str(v[0]))) if b.startswith("24") else None
                ks = f"{kk:.1f} %" if kk is not None else "-"
                gs = f"{100 * (kk - k147) / (km - k147):+.0f} %" if kk is not None else "-"
                tag = " (default)" if v == cells[dkey] else ""
                L.append(f"| {b} | {d} | {v[0]} / {v[1]}{tag} | {x:.1f} (x{x / ref:.3f}) | {lsb:.1f} | {ks} | {gs} |")
                if d not in best.get(b, {}) or x < best[b][d][1]: best.setdefault(b, {})[d] = (v, x, x / ref)
    L.append("BEST SPLIT by bytes: " + "; ".join(f"{b} {d} {v[0]}/{v[1]} {x:.1f} (x{r:.3f})" for b in best
                                               for d, (v, x, r) in best[b].items()))
    L.append(f"({time.time() - t0:.0f} s; self-check: defaults 24 GB 35.7 / 19.4 / 13.9, 12 GB 83.1 / 72.6 / 52.9; 11-bit "
             f"defaults (154 / 49, 53 / 34) must equal capacity_11bit.txt's corrected 11-bit columns)")
    os.makedirs(T.DST, exist_ok=True)
    open(os.path.join(T.DST, "pool_split.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
