"""TRELLIS FAST STORE x DRAFT17 price table (NewIdeas, Wed 4 Nov 2026 14:1x; CPU only; extends trellis_store.py, whose
U1 / U2 verdict stands). C's rule (QOM cf 0 + gated last row) with every last row gated ('all' = a drafted bonus row), so
no row is exact: all rows' top-2 load through one pool of the store's base records (Cl 0, equal RAM per layer). Lines:
24 GB (147 x 2,162,688 + 46 x 770,048 B per layer) and 12 GB (51 / 32). Stores: MSB-3 11-bit (packed 2,064,384 B, the
reference), TRL-3 3.07 b, gate/up TRL-2 + down TRL-3 2.40 b, TRL-2 2.06 b (W = 3 x 2560 x 640, 16 KB-aligned parts).
Keep = pool_quality's LRU kept share (MASS) of sure-row rank 2-9 pairs (the 'engine' rule's pool; a proxy).
Writes ~/NewIdeas/data/ledger/trellis_draft17.txt. Usage (BigMoE venv): python trellis_draft17.py"""
import os, sys, time
import multiprocessing as mp
from collections import OrderedDict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ten_x_ledger as T  # noqa: E402
import ledger_lastrow as LL  # noqa: E402
import pool_quality as PQ  # noqa: E402
BUDGET = {"24 GB": 147 * 2162688 + 46 * 770048, "12 GB": 51 * 2162688 + 32 * 770048}
STORES = {"MSB-3 11-bit": 2064384, "TRL-3 3.07 b": 1900544, "TRL-2/3 2.40 b": 1474560, "TRL-2 2.06 b": 1277952}


def replay(task):
    dom, line, rec = task
    C = BUDGET[line] // rec
    bp = [OrderedDict() for _ in range(48)]
    mb = kept = 0
    for w in T.W[dom]:
        ids = w["ids"]
        for p in range(w["a"]):
            for l in range(48):
                for e in ids[p, l].tolist(): T.touch(bp[l], e, C)
        for rows, cfs, kp in LL.blocks_conv(w, "all"):
            for l in range(48):
                U = set()
                for r, c in zip(rows, cfs):
                    U.update(ids[r, l, :2 if c >= 0.0 else 10].tolist())
                mb += sum(1 for e in U if e not in bp[l])
                for e in U: T.touch(bp[l], e, C)
            kept += kp
    return task, (C, mb * rec / kept / 1e6)


def keep(task):
    dom, c = task
    n, mem, res, wn, wmem, wres, ld, ms = PQ.run((dom, "lru", str(c)))[1]
    return task, 100 * (wmem + wres) / wn


def job(t):
    return replay(t[1:]) if t[0] == "b" else keep(t[1:])


def main():
    t0 = time.time(); T.W = T.windows()
    slots = sorted({BUDGET[ln] // r for ln in BUDGET for r in STORES.values()})
    tasks = ([("b", d, ln, r) for d in T.W for ln in BUDGET for r in STORES.values()]
             + [("k", d, c) for d in T.W for c in slots + [147, "m"]])
    with mp.get_context("fork").Pool(int(os.environ.get("PQ_PROCS", "3"))) as P:
        R = dict(P.map(job, tasks, chunksize=1))
    L = ["TRELLIS x DRAFT17: C's rule with every last row gated (no exact rows), SSD MB per kept token, the line's pool RAM "
         "spent on each store's base records (Cl 0); keep = LRU kept share (MASS) of sure-row rank 2-9 pairs at those slots",
         "| line | store | slots | code | IFEval | multi | keep code / IFEval / multi | keep gap closed (vs 147 -> 45 GB pool) |",
         "|---|---|---|---|---|---|---|---|"]
    for ln in BUDGET:
        for nm, r in STORES.items():
            C = BUDGET[ln] // r
            x = {d: R[(d, ln, r)][1] for d in T.W}
            k = {d: R[(d, C)] for d in T.W}
            g = {d: 100 * (k[d] - R[(d, 147)]) / (R[(d, "m")] - R[(d, 147)]) for d in T.W}
            L.append(f"| {ln} | {nm} | {C} | {x['code']:.1f} | {x['ifeval']:.1f} | {x['multi']:.1f} | "
                     f"{k['code']:.1f} / {k['ifeval']:.1f} / {k['multi']:.1f} % | {g['code']:+.0f} / {g['ifeval']:+.0f} / {g['multi']:+.0f} % |")
    L.append(f"({time.time() - t0:.0f} s; cross-checks: 24 GB MSB-3 11-bit code = draft17.txt's 171 / 0 'all' 19.5 minus its LSB "
             f"reads (none here); 24 GB TRL-2 / 2.40 b code = draft17.txt's 4.0 / 7.2)")
    os.makedirs(T.DST, exist_ok=True)
    open(os.path.join(T.DST, "trellis_draft17.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
