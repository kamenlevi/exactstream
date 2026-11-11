"""TRELLIS FAST STORE byte test (NewIdeas, Wed 4 Nov 2026 13:4x; pre-registered QUEUE 13:4x; CPU only, no GPU).
C's rule (QOM cf 0 + 3-bit-class sure rows + the gated last row, engine convention; ledger_top1's loop) with the 24 GB
pool RAM (147 x 2,162,688 + 46 x 770,048 B per layer) spent on base records of a smaller FAST store, Cl 0:
  MSB-3 11-bit 2,064,384 B (171 slots), TRL-3 3.07 b 1,900,544 B (185), gate/up TRL-2 + down TRL-3 2.40 b 1,474,560 B (239),
  TRL-2 2.06 b 1,277,952 B (276); W = 3 x 2560 x 640 weights per expert, 16 KB-aligned parts.
Unsure rows (cf < 0: last rows with no next draft):
  U1  served from the same pool (all 10 ranks, no exact rows; optimistic)
  U2  read from the v3 store on every use: 10 MSB parts (2,064,384 B) + 2 LSB parts (720,896 B), no pool (conservative)
Keep = pool_quality's LRU kept share (MASS) of sure-row rank 2-9 pairs at each slot count.
Writes ~/NewIdeas/data/ledger/trellis_store.txt. Usage (BigMoE venv): python trellis_store.py
"""
import os, sys, time
import multiprocessing as mp
from collections import OrderedDict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ten_x_ledger as T  # noqa: E402
import ledger_lastrow as LL  # noqa: E402
import pool_quality as PQ  # noqa: E402
B = 147 * 2162688 + 46 * 770048                                # 24 GB pool bytes per layer
MR11, LR11 = 2064384, 720896
STORES = {"MSB-3 11-bit": 2064384, "TRL-3 3.07 b": 1900544, "TRL-2/3 2.40 b": 1474560, "TRL-2 2.06 b": 1277952}


def replay(task):
    dom, rec, mode = task
    C = B // rec
    bp = [OrderedDict() for _ in range(48)]
    mb = vb = kept = 0                                         # pool misses (records), v3 bytes (unsure rows, U2)
    for w in T.W[dom]:
        ids = w["ids"]
        for p in range(w["a"]):
            for l in range(48):
                for e in ids[p, l].tolist(): T.touch(bp[l], e, C)
        for rows, cfs, kp in LL.blocks_conv(w, "engine"):
            sure = [c >= 0.0 for c in cfs]
            for l in range(48):
                Ub, Uv, Uv2 = set(), set(), set()
                for r, s_ in zip(rows, sure):
                    if s_: Ub.update(ids[r, l, :2].tolist())
                    elif mode == "U1": Ub.update(ids[r, l, :10].tolist())
                    else: Uv.update(ids[r, l, :10].tolist()); Uv2.update(ids[r, l, :2].tolist())
                mb += sum(1 for e in Ub if e not in bp[l])
                for e in Ub: T.touch(bp[l], e, C)
                vb += len(Uv) * MR11 + len(Uv2) * LR11
            kept += kp
    return task, (C, mb, vb, kept)


def keep(task):
    dom, c = task
    n, mem, res, wn, wmem, wres, ld, ms = PQ.run((dom, "lru", str(c)))[1]
    return task, 100 * (wmem + wres) / wn


def job(t):
    return replay(t[1:]) if t[0] == "b" else keep(t[1:])


def main():
    t0 = time.time(); T.W = T.windows()
    slots = sorted({B // r for r in STORES.values()} | {147})
    tasks = ([("b", d, r, m) for d in T.W for r in STORES.values() for m in ("U1", "U2")]
             + [("k", d, c) for d in T.W for c in slots] + [("k", d, "m") for d in T.W])
    with mp.get_context("fork").Pool(int(os.environ.get("PQ_PROCS", "3"))) as P:
        R = dict(P.map(job, tasks, chunksize=1))
    L = ["TRELLIS FAST STORE: C's rule (QOM cf 0 + gated last row, engine convention) with the 24 GB pool RAM spent on base "
         "records of each store (Cl 0); SSD MB per kept token; U1 = unsure rows from the same pool, U2 = unsure rows read from "
         "v3 every use; keep = LRU kept share (MASS) of sure-row rank 2-9 pairs; gap = of the keep gap 147 -> the 45 GB pool",
         "| domain | store | slots | U1 | U2 | of U2 from v3 | keep | gap closed |", "|---|---|---|---|---|---|---|---|"]
    for d in T.W:
        k147, km = R[(d, 147)], R[(d, "m")]
        for nm, rec in STORES.items():
            C, mb1, _, kp = R[(d, rec, "U1")]; _, mb2, vb2, kp2 = R[(d, rec, "U2")]
            u1 = mb1 * rec / kp / 1e6; u2 = (mb2 * rec + vb2) / kp2 / 1e6; kk = R[(d, C)]
            L.append(f"| {d} | {nm} | {C} | {u1:.1f} | {u2:.1f} | {vb2 / kp2 / 1e6:.1f} | {kk:.1f} % | "
                     f"{100 * (kk - k147) / (km - k147):+.0f} % |")
    L.append(f"({time.time() - t0:.0f} s; cross-check: MSB-3 11-bit U2 is POOLSPLIT's 171 / 0 cell with the v3 reads "
             f"uncached, so it must be >= 29.3 / 14.5 / 10.6)")
    os.makedirs(T.DST, exist_ok=True)
    open(os.path.join(T.DST, "trellis_store.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
