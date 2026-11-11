"""TRELLIS FAST STORE, INFO add-on U3 (NewIdeas, Wed 4 Nov 2026 13:4x; QUEUE 13:44 add-on; CPU only).
Code only: unsure rows (no next draft) are read through a small v3 pool (MSB + LSB of an expert as one 2,785,280 B unit,
LRU, V slots per layer, ranks 0-9 of the row: ranks 0-1 need both parts, ranks 2-9 the MSB part, counted as one unit for
simplicity: conservative); the rest of the 24 GB pool RAM holds trellis records for the sure rows (top-2 + residents).
Writes ~/NewIdeas/data/ledger/trellis_u3.txt."""
import os, sys, time
import multiprocessing as mp
from collections import OrderedDict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ten_x_ledger as T  # noqa: E402
import ledger_lastrow as LL  # noqa: E402
B = 147 * 2162688 + 46 * 770048
V3 = 2064384 + 720896
STORES = {"TRL-2/3 2.40 b": 1474560, "TRL-2 2.06 b": 1277952}


def replay(task):
    rec, V = task
    C = (B - V * V3) // rec
    bp = [OrderedDict() for _ in range(48)]; vp = [OrderedDict() for _ in range(48)]
    mb = mv = kept = 0
    for w in T.W["code"]:
        ids = w["ids"]
        for p in range(w["a"]):
            for l in range(48):
                for e in ids[p, l].tolist(): T.touch(bp[l], e, C)
        for rows, cfs, kp in LL.blocks_conv(w, "engine"):
            for l in range(48):
                Ub, Uv = set(), set()
                for r, c in zip(rows, cfs):
                    if c >= 0.0: Ub.update(ids[r, l, :2].tolist())
                    else: Uv.update(ids[r, l, :10].tolist())
                mb += sum(1 for e in Ub if e not in bp[l])
                for e in Ub: T.touch(bp[l], e, C)
                if V: mv += sum(1 for e in Uv if e not in vp[l]); [T.touch(vp[l], e, V) for e in Uv]
                else: mv += len(Uv)
            kept += kp
    return task, (C, (mb * rec + mv * V3) / kept / 1e6, mv * V3 / kept / 1e6)


def main():
    t0 = time.time(); T.W = {"code": T.windows()["code"]}
    tasks = [(r, V) for r in STORES.values() for V in (0, 8, 16, 32)]
    with mp.get_context("fork").Pool(3) as P:
        R = dict(P.map(replay, tasks, chunksize=1))
    L = ["TRELLIS U3 (code, 24 GB): unsure rows through a V-slot v3 pool (2,785,280 B units), trellis slots from the rest; "
         "SSD MB per kept token", "| store | V | trellis slots | MB per kept | of it v3 |", "|---|---|---|---|---|"]
    for nm, r in STORES.items():
        for V in (0, 8, 16, 32):
            C, x, v = R[(r, V)]; L.append(f"| {nm} | {V} | {C} | {x:.1f} | {v:.1f} |")
    L.append(f"({time.time() - t0:.0f} s; V 0 counts every unsure expert as a full v3 unit (MSB + LSB), so its v3 part is "
             f"~x1.26 trellis_store's U2 39.2 = ~49: conservative by design)")
    open(os.path.join(T.DST, "trellis_u3.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
