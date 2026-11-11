"""DRAFT17 (NewIdeas, Wed 4 Nov 2026 13:5x; pre-registered QUEUE 13:5x; CPU only): a 17th draft gates every last row.
C's rule (QOM cf 0 + 3-bit sure rows + gated last row) on code, blocks_conv 'engine' (today: the last row gated by its
next draft when one exists) vs 'all' (every last row gated: what a drafted bonus row gives at cf 0). Pools at equal RAM
(records3 or the packed 11-bit parts; trellis records with no LSB pool, unsure rows read from v3 = trellis_store's U2).
Writes ~/NewIdeas/data/ledger/draft17.txt. Usage (BigMoE venv): python draft17.py"""
import os, sys, time
import multiprocessing as mp
from collections import OrderedDict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ten_x_ledger as T  # noqa: E402
import ledger_lastrow as LL  # noqa: E402
LINES = {"24 GB 147 / 46": (147, 46, 2162688, 770048), "24 GB 163 / 0": (163, 0, 2162688, 770048),
         "24 GB 11-bit 171 / 0": (171, 0, 2064384, 720896), "12 GB 62 / 0": (62, 0, 2162688, 770048),
         "24 GB TRL-2/3 2.40 b 239": (239, 0, 1474560, 720896), "24 GB TRL-2 2.06 b 276": (276, 0, 1277952, 720896)}


def replay(task):
    line, conv = task
    Cm, Cl, mr, lr = LINES[line]
    trl = "TRL" in line
    bp = [OrderedDict() for _ in range(48)]; lp = [OrderedDict() for _ in range(48)]
    mb = ml = vb = kept = 0
    for w in T.W["code"]:
        ids = w["ids"]
        for p in range(w["a"]):
            for l in range(48):
                for e in ids[p, l].tolist(): T.touch(bp[l], e, Cm)
                if Cl:
                    for e in ids[p, l, :2].tolist(): T.touch(lp[l], e, Cl)
        for rows, cfs, kp in LL.blocks_conv(w, conv):
            sure = [c >= 0.0 for c in cfs]
            for l in range(48):
                Ub, Ul, Uv, Uv2 = set(), set(), set(), set()
                for r, s_ in zip(rows, sure):
                    if s_: Ub.update(ids[r, l, :2].tolist())
                    elif trl: Uv.update(ids[r, l, :10].tolist()); Uv2.update(ids[r, l, :2].tolist())   # U2: v3 reads
                    else: Ub.update(ids[r, l, :10].tolist()); Ul.update(ids[r, l, :2].tolist())
                mb += sum(1 for e in Ub if e not in bp[l])
                for e in Ub: T.touch(bp[l], e, Cm)
                if Cl:
                    ml += sum(1 for e in Ul if e not in lp[l])
                    for e in Ul: T.touch(lp[l], e, Cl)
                else: ml += len(Ul)
                vb += len(Uv) * 2064384 + len(Uv2) * 720896
            kept += kp
    return task, (mb * mr + ml * lr + vb) / kept / 1e6


def main():
    t0 = time.time(); T.W = {"code": T.windows()["code"]}
    tasks = [(ln, c) for ln in LINES for c in ("engine", "all")]
    with mp.get_context("fork").Pool(int(os.environ.get("PQ_PROCS", "2"))) as P:
        R = dict(P.map(replay, tasks, chunksize=1))
    L = ["DRAFT17 (code): C's rule, SSD MB per kept token with the last row gated only when a next draft exists ('engine') "
         "vs always ('all' = a drafted bonus row at cf 0); trellis lines read unsure rows from v3 (U2)",
         "| line | engine | all | x |", "|---|---|---|---|"]
    for ln in LINES:
        e, a = R[(ln, "engine")], R[(ln, "all")]; L.append(f"| {ln} | {e:.1f} | {a:.1f} | x{a / e:.3f} |")
    L.append(f"({time.time() - t0:.0f} s; self-check: engine column = 35.7 / 32.8 / 29.3 / 75.6 / 46.3 / 43.2 from "
             f"ledger_top1, pool_split, trellis_store)")
    open(os.path.join(T.DST, "draft17.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
