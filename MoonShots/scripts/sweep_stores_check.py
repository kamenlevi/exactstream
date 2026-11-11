"""Independent re-derivation of THE SWEEP v1's store prices (NewIdeas 14:41; ~/NewIdeas/research/SWEEP.md rows 1-4), with
my own replay (draft17_check.py's: Breakthrough's capture checks at t 0.3, plain LRU, far-row stand-ins for wasted rows),
none of NewIdeas' scripts.
Rule: QOM cf 0 + 3-bit-class sure rows + every last row gated (DRAFT17): every row is sure and loads its top-2 (ranks 0-1).
Pool: one LRU pool per layer of the store's base records at equal RAM, no LSB pool: 24 GB = 147 x 2,162,688 + 46 x 770,048
B per layer, 12 GB = 51 / 32; slots = floor(RAM / record). Warm-up: each window's prompt rows (top-10) enter the pool, and
the pool carries across a domain's windows. Decode bytes: misses x the record's bytes, per kept token.
Prefill share (24 GB, the 11-bit store, 171 slots): the prompt rows' top-10 through the same pool, each miss read at 4 bits
(MSB + LSB 11-bit parts: 2,064,384 + 720,896 B), per decode kept token.
NewIdeas' numbers (24 GB; 12 GB; code / IFEval / multi): 11-bit / GPTQ-3 19.5 / 14.3 / 10.6; 52.0 / 54.4 / 39.4. TRL-3
15.8 / 11.4 / 8.5; 44.9 / 45.2 / 32.8. TRL-2 + 3 7.2 / 5.0 / 4.1; 28.5 / 25.3 / 18.4. TRL-2 4.0 / 3.0 / 2.7; 21.7 / 18.0 /
13.0. Prefill 117.9 / 43.1 / 17.0. Disagreements over ~5 % go to QUEUE as found.
usage: [BigMoE venv] python sweep_stores_check.py -> data/sweep_stores_check.txt"""
import glob, os, time
from collections import OrderedDict
import multiprocessing as mp
import numpy as np

CAP = os.path.expanduser("~/FableGod/decision_map/out/moonshots")
NL, MR0, LR0 = 48, 2162688, 770048
RAM = {"24 GB": 147 * MR0 + 46 * LR0, "12 GB": 51 * MR0 + 32 * LR0}
STORES = {"11-bit / GPTQ-3": 2064384, "TRL-3 3.07 b": 1900544, "TRL-2+3 2.40 b": 1474560, "TRL-2 2.06 b": 1277952}
THEIRS = {("11-bit / GPTQ-3", "24 GB"): (19.5, 14.3, 10.6), ("11-bit / GPTQ-3", "12 GB"): (52.0, 54.4, 39.4),
          ("TRL-3 3.07 b", "24 GB"): (15.8, 11.4, 8.5), ("TRL-3 3.07 b", "12 GB"): (44.9, 45.2, 32.8),
          ("TRL-2+3 2.40 b", "24 GB"): (7.2, 5.0, 4.1), ("TRL-2+3 2.40 b", "12 GB"): (28.5, 25.3, 18.4),
          ("TRL-2 2.06 b", "24 GB"): (4.0, 3.0, 2.7), ("TRL-2 2.06 b", "12 GB"): (21.7, 18.0, 13.0)}
PREFILL_THEIRS = (117.9, 43.1, 17.0)
PRE_REC = 2064384 + 720896                                        # a prefill miss at 4 bits (MSB + LSB, 11-bit parts)
DOMS = ("code", "ifeval", "multi")
STAND = os.environ.get("STAND", "far")
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def replay(task):
    dom, store, pool = task
    rec = STORES[store]
    C = RAM[pool] // rec
    bp = [OrderedDict() for _ in range(NL)]
    byt = pre = kept = 0.0

    def touch(c, e):
        if e in c:
            c.move_to_end(e); return 0
        c[e] = 1
        if len(c) > C:
            c.popitem(last=False)
        return 1

    for f in sorted(glob.glob(os.path.join(CAP, f"tf_{dom}_*.npz"))):
        z = np.load(f)
        ids, g, R = z["ids"], int(z["gen"]), z["ids"].shape[0]
        for p in range(g - 1):                                    # the prompt (prefill): top-10 through the pool
            for l in range(NL):
                for e in ids[p, l].tolist():
                    pre += PRE_REC * touch(bp[l], e)
        for a, n, k in z["checks"].tolist():
            off = {"far": R // 2, "p97": 97}.get(STAND, 0)       # wasted rows' stand-in (STAND env: far / p97 / path)
            rows = [min(a + j, R - 1) if j < k else ((a + j + off) % R if off else min(a + j, R - 1)) for j in range(n)]
            for l in range(NL):
                need = set()
                for r in rows:
                    need.update(ids[r, l, :2].tolist())                   # every row sure (DRAFT17): ranks 0-1
                miss = [e for e in need if e not in bp[l]]
                byt += rec * len(miss)
                for e in sorted(need):
                    touch(bp[l], e)
            kept += k
    return task, byt / kept / 1e6, pre / kept / 1e6, int(C)


def main():
    t0 = time.time()
    tasks = [(d, s, p) for d in DOMS for s in STORES for p in RAM]
    with mp.get_context("fork").Pool(4) as pool:
        res = {t: (v, pf, c) for t, v, pf, c in pool.map(replay, tasks)}
    say(f"== THE SWEEP v1 store prices, independent re-derivation (MoonShots, {time.strftime('%a %H:%M')}): decode SSD MB per kept "
        f"token, code / IFEval / multi (NewIdeas' in brackets; x = mine / theirs); wasted-row stand-in: {STAND}")
    flags = []
    for s in STORES:
        for p in RAM:
            mine = [res[(d, s, p)][0] for d in DOMS]
            th = THEIRS[(s, p)]
            ratio = [m / t for m, t in zip(mine, th)]
            fl = [d for d, r in zip(DOMS, ratio) if abs(r - 1) > 0.05]
            flags += [(s, p, d) for d in fl]
            say(f"  {s:16s} {p} ({res[(DOMS[0], s, p)][2]} slots): " + " / ".join(f"{m:5.1f}" for m in mine)
                + f"  [{' / '.join(f'{t:.1f}' for t in th)}]  x " + " / ".join(f"{r:.2f}" for r in ratio)
                + (f"  <- > 5 % on {', '.join(fl)}" if fl else ""))
    pf = [res[(d, "11-bit / GPTQ-3", "24 GB")][1] for d in DOMS]
    say(f"  prefill share at 24 GB, 11-bit store (171 slots), 4-bit reads: " + " / ".join(f"{x:.1f}" for x in pf)
        + f" MB per kept token  [{' / '.join(f'{x:.1f}' for x in PREFILL_THEIRS)}]  x "
        + " / ".join(f"{m / t:.2f}" for m, t in zip(pf, PREFILL_THEIRS)))
    say(f"  {len(flags)} cells differ by more than 5 %  ({time.time() - t0:.0f} s)")
    open(os.path.expanduser("~/MoonShots/data/sweep_stores_check.txt"), "a").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
