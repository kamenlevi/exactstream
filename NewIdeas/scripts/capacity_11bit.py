"""11-bit's capacity under QOM (NewIdeas, Wed 4 Nov 2026 08:59; pre-registered QUEUE 08:59; CPU only).
CORRECTED Wed 13:4x: the 08:59 run took MoonShots' -17.6 % (M-27, a replay at the 45 GB C 256 pools that already holds
~4.6 % more parts in the same RAM) as a per-byte factor x0.824 AND added 1 / 0.824 more slots: double counting. The
packed store's real part sizes are 2,064,384 / 720,896 B (MoonShots' packed45_256.json part_bytes) vs records3's
2,162,688 / 770,048 B: x0.955 / x0.936 per part, and 24 GB's 147 / 46 slots become 154 / 49 at equal RAM (12 GB 51 / 32 ->
53 / 34). Bytes: ledger_top1's replay of C's rule (QOM cf 0 + 3-bit sure rows + the gated last row, engine convention)
with each pool line's own part sizes. Keep: pool_quality's LRU kept share of sure-row rank 2-9 pairs at 147 / 154 slots
and at the 45 GB pool. Writes ~/NewIdeas/data/ledger/capacity_11bit.txt (the 08:59 output is kept as
capacity_11bit_0859_wrong.txt)."""
import os, sys, time
import multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ten_x_ledger as T  # noqa: E402
import ledger_top1 as LT  # noqa: E402
import pool_quality as PQ  # noqa: E402
MR11, LR11 = 2064384, 720896                                  # M-27b PACKED11 part bytes (packed45_256.json)
POOLS = {"24 GB": ([147] * 48, [46] * 48), "24 GB 11-bit": ([154] * 48, [49] * 48),
         "12 GB": ([51] * 48, [32] * 48), "12 GB 11-bit": ([53] * 48, [34] * 48)}
RULE = ("QOM cf 0 + 3b (top-2)", 0.0, 2, None)


def job(t):
    if t[0] == "bytes":
        T.POOLS = POOLS
        if "11-bit" in t[2]: T.MR, T.LR = MR11, LR11            # this worker's own part sizes (fork: no leak back)
        return t, LT.replay((t[1], t[2], RULE, "engine"))[1]
    return t, PQ.run((t[1], "lru", t[2]))[1]


def main():
    t0 = time.time(); T.W = T.windows()
    tasks = [("bytes", d, p) for d in T.W for p in POOLS] + [("keep", d, c) for d in T.W for c in ("147", "154", "m")]
    with mp.get_context("fork").Pool(int(os.environ.get("PQ_PROCS", "3")), maxtasksperchild=1) as P:
        R = dict(P.map(job, tasks, chunksize=1))
    L = ["11-BIT CAPACITY under QOM (CORRECTED 13:4x): C's rule (QOM cf 0 + 3-bit + the gated last row), SSD MB per kept "
         "token (replay, engine convention; 11-bit lines with the packed part sizes 2,064,384 / 720,896 B on 154 / 49 and 53 / "
         "34 slots at equal RAM) and the LRU kept share of sure-row rank 2-9 pairs (MASS-weighted)",
         "| domain | 24 GB | 24 GB 11-bit (x) | 12 GB | 12 GB 11-bit (x) | kept 147 | kept 154 | kept 45 GB pool | "
         "gap closed at 154 |", "|---|---|---|---|---|---|---|---|---|"]
    for d in T.W:
        b = {p: R[("bytes", d, p)] for p in POOLS}
        k = {}
        for c in ("147", "154", "m"):
            n, mem, res, wn, wmem, wres, ld, ms = R[("keep", d, c)]; k[c] = 100 * (wmem + wres) / wn
        L.append(f"| {d} | {b['24 GB']:.1f} | {b['24 GB 11-bit']:.1f} (x{b['24 GB 11-bit'] / b['24 GB']:.3f}) | {b['12 GB']:.1f} | "
                 f"{b['12 GB 11-bit']:.1f} (x{b['12 GB 11-bit'] / b['12 GB']:.3f}) | {k['147']:.1f} % | {k['154']:.1f} % | "
                 f"{k['m']:.1f} % | {100 * (k['154'] - k['147']) / (k['m'] - k['147']):+.0f} % |")
    L.append(f"({time.time() - t0:.0f} s; 24 GB without 11-bit must equal ledger_top1's engine row 35.7 / 19.4 / 13.9)")
    open(os.path.expanduser("~/NewIdeas/data/ledger/capacity_11bit.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
