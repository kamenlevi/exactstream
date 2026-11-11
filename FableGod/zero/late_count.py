"""LOSSY350_ZERO follow-up (zero session, Thu 5 Nov 2026). CPU only, one process, numpy, seconds.
Counts the DEMAND misses of today's 45 GB FAST preset (QOM0BQ; the split forms load the same experts) per live block
on the 36 decision-map windows, with Fable C's replay conventions (bytes10_replay / audit_split_replay: plain LRU per
layer, prompt rows warm the pool, one session per domain, live-cut blocks, the bonus row gated by the next draft's cf).
For each missed expert: the best rank at which a row of the block demands it (0, 1, or 2-9 = an undrafted row's tail).
Reports per block: misses by rank class, stalled layers (>= 1 miss), stalled layers if only rank-0 misses are waited,
(row, layer) pairs that have a late expert, the gate mass that would arrive late, and QOM's dropped optional tails.
Reads only; writes late_count.out / .json beside this file."""
import json, os, sys, time
from collections import OrderedDict
import numpy as np
sys.path.insert(0, os.path.expanduser("~/FableGod/fablec"))
from bytes10_replay import windows, live_blocks, POOLS, MR  # noqa: E402

RANK_W = [0.294, 0.193, 0.143, 0.109, 0.084, 0.064, 0.048, 0.033, 0.021, 0.010]   # AUDIT_1105 7.2 (cloud3 P2 s4)
HERE = os.path.dirname(os.path.abspath(__file__))


def run(ws, Cm):
    pools = [OrderedDict() for _ in range(48)]
    B = []                                   # per block: dict of counts
    for w in ws:
        ids = w["ids"]; a = w["a"]
        for l in range(48):
            c = pools[l]
            for e in ids[:a, l].ravel().tolist():
                if e in c: c.move_to_end(e)
                else:
                    c[e] = 1
                    if len(c) > Cm[l]: c.popitem(last=False)
        for rows, cfs, kp in live_blocks(w, bonus="next"):
            E = ids[rows]; R = len(rows)
            k0 = [10 if c < 0 else 2 for c in cfs]
            d = dict(kept=kp, rows=R, miss=0, r0=0, r1=0, tail=0, stalled=0, stalled_r0=0, rowlayer=0,
                     late_mass=0.0, late_mass_r1=0.0, dropped=0, undrafted=sum(1 for c in cfs if c < 0))
            for l in range(48):
                c = pools[l]
                best = {}                                     # expert -> best rank demanded
                for r in range(R):
                    row = E[r, l].tolist()
                    for rk in range(k0[r]):
                        e = row[rk]
                        if e not in best or rk < best[e]: best[e] = rk
                missed = {e: rk for e, rk in best.items() if e not in c}
                if missed:
                    d["stalled"] += 1
                    if any(rk == 0 for rk in missed.values()): d["stalled_r0"] += 1
                    d["miss"] += len(missed)
                    for rk in missed.values():
                        if rk == 0: d["r0"] += 1
                        elif rk == 1: d["r1"] += 1
                        else: d["tail"] += 1
                    for r in range(R):                        # rows that would see a late term
                        row = E[r, l].tolist(); m = 0.0; m1 = 0.0
                        for rk in range(k0[r]):
                            if row[rk] in missed:
                                m += RANK_W[rk]
                                if rk >= 1: m1 += RANK_W[rk]
                        if m > 0: d["rowlayer"] += 1; d["late_mass"] += m; d["late_mass_r1"] += m1
                loads = set(best)
                for e in loads:                               # demand reads (the replay's order)
                    if e in c: c.move_to_end(e)
                    else:
                        c[e] = 1
                        if len(c) > Cm[l]: c.popitem(last=False)
                keepset = set(); drop = set()
                for r in range(R):
                    row = E[r, l].tolist(); kk = k0[r]
                    keepset.update(row[:kk])
                    for e in row[kk:]:
                        if e in loads or e in c: keepset.add(e)
                        else: drop.add(e)
                d["dropped"] += len(drop - keepset)
                for e in keepset:
                    if e in c: c.move_to_end(e)
            B.append(d)
    return B


def summ(B):
    A = lambda k: np.array([b[k] for b in B], dtype=float)
    kept = A("kept").sum(); n = len(B); rows = A("rows")
    m = A("miss")
    return dict(blocks=n, kept_per_block=kept / n, rows_per_block=rows.mean(),
                mb_per_kept=m.sum() * MR / 1e6 / kept,
                miss_mean=m.mean(), miss_med=float(np.median(m)), miss_p90=float(np.percentile(m, 90)),
                zero_miss_share=float((m == 0).mean()),
                r0=A("r0").mean(), r1=A("r1").mean(), tail=A("tail").mean(),
                stalled=A("stalled").mean(), stalled_r0=A("stalled_r0").mean(),
                rowlayer_share=A("rowlayer").sum() / (rows.sum() * 48),
                late_events_per_row=A("rowlayer").sum() / rows.sum(),
                late_mass_per_rowlayer=A("late_mass").sum() / (rows.sum() * 48),
                late_mass_r1only_per_rowlayer=A("late_mass_r1").sum() / (rows.sum() * 48),
                dropped_per_block=A("dropped").mean(), dropped_mb_per_block=A("dropped").mean() * MR / 1e6,
                undrafted_rows_per_block=A("undrafted").mean())


if __name__ == "__main__":
    t0 = time.time(); W = windows(); res = {}; L = []
    L.append("| domain | pool | blocks | kept / rows per block | SSD MB per kept | demand misses per block mean / median / p90 | blocks with 0 | "
             "rank 0 / rank 1 / undrafted-row tails | stalled layers (all / if only rank 0 waits) | late (row,layer) share; events per row | "
             "MoE mass late per row-layer (all / if rank 0 waits) | QOM-dropped tails per block (MB) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for pool in ("45", "24"):
        Cm = POOLS[pool][0]
        for dom in ("code", "ifeval", "multi"):
            s = summ(run(W[dom], Cm)); res[f"{dom}|{pool}"] = s
            L.append(f"| {dom} | {pool} GB | {s['blocks']} | {s['kept_per_block']:.1f} / {s['rows_per_block']:.1f} | {s['mb_per_kept']:.1f} | "
                     f"{s['miss_mean']:.1f} / {s['miss_med']:.0f} / {s['miss_p90']:.0f} | {100 * s['zero_miss_share']:.0f} % | "
                     f"{s['r0']:.1f} / {s['r1']:.1f} / {s['tail']:.1f} | {s['stalled']:.1f} / {s['stalled_r0']:.1f} | "
                     f"{100 * s['rowlayer_share']:.2f} %; {s['late_events_per_row']:.2f} | "
                     f"{100 * s['late_mass_per_rowlayer']:.3f} % / {100 * s['late_mass_r1only_per_rowlayer']:.3f} % | "
                     f"{s['dropped_per_block']:.0f} ({s['dropped_mb_per_block']:.0f}) |")
            print(L[-1], flush=True)
    L.append(f"({time.time() - t0:.0f} s; rule QOM0BQ loads, pools = bytes10_replay.POOLS; MR {MR} bytes)")
    open(os.path.join(HERE, "late_count.out"), "w").write("\n".join(L) + "\n")
    json.dump(res, open(os.path.join(HERE, "late_count.json"), "w"), indent=1)
    print(L[-1])
