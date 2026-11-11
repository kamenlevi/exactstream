"""LOSSY350_ZERO follow-up 3 (zero session, Thu 5 Nov 2026): the `late` mask of NEVERWAIT_TEST.md section 2.3, counted the
way the driver will print it: late[p, l, r] = keep[p, l, r] AND expert in M_l (the check's loads that are not in the pool
at the check's start), on the check's REALIZED rows, every kept rank (ranks 0-1, kept members, kept residents cannot be
late, an undrafted row's ranks 2-9). Fable C's replay conventions (bytes10_replay.live_blocks), QOM0's keep rule
(dm_oea_masks' qom arm: loads first, then members | residents-after-loads). CPU only, seconds.
Writes late_terms.out beside this file."""
import os, sys, time
from collections import OrderedDict
import numpy as np
sys.path.insert(0, os.path.expanduser("~/FableGod/fablec"))
from bytes10_replay import windows, live_blocks, POOLS, MR  # noqa: E402

RANK_W = np.array([0.294, 0.193, 0.143, 0.109, 0.084, 0.064, 0.048, 0.033, 0.021, 0.010])
HERE = os.path.dirname(os.path.abspath(__file__))


def run(ws, Cm, bonus):
    pools = [OrderedDict() for _ in range(48)]
    S = dict(blocks=0, kept=0, real=0, miss=0, late=0, late_r1=0, late_mass=0.0, late_mass_r1=0.0, kept_terms=0,
             kept_mass=0.0, rows_hit=0, rows_hit_r1=0, late_top2=0, late_member=0, late_tail=0)
    for w in ws:
        ids = w["ids"]; a = w["a"]
        for l in range(48):
            c = pools[l]
            for e in ids[:a, l].ravel().tolist():
                if e in c: c.move_to_end(e)
                else:
                    c[e] = 1
                    if len(c) > Cm[l]: c.popitem(last=False)
        for rows, cfs, kp in live_blocks(w, bonus=bonus):
            E = ids[rows]; R = len(rows)
            nreal = 1
            while nreal < R and rows[nreal] == rows[0] + nreal: nreal += 1
            k0 = [10 if c < 0 else 2 for c in cfs]
            S["blocks"] += 1; S["kept"] += kp; S["real"] += nreal
            hit = np.zeros(nreal, bool); hit1 = np.zeros(nreal, bool)
            for l in range(48):
                c = pools[l]
                loads = set(); r0 = set()
                for r in range(R):
                    row = E[r, l].tolist(); loads.update(row[:k0[r]]); r0.add(row[0])
                M = {e for e in loads if e not in c}
                S["miss"] += len(M)
                for e in sorted(loads):                       # the builder's order: np.unique, touched one by one
                    if e in c: c.move_to_end(e)
                    else:
                        c[e] = 1
                        if len(c) > Cm[l]: c.popitem(last=False)
                for r in range(R):
                    row = E[r, l].tolist(); kk = k0[r]
                    for rk in range(10):
                        e = row[rk]
                        kept = rk < kk or e in loads or e in c
                        if not kept: continue
                        if r < nreal:
                            S["kept_terms"] += 1; S["kept_mass"] += RANK_W[rk]
                            if e in M:
                                S["late"] += 1; S["late_mass"] += RANK_W[rk]; hit[r] = True
                                if rk < 2 and kk == 2: S["late_top2"] += 1
                                elif rk < kk: S["late_tail"] += 1 if rk >= 2 else 0; S["late_top2"] += 1 if rk < 2 else 0
                                else: S["late_member"] += 1
                                if e not in r0:
                                    S["late_r1"] += 1; S["late_mass_r1"] += RANK_W[rk]; hit1[r] = True
                    for e in row[kk:]:
                        if e in c and (e in loads or True): c.move_to_end(e) if (e in loads or e in c) else None
            S["rows_hit"] += int(hit.sum()); S["rows_hit_r1"] += int(hit1.sum())
    return S


if __name__ == "__main__":
    W = windows(); L = []; t0 = time.time()
    L.append("| domain | pool | bonus | kept / realized rows per block | demand misses per kept | late terms per realized row: nw1 (top-2 / members / undrafted tails) | nw1r1 | "
             "late MoE mass per realized row-layer: nw1 / nw1r1 | share of the row's KEPT mass that is late: nw1 / nw1r1 | realized rows with >= 1 late term: nw1 / nw1r1 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, Cm in [("208", [208] * 48), ("224", [224] * 48), ("laysz45_256", POOLS["45"][0]), ("147", [147] * 48)]:
        for bonus in ("all", "next"):
            if name not in ("208", "laysz45_256") and bonus == "next": continue
            for dom in ("code", "ifeval", "multi"):
                s = run(W[dom], Cm, bonus); rl = s["real"] * 48
                L.append(f"| {dom} | {name} | {'_ball' if bonus == 'all' else '_bq'} | {s['kept'] / s['blocks']:.1f} / {s['real'] / s['blocks']:.1f} | {s['miss'] / s['kept']:.2f} | "
                         f"{s['late'] / s['real']:.2f} ({s['late_top2'] / s['real']:.2f} / {s['late_member'] / s['real']:.2f} / {s['late_tail'] / s['real']:.2f}) | {s['late_r1'] / s['real']:.2f} | "
                         f"{100 * s['late_mass'] / rl:.2f} % / {100 * s['late_mass_r1'] / rl:.2f} % | {100 * s['late_mass'] / s['kept_mass']:.2f} % / {100 * s['late_mass_r1'] / s['kept_mass']:.2f} % | "
                         f"{100 * s['rows_hit'] / s['real']:.0f} % / {100 * s['rows_hit_r1'] / s['real']:.0f} % |")
                print(L[-1], flush=True)
    L.append(f"({time.time() - t0:.0f} s; RANK_W = AUDIT_1105 7.2's mean renormalised weight by rank; MR {MR})")
    open(os.path.join(HERE, "late_terms.out"), "w").write("\n".join(L) + "\n"); print(L[-1])
