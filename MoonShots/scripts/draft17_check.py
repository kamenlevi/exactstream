"""Independent check of NewIdeas' DRAFT17 byte claim (commit 80fd4bd), for the orchestrator (Wed 14:1x): CPU only, my own
replay and check source, none of NewIdeas' scripts.
Claim: in the FAST C stack at the 24 GB pool (QOM cf 0 + 3-bit sure rows + gated last row, 147 / 46 slots) a check's last
row is gated only when a next draft exists; on code ~32 % of segments have none, those last rows run exact (full top-10 +
LSB) and carry ~a third of code's bytes; drafting a 17th position gates them all: code 35.7 -> 25.2 MB per kept token
(x0.706), 12 GB x0.742, chat unchanged.
Mine: Breakthrough's capture (~/FableGod/decision_map/out/moonshots: the 125B's teacher-forced routing ids [T-1, 48, 10],
the engine's t 0.3 cut checks (anchor, rows n, kept k) from its own DFlash2 walk; not MTP Head's HPD-0 walks). Plain LRU
per layer (base pool = the 3-bit MSB part of every needed expert, LSB pool = ranks 0-1's sidecar part), warmed by each
window's prompt rows (10 ranks + LSB), carried across a domain's windows. Rows of a check: j < k on the answer's true path
(ids[a + j]); j >= k come after the first wrong draft and use a far row of the window as their stand-in (ids[(p + R/2) %
R]; the true-path stand-in is reported as a sensitivity). QOM cf 0: a row with a draft to verify is sure: it needs its
ranks 0-1 in the base pool only (3-bit, no LSB). The last row (j = n - 1): map = exact; engine = sure iff n < 16 (the cut
left a next draft); draft17 = sure. Exact rows need all 10 ranks + the LSB part of ranks 0-1.
Bytes: base misses x 2,162,688 + LSB misses x 770,048 (records3 parts), per kept token.
usage: [BigMoE venv] python draft17_check.py -> data/draft17_check.txt"""
import glob, os, time
from collections import OrderedDict
import multiprocessing as mp
import numpy as np

CAP = os.path.expanduser("~/FableGod/decision_map/out/moonshots")
MR, LR, NL = 2162688, 770048, 48
POOLS = {"24 GB": (147, 46), "12 GB": (51, 32)}
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def touch(c, e, cap):
    if e in c:
        c.move_to_end(e)
    else:
        c[e] = 1
        if len(c) > cap:
            c.popitem(last=False)


def replay(task):
    dom, pool, conv, stand = task
    Cm, Cl = POOLS[pool]
    bp = [OrderedDict() for _ in range(NL)]
    lp = [OrderedDict() for _ in range(NL)]
    byt = kept = 0.0
    nchk = nexact_last = 0
    for f in sorted(glob.glob(os.path.join(CAP, f"tf_{dom}_*.npz"))):
        z = np.load(f)
        ids, g, R = z["ids"], int(z["gen"]), z["ids"].shape[0]
        for p in range(g - 1):                                   # the prompt warms both pools
            for l in range(NL):
                for e in ids[p, l].tolist(): touch(bp[l], e, Cm)
                for e in ids[p, l, :2].tolist(): touch(lp[l], e, Cl)
        for a, n, k in z["checks"].tolist():
            rows, sure = [], []
            for j in range(n):
                p = a + j
                if j < k:
                    rows.append(min(p, R - 1))                   # a final check may reach past the last routed row
                else:
                    rows.append((p + R // 2) % R if stand == "far" else min(p, R - 1))
                last = j == n - 1
                s = (not last) or (conv == "engine" and n < 16) or conv == "draft17"
                sure.append(s)
            nexact_last += not sure[-1]
            for l in range(NL):
                ub, ul = set(), set()
                for r, s in zip(rows, sure):
                    if s:
                        ub.update(ids[r, l, :2].tolist())
                    else:
                        ub.update(ids[r, l].tolist()); ul.update(ids[r, l, :2].tolist())
                byt += MR * sum(1 for e in ub if e not in bp[l]) + LR * sum(1 for e in ul if e not in lp[l])
                for e in sorted(ub): touch(bp[l], e, Cm)
                for e in sorted(ul): touch(lp[l], e, Cl)
            kept += k; nchk += 1
    return task, byt / kept / 1e6, nexact_last / nchk


def main():
    t0 = time.time()
    tasks = [(d, p, c, s) for d in ("code", "ifeval", "multi") for p in POOLS for c in ("map", "engine", "draft17")
             for s in ("far", "path")]
    with mp.get_context("fork").Pool(4) as pool:
        res = {t: (v, x) for t, v, x in pool.map(replay, tasks)}
    say(f"== DRAFT17 independent check (MoonShots, {time.strftime('%a %H:%M')}): FAST C stack (QOM cf 0 + 3-bit sure rows), "
        f"plain LRU, Breakthrough's capture checks; SSD MB per kept token (wasted-row stand-in: far row | true path)")
    for d in ("code", "ifeval", "multi"):
        for p in POOLS:
            m, e, x = (res[(d, p, c, "far")][0] for c in ("map", "engine", "draft17"))
            mt, et, xt = (res[(d, p, c, "path")][0] for c in ("map", "engine", "draft17"))
            share = res[(d, p, "engine", "far")][1]
            say(f"  {d:6s} {p}: map {m:6.1f} | engine {e:6.1f} | draft17 {x:6.1f} -> x{x / e:.3f} "
                f"({100 * (1 - x / e):.0f} % of the engine's bytes come from its exact last rows; {100 * share:.1f} % of "
                f"checks keep one) | true-path stand-in: {mt:.1f} / {et:.1f} / {xt:.1f} (x{xt / et:.3f})")
    say(f"  ({time.time() - t0:.0f} s)")
    open(os.path.expanduser("~/MoonShots/data/draft17_check.txt"), "a").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
