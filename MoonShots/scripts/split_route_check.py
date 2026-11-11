"""SPLITCHECK (MoonShots, Thu 2026-11-05; the orchestrator 07:1x, NewIdeas' ask before Kamen sees SWEEP v1.3): independent
re-derivation of SWEEP rank 5's bytes, the split-threshold chat route 'drafted 0.3 + bonus gate 0.4' (NewIdeas
scripts/split_price.py, data/ledger/split_price.txt: 53.9 / 62.4 / 64.0 MB per kept token at 24 GB, x0.71 / 0.73 of the
cf 0.5 route on chat). My own replay = M-37's (scripts/chat_route_check.py, c388167): Breakthrough's capture checks at
t 0.3, the live chain's dcf, plain LRU per layer, prefill warm-up, wasted rows by stand-ins (p+97 as NewIdeas, far rows as
a sensitivity), none of NewIdeas' scripts.
Rule (10a form, 4-bit sure rows): a drafted row j (j < the check's last row) is sure iff its draft's drafter cf >= td; the
check's last (bonus) row is sure iff the next draft's cf >= tb (none for a full 16-row check: unsure). Sure rows load their
top-2 experts and those top-2's LSB parts; unsure rows read REF2-T's way (top-10 MSB + top-2 LSB). 24 GB on 11-bit parts:
Cm = (B - Cl x 720,896) // 2,064,384, B = 147 x 2,162,688 + 46 x 770,048 per layer, best Cl per domain.
Variants (NewIdeas' table): (0.5, 0.5) and (0.3, 0.3) as checks; (0.3, 0.4) = rank 5; (0.3, 0.5) V1; (0.5, 0.3) V2.
usage: [BigMoE venv] python split_route_check.py -> data/split_route_check.txt"""
import glob, os, time
from collections import OrderedDict
import multiprocessing as mp
import numpy as np

CAP = os.path.expanduser("~/FableGod/decision_map/out/moonshots")
NL, MSB, LSB = 48, 2064384, 720896
B = 147 * 2162688 + 46 * 770048
CLS = (0, 23, 46, 69, 92, 138)
DOMS = ("code", "ifeval", "multi")
VARS = {"cf 0.5 route (check)": (0.5, 0.5), "cf 0.3 route (check)": (0.3, 0.3), "rank 5: drafted 0.3 + bonus 0.4": (0.3, 0.4),
        "V1 drafted 0.3 + bonus 0.5": (0.3, 0.5), "V2 drafted 0.5 + bonus 0.3": (0.5, 0.3)}
THEIRS = {"cf 0.5 route (check)": (61.8, 87.9, 87.2), "cf 0.3 route (check)": (50.2, 49.2, 51.7),
          "rank 5: drafted 0.3 + bonus 0.4": (53.9, 62.4, 64.0), "V1 drafted 0.3 + bonus 0.5": (56.9, 73.3, 71.9),
          "V2 drafted 0.5 + bonus 0.3": (55.4, 66.0, 69.2)}
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def replay(task):
    dom, var, cl, stand = task
    td, tb = VARS[var]
    cm = (B - cl * LSB) // MSB
    bp = [OrderedDict() for _ in range(NL)]
    lp = [OrderedDict() for _ in range(NL)]
    byt = kept = 0.0

    def touch(c, e, cap):
        if cap <= 0:
            return
        if e in c:
            c.move_to_end(e)
        else:
            c[e] = 1
            if len(c) > cap:
                c.popitem(last=False)

    for f in sorted(glob.glob(os.path.join(CAP, f"tf_{dom}_*.npz"))):
        z = np.load(f)
        ids, g, R, dcf = z["ids"], int(z["gen"]), z["ids"].shape[0], z["dcf"]
        for p in range(g - 1):                                        # the prompt (prefill) warms both pools
            for l in range(NL):
                for e in ids[p, l].tolist(): touch(bp[l], e, cm)
                for e in ids[p, l, :2].tolist(): touch(lp[l], e, cl)
        off = {"far": R // 2, "p97": 97}[stand]
        for ci, (a, n, k) in enumerate(z["checks"].tolist()):
            rows, sure = [], []
            for j in range(n):
                rows.append(min(a + j, R - 1) if j < k else (a + j + off) % R)
                cf = float(dcf[ci, j]) if j < dcf.shape[1] else None   # row j verifies draft j + 1 (the last row: the next draft)
                sure.append(cf is not None and cf >= (tb if j == n - 1 else td))
            for l in range(NL):
                nm, nl = set(), set()
                for r, s in zip(rows, sure):
                    if s:
                        nm.update(ids[r, l, :2].tolist()); nl.update(ids[r, l, :2].tolist())
                    else:
                        nm.update(ids[r, l].tolist()); nl.update(ids[r, l, :2].tolist())
                byt += MSB * sum(1 for e in nm if e not in bp[l]) + LSB * sum(1 for e in nl if e not in lp[l])
                for e in sorted(nm): touch(bp[l], e, cm)
                for e in sorted(nl): touch(lp[l], e, cl)
            kept += k
    return task, byt / kept / 1e6


def main():
    t0 = time.time()
    tasks = [(d, v, c, s) for d in DOMS for v in VARS for c in CLS for s in ("p97", "far")]
    with mp.get_context("fork").Pool(int(os.environ.get("WORKERS", "4"))) as pool:
        res = {t: v for t, v in pool.map(replay, tasks)}
    say(f"== SWEEP rank 5 bytes, independent re-derivation (MoonShots, {time.strftime('%a %H:%M')}): decode SSD MB per kept token "
        f"at 24 GB on 11-bit parts, 4-bit sure rows, best Cl per domain (code / IFEval / multi); NewIdeas' split_price.txt in brackets")
    ref = {}
    for s in ("p97", "far"):
        say(f"  wasted-row stand-in {s}:")
        for v in VARS:
            best = [min((res[(d, v, c, s)], c) for c in CLS) for d in DOMS]
            if v.startswith("cf 0.5"):
                ref[s] = best
            say(f"    {v:33s}: " + " / ".join(f"{x:6.1f} (Cl {c})" for x, c in best) + f"  [{' / '.join(f'{x:.1f}' for x in THEIRS[v])}]"
                + "  x mine/theirs " + " / ".join(f"{x / t:.3f}" for (x, _), t in zip(best, THEIRS[v]))
                + "  chat x the cf 0.5 route " + " / ".join(f"{best[i][0] / ref[s][i][0]:.2f}" for i in (1, 2)))
    say(f"  ({time.time() - t0:.0f} s)")
    open(os.path.expanduser("~/MoonShots/data/split_route_check.txt"), "a").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
