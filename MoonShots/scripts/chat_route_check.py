"""Independent re-derivation of THE SWEEP's chat row 10a / 10b (NewIdeas ad0bde0; the orchestrator's re-derivation rule),
with my own replay (sweep_stores_check.py's: Breakthrough's capture checks at t 0.3, plain LRU per layer, prefill warm-up,
far-row stand-ins for wasted rows, p+97 as a sensitivity), none of NewIdeas' scripts.
Rule (Fable C's 1(c)): QOM at cf 0.5. A row whose draft has drafter cf >= 0.5 (the live chain's dcf at that depth) is sure:
it loads its top-2 and uses resident ranks 2-9. The bonus (last) row is sure iff the draft it verifies has cf >= 0.5; a
full 16-row check's last row verifies none and is unsure. Unsure rows read REF2-T's way: top-10 MSB parts + top-2 LSB
parts. 10a: sure rows also read their top-2's LSB parts (4-bit); 10b: sure rows read no LSB part. EXACT: every row unsure.
Check: today's cf 0 stack (every row sure, DRAFT17, no LSB for sure rows) at Cl 0.
Layout: 24 GB per layer B = 147 x 2,162,688 + 46 x 770,048 B on 11-bit parts (MSB 2,064,384, LSB 720,896):
Cm = (B - Cl x 720,896) // 2,064,384, Cl in {0, 23, 46, 69, 92, 138}; the prefill warms both pools.
NewIdeas (SSD MB per kept token in decode, code / IFEval / multi): 10a best 61.8 / 87.9 / 87.2 (Cl 92 / 69 / 69); 10b best
44.5 / 73.0 / 75.9 (Cl 23); EXACT best 190.6 / 176.8 / 155.1; cf 0 check 19.5 / 14.3 / 10.6.
usage: [BigMoE venv] python chat_route_check.py -> data/chat_route_check.txt"""
import glob, os, time
from collections import OrderedDict
import multiprocessing as mp
import numpy as np

CAP = os.path.expanduser("~/FableGod/decision_map/out/moonshots")
NL, MSB, LSB = 48, 2064384, 720896
B = 147 * 2162688 + 46 * 770048
CLS = (0, 23, 46, 69, 92, 138)
DOMS = ("code", "ifeval", "multi")
RULES = ("10a", "10b", "exact", "cf0d17")
THEIRS = {"10a": ((61.8, 87.9, 87.2), (92, 69, 69)), "10b": ((44.5, 73.0, 75.9), (23, 23, 23)),
          "exact": ((190.6, 176.8, 155.1), None), "cf0d17": ((19.5, 14.3, 10.6), (0, 0, 0))}
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def replay(task):
    dom, rule, cl, stand = task
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
        for p in range(g - 1):                                    # the prompt (prefill) warms both pools
            for l in range(NL):
                for e in ids[p, l].tolist(): touch(bp[l], e, cm)
                for e in ids[p, l, :2].tolist(): touch(lp[l], e, cl)
        off = {"far": R // 2, "p97": 97}[stand]
        for ci, (a, n, k) in enumerate(z["checks"].tolist()):
            rows, sure = [], []
            for j in range(n):
                rows.append(min(a + j, R - 1) if j < k else (a + j + off) % R)
                cf = float(dcf[ci, j]) if j < dcf.shape[1] else None         # row j verifies draft j + 1
                if rule == "exact":
                    s = False
                elif rule == "cf0d17":
                    s = True
                else:
                    s = cf is not None and cf >= 0.5
                sure.append(s)
            for l in range(NL):
                nm, nl = set(), set()
                for r, s in zip(rows, sure):
                    if s:
                        nm.update(ids[r, l, :2].tolist())
                        if rule == "10a":
                            nl.update(ids[r, l, :2].tolist())
                    else:
                        nm.update(ids[r, l].tolist()); nl.update(ids[r, l, :2].tolist())
                byt += MSB * sum(1 for e in nm if e not in bp[l]) + LSB * sum(1 for e in nl if e not in lp[l])
                for e in sorted(nm): touch(bp[l], e, cm)
                for e in sorted(nl): touch(lp[l], e, cl)
            kept += k
    return task, byt / kept / 1e6


def main():
    t0 = time.time()
    tasks = [(d, r, c, s) for d in DOMS for r in RULES for c in (CLS if r != "cf0d17" else (0,)) for s in ("far", "p97")]
    with mp.get_context("fork").Pool(4) as pool:
        res = {t: v for t, v in pool.map(replay, tasks)}
    say(f"== THE SWEEP chat row 10a / 10b, independent re-derivation (MoonShots, {time.strftime('%a %H:%M')}): decode SSD MB per "
        f"kept token at 24 GB on 11-bit parts, best Cl per domain (wasted-row stand-in: far | p97); NewIdeas' in brackets")
    for r in RULES:
        th, tcl = THEIRS[r]
        for s in ("far", "p97"):
            best = []
            for d in DOMS:
                cand = [(res[(d, r, c, s)], c) for c in (CLS if r != "cf0d17" else (0,))]
                best.append(min(cand))
            say(f"  {r:6s} [{s:3s}]: " + " / ".join(f"{v:6.1f} (Cl {c})" for v, c in best)
                + f"  [{' / '.join(f'{x:.1f}' for x in th)}]  x " + " / ".join(f"{v / t:.2f}" for (v, _), t in zip(best, th)))
    say("  grid (far stand-in), code / IFEval / multi per Cl:")
    for r in ("10a", "10b", "exact"):
        say(f"    {r:6s}: " + "; ".join(f"Cl {c}: " + " / ".join(f"{res[(d, r, c, 'far')]:.1f}" for d in DOMS) for c in CLS))
    say(f"  ({time.time() - t0:.0f} s)")
    open(os.path.expanduser("~/MoonShots/data/chat_route_check.txt"), "a").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
