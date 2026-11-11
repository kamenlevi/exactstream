"""The agent FAIL's cause bisect (docs/PREREG_agentbisect_2026-11-06.md): per arm and order, the tfwin scores of its rows
(results/bisect_<arm>_<order>/) -> blunders (the reference's p of a committed token < 0.01), deviations, and the lines:
an arm's forward blunders <= 20 % of base's forward = this lever carries the failure; >= 50 % = not the main cause; else
partial (weak if base has fewer than 10). The sharp test: qom's two orders token-identical on all three turns.
  .venv/bin/python tools/agentbisect_report.py"""
import gzip, json, math, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
LN = math.log(0.01)
ARMS = ("base", "qom", "d17", "hc8", "nax", "small")


def load(run):
    p = os.path.join(RES, run, "decode.jsonl")
    if not os.path.exists(p):
        return None
    rows = {r["key"]: r for r in map(json.loads, open(p))}
    sc = {}
    for k in rows:
        f = os.path.join(RES, run, "tfwin", k.replace("/", "__") + ".json.gz")
        if os.path.exists(f):
            sc[k] = json.load(gzip.open(f, "rt"))
    return rows, sc


def stats(x):
    rows, sc = x
    tok = sum(len(rows[k]["completion_ids"]) for k in sc)
    dev = sum(sum(a != b for a, b in zip(sc[k]["am"], rows[k]["completion_ids"])) for k in sc)
    bl = sum(sum(lp < LN for lp in sc[k]["lp"]) for k in sc)
    return dict(rows=len(rows), scored=len(sc), tok=tok, dev=dev, bl=bl)


def first_diff(a, b):
    return next((i for i, (u, v) in enumerate(zip(a, b)) if u != v), None if len(a) == len(b) else min(len(a), len(b)))


def main():
    R = {}
    for arm in ARMS:
        for o in ("fwd", "rev"):
            x = load(f"bisect_{arm}_{o}")
            if x:
                R[(arm, o)] = x
    print("== per arm and order (three turns: write12-14)")
    for (arm, o), x in sorted(R.items()):
        s = stats(x)
        print(f"{arm:5s} {o}: {s['scored']}/{s['rows']} rows scored, {s['tok']} tokens, deviations {s['dev']} "
              f"({100 * s['dev'] / max(s['tok'], 1):.2f} %), blunders {s['bl']}")
    if ("base", "fwd") not in R:
        print("BISECT: INCOMPLETE (no base forward)")
        return
    b0 = stats(R[("base", "fwd")])["bl"]
    weak = b0 < 10
    print(f"\n== the lines (vs base forward: {b0} blunders{'; weak: fewer than 10' if weak else ''})")
    for arm in ARMS[1:]:
        if (arm, "fwd") not in R:
            print(f"BISECT {arm}: not run")
            continue
        b = stats(R[(arm, "fwd")])["bl"]
        r = b / b0 if b0 else float("inf") if b else 0.0
        v = "carries it" if r <= 0.20 else "not the main cause" if r >= 0.50 else "partial"
        print(f"BISECT {arm}: {b} blunders = {100 * r:.0f} % of base: {v}{' (weak)' if weak else ''}")
    for arm in ("qom", "base"):
        if (arm, "fwd") in R and (arm, "rev") in R:
            a, b = R[(arm, "fwd")][0], R[(arm, "rev")][0]
            same = [k for k in a if k in b and a[k]["completion_ids"] == b[k]["completion_ids"]]
            print(f"BISECT sharp test {arm}: the two orders token-identical on {len(same)} of {len(set(a) & set(b))} turns"
                  + ("" if len(same) == len(set(a) & set(b)) else "; first differences " + ", ".join(
                      f"{k.split('/')[1]} @{first_diff(a[k]['completion_ids'], b[k]['completion_ids'])}"
                      for k in sorted(set(a) & set(b)) if k not in same)))
    print("\n== each arm's forward tokens vs base's forward (first divergence)")
    base = R[("base", "fwd")][0]
    for arm in ARMS[1:]:
        if (arm, "fwd") in R:
            a = R[(arm, "fwd")][0]
            print(f"  {arm}: " + ", ".join(f"{k.split('/')[1]} @{first_diff(a[k]['completion_ids'], base[k]['completion_ids'])}"
                                         for k in sorted(set(a) & set(base))))


if __name__ == "__main__":
    main()
