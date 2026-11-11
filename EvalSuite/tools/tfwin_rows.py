"""Per-row and per-language breakdown of a tfwin-scored run (docs/RESULT_chat_m1b_2026-11-08.md): deviations (committed
token != the reference's argmax along the run's own prefix) and blunders (the reference's p of the committed token
< 0.01), with the first positions of each, and whether another run (e.g. the other prompt order) has the same tokens.
  .venv/bin/python tools/tfwin_rows.py <tfwin run> [<other run>]"""
import glob, gzip, json, math, os, sys
from collections import defaultdict

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
B = math.log(0.01)


def rows(run):
    return {r["key"]: r for r in map(json.loads, open(f"results/{run}/decode.jsonl"))}


def main(run, other=None):
    R = rows(run)
    O = rows(other) if other else {}
    lang = defaultdict(lambda: [0, 0, 0])
    for f in sorted(glob.glob(f"results/{run}/tfwin/*.json.gz")):
        d = json.load(gzip.open(f))
        k, c = d["key"], R[d["key"]]["completion_ids"]
        n = min(len(d["lp"]), len(c))
        dev = [i for i in range(n) if d["am"][i] != c[i]]
        bl = [i for i in range(n) if d["lp"][i] < B]
        g = lang[k.split("/")[-1].rstrip("0123456789")]
        g[0] += n; g[1] += len(dev); g[2] += len(bl)
        same = "" if not O else f"; == {other}: {O.get(k, {}).get('completion_ids') == c}"
        print(f"{k}: {n} tokens, deviations {len(dev)} (first {dev[:4]}), blunders {len(bl)} (first {bl[:4]}){same}")
    t = [sum(v[i] for v in lang.values()) for i in range(3)]
    for g, (n, dv, bl) in sorted(lang.items()) + [("all", t)]:
        print(f"{g}: {n} tokens, deviations {100 * dv / n:.2f} %, blunders {bl} = {1e4 * bl / n:.1f} per 10K"
              + (f", {100 * bl / dv:.0f} % of deviations" if dv else ""))


if __name__ == "__main__":
    main(*sys.argv[1:])
