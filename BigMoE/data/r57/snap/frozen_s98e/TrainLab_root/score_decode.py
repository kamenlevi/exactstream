"""Score TrainLab decode arms with the Eval suite's own scorers (run with ~/EvalSuite/.venv/bin/python, CPU).
Per arm and set: accuracy, paired delta vs the exact arm, answer agreement with it (math: same extracted answer;
IFEval: same pass/fail), truncations. usage: score_decode.py ARM [ARM ...]  (first arm = the reference, e.g. exact)"""
import json, os, sys
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
from evalsuite import decode_report as R

D = os.path.expanduser("~/TrainLab/data/decode")


def main():
    arms = sys.argv[1:]
    its = {}
    for s in ("math", "ifeval"):
        for l in open(os.path.expanduser(f"~/EvalSuite/data/items/dec_{s}.jsonl")):
            it = json.loads(l); its[it["key"]] = it
    scored = {}
    for a in arms:
        rows = {r["key"]: r for r in map(json.loads, open(os.path.join(D, a + ".jsonl")))}
        sc = {}
        sc.update(R.score_math(rows, its))
        sc.update(R.score_ifeval(rows, its))
        scored[a] = (rows, sc)
    ref_rows, ref = scored[arms[0]]
    for a in arms:
        rows, sc = scored[a]
        line = [f"{a:28s}"]
        for s in ("math", "ifeval"):
            keys = [k for k in sc if its[k]["set"] == s and k in ref]
            if not keys:
                continue
            acc = sum(sc[k][1] for k in keys) / len(keys)
            racc = sum(ref[k][1] for k in keys) / len(keys)
            agree = sum(sc[k][0] == ref[k][0] for k in keys) / len(keys)
            trunc = sum(rows[k]["finish"] == "length" for k in keys)
            line.append(f"{s} {acc:.3f} ({acc - racc:+.3f} vs ref, n {len(keys)}) agree {agree:.3f} trunc {trunc}")
        print(" | ".join(line))


if __name__ == "__main__":
    main()
