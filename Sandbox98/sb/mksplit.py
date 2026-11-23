"""Pin the Magicoder split (data/split.json) in the prompts' seeded order: base = the first --nbase captured answers (the
base drafter trains on them), pool = the next --npool (screens train on these: new to the base, as TAPSQ trained on files
t30b never saw), val = the last --nval (in-distribution held-out). MBPP and HumanEval stay held out entirely.
usage: mksplit.py [--nbase 380 --npool 120 --nval 40]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sb import core as C

MAGIC = os.path.expanduser("~/datasets/deepdraft/magicoder/prompts_py_v1.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbase", type=int, default=380); ap.add_argument("--npool", type=int, default=120)
    ap.add_argument("--nval", type=int, default=40)
    a = ap.parse_args()
    keys = []
    for l in open(MAGIC):
        k = json.loads(l)["key"]
        if os.path.exists(os.path.join(C.DATA, "mc", k + ".npz")):
            keys.append(k)
        if len(keys) >= a.nbase + a.npool + a.nval:
            break
    assert len(keys) == a.nbase + a.npool + a.nval, len(keys)
    s = {"base": keys[:a.nbase], "pool": keys[a.nbase:a.nbase + a.npool], "val": keys[a.nbase + a.npool:],
         "train": keys[:a.nbase], "rule": "prompts_py_v1 order; base / pool / val; MBPP + HumanEval held out"}
    json.dump(s, open(os.path.join(C.DATA, "split.json"), "w"), indent=1)
    print({k: len(v) for k, v in s.items() if isinstance(v, list)})


if __name__ == "__main__":
    main()
