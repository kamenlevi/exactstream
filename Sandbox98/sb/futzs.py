"""BREAK98 Screen 1, FAR's zero-shot print: the frozen 2B's OWN top-1 at placeholder offset p (stored by sb/futgen.py as
top1) against the true token it predicts, ids[s + p + 1] (the head's depth-p target), per offset 1..8, MBPP-50 / HE-30.
Run before a variant's states are deleted. usage: futzs.py --variant A [--out results/b98s1_zeroshot_A.json]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx
from sb import core as C
from sb.futstate import K, fut_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True); ap.add_argument("--nm", type=int, default=50); ap.add_argument("--nh", type=int, default=30)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = {"variant": a.variant, "sets": {}}
    for st, n in (("mbpp", a.nm), ("he", a.nh)):
        r = np.zeros(K, np.int64); v = np.zeros(K, np.int64)
        for f in C.files(st)[:n]:
            ids = np.asarray(np.load(f)["ids"], np.int64)
            z = mx.load(fut_path(a.variant, f))
            anc = np.array(z["anc"]).astype(np.int64); t1 = np.array(z["top1"]).astype(np.int64)
            for p in range(1, K + 1):
                ok = anc + p + 1 <= len(ids) - 1
                r[p - 1] += int((t1[ok, p - 1] == ids[anc[ok] + p + 1]).sum()); v[p - 1] += int(ok.sum())
        out["sets"][st] = {"acc": (r / np.maximum(v, 1)).tolist(), "n": v.tolist()}
        print(f"zero-shot {a.variant} {st}: the 2B's own top-1 at placeholder offset 1..8 = "
              + " ".join(f"{100 * x:.1f}" for x in out["sets"][st]["acc"]) + f" % (n {v[0]}..{v[-1]})", flush=True)
    json.dump(out, open(a.out or f"results/b98s1_zeroshot_{a.variant}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
