"""P-OFF (REPORT 'P-OFF'): the training folder = symlinks to L1-20's 109 windows (~/Leap/data/l120, incl. its 10 val) plus
the first extracted off-policy windows (ext/poff/poff_windows.jsonl order = the seed-0 shuffle) until --tokens assistant
tokens. Usage: python sticky/poff_data.py [--tokens 160000] [--out ~/TrainLab/ext/poff/train160]"""
import argparse, glob, json, os
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, default=160_000); ap.add_argument("--out", default="~/TrainLab/ext/poff/train160")
a = ap.parse_args()
out, X = os.path.expanduser(a.out), os.path.expanduser("~/TrainLab/ext/poff/x")
os.makedirs(out, exist_ok=True)
for p in glob.glob(os.path.expanduser("~/Leap/data/l120/*.npz")):
    q = os.path.join(out, os.path.basename(p))
    if not os.path.exists(q):
        os.symlink(p, q)
n, k, on_n, on_d = 0, 0, 0, 0
for line in open(os.path.expanduser("~/TrainLab/ext/poff/poff_windows.jsonl")):
    r = json.loads(line)
    p = os.path.join(X, r["key"] + ".npz")
    if not os.path.exists(p):
        raise SystemExit(f"missing extraction {p}: stop (the folder would not be the pre-registered prefix)")
    q = os.path.join(out, r["key"] + ".npz")
    if not os.path.exists(q):
        os.symlink(p, q)
    z = np.load(p)
    n += int(z["asst"].sum()); k += 1
    m_ = len(z["ids"]) - 1                                   # the extractor's own on-path count, over the whole prefix
    msk = z["asst"][1:m_ + 1].astype(bool)
    hit = z["tk"][:m_, 0] == z["ids"][1:m_ + 1]
    on_n += int(hit[msk].sum()); on_d += int(msk.sum())
    if n >= a.tokens:
        break
print(f"P-OFF data: {out}: L1-20's windows + {k} off-policy windows, {n} assistant tokens; "
      f"on-path rate (assistant) over these {k} windows {on_n / max(1, on_d):.4f}")
