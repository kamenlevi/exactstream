"""E2 / D2 training sets (MTP Head, Mon 11-02; REPORT 'E2 / D2 DATASLOPE-MBPP'): L1-20's own data (~/Leap/data/l120, all 109
windows: 99 train + the 10 val) plus the 125B's MBPP answers (data/dlog/d1102mbpp, index order = capture order), nested:
x2 = the first MBPP files adding >= 16,171 code answer tokens (one doubling of L1-20's code), x4 = the first adding >= 48,513
(two). Symlinked into data/e2_x2/, data/e2_x4/ (the trainer reads every *.npz in a folder). Exit 0: both built; 2: x2 only;
1: not enough for x2."""
import json, os, sys

L120 = os.path.expanduser("~/Leap/data/l120")
D = os.path.expanduser("~/TrainLab/data/dlog/d1102mbpp")
CODE = 16171
need = {"x2": CODE, "x4": 3 * CODE}
idx = [json.loads(l) for l in open(os.path.join(D, "index.jsonl"))]
idx = [r for r in idx if r["lang"] == "code" and os.path.exists(os.path.join(D, r["key"].replace("/", "__") + ".npz"))]
have = sum(r["n"] for r in idx)
built = []
for arm, n_need in need.items():
    out = os.path.expanduser(f"~/TrainLab/data/e2_{arm}")
    if have < n_need:
        print(f"e2_sets: {arm} needs {n_need} MBPP answer tokens, captured {have}: not built")
        continue
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        os.remove(os.path.join(out, f))
    for f in sorted(os.listdir(L120)):
        if f.endswith(".npz"):
            os.symlink(os.path.join(L120, f), os.path.join(out, f))
    tot = nf = 0
    for r in idx:
        if tot >= n_need:
            break
        fn = r["key"].replace("/", "__") + ".npz"
        os.symlink(os.path.join(D, fn), os.path.join(out, fn))
        tot += r["n"]; nf += 1
    print(f"e2_sets: {arm} = L1-20's 109 windows + {nf} MBPP answers / {tot} code answer tokens (need {n_need}) -> {out}")
    built.append(arm)
sys.exit(0 if built == ["x2", "x4"] else 2 if built == ["x2"] else 1)
