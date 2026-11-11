"""D-09 DSCALE subsets (MTP Head, Wed 10-28 22:1x; REPORT 'D-09 DSCALE'): nested 1/4 and 1/2 of L1-20's 99 training
windows (~/Leap/data/l120, one window per prompt: 80 HumanEval+ 'he' + 29 GSM8K 'gsm'), stratified by domain, seed 9;
all val windows kept (the same hold-outs). Writes symlink folders data/d09_q and data/d09_h (Leap's files are only read)
and prints each set's train windows and anchors (= generated tokens the trainer learns from) as a share of the full set.
Usage: python sticky/d09_subsets.py"""
import glob, math, os
import numpy as np

L = os.path.expanduser("~/Leap/data/l120")
T = os.path.expanduser("~/TrainLab/data")
wins = {}
for p in sorted(glob.glob(os.path.join(L, "*.npz"))):
    z = np.load(p)
    asst = z["asst"].astype(bool)
    n = z["tk"].shape[0]
    wins[os.path.basename(p)] = (str(z["split"]), sum(1 for a in range(n) if asst[a] and asst[a + 1]))
train = {d: sorted(k for k, (s, _) in wins.items() if s == "train" and k.startswith(d + "_")) for d in ("he", "gsm")}
val = sorted(k for k, (s, _) in wins.items() if s == "val")
rng = np.random.default_rng(9)
order = {d: [ks[i] for i in rng.permutation(len(ks))] for d, ks in train.items()}
full = sum(wins[k][1] for ks in train.values() for k in ks)
print(f"full: {sum(len(v) for v in train.values())} train windows ({len(train['he'])} he + {len(train['gsm'])} gsm), "
      f"{full} anchors; {len(val)} val windows")
for tag, frac in (("q", 0.25), ("h", 0.5)):
    pick = [k for d in ("he", "gsm") for k in order[d][:math.ceil(frac * len(order[d]))]]
    out = os.path.join(T, f"d09_{tag}")
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        os.remove(os.path.join(out, f))
    for k in pick + val:
        os.symlink(os.path.join(L, k), os.path.join(out, k))
    na = sum(wins[k][1] for k in pick)
    print(f"d09_{tag}: {len(pick)} train windows ({sum(k.startswith('he_') for k in pick)} he + "
          f"{sum(k.startswith('gsm_') for k in pick)} gsm), {na} anchors = {na / full:.3f} of the full set; + {len(val)} val")
