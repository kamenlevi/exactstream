"""How many distinct experts does a 4-row verify pass touch per layer? Consecutive decode tokens from MoEPrefetch's
125B traces (top-20 recorded; top-10 used) stand in for the verify rows. 40 = no sharing; fewer = rows share experts,
which an expert-sorted gather (v2-8) would read once instead of once per row."""
import glob, os
import numpy as np
fs = sorted(glob.glob(os.path.expanduser("~/MoEPrefetch/data/traces125/*_800.npz")))
for f in fs:
    d = np.load(f)
    u = []
    for k in d.files:                                    # one [tokens, 20] array per layer
        t = d[k][:, :10]
        u += [len(set(t[i:i + 4].ravel().tolist())) for i in range(0, t.shape[0] - 4, 4)]
    print(os.path.basename(f), "distinct experts per 4-row window (of 40):", round(float(np.mean(u)), 1))
