"""WEEK98 MODULE 2 (MoonShots, Tue 17 Nov 2026 07:5x): the paper cost of Leap's FAST small-segment path needs the real
pairs-per-segment histogram of the one-call. CPU only: exp32's blocks rebuilt from the decision map (as m2_common /
exp32_bench: layer 20, 32-row blocks of the code windows, km path, 2-D nolsb), grouped like ooo1cq's _GRP1CQ for the
all-resident case: key = (expert, tier) with tier 4 only for a kept rank-0/1 use that reads its LSB (else MSB-3), and a
new segment every 16 pairs of one key (the D8 class). usage: python3 m2_seghist.py   (numpy, seconds)"""
import json
import os

import numpy as np

MAP = os.path.expanduser("~/FableGod/decision_map/out")
LM, R4 = 20, 2
basej = json.load(open(os.path.join(MAP, "baseline.json")))
code = [f for f in sorted(os.listdir(os.path.join(MAP, "xcap"))) if basej[f]["dom"] == "code"]
sizes, nsegs, blocks = [], [], 0
for f in code:
    z = np.load(os.path.join(MAP, "xcap", f))
    tz = np.load(os.path.join(MAP, "bt", "tiers_1c", f))
    a = int(z["a"])
    n = z["ids"].shape[0]
    for b0 in range(0, n - 31, 32):
        ids = z["ids"][b0:b0 + 32, LM].astype(np.int64)
        k4 = tz["keep"][a + b0:a + b0 + 32, LM].astype(bool)
        k3 = tz["keep3"][a + b0:a + b0 + 32, LM].astype(bool)
        kept = k4 | k3
        idk = np.where(kept, ids, ids[:, :1])                 # km path: a dropped use reads its row's rank-0 id
        tier4 = np.zeros_like(kept)
        tier4[:, :R4] = k4[:, :R4]                            # rank 0-1 uses with their LSB read (not nolsb-marked)
        keys = {}
        for m in range(32):
            for r in range(10):
                k = (int(idk[m, r]), 4 if tier4[m, r] else 3)
                keys[k] = keys.get(k, 0) + 1
        seg = []
        for c in keys.values():
            while c > 0:
                seg.append(min(c, 16))
                c -= 16
        sizes += seg
        nsegs.append(len(seg))
        blocks += 1
s = np.array(sizes)
h = np.bincount(s, minlength=17)[1:]
out = dict(blocks=blocks, segs_per_block=float(np.mean(nsegs)), pairs_per_seg=float(s.mean()),
           hist={i + 1: int(h[i]) for i in range(16)}, share_le2=float((s <= 2).mean()), share_le4=float((s <= 4).mean()),
           share_le8=float((s <= 8).mean()), pairs_in_le2=float(s[s <= 2].sum() / s.sum()))
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.expanduser("~/MoonShots/week98/m2/m2_seghist.json"), "w"), indent=1)
