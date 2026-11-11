#!/usr/bin/env python3
"""s0_align.py (drafterwall, exploratory, CPU): what do DFlash2's block drafts look like AFTER their first miss?
Data: ~/TrainLab/data/tree3_map/<dom>/*.npz (every answer position is an anchor A; dfl_own[A, 0..14] = the live block
draft from A; true[A, 0..14] = the 125B's tokens). For every anchor with a first miss at depth k < m-1:
  - Hamming accuracy of the later drafts own[j] == true[j], by distance d = j - k (1..);
  - shift test: does own[j] == true[j + s] for s in (-2, -1, +1, +2) hold better than s = 0 (insert / delete misses)?
  - the block's 2nd miss distance if aligned.
Exploratory read (no PASS / KILL line): it sizes the opportunity for any 'second look' scheme."""
import numpy as np, glob, os, sys
from collections import defaultdict
TD = os.path.expanduser("~/TrainLab/data/tree3_map")
for dom in ("code", "ifeval", "multi"):
    files = sorted(glob.glob(f"{TD}/{dom}/*.npz"))
    nb = 0; kfirst = []; ham = defaultdict(lambda: [0, 0]); shift = defaultdict(lambda: [0, 0]); post_all = [0, 0]
    aligned_blocks = 0; blocks_with_tail = 0; run2 = []
    full_right = 0
    for f in files:
        z = np.load(f)
        own, true, m = z["dfl_own"], z["true"], z["m"]
        n = own.shape[0]
        for a in range(n):
            mm = int(m[a])
            if mm < 15: continue                      # full blocks only
            eq = own[a, :15] == true[a, :15]
            nb += 1
            if eq.all(): full_right += 1; kfirst.append(15); continue
            k = int(np.argmin(eq)); kfirst.append(k)
            tail = eq[k + 1:]
            if len(tail) == 0: continue
            blocks_with_tail += 1
            post_all[0] += int(tail.sum()); post_all[1] += len(tail)
            for d, v in enumerate(tail, 1):
                ham[d][0] += int(v); ham[d][1] += 1
            # shift test over the tail
            for s in (-2, -1, 0, 1, 2):
                js = np.arange(k + 1, 15); jt = js + s
                ok = (jt >= 0) & (jt < 15)
                if ok.sum() == 0: continue
                shift[s][0] += int((own[a, js[ok]] == true[a, jt[ok]]).sum()); shift[s][1] += int(ok.sum())
            # aligned = the next 3 drafts after the miss are all right (or the tail if shorter)
            w = tail[:3]
            if w.all():
                aligned_blocks += 1
                r = 0
                for v in tail:
                    if v: r += 1
                    else: break
                run2.append(r)
    kf = np.array(kfirst)
    print(f"\n== {dom}: {len(files)} windows, {nb} full blocks; kept per 16-row check (k+1, capped) {np.mean(np.minimum(kf, 15) + 1):.2f}; all-15-right {100 * full_right / nb:.1f} %")
    print(f"   after the first miss: Hamming right {100 * post_all[0] / max(post_all[1], 1):.1f} % of {post_all[1]} tail positions")
    print("   by distance d after the miss: " + " ".join(f"d{d}:{100 * ham[d][0] / ham[d][1]:.0f}%" for d in sorted(ham) if d <= 12))
    print("   shift test own[j] == true[j+s]: " + " ".join(f"s={s:+d}:{100 * shift[s][0] / max(shift[s][1], 1):.1f}%" for s in (-2, -1, 0, 1, 2)))
    print(f"   blocks whose next 3 drafts after the miss are right: {100 * aligned_blocks / max(blocks_with_tail, 1):.1f} % of {blocks_with_tail}; mean right-run after the miss in those {np.mean(run2) if run2 else float('nan'):.2f}")
