#!/usr/bin/env python3
"""t1_secondlook.py (drafterwall T1, pre-registered in PREREG.md, commit 602d66d). CPU.
After a live check's first miss, were the later rows fed the right tokens?
Data: ~/TrainLab/data/d02b_walk_hpd0_<dom>.npz (DFlash2 L1-20, official loop, block 16): per window i: i_ids (tokens),
i_checks [nc, 3] = (k accepted, m checkable, A anchor), i_dtok [nc * 15] the 15 drafts (draft j predicts ids[A + j + 1]).
Metric: over checks with a first miss at draft k and at least one later checkable draft: tail alignment = share of
drafts j in k+1 .. m-1 equal to ids[A + j + 1]; 'next 3 right' = drafts k+1 .. k+3 all right (needs m >= k + 4)."""
import numpy as np, os, sys
TD = os.path.expanduser("~/TrainLab/data")
out = []
def say(s):
    print(s); out.append(s)
res = {}
for dom in ("code", "mbpp", "math", "ifeval", "multi"):
    f = f"{TD}/d02b_walk_hpd0_{dom}.npz"
    z = np.load(f); names = list(z["names"])
    nchk = nmiss = 0; tail = [0, 0]; n3 = [0, 0]; bydist = {}; kept = []; run_after = []
    for i, nm in enumerate(names):
        if f"{i}_dtok" not in z.files: continue
        ids = z[f"{i}_ids"]; ch = z[f"{i}_checks"]; dt = z[f"{i}_dtok"].reshape(-1, 15)
        for c in range(ch.shape[0]):
            k, m, A = int(ch[c, 0]), int(ch[c, 1]), int(ch[c, 2])
            nchk += 1; kept.append(k + 1)
            if k >= m: continue                                # the whole checkable block was right
            # sanity: draft k is wrong, drafts < k right
            assert dt[c, k] != ids[A + k + 1], (dom, nm, c)
            js = np.arange(k + 1, m)
            js = js[A + js + 1 < len(ids)]
            if len(js) == 0: continue
            nmiss += 1
            eq = dt[c, js] == ids[A + js + 1]
            tail[0] += int(eq.sum()); tail[1] += len(eq)
            for d, v in enumerate(eq, 1):
                b = bydist.setdefault(d, [0, 0]); b[0] += int(v); b[1] += 1
            if len(eq) >= 3:
                n3[1] += 1; n3[0] += int(eq[:3].all())
            r = 0
            for v in eq:
                if v: r += 1
                else: break
            run_after.append(r)
    ta = 100 * tail[0] / max(tail[1], 1)
    res[dom] = ta
    say(f"{dom}: {len(names)} windows, {nchk} checks, kept per check {np.mean(kept):.2f}; checks with a first miss and a tail {nmiss}; "
        f"tail alignment {ta:.1f} % of {tail[1]} positions; next 3 right {100 * n3[0] / max(n3[1], 1):.1f} % of {n3[1]}; "
        f"mean right run right after the miss {np.mean(run_after):.2f}")
    say("    by distance after the miss: " + " ".join(f"d{d}:{100 * bydist[d][0] / bydist[d][1]:.0f}%" for d in sorted(bydist) if d <= 10))
v = "PASS" if (res["code"] >= 40 or res["ifeval"] >= 30 or res["multi"] >= 30) else ("KILL" if (res["code"] < 25 and res["ifeval"] < 15 and res["multi"] < 15) else "BETWEEN")
say(f"T1 LINE (PASS: code >= 40 % or IFEval / multi >= 30 %; KILL: code < 25 % and IFEval < 15 % and multi < 15 %): "
    f"code {res['code']:.1f} %, IFEval {res['ifeval']:.1f} %, multi {res['multi']:.1f} % -> {v}")
open(os.path.join(os.path.dirname(__file__), "..", "out", "t1_secondlook.txt"), "w").write("\n".join(out) + "\n")
