#!/usr/bin/env python3 -I
"""seed_test.py (FableGod, mission 30, idea #5 MTP-seeded block drafting). CPU replay joining D-24's native-head ranks
(rk[pos, 0]: the MTP head's depth-1 rank of the 125B's token at every answer position) with DFlash2's HPD-0 walks on the
same windows (per check: k accepted rows of m, the start A; dtok the chain's drafts, ttok the teacher-forced drafts,
dcf the chain's cf). Per check: chain kept = k + 1 (bonus); seeded kept = 1 + the teacher-forced DFlash2 run from row 1
+ 1 if the MTP token at row 0 is right, else 1 (an upper bound: the seeded continuation is at most the teacher-forced
one). Policies: always seed; seed when the chain's row-0 cf < tau; the oracle pick (the better of the two per check)."""
import numpy as np, os
D = os.path.expanduser("~/TrainLab/data")
SETS = [("code", "code"), ("ifeval", "ifeval"), ("multi", "multi"), ("math", "gsm")]
for wname, dname in SETS:
    w = np.load(f"{D}/d02b_walk_hpd0_{wname}.npz"); d = np.load(f"{D}/d24/{dname}_b5.npz", allow_pickle=True)
    dkeys = {str(k): i for i, k in enumerate(d["keys"])}
    chain = []; seed = []; cf0 = []; mtp_ok = []; row0_ok = []; nmiss = 0
    for i, nm in enumerate(w["names"]):
        key = str(nm).replace(".npz", "")
        if key not in dkeys or f"{i}_checks" not in w.files: nmiss += 1; continue
        j = dkeys[key]; rk = d[f"rk_{j}"]; gen = int(d["gen"][j]); ids = w[f"{i}_ids"]
        ch = w[f"{i}_checks"]; tt = w[f"{i}_ttok"]; dc = w[f"{i}_dcf"]
        for c in range(ch.shape[0]):
            k, m, A = int(ch[c, 0]), int(ch[c, 1]), int(ch[c, 2]); r = A - gen
            if r < 0 or r >= rk.shape[0]: continue
            ok = rk[r, 0] == 0                                     # the MTP head's depth-1 token right at A + 1
            run = 0
            for jj in range(1, m):
                if A + jj + 1 < len(ids) and tt[c, jj] == ids[A + jj + 1]: run += 1
                else: break
            chain.append(k + 1); seed.append((1 + run + 1) if ok else 1); cf0.append(float(dc[c, 0])); mtp_ok.append(bool(ok)); row0_ok.append(k >= 1)
    chain, seed, cf0, mtp_ok, row0_ok = map(np.array, (chain, seed, cf0, mtp_ok, row0_ok))
    n = len(chain)
    line = f"  {wname}: {n} checks ({nmiss} windows unmatched); chain kept {chain.mean():.2f}; MTP depth-1 right at the block start {mtp_ok.mean():.3f}, DFlash2's row 0 right {row0_ok.mean():.3f}, both wrong {np.mean(~mtp_ok & ~row0_ok):.3f}, MTP right where DFlash2 wrong {np.mean(mtp_ok & ~row0_ok):.3f}; always seed {seed.mean():.2f} ({100 * (seed.mean() / chain.mean() - 1):+.1f} %); oracle pick {np.maximum(chain, seed).mean():.2f} ({100 * (np.maximum(chain, seed).mean() / chain.mean() - 1):+.1f} %)"
    for tau in (0.5, 0.8, 0.9, 0.95):
        pick = np.where(cf0 < tau, seed, chain); line += f"; seed if cf0 < {tau}: {pick.mean():.2f} ({100 * (pick.mean() / chain.mean() - 1):+.1f} %, {100 * np.mean(cf0 < tau):.0f} % of checks)"
    print(line)
