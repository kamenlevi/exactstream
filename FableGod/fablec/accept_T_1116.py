"""THINK_1116 #1 WALL A, tiny CPU check on the map's 12 code windows (old DFlash2 HPD-0 walks; 125B top-8 log-probs).
Per band (walk-strict, at the walk's check starts): greedy accuracy vs the expected acceptance of distribution-exact
speculative sampling at temperature T, accept with prob min(1, p_T(d) / q(d)), q(d) = the drafter's own confidence (dcf)
for its draft d, p_T(d) = the 125B's softmax(logits / T) restricted to its top-8 (0 if d is outside); also the per-band share
of rows that are near-ties (125B margin < 1 nat) and the miss share that is "forgivable" (d = 125B #2 within delta nats).
A 2-point picture of q (the draft vs everything else), so it is an estimate, not a measurement of the tree engine."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/FableGod/decision_map")); sys.path.insert(0, os.path.expanduser("~/FableGod/fablec"))
import dm_gate
from bytes10_replay import windows
base = json.load(open(os.path.join(dm_gate.OUT, "baseline.json"))); W = windows()
BANDS = [(0, 0), (1, 2), (3, 6), (7, 10), (11, 14), (15, 22), (23, 30)]
TS = (0.3, 0.5, 0.7, 1.0)
for dom in ("code", "ifeval"):
    Z = np.load(os.path.join(dm_gate.TD, dm_gate.WALK[dom])); names = [str(x) for x in Z["names"]]
    acc = {b: [] for b in BANDS}; sam = {T: {b: [] for b in BANDS} for T in TS}; tie = {b: [] for b in BANDS}; forg = {b: [0, 0] for b in BANDS}
    for w in W[dom]:
        nm = w["nm"]; i = names.index(nm); b_ = base[nm]; a = int(b_["a"]); tk = np.array(b_["tk"]); lp = np.array(b_["lp"]); n = len(tk)
        checks = Z[f"{i}_checks"].tolist(); dtok = Z[f"{i}_dtok"]; dcf = Z[f"{i}_dcf"]
        for ci, (k, m, A) in enumerate(checks):
            r0 = A - a
            for j in range(min(m, dtok.shape[1])):          # walk-strict: rows 0..k were drafted while the chain was alive (k < m: row k is the miss)
                if j > k: break
                r = r0 + j
                if not (0 <= r < n): break
                d = int(dtok[ci, j]); q = float(dcf[ci, j]); right = d == int(tk[r, 0])
                band = next(bb for bb in BANDS if bb[0] <= j <= bb[1]) if j <= 30 else None
                if band is None: break
                acc[band].append(right); tie[band].append(lp[r, 0] - lp[r, 1] < 1.0)
                if not right:
                    forg[band][1] += 1; forg[band][0] += int(d == int(tk[r, 1]) and lp[r, 0] - lp[r, 1] < 1.0)
                pos = np.where(tk[r] == d)[0]
                for T in TS:
                    pT = np.exp((lp[r] - lp[r, 0]) / T); pT = pT / pT.sum()        # top-8 renormalized at T
                    pd = float(pT[pos[0]]) if len(pos) else 0.0
                    sam[T][band].append(min(1.0, pd / max(q, 1e-6)))
    print(f"{dom}: band | n | greedy right % | sampling acceptance % at T 0.3 / 0.5 / 0.7 / 1.0 | near-tie rows (< 1 nat) % | forgivable misses (d = #2 within 1 nat) % of misses")
    for b in BANDS:
        if not acc[b]: continue
        print(f"  {b[0]:2d}-{b[1]:2d} | {len(acc[b]):5d} | {100 * np.mean(acc[b]):5.1f} | " + " / ".join(f"{100 * np.mean(sam[T][b]):5.1f}" for T in TS)
              + f" | {100 * np.mean(tie[b]):4.1f} | {100 * forg[b][0] / max(forg[b][1], 1):4.1f} ({forg[b][1]})")
