#!/usr/bin/env python3 -I
"""draft_hidden.py (FableGod, mission 29). CPU replay on DFlash2 L1-20's HPD-0 walks (~/TrainLab/data/d02b_walk_hpd0_{code,
ifeval,math,multi}.npz: per check (k accepted rows, m rows, start) and the drafter's cf per row) of what a hidden draft can
cover. The next block's draft depends on the check's outcome (the accepted length k) and on the bonus token at the last
accepted row; a pre-drafted candidate is usable only if both its anchor (k) and its bonus guess are right.
  anchors ranked by the drafter's own first-miss probability (MTP Head's ANCHORCOV): P(all) = prod cf_i;
  P(first miss at j) = prod_{i<j} cf_i x (1 - cf_j); coverage at N = the outcome is among the top N anchors (N = 3 is
  capped at ANCHORCOV's measured value).
  bonus guess: at the all-accepted anchor the drafter's own next token (its accuracy at depth 15, [K]); at a first-miss
  anchor the drafter's second choice at the miss row (chat 0.34 [K R-61N]; code 0.45, math 0.40 [G]; a sweep).
Cycle per check at the floor engine: check GPU = dense 17 ms + 1.0 ms per row (code 16 rows, math 10, chat 4), x1.02
beside the ANE; the commit 3.0 ms today, 0.5 when hidden (behind the fallback draft, or pipelined layer by layer into
the next check); CPU spans 2.0 (bookkeeping, PLE staging, the first segment's build); the exposed draft: 6.3 ms (D7b)
today; a usable candidate 0.3 ms (the hand-off); the fallback 6.3. Bounds: hand-off 0.6, commit 1.0."""
import numpy as np, os
W = os.path.expanduser("~/TrainLab/data")
SETS = [("code", 16, 10.54), ("math", 10, 6.61), ("ifeval", 4, 2.03), ("multi", 4, 2.03)]
P2 = dict(code=0.527, math=0.502, ifeval=0.266, multi=0.181)      # the true token = the drafter's next candidate at its first miss [K lattice_rescue, selector order]
P3 = dict(code=0.692, math=0.641, ifeval=0.399, multi=0.275)      # within its next two [K]
ANCH = dict(code=0.782, math=0.672, ifeval=0.799, multi=0.804)

def load(name):
    z = np.load(f"{W}/d02b_walk_hpd0_{name}.npz"); out = []
    for i in range(len(z["names"])):
        if f"{i}_checks" not in z.files: continue
        ch = z[f"{i}_checks"]; cf = z[f"{i}_dcf"]
        for c in range(ch.shape[0]):
            k, m = int(ch[c, 0]), int(ch[c, 1]); out.append((k, m, cf[c, :m].astype(np.float64)))
    return out

print("## 1. Outcomes and anchor coverage (ANCHORCOV's N=3 by first-miss probability: code 0.782, math 0.672, ifeval 0.799, multi 0.804 [K])")
R = {}
for name, rows, kept in SETS:
    checks = load(name); n = len(checks)
    allacc = np.mean([k == m for k, m, _ in checks]); cov = {2: 0, 3: 0, 4: 0}
    for k, m, cf in checks:
        cfc = np.clip(cf, 1e-6, 1 - 1e-6)
        probs = {m: float(np.prod(cfc))}
        for j in range(m): probs[j] = float(np.prod(cfc[:j]) * (1 - cfc[j]))
        order = sorted(probs, key=lambda a: -probs[a])
        for N in cov:
            if k in order[:N]: cov[N] += 1
    reach = [(k, m) for k, m, _ in checks if m >= 15 and k >= 14]
    d15 = float(np.mean([k >= 15 for k, m in reach])) if reach else float("nan")
    covd = {N: cov[N] / n for N in cov}; covd[3] = min(covd[3], ANCH[name])
    note = ""
    if np.isnan(d15): d15 = 0.667; note = " (no check reaches row 14: ifeval's rate used as the chat proxy)"
    R[name] = dict(n=n, allacc=allacc, cov=covd, d15=d15, rows=rows, kept=kept)
    print(f"  {name}: {n} checks; all accepted {allacc:.3f}; anchors cover N=2 {cov[2]/n:.3f}, N=3 {cov[3]/n:.3f} (used {covd[3]:.3f}), N=4 {cov[4]/n:.3f}; the drafter's depth-15 accuracy (the bonus guess at the all-accepted anchor) {d15:.3f}{note}")

def cycle(rows, exposed, ane, commit):
    gpu = 17 + 1.0 * rows
    return gpu * (1.02 if ane else 1.0) + commit + 2.0 + exposed

print("\n## 2. Usable coverage, the exposed draft and the cycle per design [P on K; G for the second-choice rate]")
print("  | set | design | candidates | usable coverage | exposed draft ms (expected / bound) | cycle ms today -> design (bound) | tok/s today -> design (gain, bound, halved) |")
print("  |---|---|---|---|---|---|---|")
for name, rows, kept in SETS:
    r = R[name]; p2 = P2[name]; pa = r["d15"]; cov3 = r["cov"][3]; cov4 = r["cov"][4]
    base = cycle(rows, 6.3, False, 3.0); tps0 = 1e3 * kept / base
    D = []
    def add(label, N, usable, ane=True, exposed=None, bound=None):
        if exposed is None:
            exposed = usable * 0.3 + (1 - usable) * 6.3; bound = usable * 0.6 + (1 - usable) * 6.3
        D.append((label, N, usable, exposed, bound, ane))
    add("A  ANE-3 anchors (cloud's, first-miss-probability anchors) + D", 3, r["allacc"] * pa + (cov3 - r["allacc"]) * p2)
    add("A4 ANE-4 anchors + D", 4, r["allacc"] * pa + (cov4 - r["allacc"]) * p2)
    add("A5 ANE-5: 3 anchors, #2 and #3 at the two miss anchors + D", 5, r["allacc"] * pa + (cov3 - r["allacc"]) * P3[name])
    add("A7 ANE-7: 4 anchors, #2 and #3 at the three miss anchors + D", 7, r["allacc"] * pa + (cov4 - r["allacc"]) * P3[name])
    for pl in (0.6, 0.8):
        add(f"L  ANE-3 + a bonus guess at {pl} at the miss anchors (a lens or a better second choice) [G] + D", 3, r["allacc"] * pa + (cov3 - r["allacc"]) * pl)
    add("D  the commit hidden behind the serial draft (no ANE)", 0, 0.0, ane=False, exposed=6.3, bound=6.3)
    add("B  the MTP head's depth-1 after the check, hidden behind the pipelined commit (0.5 exposed) + the serial draft", 0, 0.0, ane=False, exposed=6.8, bound=7.8)
    for label, N, usable, exposed, bound, ane in D:
        c = cycle(rows, exposed, ane, 0.5); cb = cycle(rows, bound, ane, 1.0)
        tps = 1e3 * kept / c; g = tps / tps0 - 1; gb = (1e3 * kept / cb) / tps0 - 1
        print(f"  | {name} | {label} | {N} | {usable:.2f} | {exposed:.1f} / {bound:.1f} | {base:.1f} -> {c:.1f} ({cb:.1f}) | {tps0:.0f} -> {tps:.0f} ({100 * g:+.1f} %, {100 * gb:+.1f}, halved {100 * g / 2:+.1f} %) |")

print("\n## 3. Sensitivity of A5 + D to the top-3 rescue rate at the miss rows (measured: code 0.69, math 0.64, ifeval 0.40, multi 0.28) (exposed draft ms, averaged over outcomes)")
for name, rows, kept in SETS:
    r = R[name]; line = f"  {name}: "
    for p3 in (0.3, 0.45, 0.6, 0.8, 0.95):
        u = r["allacc"] * r["d15"] + (r["cov"][3] - r["allacc"]) * p3
        line += f"top-3 rescue {p3}: usable {u:.2f}, exposed {u * 0.3 + (1 - u) * 6.3:.1f} ms; "
    print(line)
print("\n## 4. What <= 1 ms of average exposed draft needs: usable >= (6.3 - 1.0) / (6.3 - 0.3) = 0.88 of outcomes, or a fallback draft <= 1.0 + (1 - u) ... i.e. at u 0.57 a fallback of <= 1.5 ms")
