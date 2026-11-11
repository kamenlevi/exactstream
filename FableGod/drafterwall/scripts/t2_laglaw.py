#!/usr/bin/env python3
"""t2_laglaw.py (drafterwall T2, pre-registered in PREREG.md, commit 602d66d). CPU.
(b) unseen files: the MTP head b5 teacher-forced to depth 31 (data/d23h/<set>_b5.npz: rt_i [n, 31] = 1 where its
    argmax is the 125B's token; row r is the check anchored at A = r + 1 + a0, depth index j predicts ids[A + 1 + j]).
    - accuracy by depth band;
    - self-check: the real chain's kept per 32-row check (leading ones of rt[r]) must reproduce TrainLab's 12.55 / 6.23;
    - 'lag-2 everywhere': a drafter right at a position exactly when the reader is right there at depth 0.
(a) confirmation on the 12 map windows per domain (tree3_map): the same with DFlash2, the MTP head, and either-right."""
import numpy as np, os, glob
TD = os.path.expanduser("~/TrainLab/data"); out = []
def say(s=""):
    print(s); out.append(s)
BANDS = [(0, 0), (1, 2), (3, 6), (7, 10), (11, 14), (15, 22), (23, 30)]
def loop_kept(first_ok, chain_ok_fn, n_pos, cap=31):
    """first position index p0 = 0; chain_ok_fn(p) -> list of bools for the draft chain starting at position p."""
    kept = []; p = 0
    while p < n_pos:
        ok = chain_ok_fn(p)[:cap]; k = 0
        while k < len(ok) and ok[k]: k += 1
        kept.append(k + 1); p += k + 1
    return kept
res = {}
say("## (b) unseen: MTP head b5 to depth 31 (TrainLab d23h tables)")
for st in ("code", "mbpp"):
    z = np.load(f"{TD}/d23h/{st}_b5.npz"); nw = len(z["keys"])
    band = {b: [0, 0] for b in BANDS}; real, lag2 = [], []
    for i in range(nw):
        rt = z[f"rt_{i}"].astype(bool); n = rt.shape[0]
        for lo, hi in BANDS:
            band[(lo, hi)][0] += int(rt[:, lo:hi + 1].sum()); band[(lo, hi)][1] += rt[:, lo:hi + 1].size
        # anchors start at gen: row r0 = gen - 1 - a0
        r0 = int(z["gen"][i]) - 1 - int(z["a0"][i]); assert r0 >= 0
        sub = rt[r0:]; m = sub.shape[0]
        real += loop_kept(None, lambda p: [bool(v) for v in sub[p, :min(31, m - p)]], m)
        d0 = sub[:, 0]
        lag2 += loop_kept(None, lambda p: [bool(v) for v in d0[p:p + 31]], m)
    acc = {b: band[b][0] / band[b][1] for b in BANDS}
    res["d23h_" + st] = (acc, float(np.mean(real)), float(np.mean(lag2)))
    say(f"{st}: {nw} windows; MTP top-1 by depth: " + ", ".join(f"{lo}-{hi}: {100 * acc[(lo, hi)]:.1f}" for lo, hi in BANDS))
    say(f"    self-check, real chain kept per 32-row check: {np.mean(real):.2f} (TrainLab: code 12.55, mbpp 6.23); checks {len(real)}")
    say(f"    lag-2 everywhere (right iff right at depth 0): kept per 32-row check {np.mean(lag2):.2f}  (x{np.mean(lag2) / np.mean(real):.2f} the real chain); checks {len(lag2)}")
for st in ("gsm", "mgsm"):
    f = f"{TD}/d24/{st}_b5.npz"
    if not os.path.exists(f): continue
    z = np.load(f); ks = [k for k in z.files if k.endswith("_0")]
    say(f"{st}: d24 keys {ks}")
    tot = {b: [0, 0] for b in BANDS[:5]}
    for i in range(len(z["keys"])):
        a = z[f"rk_{i}"]                                     # rank of the true token: 0 = top-1, 3 = 3 or worse, -1 = no token
        for lo, hi in BANDS[:5]:
            blk = a[:, lo:hi + 1]; tot[(lo, hi)][0] += int((blk == 0).sum()); tot[(lo, hi)][1] += int((blk >= 0).sum())
    res[st] = {b: tot[b][0] / tot[b][1] for b in tot}
    say(f"    MTP top-1 by depth: " + ", ".join(f"{lo}-{hi}: {100 * tot[(lo, hi)][0] / tot[(lo, hi)][1]:.1f}" for lo, hi in BANDS[:5]))

say("\n## (a) confirmation: the 12 map windows per domain (tree3_map), both readers")
for dom in ("code", "ifeval", "multi"):
    K = {k: [] for k in ("dfl", "mtp", "either", "dfl_lag2", "mtp_lag2", "either_lag2", "top3_lag2")}
    for f in sorted(glob.glob(f"{TD}/tree3_map/{dom}/*.npz")):
        z = np.load(f); true = z["true"]; n = true.shape[0]
        d = (z["dfl_t3"][..., 0] == true) & (true >= 0); m = (z["mtp_t3"][..., 0] == true) & (true >= 0)
        t3 = ((z["dfl_t3"] == true[..., None]).any(-1) | (z["mtp_t3"] == true[..., None]).any(-1)) & (true >= 0)
        mm = z["m"]
        for name, tab in (("dfl", d), ("mtp", m), ("either", d | m)):
            K[name] += loop_kept(None, lambda p, tab=tab: [bool(v) for v in tab[p, :min(15, int(mm[p]))]], n, cap=15)
        for name, tab in (("dfl_lag2", d[:, 0]), ("mtp_lag2", m[:, 0]), ("either_lag2", (d | m)[:, 0]), ("top3_lag2", t3[:, 0])):
            K[name] += loop_kept(None, lambda p, tab=tab: [bool(v) for v in tab[p:p + 31]], n, cap=31)
    res[dom] = {k: float(np.mean(v)) for k, v in K.items()}
    say(f"{dom}: kept per 16-row check on the tables: DFlash2 {res[dom]['dfl']:.2f}, MTP {res[dom]['mtp']:.2f}, either-right oracle {res[dom]['either']:.2f}")
    say(f"    lag-2 everywhere, kept per 32-row check: DFlash2 {res[dom]['dfl_lag2']:.2f}, MTP {res[dom]['mtp_lag2']:.2f}, either {res[dom]['either_lag2']:.2f}, either top-3 (a 3-wide tree) {res[dom]['top3_lag2']:.2f}")

ac, rc, lc = res["d23h_code"]; am, rm, lm = res["d23h_mbpp"]
pa = ac[(0, 0)] >= 0.96 and (ac[(0, 0)] - ac[(7, 10)]) >= 0.06
pb = am[(0, 0)] <= 0.95 and (am[(0, 0)] - am[(7, 10)]) >= 0.06
pc1 = lc >= 20; pc2 = res["code"]["either_lag2"] >= 25; pc3 = lm < 16
kill = (ac[(0, 0)] - ac[(7, 10)]) < 0.03 or (am[(0, 0)] - am[(7, 10)]) < 0.03 or lc < 14
verdict = "PASS" if (pa and pb and pc1 and pc2) else ("KILL" if kill else "BETWEEN")
say(f"\nT2 predictions: P-a {pa} (HumanEval+ depth 0 {100 * ac[(0, 0)]:.1f}, depths 7-10 {100 * ac[(7, 10)]:.1f}); P-b {pb} (MBPP {100 * am[(0, 0)]:.1f} -> {100 * am[(7, 10)]:.1f}); "
    f"P-c: lag-2 MTP on HumanEval+ {lc:.2f} >= 20 {pc1}; either on the 12 map windows {res['code']['either_lag2']:.2f} >= 25 {pc2}; MBPP {lm:.2f} < 16 {pc3}")
say(f"T2 LINE (PASS: P-a, P-b and both HumanEval+ parts of P-c; KILL: decay < 3 points or lag-2 keeps < 14): {verdict}")
open(os.path.join(os.path.dirname(__file__), "..", "out", "t2_laglaw.txt"), "w").write("\n".join(out) + "\n")
