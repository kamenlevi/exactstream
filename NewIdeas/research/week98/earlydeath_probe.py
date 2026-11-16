"""EARLY DEATH probe, stage 1 (NewIdeas, Mon 16 Nov 16:3x). CPU only, read-only. EARLYDEATH_PREREG.md amendment 1.
Stage 1 (no capture needed): the CHAIN version's units, the REAL first-miss positions, the perfect-flag byte bound on the
real walk, and the DRAFTER-ONLY flag baseline (the bar a target-state probe must beat by 10 points). HE+ only: the
decision map's 24 code windows are the same files as BigMoE's heplus24 tree dump, so per-row routing is available.
Stage 2 (Wed, after BigMoE's capture): add P-lens / P-tuned / P-comb scores from the layer-8/16/24 states.
Chain from anchor a0: draft j (1-based) sits at position a0 + 1 + j; row r of the check sits at position a0 + 1 + r.
Draft 1 = the root's top-1; draft j >= 2 = the top-1 next token of the on-true depth-(j-2) node (correct iff its
ynext_rank == 0). kept = j* (the first rejected draft's index) or 32.
usage: taskpolicy -b ~/BigMoE/.venv/bin/python -I -B earlydeath_probe.py"""
import os, sys, json
import numpy as np
sys.dont_write_bytecode = True
D = os.path.expanduser("~/BigMoE/data/deepdraft/ceiling/tree_t30b_heplus24_n127k8.npz")
MAP = os.path.expanduser("~/FableGod/decision_map")
OUT = os.path.expanduser("~/NewIdeas/research/week98/earlydeath_stage1")
NL, ROWS, S_, LAM = 48, 32, 0.98, 1.8e-2
B4, B3 = 2.784, 2.064
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)


def tier_mb(ids, w, rows):
    """expert MB per layer, C6 rule on these check rows (weights s^r by row order)"""
    rw = S_ ** np.arange(len(rows)); out = np.zeros(NL)
    for l in range(NL):
        u, inv = np.unique(ids[rows, l].ravel(), return_inverse=True)
        imp = np.bincount(inv, weights=(w[rows, l] ** 2 * rw[:, None]).ravel(), minlength=len(u))
        n4 = int((imp > 2.775 * LAM).sum()); n3 = int((imp > 0.8457 * LAM).sum()) - n4
        out[l] = n4 * B4 + n3 * B3
    return out


def main():
    z = np.load(D, allow_pickle=True)
    files = [str(f) for f in z["files"]]
    fi, an, dp, ot = z["fi"], z["an"], z["depth"], z["on_true"].astype(bool)
    yr, tp, yy = z["ynext_rank"], z["top_p"][:, 0], z["y"]
    a_fi, a_an, rti, rtp = z["a_fi"], z["a_an"], z["root_top_ids"][:, 0], z["root_top_p"][:, 0]
    # per anchor: the on-true node per depth -> (ynext_rank, top_p[0]); the depth-0 true token y
    key = fi * 10_000_000 + an
    order = np.lexsort((dp, ~ot, key)); key_s = key[order]
    b = np.flatnonzero(np.append(True, np.diff(key_s) != 0)); b = np.append(b, len(key_s))
    A = {}
    for b0, b1 in zip(b[:-1], b[1:]):
        idx = order[b0:b1]; idt = idx[ot[idx]]
        dd = {int(dp[i]): (int(yr[i]), float(tp[i])) for i in idt}
        y0 = int(yy[idx[dp[idx] == 0][0]]) if (dp[idx] == 0).any() else -1
        A[int(key_s[b0])] = (dd, y0)
    root = {int(f) * 10_000_000 + int(a): (int(t), float(p)) for f, a, t, p in zip(a_fi, a_an, rti, rtp)}

    def chain(k):
        """(kept, conf[1..31]) for anchor key k"""
        dd, y0 = A.get(k, ({}, -1)); t1, p1 = root[k]
        conf = np.ones(ROWS); conf[1] = p1
        kept = None
        if t1 != y0:
            kept = 1
        for j in range(2, ROWS):
            nd = dd.get(j - 2)
            if nd is None:
                kept = kept or j - 1 if kept is None else kept
                break
            conf[j] = nd[1]
            if kept is None and nd[0] != 0:
                kept = j
        return (kept if kept is not None else ROWS), conf

    say("EARLY DEATH stage 1 (HE+ heplus24 = the decision map's 24 code windows; C6 tiers, real pieces)")
    checks = []          # (file, a0, kept, conf)
    for f_i, f in enumerate(files):
        ks = sorted(int(a) for a in a_an[a_fi == f_i])
        if not ks:
            continue
        p, last = ks[0], ks[-1]
        while p <= last:
            k = f_i * 10_000_000 + p
            if k not in root:
                p += 1; continue
            kept, conf = chain(k)
            checks.append((f, p, kept, conf)); p += kept
    kept_all = np.array([c[2] for c in checks])
    say(f"  chain walk: {len(checks)} checks, mean kept {kept_all.mean():.2f} (tree-32 walk 16.26; chain-32 t30b 13.77)")
    # bytes: rows = positions a0+1 .. a0+32 in the map's row index
    cache = {}
    tot_full, tot_perf = 0.0, {8: 0.0, 16: 0.0, 24: 0.0}
    per = []             # per check: (MB per layer for prefix k = 1..32)
    n_ok = 0
    for f, a0, kept, conf in checks:
        if f not in cache:
            zz = np.load(os.path.join(MAP, "out_code24", "scores", f))
            cache[f] = (zz["ids"].astype(np.int64), zz["scores"].astype(np.float64))
        ids, w = cache[f]
        r0 = a0 + 1
        if r0 + ROWS > ids.shape[0]:
            continue
        M = np.zeros((ROWS + 1, NL))
        for k in range(1, ROWS + 1):
            M[k] = tier_mb(ids, w, np.arange(r0, r0 + k))
        per.append((M, kept, conf)); n_ok += 1
    full = np.mean([M[ROWS].sum() for M, _, _ in per])
    say(f"  {n_ok} checks with full rows in the map; C6 expert MB per 32-row check {full:.0f}")
    res = {"checks": n_ok, "full_MB": float(full), "mean_kept": float(kept_all.mean())}
    for L in (8, 16, 24):
        ed = np.mean([M[ROWS, :L + 1].sum() + M[kept, L + 1:].sum() for M, kept, _ in per])
        res[f"perfect_L{L}"] = float(1 - ed / full)
        say(f"  PERFECT flag after layer {L + 1:2d} (real first misses): expert bytes -{100 * (1 - ed / full):.1f} %")
    # drafter-only flag: flag at the first draft j with conf_j < tau -> keep rows 0..j-1 for layers > L
    L = 16
    perf = res["perfect_L16"]
    curve = []
    for tau in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        sv, kl = [], []
        for M, kept, conf in per:
            js = [j for j in range(1, ROWS) if conf[j] < tau]
            jf = js[0] if js else ROWS
            k_rows = min(jf, ROWS)
            sv.append(M[ROWS, :L + 1].sum() + M[k_rows, L + 1:].sum())
            kl.append(min(jf, kept) if jf < kept else kept)
        s_ = 1 - np.mean(sv) / full; loss = 1 - np.mean(kl) / np.mean([k for _, k, _ in per])
        curve.append((tau, s_, loss))
        say(f"  DRAFTER-ONLY flag tau {tau:.2f}: expert bytes -{100 * s_:5.1f} % ({100 * s_ / perf:5.1f} % of perfect), "
            f"kept loss {100 * loss:5.2f} %")
    ok = [c for c in curve if c[2] <= 0.02]
    best = max(ok, key=lambda c: c[1]) if ok else (None, 0.0, 0.0)
    res["drafter_only_at_2pct"] = {"tau": best[0], "saving": float(best[1]), "share_of_perfect": float(best[1] / perf)}
    say(f"\n  BASELINE for Wed's probe: drafter-only at <= 2 % kept loss = -{100 * best[1]:.1f} % expert bytes = "
        f"{100 * best[1] / perf:.1f} % of perfect. A target-state probe must reach >= 50 % of perfect (PASS) and beat "
        f"this by >= 10 points.")
    json.dump(res, open(OUT + ".json", "w"), indent=1)
    open(OUT + ".txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
