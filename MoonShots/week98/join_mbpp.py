"""WEEK98 lever 1 v1 (audit and repair), MoonShots Sun 15 Nov 2026: join t30b's per-cell chain results on MBPP 100 (BigMoE's
CEILING cells, read-only) with Qwen3.5-2B's teacher-forced per-row picks (rows2b_mbpp.npz), and score the combined drafter:
at each cell, the 2B's token replaces the head's where they disagree and the 2B's own margin >= tau.

The head's drafted token isn't stored in the cells, and it isn't needed:
- head right, 2B right: same token, right;
- head right, 2B wrong: they disagree; an override (margin >= tau) makes it wrong;
- head wrong, 2B right: they disagree; an override makes it right;
- head wrong, 2B wrong: wrong either way.
Cell layout: dd_ceiling.py cmd_misses. For each file (sorted d1102mbpp [:100]), each band index j = 0..30 and anchor
s = a0 + i (a0 = max(0, gen - 2)), the target row is s + j + 1 (it predicts ids[row + 1]). Rows run up to len(lp) - 1.
The rebuilt rows are validated against the cells' own margin m (exact match required).

Walk: per anchor, accept depths in order until the first wrong one (max 31); at most P successful repairs per block
(P = the 2B's audit passes per check: 1, 2, or unlimited). Reported: bands 0 / 1-2 / 3-6 / 7-10 / 11-14 / 15-22 / 23-30 for
the head and the combined drafter, and the mean accepted per anchor.
LINE (LEVERS_A_MOONSHOTS.md, written first): PASS if MBPP 23-30 >= 82 with depth 0 within 0.3 pt of t30b, at the best tau.
usage: ~/BigMoE/.venv/bin/python -I join_mbpp.py"""
import glob, json, os
import numpy as np

H = os.path.expanduser
CELLS = H("~/BigMoE/data/deepdraft/ceiling/cells_t30b_mbpp100.npz")
MBPP = sorted(glob.glob(H("~/TrainLab/data/dlog/d1102mbpp/dlog__mbpp*.npz")))[:100]
R2B = H("~/MoonShots/week98/rows2b_mbpp.npz")
OUT = H("~/MoonShots/week98")
BANDS = [(0, 0), (1, 2), (3, 6), (7, 10), (11, 14), (15, 22), (23, 30)]


def main():
    C = np.load(CELLS)
    depth, right, mcell = C["depth"], C["right"], C["m"]
    # rebuild (file, anchor, depth, row) in the order dd_ceiling.py appended the cells: file-major, then depth, then anchor.
    # Depths 0-13 hold every candidate row; from depth 14 the eval drops a few cells (1-24 per depth over 100 files), so
    # each (file, depth) block is aligned to its candidate rows by the cells' own margin m, an exact per-row fingerprint.
    starts = np.r_[0, np.flatnonzero(np.diff(depth) < 0) + 1, len(depth)]
    assert len(starts) - 1 == len(MBPP), (len(starts) - 1, len(MBPP))
    fi = np.empty(len(depth), np.int64); rw = np.empty(len(depth), np.int64)
    for k, f in enumerate(MBPP):
        z = np.load(f)
        lp = np.asarray(z["lp"], np.float32); L, gen = len(lp), int(z["gen"]); a0 = max(0, gen - 2)
        for j in range(31):
            seg = np.arange(starts[k], starts[k + 1])
            seg = seg[depth[seg] == j]
            rows = np.arange(a0 + j + 1, L)
            rows = rows[rows >= gen - 1]
            cm = lp[rows, 0] - lp[rows, 1]
            p = 0
            for c in seg:
                while p < len(rows) and cm[p] != mcell[c]:
                    p += 1
                assert p < len(rows), f"file {k} depth {j}: cell {c} has no matching row"
                fi[c], rw[c] = k, rows[p]; p += 1
    an = rw - depth - 1                                     # the anchor row s (target row = s + j + 1)
    print(f"cells aligned by margin: {len(rw)} (every cell matched exactly)")

    # the 2B per row
    R = np.load(R2B)
    names = [os.path.basename(f) for f in MBPP]
    pos = {nm: i for i, nm in enumerate(names)}
    key = {}
    for nm, r, t1, mg, t0 in zip(R["fname"], R["row"], R["top1"], R["margin2b"], R["tk0"]):
        key[(pos[str(nm)], int(r))] = (bool(t1 == t0), float(mg))
    miss = [(int(a), int(b)) for a, b in zip(fi, rw) if (int(a), int(b)) not in key]
    assert not miss, f"{len(miss)} cells have no 2B row, e.g. {miss[:3]}"
    r2b = np.array([key[(int(a), int(b))][0] for a, b in zip(fi, rw)])
    m2b = np.array([key[(int(a), int(b))][1] for a, b in zip(fi, rw)], np.float32)

    def bands(ok):
        return {f"{a}-{b}" if a != b else str(a): round(100 * float(ok[(depth >= a) & (depth <= b)].mean()), 2) for a, b in BANDS}

    def walk(ok_head, ok_comb_repair, P):
        """mean accepted per anchor: depths in order until the first wrong; a repair (head wrong -> combined right) counts
        against the budget P; when it is spent, the head's own result stands"""
        acc = []
        order = np.lexsort((depth, an, fi))
        f_, a_ = fi[order], an[order]
        h_, c_ = ok_head[order], ok_comb_repair[order]
        starts = np.r_[0, np.flatnonzero((np.diff(f_) != 0) | (np.diff(a_) != 0)) + 1, len(f_)]
        for s0, s1 in zip(starts[:-1], starts[1:]):
            n, used = 0, 0
            for j in range(s0, s1):
                if P is not None and used >= P:               # no audit pass left: the head's own result stands
                    if h_[j]:
                        n += 1; continue
                    break
                if h_[j] and c_[j]:
                    n += 1                                    # right, no override needed (or an override that stays right)
                elif not h_[j] and c_[j]:
                    used += 1; n += 1                         # a repair: the next rows need a new audit pass
                else:
                    break                                     # a wrong row (or a harmful override) ends the block
            acc.append(n)
        return float(np.mean(acc))

    res = {"head": dict(bands=bands(right), accepted=walk(right, right, 0))}
    print(f"head (t30b chain): {res['head']['bands']}  mean accepted {res['head']['accepted']:.2f}")
    for tau in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0):
        ov = m2b >= tau
        comb = np.where(ov, np.where(right, r2b, r2b), right)    # override -> the 2B's correctness; else the head's
        comb = np.where(ov & ~right & ~r2b, False, comb)
        comb = np.where(ov & right & r2b, True, comb)            # same token: right
        res[f"tau{tau:g}"] = dict(bands=bands(comb), acc_P1=walk(right, comb, 1), acc_P2=walk(right, comb, 2),
                                  acc_inf=walk(right, comb, None), override_rate=round(float(ov.mean()), 4))
        r_ = res[f"tau{tau:g}"]
        print(f"tau {tau:>3g}: {r_['bands']}  accepted P1 {r_['acc_P1']:.2f} P2 {r_['acc_P2']:.2f} all {r_['acc_inf']:.2f}"
              f"  (2B margin >= tau on {100 * r_['override_rate']:.1f} % of cells)", flush=True)
    # a depth schedule (chosen AFTER seeing the fixed-tau rows above, so it is not a held-out result: HE+ must confirm it)
    for name, sch in (("sched_4_2_1_0.5", {0: 4.0, 1: 2.0, 2: 2.0, **{j: 1.0 for j in range(3, 7)}, **{j: 0.5 for j in range(7, 31)}}),
                      ("sched_4_2_2_1", {0: 4.0, **{j: 2.0 for j in range(1, 7)}, **{j: 1.0 for j in range(7, 31)}})):
        tau_c = np.array([sch[int(d)] for d in depth], np.float32)
        ov = m2b >= tau_c
        comb = np.where(ov, r2b, right)
        res[name] = dict(bands=bands(comb), acc_P1=walk(right, comb, 1), acc_P2=walk(right, comb, 2),
                         acc_inf=walk(right, comb, None), override_rate=round(float(ov.mean()), 4))
        r_ = res[name]
        print(f"{name}: {r_['bands']}  accepted P1 {r_['acc_P1']:.2f} P2 {r_['acc_P2']:.2f} all {r_['acc_inf']:.2f}", flush=True)
    json.dump(res, open(os.path.join(OUT, "join_mbpp.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
