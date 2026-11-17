"""WEEK98 PILLAR 3, M3 scoring (MoonShots, Mon 16 Nov 2026; NewIdeas 08:4x: kept per cut at tree-16 AND tree-32 under
certify98's walk). For each cut's dump (week98/p3/tree_t30b_<set>_<cut>.npz): kept per check with rows 16 / 32 (15 / 31
best-first nodes + the bonus), from NewIdeas' treek_final.py / tree_audit.py imported read-only (its S9c assert applies
to the full head, i.e. the reference dump and the gate dump only), against the full head (BigMoE's
tree_t30b_<set>.npz). Per window as well, for a window-clustered 95 % CI of cut / full (2,000 resamples, seed 1115).
Line (pre-registered, LEVERS_A "PILLAR 3"; NewIdeas 08:4x): within 1 % kept of the full head on both sets (tree-32;
tree-16 reported beside it). Also: each cut's screened-kernel bitwise check and pick-change share (from its dump json).
TRUE-STATE (lab): teacher-forced, the 125B's true state at every anchor.
usage: ~/BigMoE/.venv/bin/python -I -B p3_score.py [--cuts none,C1,C3,C8,Q4,H8]"""
import argparse, json, os, sys
import numpy as np

sys.dont_write_bytecode = True
H = os.path.expanduser
sys.path.insert(0, H("~/NewIdeas/research/week98"))
import tree_audit as TA                                     # noqa: E402  (read-only)
import treek_final as TF                                    # noqa: E402  (read-only)

P3 = H("~/TrainLab/data/s22/m3")   # TrainLab COPY of MoonShots' week98/p3_score.py (S22): only this path changed


def per_window(W, dump, rows):
    """treek_final.kept_by_rows' walk, kept lists per window"""
    K = rows - 1
    fi, an, nd, ot, dp = (dump[k] for k in ("fi", "an", "node", "on_true", "depth"))
    hit = ot & (nd < K)
    acc = {}
    for f, a_, d in zip(fi[hit], an[hit], dp[hit]):
        acc[(int(f), int(a_))] = max(acc.get((int(f), int(a_)), 0), int(d) + 1)
    out = []
    for X in W:
        ks, p = [], 1
        while p < X["n"]:
            key = (X["fi"], int(X["an"][p])) if p < len(X["an"]) else None
            a = min(acc.get(key, 0), K) if key else 0
            ks.append(a + 1)
            p += a + 1
        out.append(ks)
    return out


BUCKETS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 7), (8, 15), (16, 99)]


def agreement(ref, d):
    """the pre-registered per-depth agreement: the cut head's top-1 / top-3 set vs the full head's, at the states both
    trees share on the true path. Prediction depth j = 0 is the root (the anchor's own state); j = depth + 1 for a
    true-path node (its top-k predicts the next true token). Deeper j only where both trees drafted the true path."""
    def top3(ids, p):
        return np.sort(np.take_along_axis(ids, np.argsort(-p, axis=1, kind="stable")[:, :3], 1), axis=1)
    rows = {}
    kr = {(int(f), int(a)): i for i, (f, a) in enumerate(zip(ref["a_fi"], ref["a_an"]))}
    kc = {(int(f), int(a)): i for i, (f, a) in enumerate(zip(d["a_fi"], d["a_an"]))}
    com = [k for k in kr if k in kc]
    ri, ci = np.array([kr[k] for k in com]), np.array([kc[k] for k in com])
    R3, C3 = top3(ref["root_top_ids"], ref["root_top_p"])[ri], top3(d["root_top_ids"], d["root_top_p"])[ci]
    t1 = ref["root_top_ids"][ri, np.argmax(ref["root_top_p"][ri], 1)] == d["root_top_ids"][ci, np.argmax(d["root_top_p"][ci], 1)]
    J, T1, S3 = [np.zeros(len(ri), int)], [t1], [(R3 == C3).all(1)]

    def nodes(z):
        m = np.flatnonzero(z["on_true"])
        return {(int(z["fi"][i]), int(z["an"][i]), int(z["depth"][i])): i for i in m}
    nr, nc = nodes(ref), nodes(d)
    com = [k for k in nr if k in nc]
    if com:
        ri, ci = np.array([nr[k] for k in com]), np.array([nc[k] for k in com])
        R3, C3 = top3(ref["top_ids"][ri], ref["top_p"][ri]), top3(d["top_ids"][ci], d["top_p"][ci])
        t1 = ref["top_ids"][ri, np.argmax(ref["top_p"][ri], 1)] == d["top_ids"][ci, np.argmax(d["top_p"][ci], 1)]
        J.append(np.array([k[2] + 1 for k in com])); T1.append(t1); S3.append((R3 == C3).all(1))
    J, T1, S3 = np.concatenate(J), np.concatenate(T1), np.concatenate(S3)
    for lo, hi in BUCKETS:
        m = (J >= lo) & (J <= hi)
        if m.any():
            rows[f"{lo}" if lo == hi else f"{lo}-{hi}"] = [int(m.sum()), round(float(T1[m].mean()), 4), round(float(S3[m].mean()), 4)]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuts", default="none,C1,C3,C8,Q4,H8")
    ap.add_argument("--tag", default="m3", help="outputs p3/<tag>_scores.{json,txt} (q8: NewIdeas' 21:1x ask)")
    a = ap.parse_args()
    TA.say = lambda s="": None
    rng = np.random.default_rng(1115)
    res, lines = {}, []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)
    for s, (tag, rtag, pub32, pub16) in TA.SETS.items():
        W = TA.load_windows(tag, pub32)
        ref = dict(np.load(os.path.join(TA.CE, f"tree_t30b_{tag}.npz"), allow_pickle=True))   # in memory: NpzFile re-reads per access
        R = {r: per_window(W, ref, r) for r in (16, 32)}
        chk = TF.kept_by_rows(W, ref, [16, 32])
        for r in (16, 32):
            assert abs(np.mean(sum(R[r], [])) - chk[r]["kept"]) < 1e-9, "per-window walk != treek_final's"
        assert abs(chk[16]["kept"] - pub16) < 0.01, f"{s}: full head's 16-row tree {chk[16]['kept']:.3f} != S9c {pub16}"
        idx = rng.integers(0, len(W), (2000, len(W)))
        say(f"=== {s} ({tag}, {len(W)} windows): full head kept rows 16 {chk[16]['kept']:.3f}, rows 32 {chk[32]['kept']:.3f}")
        res[s] = {"full": {r: round(chk[r]["kept"], 4) for r in (16, 32)}}
        for cut in a.cuts.split(","):
            f = os.path.join(P3, f"tree_t30b_{tag}_{cut}.npz")
            if not os.path.exists(f):
                say(f"  {cut}: no dump")
                continue
            d = dict(np.load(f, allow_pickle=True))
            row = {}
            for r in (16, 32):
                C = per_window(W, d, r)
                cs, cn = np.array([sum(x) for x in C], float), np.array([len(x) for x in C], float)
                rs, rn = np.array([sum(x) for x in R[r]], float), np.array([len(x) for x in R[r]], float)
                ratio = (cs.sum() / cn.sum()) / (rs.sum() / rn.sum())
                bs = (cs[idx].sum(1) / cn[idx].sum(1)) / (rs[idx].sum(1) / rn[idx].sum(1))
                row[r] = dict(kept=round(float(cs.sum() / cn.sum()), 4), ratio=round(float(ratio), 4),
                              ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)])
            jf = os.path.join(P3, f"tree_t30b_{tag}_{cut}.json")
            J = json.load(open(jf)) if os.path.exists(jf) else {}
            row["screen_kernel_bitwise_windows"] = J.get("screen_kernel_bitwise_windows")
            row["full_top1_outside_screened_set"] = J.get("full_top1_outside_screened_set")
            row["within_1pct_tree32"] = bool(row[32]["ratio"] >= 0.99)
            row["within_1pct_16_and_32"] = bool(row[16]["ratio"] >= 0.99 and row[32]["ratio"] >= 0.99)
            row["agree_by_depth"] = agreement(ref, d)                   # bucket: [n, top-1 same, top-3 set same]
            res[s][cut] = row
            say(f"  {cut:4s}: rows 16 kept {row[16]['kept']:.3f} (x{row[16]['ratio']:.4f} [{row[16]['ci95'][0]:.4f}, "
                f"{row[16]['ci95'][1]:.4f}]) | rows 32 kept {row[32]['kept']:.3f} (x{row[32]['ratio']:.4f} "
                f"[{row[32]['ci95'][0]:.4f}, {row[32]['ci95'][1]:.4f}]) | within 1 % at 32: {row['within_1pct_tree32']}"
                f", at 16 and 32: {row['within_1pct_16_and_32']}"
                + (f" | screened kernel bitwise {row['screen_kernel_bitwise_windows']} windows, full top-1 outside the "
                   f"set {100 * row['full_top1_outside_screened_set']:.3f} %" if row["screen_kernel_bitwise_windows"] else ""))
            say("        agreement with the full head by prediction depth j (n, top-1 same, top-3 set same): " + "; ".join(
                f"j {b}: {v[0]}, {v[1]:.3f}, {v[2]:.3f}" for b, v in row["agree_by_depth"].items()))
    json.dump(res, open(os.path.join(P3, f"{a.tag}_scores.json"), "w"), indent=1)
    open(os.path.join(P3, f"{a.tag}_scores.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
