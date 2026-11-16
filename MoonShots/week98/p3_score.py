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

P3 = H("~/MoonShots/week98/p3")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuts", default="none,C1,C3,C8,Q4,H8")
    a = ap.parse_args()
    TA.say = lambda s="": None
    rng = np.random.default_rng(1115)
    res, lines = {}, []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)
    for s, (tag, rtag, pub32, pub16) in TA.SETS.items():
        W = TA.load_windows(tag, pub32)
        ref = np.load(os.path.join(TA.CE, f"tree_t30b_{tag}.npz"), allow_pickle=True)
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
            d = np.load(f, allow_pickle=True)
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
            res[s][cut] = row
            say(f"  {cut:4s}: rows 16 kept {row[16]['kept']:.3f} (x{row[16]['ratio']:.4f} [{row[16]['ci95'][0]:.4f}, "
                f"{row[16]['ci95'][1]:.4f}]) | rows 32 kept {row[32]['kept']:.3f} (x{row[32]['ratio']:.4f} "
                f"[{row[32]['ci95'][0]:.4f}, {row[32]['ci95'][1]:.4f}]) | within 1 % at 32: {row['within_1pct_tree32']}"
                + (f" | screened kernel bitwise {row['screen_kernel_bitwise_windows']} windows, full top-1 outside the "
                   f"set {100 * row['full_top1_outside_screened_set']:.3f} %" if row["screen_kernel_bitwise_windows"] else ""))
    json.dump(res, open(os.path.join(P3, "m3_scores.json"), "w"), indent=1)
    open(os.path.join(P3, "m3_scores.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
