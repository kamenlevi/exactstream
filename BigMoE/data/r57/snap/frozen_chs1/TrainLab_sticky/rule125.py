"""The Eval suite's own per-part noninferiority rules (margins.json, evalsuite.report's cluster bootstrap, read-only
import) applied to budget125_tf's per-position results (data/budget125_tf_R<R>[_<variant>]_pos.npz): per arm and part,
against the accepted build (ar4's layer-major KLD / top-1 agreement at the SAME (sequence, position), results/ar4/tf.npz):
- mean KLD ratio: the 90 % CI upper bound of sum(KLD arm) / sum(KLD ar4) <= 1.10 (clusters = sequences);
- P99.9 KLD ratio: the 90 % CI upper bound <= 1.10;
- top-1 agreement: the 90 % CI lower bound of (agree arm - agree ar4) >= -0.5 points.
CPU. usage: rule125.py [R] [variant]  -> stdout (+ data/rule125_<variant>.txt)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
from evalsuite.report import cluster_boot, quantile_ratio, MARGINS  # noqa: E402

R = int(sys.argv[1]) if len(sys.argv) > 1 else 4
VT = sys.argv[2] if len(sys.argv) > 2 else "ref"
z = np.load(os.path.expanduser(f"~/TrainLab/data/budget125_tf_R{R}{'' if VT == 'ref' else '_' + VT}_pos.npz"))
a = np.load(os.path.expanduser("~/EvalSuite/results/ar4/tf.npz"))
key = {(int(s), int(p)): j for j, (s, p) in enumerate(zip(a["seq"], a["pos"]))}
bt, nm = MARGINS["bootstrap"], MARGINS["noninferiority"]
parts = list(z["parts"])
arms = sorted({k.split("__")[0] for k in z.files if "__" in k}, key=lambda s: (s != "exact", s))
out = [f"Eval-suite rules vs ar4 at the same positions (R={R}, variant {VT}; mean-ratio max {nm['kld_mean_ratio_max']}, "
       f"P99.9 max {nm['kld_p999_ratio_max']}, top-1 drop max {nm['top1_agree_drop_max_points']} pt; 90 % cluster CI)"]
for arm in arms:
    seq, pos, part = z[f"{arm}__seq"], z[f"{arm}__pos"], z[f"{arm}__part"]
    idx = np.array([key.get((int(s), int(p)), -1) for s, p in zip(seq, pos)])
    ok = idx >= 0
    cells, all_pass = [], True
    for pi, pn in enumerate(parts):
        m = ok & (part == pi)
        if not m.any():
            continue
        kv, ka = z[f"{arm}__kld"][m].astype(np.float64), a["kld"][idx[m]].astype(np.float64)
        gv, ga = z[f"{arm}__agree"][m].astype(np.float64), a["agree"][idx[m]].astype(np.float64)
        cl = seq[m]
        e, lo, hi = cluster_boot(kv, ka, cl, bt["resamples"], bt["seed"])
        q, qlo, qhi = quantile_ratio(kv, ka, cl, 0.999, bt["quantile_resamples"], bt["seed"])
        d, dlo, dhi = cluster_boot(gv, ga, cl, bt["resamples"], bt["seed"], stat="diff")
        p_mean, p_q, p_ag = hi <= nm["kld_mean_ratio_max"], qhi <= nm["kld_p999_ratio_max"], dlo * 100 >= -nm["top1_agree_drop_max_points"]
        all_pass &= p_mean and p_q and p_ag
        cells.append(f"{pn[3:]}: KLD x{e:.2f} [{lo:.2f}, {hi:.2f}] {'ok' if p_mean else 'FAIL'}, P99.9 x{q:.2f} [..{qhi:.2f}] "
                     f"{'ok' if p_q else 'FAIL'}, top-1 {d * 100:+.1f} [{dlo * 100:+.1f}..] pt {'ok' if p_ag else 'FAIL'} "
                     f"(n {int(m.sum())}, {len(np.unique(cl))} seqs)")
    out.append(f"{arm:6s} {'PASS' if all_pass else 'FAIL'} | " + " | ".join(cells))
print("\n".join(out))
open(os.path.expanduser(f"~/TrainLab/data/rule125_{VT}.txt"), "w").write("\n".join(out) + "\n")
