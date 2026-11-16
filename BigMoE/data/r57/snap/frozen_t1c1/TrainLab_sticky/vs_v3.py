"""An arm against turn18's v3 arm (p4433333333, data/budget125_tf_R4_plan4433333333_mtp_b2_pos.npz) at the SAME
(sequence, position), per Eval-suite part, with the suite's cluster bootstrap (read-only import, as rule125.py):
mean KLD ratio arm / v3 [90 % CI], P99.9 ratio, top-1 agreement difference. NewIdeas' 3B kill: a mean ratio > 1.5 on
any part. CPU. usage: vs_v3.py <variant suffix> [arm ...]   e.g. vs_v3.py plan4433333333_mtp_b3b 3b140t32 p3333333333"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
from evalsuite.report import cluster_boot, quantile_ratio, MARGINS  # noqa: E402

D = os.path.expanduser("~/TrainLab/data")
z = np.load(f"{D}/budget125_tf_R4_{sys.argv[1]}_pos.npz")
v = np.load(f"{D}/budget125_tf_R4_plan4433333333_mtp_b2_pos.npz")
key = {(int(s), int(p)): j for j, (s, p) in enumerate(zip(v["p4433333333__seq"], v["p4433333333__pos"]))}
bt = MARGINS["bootstrap"]
arms = sys.argv[2:] or sorted({k.split("__")[0] for k in z.files if "__" in k})
parts = list(z["parts"])
out = [f"vs turn18's v3 arm (p4433333333) at the same positions ({sys.argv[1]}; 90 % cluster CI; kill: mean ratio > 1.5)"]
for arm in arms:
    seq, pos, part = z[f"{arm}__seq"], z[f"{arm}__pos"], z[f"{arm}__part"]
    idx = np.array([key.get((int(s), int(p)), -1) for s, p in zip(seq, pos)])
    cells, worst = [], 0.0
    for pi, pn in enumerate(parts):
        m = (idx >= 0) & (part == pi)
        if not m.any():
            continue
        kv, kr = z[f"{arm}__kld"][m].astype(np.float64), v["p4433333333__kld"][idx[m]].astype(np.float64)
        gv, gr = z[f"{arm}__agree"][m].astype(np.float64), v["p4433333333__agree"][idx[m]].astype(np.float64)
        e, lo, hi = cluster_boot(kv, kr, seq[m], bt["resamples"], bt["seed"])
        q, qlo, qhi = quantile_ratio(kv, kr, seq[m], 0.999, bt["quantile_resamples"], bt["seed"])
        d, dlo, dhi = cluster_boot(gv, gr, seq[m], bt["resamples"], bt["seed"], stat="diff")
        worst = max(worst, e)
        cells.append(f"{pn[3:]}: KLD x{e:.2f} [{lo:.2f}, {hi:.2f}], P99.9 x{q:.2f} [..{qhi:.2f}], top-1 {d * 100:+.1f} "
                     f"[{dlo * 100:+.1f}, {dhi * 100:+.1f}] pt (n {int(m.sum())}, {len(np.unique(seq[m]))} seqs)")
    out.append(f"{arm:12s} worst x{worst:.2f} {'KILL' if worst > 1.5 else 'ok'} | " + " | ".join(cells))
print("\n".join(out))
open(f"{D}/vs_v3_{sys.argv[1]}.txt", "w").write("\n".join(out) + "\n")
