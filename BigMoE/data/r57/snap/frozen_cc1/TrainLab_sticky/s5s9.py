"""S5 (HC8) + S9 (top-8) read-out (NewIdeas Sun 09:45, 350 STACK; CPU, read-only import of evalsuite.report as
rule125.py). budget125_tf's arms p4433333333h (v3 + HC8), p44333333 (v3 + top-8), p44333333h (both), at v3's tiers and
turn18's positions; v3 itself (p4433333333) from the same run when it has it, else turn18's arm at the same (sequence,
position) (data/budget125_tf_R4_plan4433333333_mtp_b2_pos.npz; turns/turn21.sh's smoke checks it is bit-identical).
1. The Eval suite's strict per-part rules vs ar4 at the same positions (rule125.py / ugb_rule.py): the 90 % cluster-CI
   upper bounds of the mean KLD ratio and of the P99.9 ratio <= 1.10, the lower bound of the top-1 agreement change
   >= -0.5 points, in every part.
2. Each arm's mean KLD as x of v3 at the same positions [90 % cluster CI]; HC8's own share twice: v3 + HC8 over v3 and
   top-8 + HC8 over top-8.
Pre-registered (NewIdeas 09:45, written down 26 Oct 03:xx before the run):
   S5: HC8 NOT FREE if v3 + HC8 over v3, averaged over the four parts, is > 1.05, or any part's 90 % CI lower bound is
       > 1.05 -> HC8 stays a 12 GB-only setting.
   S9: top-8 FAILS if p44333333 fails the strict rules in any part -> an opt-in only, judged by the task bar (Eval suite).
   Both pass: two near-free main-track pieces; adopting them is Kamen's call.
R-26 (REPORT 'R-26 DYNK-Q'): DYNK arms qTTT in the file get the same strict rules, x of v3, and a verdict line each.
usage: s5s9.py [variant suffix]   (default plan4433333333_mtp_s5s9)  -> data/s5s9_<suffix>.txt"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
from evalsuite.report import cluster_boot, quantile_ratio, MARGINS  # noqa: E402

D = os.path.expanduser(os.environ.get("TL_DATA", "~/TrainLab/data"))
SUF = sys.argv[1] if len(sys.argv) > 1 else "plan4433333333_mtp_s5s9"
V3, HC8, TK8, BOTH = "p4433333333", "p4433333333h", "p44333333", "p44333333h"
NAME = {V3: "v3", HC8: "v3 + HC8", TK8: "v3 + top-8", BOTH: "v3 + HC8 + top-8"}
z = dict(np.load(f"{D}/budget125_tf_R4_{SUF}_pos.npz"))
v3_src = "this run"
if f"{V3}__kld" not in z:
    t18 = np.load(f"{D}/budget125_tf_R4_plan4433333333_mtp_b2_pos.npz")
    z.update({k: t18[k] for k in t18.files if k.startswith(V3 + "__")})
    here = [str(p) for p in z["parts"]]                       # turn18's part index -> this file's (-1: not run here;
    pm = np.array([here.index(str(q)) if str(q) in here else -1 for q in t18["parts"]], np.int8)   # R-26 runs 3 parts)
    z[f"{V3}__part"] = pm[t18[f"{V3}__part"]]
    v3_src = "turn18 (b2)"
a4 = np.load(os.path.expanduser(os.environ.get("TL_AR4", "~/EvalSuite/results/ar4/tf.npz")))
bt, nm = MARGINS["bootstrap"], MARGINS["noninferiority"]
parts = [str(x) for x in z["parts"]]
arms = [x for x in (V3, HC8, TK8, BOTH) if f"{x}__kld" in z] + sorted(   # R-26 DYNK qTTT / R-31 UNIQ uTTT[n] (TTT /
    k[:-5] for k in z if k[0] in "qu" and k.endswith("__kld") and int(k[1:-5].rstrip("n")) > 0)   # 1000; 0 = smokes)
NAME.update({a: f"v3 + DYNK {int(a[1:]) / 1000:.3f}" for a in arms if a.startswith("q")})
NAME.update({a: f"v3 + UNIQ {int(a[1:].rstrip('n')) / 1000:.3f}{' raw' if a.endswith('n') else ''}" for a in arms if a.startswith("u")})
AR4 = {(int(s), int(p)): j for j, (s, p) in enumerate(zip(a4["seq"], a4["pos"]))}


def paired(arm, ref, pi):
    """(arm KLD, ref KLD, arm agree, ref agree, sequence) at the positions of part pi that both have; ref None = ar4"""
    seq, pos, part = z[f"{arm}__seq"], z[f"{arm}__pos"], z[f"{arm}__part"]
    if ref is None:
        rk, rg, key = a4["kld"], a4["agree"], AR4
    else:
        rk, rg = z[f"{ref}__kld"], z[f"{ref}__agree"]
        key = {(int(s), int(p)): j for j, (s, p) in enumerate(zip(z[f"{ref}__seq"], z[f"{ref}__pos"]))}
    j = np.array([key.get((int(s), int(p)), -1) for s, p in zip(seq, pos)])
    m = (j >= 0) & (part == pi)
    f = lambda x: x.astype(np.float64)
    return f(z[f"{arm}__kld"][m]), f(rk[j[m]]), f(z[f"{arm}__agree"][m]), f(rg[j[m]]), seq[m]


def ratio(arm, ref, pi):
    ka, kr, _, _, cl = paired(arm, ref, pi)
    return cluster_boot(ka, kr, cl, bt["resamples"], bt["seed"])


out = [f"S5 (HC8) + S9 (top-8) strict tf screen ({SUF}; v3 from {v3_src}); 90 % cluster CIs over sequences",
       f"1. Eval-suite rules vs ar4 at the same positions (mean-ratio max {nm['kld_mean_ratio_max']}, P99.9 max "
       f"{nm['kld_p999_ratio_max']}, top-1 drop max {nm['top1_agree_drop_max_points']} pt)"]
strict = {}
for arm in arms:
    cells, ok_all = [], True
    for pi, pn in enumerate(parts):
        ka, kr, ga, gr, cl = paired(arm, None, pi)
        if not len(ka):
            continue
        e, lo, hi = cluster_boot(ka, kr, cl, bt["resamples"], bt["seed"])
        q, qlo, qhi = quantile_ratio(ka, kr, cl, 0.999, bt["quantile_resamples"], bt["seed"])
        d, dlo, dhi = cluster_boot(ga, gr, cl, bt["resamples"], bt["seed"], stat="diff")
        p_m, p_q, p_a = hi <= nm["kld_mean_ratio_max"], qhi <= nm["kld_p999_ratio_max"], dlo * 100 >= -nm["top1_agree_drop_max_points"]
        ok_all &= p_m and p_q and p_a
        cells.append(f"{pn[3:]}: KLD x{e:.2f} [..{hi:.2f}] {'ok' if p_m else 'FAIL'}, P99.9 x{q:.2f} [..{qhi:.2f}] "
                     f"{'ok' if p_q else 'FAIL'}, top-1 {d * 100:+.1f} [{dlo * 100:+.1f}..] pt {'ok' if p_a else 'FAIL'} "
                     f"(n {len(ka)}, {len(np.unique(cl))} seqs)")
    strict[arm] = ok_all
    out.append(f"  {NAME[arm]:17s} {'PASS' if ok_all else 'FAIL'} | " + " | ".join(cells))
out.append("2. mean KLD as x of v3 at the same positions")
for arm in arms[1:] if arms[0] == V3 else arms:
    rs = [ratio(arm, V3, pi) for pi in range(len(parts))]
    out.append(f"  {NAME[arm]:17s} " + " | ".join(f"{pn[3:]} x{e:.3f} [{lo:.3f}, {hi:.3f}]" for pn, (e, lo, hi) in zip(parts, rs))
               + f" | mean of the parts x{np.mean([r[0] for r in rs]):.3f}")
verdict = []
if HC8 in arms:
    rs = [ratio(HC8, V3, pi) for pi in range(len(parts))]
    mean, worst_lo = float(np.mean([r[0] for r in rs])), max(r[1] for r in rs)
    if BOTH in arms and TK8 in arms:
        r2 = [ratio(BOTH, TK8, pi) for pi in range(len(parts))]
        out.append(f"  HC8 on top of top-8 (both / top-8): " + " | ".join(f"{pn[3:]} x{e:.3f} [{lo:.3f}, {hi:.3f}]"
                   for pn, (e, lo, hi) in zip(parts, r2)) + f" | mean x{np.mean([r[0] for r in r2]):.3f}")
    free = mean <= 1.05 and worst_lo <= 1.05
    verdict.append(f"S5 HC8: v3 + HC8 = x{mean:.3f} of v3 over the four parts (highest part CI lower bound x{worst_lo:.3f}) "
                   f"-> {'FREE (<= 5 %)' if free else 'NOT FREE (> 5 %): stays a 12 GB-only setting'}")
if TK8 in arms:
    verdict.append(f"S9 top-8: the strict rules {'PASS in every part: a near-free main-track piece (-16 % union)' if strict[TK8] else 'FAIL: an opt-in only, judged by the task bar (Eval suite)'}")
if BOTH in arms:
    verdict.append(f"v3 + HC8 + top-8: the strict rules {'PASS' if strict[BOTH] else 'FAIL'}")
for a in arms:                                       # R-26 / R-31 (REPORT 'R-26 DYNK-Q', 'R-31 UNIQ-Q'): the parts run
    if a[0] in "qu":
        verdict.append(f"{'R-26' if a[0] == 'q' else 'R-31'} {NAME[a]}: the strict rules {'PASS' if strict[a] else 'FAIL'} in "
                       f"the parts run ({', '.join(p[3:] for p in parts)}; fixed top-8 failed ifeval, code and multi at x2.23 "
                       f"of v3){'; v3 in this run: ' + ('PASS' if strict.get(V3) else 'FAIL') if v3_src == 'this run' else ''}")
> (a quote from the project owner, kept private)
print("\n".join(out))
open(f"{D}/s5s9_{SUF}.txt", "w").write("\n".join(out) + "\n")
