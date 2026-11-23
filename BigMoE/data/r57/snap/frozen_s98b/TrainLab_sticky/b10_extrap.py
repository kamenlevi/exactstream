"""CPU: B10 v2 layer-sliced read-out of budget125_tf's dense layer arms, per part, against turn18's v3 arm at the SAME
(sequence, position) (data/budget125_tf_R4_plan4433333333_mtp_b2_pos.npz), 90 % CIs from a sequence-cluster bootstrap.
Two designs (arm names: dense_layers.py / budget125_tf.py):
  vs v3      t<L> (layer L trellis, the other layers v3; NewIdeas 07:00): excess = mean KLD(arm) / mean KLD(v3) - 1.
             All 48 layers: x(1 + 48 x the mean single-layer excess). Its noise is the cross term between layer L's
             change and v3's other 47 layers and the R-row floor, both much larger than one layer's ~1 % excess.
  isolated   d<L>p<PLAN> / t<L>p<PLAN> [/ x<L>p<PLAN>] after a first arm p<PLAN> (all 4-bit) with TL_LOCAL=1: lkld =
             KLD(first arm || arm) over the full vocabulary = layer L's damage alone (d: v3's tiers at L, t: trellis,
             x: the dense path's own floor at exact 4-bit). Ratio r_L = (t - x) / (d - x) (x = 0 when absent).
             All 48 layers, scaled on v3's measured KLD V = F + Q (F = the first arm's KLD vs the cached reference =
             the R-row floor, Q = V - F = v3's damage): x((F + mean r x Q) / V); also 48 x the mean (t - d) / V.
Kill (NewIdeas): > 1.5x v3 per part with every layer trellis-coded.
Several runs (one layer per process, turns/turn20.sh) merge: an arm that is in more than one file (the first arm) must be
bit-identical there (the engine is deterministic); the first file's first arm is the reference.
usage: b10_extrap.py <variant suffix>[,<suffix>...]   e.g. b10_extrap.py plan4433333333_mtp_b10L2,plan4433333333_mtp_b10L24
       -> data/b10_extrap_<suffix, or the suffixes' common start + "all">.txt"""
import os, re, sys
import numpy as np

D = os.path.expanduser(os.environ.get("TL_DATA", "~/TrainLab/data"))
NL, B, SEED = 48, 2000, 0
SUF = sys.argv[1].split(",")
z = {}
for sf in SUF:
    f = np.load(f"{D}/budget125_tf_R4_{sf}_pos.npz")
    for k in f.files:
        if k in z:
            assert np.array_equal(z[k], f[k]), f"{k} differs between the runs"
        else:
            z[k] = f[k]
v = np.load(f"{D}/budget125_tf_R4_plan4433333333_mtp_b2_pos.npz")
key = {(int(s), int(p)): j for j, (s, p) in enumerate(zip(v["p4433333333__seq"], v["p4433333333__pos"]))}
parts = [str(x) for x in z["parts"]]
arms = [a for a in dict.fromkeys(k.split("__")[0] for k in z if "__" in k)]
rng = np.random.default_rng(SEED)
ISO = re.compile(r"^([dxt])(\d+)p(\d+)$")


def per_seq(arm, pn, field="kld"):
    """(seq ids, per-sequence sums of the arm's field, of v3's KLD, position counts) at the shared positions of part pn"""
    seq, pos, part, f = (z[f"{arm}__{x}"] for x in ("seq", "pos", "part", field))
    j = np.array([key.get((int(s), int(p)), -1) for s, p in zip(seq, pos)])
    m = (j >= 0) & (part == parts.index(pn))
    us = np.unique(seq[m])
    fa = np.array([f[m & (seq == s)].astype(np.float64).sum() for s in us])
    kv = np.array([v["p4433333333__kld"][j[m & (seq == s)]].astype(np.float64).sum() for s in us])
    return us, fa, kv, np.array([(m & (seq == s)).sum() for s in us])


lines = [f"B10 v2 read-out ({', '.join(SUF)}) vs turn18's v3 at the same positions; 90 % CIs over sequences"]
for pn in parts:
    have = [a for a in arms if (z[f"{a}__part"] == parts.index(pn)).any()]
    if not have:
        continue
    us, _, kv, n = per_seq(have[0], pn)
    idx = [rng.integers(0, len(us), len(us)) for _ in range(B)]
    V = kv.sum() / n.sum()
    lines.append(f"{pn[3:]} ({len(us)} seqs, {n.sum()} positions, v3 mean KLD {V:.5f}):")
    single, ex_draw = [], []                                       # design 1: vs v3
    for a in have:
        if a[0] != "t" or not a[1:].replace(".", "").isdigit():
            continue
        sa, ka, kv_, _ = per_seq(a, pn)
        assert (sa == us).all(), "arms cover different sequences"
        ex, bs = ka.sum() / kv_.sum() - 1, np.array([ka[i].sum() / kv_[i].sum() - 1 for i in idx])
        lines.append(f"  vs v3     {a:14s} excess {ex:+.4f} [{np.quantile(bs, 0.05):+.4f}, {np.quantile(bs, 0.95):+.4f}]")
        if "." in a:
            Ls = a[1:].split(".")
            if all(f"t{L}" in have for L in Ls):
                parts_ex = [per_seq(f"t{L}", pn)[1].sum() / kv_.sum() - 1 for L in Ls]
                lines.append(f"            additivity {a}: {ex:+.4f} vs the sum of its single layers {sum(parts_ex):+.4f}")
        else:
            single.append(a); ex_draw.append(bs)
    if single:
        r, rhi = 1 + NL * np.mean([per_seq(a, pn)[1].sum() / kv.sum() - 1 for a in single]), 1 + NL * np.quantile(np.mean(ex_draw, 0), 0.95)
        lines.append(f"  vs v3     -> all {NL} layers ({', '.join(single)}): x{r:.2f} of v3 (90 % upper x{rhi:.2f}) "
                     f"{'KILL (> 1.5)' if r > 1.5 else 'ok'}")
    iso = {}                                                       # design 2: isolated layers, lkld vs the first arm
    for a in have:
        mm = ISO.match(a)
        if mm and f"{a}__lkld" in z:
            iso.setdefault(mm.group(2), {})[mm.group(1)] = per_seq(a, pn, "lkld")[1]
    lay = sorted((L for L in iso if "d" in iso[L] and "t" in iso[L]), key=int)
    if not lay:
        continue
    ref = arms[0]                                                  # the first arm: the all-4-bit R-row floor F
    _, kf, _, _ = per_seq(ref, pn)

    def stats(i):
        """one draw (sequence indices i): per layer (d, t, x) means and r; F, Q"""
        N = n[i].sum()
        out = {}
        for L in lay:
            x = iso[L]["x"][i].sum() / N if "x" in iso[L] else 0.0
            d, t = iso[L]["d"][i].sum() / N, iso[L]["t"][i].sum() / N
            out[L] = (d, t, x, (t - x) / (d - x))
        F, Vi = kf[i].sum() / N, kv[i].sum() / N
        rbar = np.mean([out[L][3] for L in lay])
        return out, F, Vi, rbar, (F + rbar * (Vi - F)) / Vi, 1 + NL * np.mean([(out[L][1] - out[L][0]) / Vi for L in lay])

    allp = np.arange(len(us))
    o, F, Vi, rbar, pred, pred48 = stats(allp)
    bs = [stats(i) for i in idx]
    q = lambda vals: f"[{np.quantile(vals, 0.05):.3g}, {np.quantile(vals, 0.95):.3g}]"
    lines.append(f"  isolated  first arm {ref}: R-row floor F {F:.5f} = {F / Vi:.0%} of v3; v3's damage Q {Vi - F:.5f}")
    for L in lay:
        d, t, x, r = o[L]
        lines.append(f"  isolated  layer {L:>2s}: v3 tiers {d:.2e} (x48 = {NL * d / (Vi - F):.2f} of Q), trellis {t:.2e}"
                     + (f", dense floor {x:.1e}" if "x" in iso[L] else "")
                     + f" -> r {r:.2f} {q([b[0][L][3] for b in bs])}")
    lines.append(f"  isolated  -> all {NL} layers (mean r {rbar:.2f} {q([b[3] for b in bs])}): x{pred:.2f} of v3 "
                 f"{q([b[4] for b in bs])}; 48 x the mean (t - d): x{pred48:.2f} {q([b[5] for b in bs])} "
                 f"{'KILL (> 1.5)' if pred > 1.5 else 'ok'}")
print("\n".join(lines))
open(f"{D}/b10_extrap_{SUF[0] if len(SUF) == 1 else os.path.commonprefix(SUF).rstrip('0123456789') + 'all'}.txt", "w").write("\n".join(lines) + "\n")
