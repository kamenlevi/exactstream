"""Compare run.py tf arms (data/defer/tf_ARM_SS.npz) per verify width S: bit-identity checks (dh10 vs v3, force vs ref)
and the KL of the non-exact arms vs v3 (mean / max over positions, top-1 agreement). Exit 1 if a pair is neither
bit-identical nor CLOSE (KL < 1e-3, same top-1 everywhere: a summation-order difference, not a wrong expert).
usage: tf_compare.py"""
import os, sys
import numpy as np
D = os.path.expanduser("~/MoonShots/data/defer")
ALL = {f[3:-4]: np.load(os.path.join(D, f))["logits"] for f in sorted(os.listdir(D)) if f.startswith("tf_") and f.endswith(".npz")}


def kl(p_lg, q_lg):
    """KL(p || q) per position from logits (p = reference)"""
    lp = p_lg - p_lg.max(-1, keepdims=True); lp -= np.log(np.exp(lp).sum(-1, keepdims=True))
    lq = q_lg - q_lg.max(-1, keepdims=True); lq -= np.log(np.exp(lq).sum(-1, keepdims=True))
    return (np.exp(lp) * (lp - lq)).sum(-1)


bad = 0
for Sx in sorted({k.rsplit("_S", 1)[1] for k in ALL}, key=int):
    arms = {k.rsplit("_S", 1)[0]: v for k, v in ALL.items() if k.endswith(f"_S{Sx}")}
    print(f"== S={Sx}: arms", ", ".join(f"{k} {v.shape}" for k, v in arms.items()))
    for a, b in (("dh10", "v3"), ("force", "ref"), ("force_sub", "ref")):
        if a in arms and b in arms:
            d = np.abs(arms[a].astype(np.float64) - arms[b].astype(np.float64))
            same = bool((arms[a] == arms[b]).all())
            k = kl(arms[b].astype(np.float64), arms[a].astype(np.float64))
            close = k.mean() < 1e-3 and (arms[a].argmax(-1) == arms[b].argmax(-1)).all()   # summation order only
            v = "BIT-IDENTICAL" if same else ("CLOSE (not bitwise)" if close else "DIFFER")
            print(f"{a} vs {b}: {v} (max |dlogit| {d.max():.3g}, KL mean {k.mean():.2e}, positions differing "
                  f"{int((d.max(-1) > 0).sum())} of {d.shape[0]})")
            bad += not (same or close)
    if "v3" in arms:
        for a in arms:
            if a != "v3":
                k = kl(arms["v3"].astype(np.float64), arms[a].astype(np.float64))
                agree = (arms[a].argmax(-1) == arms["v3"].argmax(-1)).mean()
                print(f"KL(v3 || {a}): mean {k.mean():.5f}, max {k.max():.4f}; top-1 agreement {agree:.3f}")
sys.exit(1 if bad else 0)
