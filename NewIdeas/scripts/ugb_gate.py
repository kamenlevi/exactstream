"""NEW bet, kill test (CPU): uncertainty-gated budget (UGB). Run the per-pass shared-expert budget only where the model is
confident; elsewhere natural routing. Prior art checked 10-22 (MoE-Spec 2602.16052 per-layer verify caps; AcceptMoE 2608.02989
self-sizing commitment-weighted sets with a row-0 anchor, no uncertainty fallback; EcoSpec 2607.12696 lossless draft choice;
EVICT 2605.00342; CARE 2607.26052 / DA-MoE 2409.06669 adapt k by ROUTER confidence): none gates the verify-pass budget by the
model's / drafter's own uncertainty with a fallback to the full set.
Input: TrainLab turn15 per-position arrays (v3 tiers, 4-row tf passes): ~/TrainLab/data/budget125_tf_R4_plan4433333333_pos.npz
(<arm>__kld / __agree / __lp1 / __margin / __seq / __pos / __part for arms exact, b22, b20, b18).
Gate signal = the EXACT arm's own top-1 log-prob or margin at the same position (oracle-ish: a real gate would use the MTP
drafter's confidence, known before the pass). For one global threshold: budget KLD where confident, exact-arm KLD elsewhere.
Pre-registered kill line: no threshold gives all four parts <= the strict targets (code 0.0025 / gsm 0.0045 / ifeval 0.0204 /
multi 0.0205; turn15's arms are ON v3, so KLD vs the 4-bit ref already includes v3) while the budget covers >= 80 % of code
positions -> KILL. Else: next = the realistic gate (MTP drafter confidence) on the GPU."""
import os, sys
import numpy as np
P = os.path.expanduser("~/TrainLab/data/budget125_tf_R4_plan4433333333_pos.npz")
TGT = dict(tf_code=0.0025, tf_gsm=0.0045, tf_ifeval=0.0204, tf_multi=0.0205)
d = np.load(P, allow_pickle=True)
parts = [str(x) for x in d["parts"]] if "parts" in d.files else ["tf_gsm", "tf_ifeval", "tf_code", "tf_multi"]
def key(a, f): return f"{a}__{f}"
ex = {f: d[key("exact", f)] for f in ("kld", "lp1", "margin", "seq", "pos", "part")}
idx = {(int(s), int(p)): i for i, (s, p) in enumerate(zip(ex["seq"], ex["pos"]))}
arms = sorted({k.split("__")[0] for k in d.files if "__" in k} - {"exact"})
print(f"positions (exact arm): {len(ex['kld'])}; per part:",
      {parts[j]: int((ex['part'] == j).sum()) for j in range(len(parts))})
for sig in ("lp1", "margin"):
    for a in arms:
        s, p = d[key(a, "seq")], d[key(a, "pos")]
        j = np.array([idx.get((int(x), int(y)), -1) for x, y in zip(s, p)])
        ok = j >= 0
        kb, ke = d[key(a, "kld")][ok], ex["kld"][j[ok]]
        g, part = ex[sig][j[ok]], ex["part"][j[ok]]
        print(f"\n== {a} (gate = exact arm's {sig}; {ok.sum()} joined positions)")
        base = {parts[q]: (kb[part == q].mean() / TGT[parts[q]]) for q in range(len(parts)) if (part == q).any()}
        print("   ungated KLD / target:", {k: round(v, 2) for k, v in base.items()})
        best = None
        for thr in np.quantile(g, np.linspace(0.0, 0.95, 39)):
            conf = g >= thr
            k = np.where(conf, kb, ke)
            ratio = {parts[q]: k[part == q].mean() / TGT[parts[q]] for q in range(len(parts)) if (part == q).any()}
            cov = {parts[q]: conf[part == q].mean() for q in range(len(parts)) if (part == q).any()}
            ok_all = max(ratio.values()) <= 1.0
            if ok_all and cov.get("tf_code", 0) >= 0.8 and (best is None or np.mean(list(cov.values())) > best[2]):
                best = (thr, ratio, np.mean(list(cov.values())), cov)
        if best is None:
            print("   no threshold passes every part with >= 80 % code coverage -> KILL for this arm / signal")
        else:
            thr, ratio, _, cov = best
            print(f"   PASS at threshold {thr:.3f}: KLD/target", {k: round(v, 2) for k, v in ratio.items()},
                  "| budget coverage", {k: f"{v:.0%}" for k, v in cov.items()},
                  "| per-pass coverage ~cov^4", {k: f"{v ** 4:.0%}" for k, v in cov.items()})
