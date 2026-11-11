#!/usr/bin/env python3 -I
"""assumption_hunt.py (FableGod, mission 31). CPU measurements of the project's assumptions on recorded data:
 1. hidden-state near-repeats (d02a taps f[n, 5, 2560] at layers 5/15/25/35/45: cosine to the previous token and the best
    of the last 64) and activation sparsity (share of channels under 5 % of the row's RMS);
 2. expert-set locality from the decision map's routing captures (ids [tokens, 48, 10]): within a 16-token check (distinct
    experts per layer / 160 uses), between consecutive checks (share of the next check's distinct experts already in the
    last check's set, and in the last 4 checks' union);
 3. bigram-keyed routing: for a token whose (previous id, id) bigram occurred earlier in the window, the Jaccard of its
    top-10 with the earlier occurrence's, per layer, against random pairs; and the top-1 expert agreement;
 4. consecutive checks' accepted lengths from the HPD-0 walks (does the last check predict the next: a branch predictor);
 5. the decision map's reuse arms (the previous position's MoE / GDN / attention output reused): flips by class."""
import numpy as np, glob, os, json, math
T = os.path.expanduser("~/TrainLab/data"); O = os.path.expanduser("~/FableGod/decision_map/out")
print("## 1. hidden-state near-repeats and activation sparsity (taps; 3 files: agent_bash17, mgsm_de108, tf windows absent here)")
for f in ("agent_bash17.npz", "mgsm_de108.npz", "agent_text23.npz"):
    p = f"{T}/d02a/{f}"
    if not os.path.exists(p): continue
    F = np.load(p)["f"].astype(np.float32); n = F.shape[0]
    line = f"  {f} ({n} rows): "
    for li, lay in enumerate((5, 15, 25, 35, 45)):
        X = F[:, li, :]; Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        prev = np.sum(Xn[1:] * Xn[:-1], 1)
        best = []
        for t in range(64, n, 7):
            best.append(np.max(Xn[t - 64:t] @ Xn[t]))
        best = np.array(best)
        sp = np.mean(np.abs(X) < 0.05 * np.sqrt(np.mean(X ** 2, 1, keepdims=True)))
        line += f"L{lay}: cos(prev) mean {prev.mean():.3f}, >= 0.95 {100 * np.mean(prev >= 0.95):.1f} %; best of last 64 >= 0.99 {100 * np.mean(best >= 0.99):.1f} %, >= 0.95 {100 * np.mean(best >= 0.95):.1f} %; channels < 5 % RMS {100 * sp:.1f} % | "
    print(line)

print("\n## 2. expert-set locality from 36 routing captures (ids [tokens, 48, 10])")
caps = sorted(glob.glob(f"{O}/routing/*.npz")); per = {}
for f in caps:
    z = np.load(f); ids = z["ids"].astype(np.int32); n = ids.shape[0]; dom = os.path.basename(f).split("_")[1]
    d = per.setdefault(dom, dict(within=[], prev1=[], prev4=[], n=0))
    sets = []
    for s in range(0, n - 16 + 1, 16):
        w = ids[s:s + 16]; S = [set(w[:, L, :].reshape(-1).tolist()) for L in range(48)]
        d["within"].append(np.mean([len(S[L]) for L in range(48)]) / 160)
        if sets:
            d["prev1"].append(np.mean([len(S[L] & sets[-1][L]) / len(S[L]) for L in range(48)]))
            u4 = [set().union(*[sets[-k][L] for k in range(1, min(4, len(sets)) + 1)]) for L in range(48)]
            d["prev4"].append(np.mean([len(S[L] & u4[L]) / len(S[L]) for L in range(48)]))
        sets.append(S); d["n"] += 1
for dom, d in per.items():
    print(f"  {dom}: {d['n']} checks; distinct experts per layer / 160 uses {np.mean(d['within']):.3f} (so {100 * (1 - np.mean(d['within'])):.0f} % of a check's uses repeat an expert another row uses); the next check's experts already in the last check {np.mean(d['prev1']):.3f}, in the last 4 checks {np.mean(d['prev4']):.3f}")

print("\n## 3. bigram-keyed routing repeats (a token whose (prev id, id) occurred earlier in the window)")
for dom_sel in ("code", "ifeval", "multi"):
    J = []; J0 = []; top1 = []; nb = 0; ntok = 0
    for f in caps:
        if os.path.basename(f).split("_")[1] != dom_sel: continue
        z = np.load(f); ids = z["ids"].astype(np.int32); tok = z["tokens"].astype(np.int64); n = ids.shape[0]; ntok += n
        seen = {}
        rng = np.random.default_rng(0)
        for t in range(1, n):
            key = (int(tok[t - 1]), int(tok[t]))
            if key in seen:
                t0 = seen[key]; nb += 1
                J.append(np.mean([len(set(ids[t, L]) & set(ids[t0, L])) / len(set(ids[t, L]) | set(ids[t0, L])) for L in range(48)]))
                top1.append(np.mean([ids[t, L, 0] == ids[t0, L, 0] for L in range(48)]))
                r = int(rng.integers(0, t)); J0.append(np.mean([len(set(ids[t, L]) & set(ids[r, L])) / len(set(ids[t, L]) | set(ids[r, L])) for L in range(48)]))
            seen.setdefault(key, t)
    if J: print(f"  {dom_sel}: {nb} of {ntok} tokens repeat an earlier bigram ({100 * nb / ntok:.1f} %); their top-10 Jaccard with the earlier occurrence {np.mean(J):.3f} (random pair {np.mean(J0):.3f}); top-1 expert equal {100 * np.mean(top1):.1f} % of layers")

print("\n## 4. consecutive checks' accepted lengths (HPD-0 walks): a branch predictor's signal")
for s in ("code", "math", "ifeval"):
    w = np.load(f"{T}/d02b_walk_hpd0_{s}.npz"); ks = []
    for i in range(len(w["names"])):
        if f"{i}_checks" in w.files: ks.append(w[f"{i}_checks"][:, 0].astype(int))
    a = np.concatenate([k[:-1] for k in ks]); b = np.concatenate([k[1:] for k in ks])
    hi = np.median(a)
    print(f"  {s}: P(k >= median {hi:.0f}) {np.mean(b >= hi):.3f}; given the last check's k >= median {np.mean(b[a >= hi] >= hi):.3f}, given k < median {np.mean(b[a < hi] >= hi):.3f}; corr(k_c, k_c+1) {np.corrcoef(a, b)[0, 1]:.3f}")

print("\n## 5. the decision map's reuse arms (the previous position's output reused): flips by class")
base = json.load(open(f"{O}/baseline.json"))
for f in ("moe_reuse_prev_L12_35.json", "moe_reuse_prev_L08_11.json", "moe_reuse_lag2_L12_35.json", "moe_reuse_lag4_L12_35.json", "moe_reuse_chain_L12_35.json", "gdn_reuse_prev.json", "att_reuse_prev.json"):
    if not os.path.exists(f"{O}/{f}"): print(f, "absent"); continue
    d = json.load(open(f"{O}/{f}")); by = {}
    for name, x in d.items():
        b = base.get(name)
        if not b or "tk" not in x: continue
        for i in range(min(len(b["tk"]), len(x["tk"]))):
            p1, p2 = math.exp(b["lp"][i][0]), math.exp(b["lp"][i][1]); cls = "sure" if p1 >= 0.99 else ("tie" if p1 - p2 <= 0.1 else "mid")
            c = by.setdefault((name.split("_")[1], cls), [0, 0]); c[0] += 1; c[1] += x["tk"][i][0] != b["tk"][i][0]
    print(f"  {f}: " + ", ".join(f"{k[0]}-{k[1]} {100 * v[1] / v[0]:.2f} %" for k, v in sorted(by.items()) if k[1] != "tie"))
