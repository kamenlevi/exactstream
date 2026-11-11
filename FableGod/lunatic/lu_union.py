#!/usr/bin/env python3 -I
"""lu_union.py (Lunatic, round 1, Tue 3 Nov 2026 00:3x EET). CPU only, recorded data only (numpy).
Reads the 36 REF2-T-class routing captures (~/FableGod/decision_map/out/routing/*.npz: ids [n-1, 48, 10] rank-sorted
top-10 expert ids per row and layer, tokens, gen) and prices, on the UNION per block (an expert read serves every row
of the block that routes to it):
 A. union(R): distinct experts per layer in R consecutive generated rows, R = 1..256, per domain; the answer's
    whole working set per layer.
 B. cross-stream: B independent windows of one domain decoded side by side (aligned 16-row blocks): union per layer
    vs B x the single-stream union (does batching parallel streams share expert reads?).
 C. per kept token: (expert, layer) groups and MB (1.9 MB per group, AUDIT_350 1.1: 5.13 GB / 56.1 / 48) plus the
    trunk 3.44 GB / kept, for block R and per-token agreement alpha (kept = accepted run + bonus, capped at R + 1).
"""
import glob, os, numpy as np
rng = np.random.default_rng(7)
D = os.path.expanduser("~/FableGod/decision_map/out/routing")
W = {}
for f in sorted(glob.glob(os.path.join(D, "*.npz"))):
    z = np.load(f, allow_pickle=True); ids = z["ids"]; gen = np.asarray(z["gen"])
    n = ids.shape[0]
    if gen.ndim == 0:
        st = max(int(gen) - 1, 0); g = ids[st:]
    else:
        m = gen.astype(bool); m = m[1:] if m.shape[0] == n + 1 else m[:n]; g = ids[m]
    dom = os.path.basename(f).split("_")[1]
    W.setdefault(dom, []).append(g.astype(np.int32))
print("windows:", {k: len(v) for k, v in W.items()}, " gen rows:", {k: int(sum(x.shape[0] for x in v)) for k, v in W.items()})

def union_R(g, R):            # mean distinct experts per layer over non-overlapping R-row blocks
    out = []
    for s in range(0, g.shape[0] - R + 1, R):
        b = g[s:s + R]                                   # [R, 48, 10]
        out.append(np.mean([np.unique(b[:, L, :]).size for L in range(48)]))
    return np.mean(out) if out else np.nan

Rs = [1, 2, 4, 8, 16, 32, 64, 128, 256]
U = {}
print("\nA. union per layer by block length R (mean over blocks and windows); rows per group = 10R/union")
print("dom    " + " ".join(f"{R:>6d}" for R in Rs) + "   whole-answer WS (rows)")
for dom, gs in W.items():
    u = [np.nanmean([union_R(g, R) for g in gs if g.shape[0] >= R]) for R in Rs]
    U[dom] = dict(zip(Rs, u))
    ws = np.mean([np.mean([np.unique(g[:, L, :]).size for L in range(48)]) for g in gs])
    print(f"{dom:6s} " + " ".join(f"{x:6.1f}" for x in u) + f"   {ws:6.1f} ({np.mean([g.shape[0] for g in gs]):.0f})")
    print("       " + " ".join(f"{10*R/x:6.2f}" for R, x in zip(Rs, u)) + "   rows/group")

print("\nB. cross-stream: B windows of one domain side by side, aligned 16-row blocks (mean per layer)")
for dom, gs in W.items():
    line = []
    for B in [1, 2, 4, 8, 12]:
        if B > len(gs): continue
        vals = []
        for _ in range(40):
            pick = rng.choice(len(gs), B, replace=False); L0 = min(gs[i].shape[0] for i in pick)
            for s in range(0, L0 - 15, 16):
                vals.append(np.mean([np.unique(np.concatenate([gs[i][s:s+16, L, :].ravel() for i in pick])).size for L in range(0, 48, 3)]))
        line.append(f"B{B} {np.mean(vals):6.1f} (x{np.mean(vals)/U[dom][16]/B:.2f} of B x single)")
    print(f"{dom:6s} " + "  ".join(line))
allg = [g for v in W.values() for g in v]
vals = []
for _ in range(60):
    pick = rng.choice(len(allg), 8, replace=False); L0 = min(allg[i].shape[0] for i in pick)
    for s in range(0, L0 - 15, 16):
        vals.append(np.mean([np.unique(np.concatenate([allg[i][s:s+16, L, :].ravel() for i in pick])).size for L in range(0, 48, 3)]))
print(f"mixed  B8 {np.mean(vals):6.1f} per layer (random 512-way draw of 8 x 66 would give {512*(1-(1-66/512)**8):.0f})")

print("\nC. per kept token at agreement alpha (kept = (1 - a^(R+1)) / (1 - a)): groups, expert GB, trunk GB, total GB")
for dom in ["code", "ifeval"]:
    for a in [0.91, 0.94, 0.96, 0.98, 0.99, 0.995]:
        row = []
        for R in [8, 16, 32, 64, 128]:
            k = (1 - a ** (R + 1)) / (1 - a); grp = U[dom][R] * 48 / k
            row.append(f"R{R}: k {k:5.1f} {grp:6.0f} g {grp*1.9e-3 + 3.44/k:5.2f} GB")
        print(f"{dom:6s} a {a:.3f}  " + " | ".join(row))
