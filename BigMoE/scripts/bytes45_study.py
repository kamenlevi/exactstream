"""Bytes-per-check study (Kamen 10-31 17:49: the two walls first; BigMoE's ask to NewIdeas 11-01): from cap45 (the exact live
REF2b DFlash2 loop at C 160, S45's six prompts, every gated call's routed ids in rank order), per 16-row check and layer:
distinct experts, and the expert bytes a check touches: 4-bit = MSB-3 2.15 + LSB-1 0.77 = 2.92 MB per expert (B12 notes;
records3 parts), REF2-T 4433333333 = 2.92 MB for an expert used at rank 0-1 by any row, 2.15 MB if only at ranks 2-9.
Rows: a check verifies the first r of its 16 rows (row 0 = the last accepted token, 1..15 = drafts); bytes for r = 1..16
show what draft truncation saves. Output: data/bytes45/bytes_study.txt"""
import glob, os
import numpy as np
D = os.path.expanduser("~/BigMoE/data/cap45")
B4, B3 = 2.92e-3, 2.15e-3                                    # GB per expert
out, R = [], (1, 2, 4, 8, 12, 16)
tot = {r: [] for r in R}; tier = []; dist16 = []
for f in sorted(glob.glob(os.path.join(D, "*_gate_0.npz"))):
    z = np.load(f)
    L, rows, off, ids, seq = z["L"], z["rows"], z["row_off"], z["ids"], z["seq"]
    chk = np.nonzero((rows == 16) & (L < 48))[0]            # trunk layer calls of 16-row checks (commit replays have k+1 rows)
    by = {}
    for c in chk:
        by.setdefault(int(seq[c]), []).append(c)
    per_check = {r: [] for r in R}; per_tier = []; d16 = []
    for s, cs in by.items():
        if len(cs) != 48:
            continue
        b = {r: 0.0 for r in R}; bt = 0.0
        for c in cs:
            x = ids[off[c]:off[c] + 16].astype(np.int64)          # [16, 10] rank order
            for r in R:
                b[r] += len(np.unique(x[:r])) * B4
            u = np.unique(x); hi = np.unique(x[:, :2])
            bt += len(hi) * B4 + (len(u) - len(hi)) * B3
            d16.append(len(u))
        for r in R:
            per_check[r].append(b[r])
        per_tier.append(bt)
    name = os.path.basename(f).replace("_gate_0.npz", "")
    out.append(f"  {name}: {len(per_tier)} checks; distinct experts per layer per 16-row check {np.mean(d16):.1f}; GB per check at "
               + ", ".join(f"r={r} {np.mean(per_check[r]):.2f}" for r in R) + f"; REF2-T 16 rows {np.mean(per_tier):.2f}")
    for r in R:
        tot[r] += per_check[r]
    tier += per_tier; dist16 += d16
m16 = np.mean(tot[16])
out.insert(0, "bytes45 study (cap45, exact live loop at C 160, S45's six prompts): expert bytes a 16-row check touches (RAM side; "
              "SSD = only the misses of those)")
out.append(f"ALL: distinct experts per layer per check {np.mean(dist16):.1f} (of 160 ids); GB per check 4-bit: "
           + ", ".join(f"r={r} {np.mean(tot[r]):.2f} ({np.mean(tot[r]) / m16:.0%})" for r in R)
           + f"; REF2-T (16 rows) {np.mean(tier):.2f} GB ({np.mean(tier) / m16 - 1:+.1%} vs 4-bit)")
open(os.path.expanduser("~/BigMoE/data/bytes45/bytes_study.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out))

# ---- lever table (BigMoE -> NewIdeas 11-01): GB per kept token vs a fixed verify cutoff r, 4-bit and REF2-T. A check's kept
# count k = the rows of its commit replay (keep_tokens = accepted + 1; layer 1 only, same seq); cutting the check to r rows keeps
# min(k, r) (acceptance is a prefix). Upper-bound view: the cutoff here is fixed per prompt, a confidence cutoff adapts per check.
lines = ["", "lever table: GB of expert bytes per KEPT token (lower = better; 350 tok/s at ~280 GB/s needs <= ~0.8)"]
RR = (2, 4, 6, 8, 12, 16)
allb = {(t, r): [0.0, 0.0] for t in (0, 1) for r in RR}
for f in sorted(glob.glob(os.path.join(D, "*_gate_0.npz"))):
    z = np.load(f)
    L, rows, off, ids, seq = z["L"], z["rows"], z["row_off"], z["ids"], z["seq"]
    rp = np.nonzero((L == 1) & (rows < 16))[0]             # REF2's commit replays only the PLE layer (1), same pass seq,
    nxt = {int(seq[b]): int(rows[b]) for b in rp}           # with keep_tokens = accepted + 1 rows = the check's new tokens
    acc = {(t, r): [0.0, 0.0] for t in (0, 1) for r in RR}
    for c in np.nonzero((rows == 16) & (L < 48))[0]:
        s = int(seq[c])
        if s not in nxt:
            continue
        k = nxt[s]
        x = ids[off[c]:off[c] + 16].astype(np.int64)
        for r in RR:
            xr = x[:r]; u = np.unique(xr); hi = np.unique(xr[:, :2])
            for t, b in ((0, len(u) * B4), (1, len(hi) * B4 + (len(u) - len(hi)) * B3)):
                acc[(t, r)][0] += b
                if L[c] == 0:
                    acc[(t, r)][1] += min(k, r)
    name = os.path.basename(f).replace("_gate_0.npz", "")
    lines.append(f"  {name}: 4-bit " + " ".join(f"r{r} {acc[(0, r)][0] / max(acc[(0, r)][1], 1):.2f}" for r in RR)
                 + " | REF2-T " + " ".join(f"r{r} {acc[(1, r)][0] / max(acc[(1, r)][1], 1):.2f}" for r in RR))
    for key, v in acc.items():
        allb[key][0] += v[0]; allb[key][1] += v[1]
lines.append("  ALL: 4-bit " + " ".join(f"r{r} {allb[(0, r)][0] / allb[(0, r)][1]:.2f}" for r in RR)
             + " | REF2-T " + " ".join(f"r{r} {allb[(1, r)][0] / allb[(1, r)][1]:.2f}" for r in RR))
open(os.path.expanduser("~/BigMoE/data/bytes45/bytes_study.txt"), "a").write("\n".join(lines) + "\n")
print("\n".join(lines))

# ---- scoreboard columns (NewIdeas 11-01 06:1x): kept per check (cap45, the gate loop at C 160, 16 rows) and the SSD side vs C
# (Fit12's R-57b-R replay, misses per 16-row check, no prefetch: C 192 545, 224 424, 256 345, 288 297, 320 268), at 2.92 MB per
# 4-bit miss (REF2-T reads ~0.6x those bytes per miss: M-08's SSD MB -40-44 %).
FIT12 = {192: 545, 224: 424, 256: 345, 288: 297, 320: 268}
lines = ["", "scoreboard columns: kept tokens per 16-row check (cap45) and SSD GB per check vs C (Fit12's replay misses x 2.92 MB)"]
kept = {}
for f in sorted(glob.glob(os.path.join(D, "*_gate_0.npz"))):
    z = np.load(f)
    L, rows = z["L"], z["rows"]
    k = rows[(L == 1) & (rows < 16)]
    kept[os.path.basename(f).replace("_gate_0.npz", "")] = float(np.mean(k)) if len(k) else float("nan")
lines.append("  kept per check: " + ", ".join(f"{n} {v:.2f}" for n, v in kept.items()) + f"; mean {np.mean(list(kept.values())):.2f}")
lines.append("  SSD GB per check (4-bit): " + ", ".join(f"C {c} {m * B4:.2f} ({m} misses)" for c, m in FIT12.items())
             + f"; SSD GB per kept token at the mean kept: " + ", ".join(f"C {c} {m * B4 / np.mean(list(kept.values())):.2f}" for c, m in FIT12.items()))
open(os.path.expanduser("~/BigMoE/data/bytes45/bytes_study.txt"), "a").write("\n".join(lines) + "\n")
print("\n".join(lines))
