"""x5_seg_merge.py (Breakthrough, Tue 17 Nov 2026 10:3x; EXPERT REWRITE, CPU count): how many one-call segments would an
UPGRADE save? Today a segment is one (expert, tier, sidecar state) group of <= 16 pairs (ooo1cq._GRP1CQ). An expert
picked at rank 0-1 by some rows (tier 4: MSB + LSB) and at ranks 2-9 by others (tier 3: MSB-3) runs as two segments,
though its LSB plane is read anyway. UPGRADE: such tier-3 pairs run at tier 4 (more accurate: 4-bit >= 3-bit; no extra
bytes), so the expert's pairs form one key. MMA, decode and threadgroup counts scale with segments. Real routing:
the decision map's code windows, 32-row blocks, every layer in the capture, the km path (dropped uses read the row's
rank-0 id; they form groups too) and nolsb (rank 0-1 uses not kept at 4 bits read MSB-3: tier 3, sidecar -2).
Usage: ~/BigMoE/.venv/bin/python -I -B scripts/xrw/x5_seg_merge.py  -> data/xrw/x5_seg_merge.txt"""
import json
import os

import numpy as np

MAP = os.path.expanduser("~/FableGod/decision_map/out")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "xrw", "x5_seg_merge.txt")
R4 = 2
RW = int(os.environ.get("X5_ROWS", "32"))


def nseg(keys):
    _, c = np.unique(keys, return_counts=True)
    return int(np.sum((c + 15) // 16))


def main():
    basej = json.load(open(os.path.join(MAP, "baseline.json")))
    code = [f for f in sorted(os.listdir(os.path.join(MAP, "xcap"))) if basej[f]["dom"] == "code"]
    rows = []
    for f in code:
        z = np.load(os.path.join(MAP, "xcap", f))
        tz = np.load(os.path.join(MAP, "bt", "tiers_1c", f))
        a = int(z["a"])
        n, NLy = z["ids"].shape[0], z["ids"].shape[1]
        for Lm in range(NLy):
            if tz["keep"].shape[1] <= Lm:
                continue
            for b0 in range(0, n - RW + 1, RW):
                ids = z["ids"][b0:b0 + RW, Lm].astype(np.int64)
                k4 = tz["keep"][a + b0:a + b0 + RW, Lm].astype(bool)
                k3 = tz["keep3"][a + b0:a + b0 + RW, Lm].astype(bool)
                kept = k4 | k3
                idk = np.where(kept, ids, ids[:, :1])
                tier = np.full(idk.shape, 3)
                tier[:, :R4] = np.where(k4[:, :R4], 4, 3)       # nolsb rank 0-1 uses run MSB-3
                e = idk.reshape(-1)
                t = tier.reshape(-1)
                now = nseg(e * 8 + t)
                up = set(e[t == 4].tolist())
                t2 = np.where(np.isin(e, list(up)), 4, t)
                merged = nseg(e * 8 + t2)
                rows.append((Lm, now, merged, len(np.unique(e)), int((t == 4).sum()), int(((t2 == 4) & (t == 3)).sum())))
    A = np.array(rows, dtype=float)
    lines = [f"X5 SEGMENT MERGE (CPU; {len(code)} code windows, every captured layer, {len(rows)} {RW}-row blocks)",
             f"  segments per call today {A[:, 1].mean():.1f}; with the tier-4 upgrade {A[:, 2].mean():.1f} "
             f"({100 * (1 - A[:, 2].sum() / A[:, 1].sum()):.1f} % fewer); distinct experts {A[:, 3].mean():.1f}; tier-4 pairs "
             f"{A[:, 4].mean():.1f} per call; tier-3 pairs upgraded {A[:, 5].mean():.1f} of {RW * 10 - A[:, 4].mean():.1f}"]
    for Lm in (2, 20, 44):
        s = A[A[:, 0] == Lm]
        if len(s):
            lines.append(f"  layer {Lm}: {s[:, 1].mean():.1f} -> {s[:, 2].mean():.1f} segments "
                         f"({100 * (1 - s[:, 2].sum() / s[:, 1].sum()):.1f} % fewer)")
    print("\n".join(lines), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "a").write("\n".join(lines) + "\n\n")


if __name__ == "__main__":
    main()
