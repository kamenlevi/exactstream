"""groups_count.py (Leap, Mon 16 Nov 2026 14:1x; CHECK25): CPU only. The lossy one-call (ooo1cq) groups its pairs by (expert,
tier); an expert used at 4-bit by some rows and at MSB-3 by others is two groups, its MSB plane read twice. Counts, on the
decision map's 32-row blocks of real 1c routing (all 48 layers, code windows): distinct experts vs kernel groups, and
the bytes each reads (records3 pieces). usage: [BigMoE venv] python -I groups_count.py"""
# CPU only: the one-call's (expert, tier) groups vs distinct experts per 32-row block, real 1c routing (decision map)
import json, os
import numpy as np
MAP = os.path.expanduser("~/FableGod/decision_map/out")
basej = json.load(open(os.path.join(MAP, "baseline.json")))
code = [f for f in sorted(os.listdir(os.path.join(MAP, "xcap"))) if basej[f]["dom"] == "code"]
MSB, LSB, R4 = 2.064384, 0.720048, 2
tot = dict(uniq=0.0, grp=0.0, blocks=0, ng=0, nu=0, mixed=0, dropx=0)
per_layer = {}
for f in code:
    z = np.load(os.path.join(MAP, "xcap", f))
    tz = np.load(os.path.join(MAP, "bt", "tiers_1c", f))
    a0 = int(z["a"])
    n = z["ids"].shape[0]
    for LM in range(z["ids"].shape[1]):
        for b0 in range(0, n - 31, 32):
            ids = z["ids"][b0:b0 + 32, LM].astype(np.int64)
            k4 = tz["keep"][a0 + b0:a0 + b0 + 32, LM].astype(bool)
            k3 = tz["keep3"][a0 + b0:a0 + b0 + 32, LM].astype(bool)
            kept = k4 | k3
            idk = np.where(kept, ids, ids[:, :1])                   # a dropped use reads its row's rank-0 expert
            rank = np.broadcast_to(np.arange(ids.shape[1]), ids.shape)
            tier = np.where((rank < R4) & k4, 4, 3)                 # rank 0-1 at 4-bit unless nolsb (MSB-3 or dropped)
            groups = set(zip(idk.reshape(-1).tolist(), tier.reshape(-1).tolist()))
            from collections import Counter
            cnt_g = Counter(zip(idk.reshape(-1).tolist(), tier.reshape(-1).tolist()))   # pairs per group
            segs = sum((c + 15) // 16 for c in cnt_g.values())                            # the kernel cuts every 16 pairs
            s_bytes = sum(((c + 15) // 16) * (MSB + (LSB if t == 4 else 0.0)) for (e, t), c in cnt_g.items())
            tot["segs"] = tot.get("segs", 0) + segs
            tot["sbytes"] = tot.get("sbytes", 0.0) + s_bytes
            gk = set(zip(idk[kept].tolist(), tier[kept].tolist()))          # groups from kept uses only (drops dropped)
            gx = groups - gk                                                 # groups that exist only because of drops
            tot["gk"] = tot.get("gk", 0.0) + sum(MSB + (LSB if t == 4 else 0.0) for e, t in gk)
            tot["gx"] = tot.get("gx", 0) + len(gx)
            ex = set(idk[kept].tolist()) if kept.any() else set(idk[:, 0].tolist())
            e4 = set(ids[:, :R4][k4[:, :R4]].tolist())
            g_bytes = sum(MSB + (LSB if t == 4 else 0.0) for e, t in groups)
            u_bytes = len(ex | {e for e, t in groups}) * MSB + len(e4) * LSB
            tot["grp"] += g_bytes; tot["uniq"] += u_bytes; tot["blocks"] += 1
            tot["ng"] += len(groups); tot["nu"] += len(ex | {e for e, t in groups})
            es = {}
            for e, t in groups:
                es.setdefault(e, set()).add(t)
            tot["mixed"] += sum(1 for e, ts in es.items() if len(ts) > 1)
            pl = per_layer.setdefault(LM, [0.0, 0.0])
            pl[0] += g_bytes; pl[1] += u_bytes
b = tot["blocks"]
print(f"{len(code)} code windows, {b} (block, layer) samples of 32 rows")
print(f"per 32-row block and layer: distinct experts {tot['nu'] / b:.1f}, kernel groups {tot['ng'] / b:.1f}, experts in two "
      f"tier groups {tot['mixed'] / b:.1f}")
print(f"bytes per block-layer: unique reads {tot['uniq'] / b:.1f} MB vs the kernel's group reads {tot['grp'] / b:.1f} MB = "
      f"x{tot['grp'] / tot['uniq']:.3f}")
r = sorted((v[0] / v[1], L) for L, v in per_layer.items())
print(f"  of which: groups that exist only for dropped uses (weight 0) {tot['gx'] / b:.1f} per block-layer; kept-use groups "
      f"alone read {tot['gk'] / b:.1f} MB = x{tot['gk'] / tot['uniq']:.3f} of unique")
print(f"  segments (groups cut every 16 pairs, each reads its weights): {tot['segs'] / b:.1f} per block-layer -> "
      f"{tot['sbytes'] / b:.1f} MB = x{tot['sbytes'] / tot['uniq']:.3f} of unique")
print(f"per layer ratio: min {r[0][0]:.3f} (L{r[0][1]}), median {r[len(r) // 2][0]:.3f}, max {r[-1][0]:.3f} (L{r[-1][1]})")
