"""Round 32j: where does the 46 ms of per-pass compute exceed the matvec law? (data/pass_profile125.json, frac 0.5)
between_L = t_seen[L+1] - t_rel[L] = L's MoE (routed + shared) + L+1's attention side + router. Bytes per interval:
moe_b[L] (L's non-routed MoE weights) + att_b[L+1] (L+1's attention-side weights) + L's unique routed experts x 2.765 MB
(uni "seq:L") + GDN recurrent state (fp32 read + write) if L+1 is GDN. Law: bytes / 300 GB/s + k x 20 us, k = dependent
big kernels; reported: measured ms, bytes-only ms, and the implied dependent-kernel count (measured - bytes) / 20 us,
by the kind of L+1. usage: layer_law125.py   (CPU, instant)"""
import collections, json
import numpy as np
d = json.load(open("/Users/kamen/MoonShots/data/pass_profile125.json"))
att_b, moe_b, kinds, gst = d["att_b"], d["moe_b"], d["kinds"], d["gdn_state"]
acc = collections.defaultdict(list)
for run in d["EV"]:
    by = collections.defaultdict(dict)
    for L, seq, kind, ts, tr in run["events"]:
        by[int(seq)][int(L)] = (ts, tr)
    uni = run["uni"]
    for seq, lay in by.items():
        if len(lay) < 48:
            continue
        for L in range(47):
            ms = (lay[L + 1][0] - lay[L][1]) * 1e3
            u = uni.get(f"{seq}:{L}")
            if u is None:
                continue
            b = moe_b[L] + att_b[L + 1] + u * 2.765e6 + (2 * gst if kinds[L + 1] == "gdn" else 0)
            acc[kinds[L + 1]].append((ms, b / 300e6, u))           # bytes / 300 GB/s in ms
for k, v in acc.items():
    a = np.array(v)
    med, bo = np.median(a[:, 0]), np.median(a[:, 1])
    print(f"next layer {k:4s}: n {len(a):5d} | measured p50 {med:.3f} ms (mean {a[:, 0].mean():.3f}, p90 {np.percentile(a[:, 0], 90):.3f}) | "
          f"bytes-only p50 {bo:.3f} ms ({bo / med:.0%}) | implied dependent kernels (p50) {(med - bo) / 0.020:.1f} x 20 us | "
          f"unique experts {a[:, 2].mean():.1f}")
allv = np.concatenate([np.array(v) for v in acc.values()])
print(f"per pass (47 intervals): measured p50-sum {np.median(allv[:, 0]) * 47:.1f} ms (mean-sum {allv[:, 0].mean() * 47:.1f}), "
      f"bytes-only {allv[:, 1].mean() * 47:.1f} ms")
