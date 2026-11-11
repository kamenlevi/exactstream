"""Gate-weight anatomy on Breakthrough/keepset/qom_vectors.npz (arm qom_0_live_ball_c147_t3; one window per domain, all 48
layers; valid = the realized live-cut rows; scores = renormalised top-10 weights; conf = drafter cf; resident = C147 pool).
Per (block, layer): the block's summed mass per expert; the number of experts covering 80 / 90 / 95 % of the mass vs the
union; MoE-Spec budget B = ceil(0.5 x union) by summed mass: the share of rows whose rank-0 / rank-1 expert is outside;
singletons vs shared: mean weight; share of the union with max row weight < 0.02 / 0.05 / 0.1; per-rank mean weight."""
import numpy as np
z = np.load("/root/350-tps/Breakthrough/keepset/qom_vectors.npz")
dom = z["domain"]; ids = z["ids"]; sc = z["scores"]; valid = z["valid"]; conf = z["conf"]; res = z["resident"]
for d in ["code", "ifeval", "multi"]:
    m = np.where(dom == d)[0]
    cov = np.zeros(3); nU = 0; nb = 0; r0_out = 0; r1_out = 0; nrows = 0; single_w = []; shared_w = []
    light = np.zeros(3); rankw = np.zeros(10); rankn = 0
    res_share = 0; res_n = 0
    for i in m:
        v = valid[i]; R = int(v.sum())
        if R == 0: continue
        I = ids[i][v]; S = sc[i][v]
        rankw += S.sum(0); rankn += R
        mass = {}; cnt = {}; maxw = {}
        for r in range(R):
            for k in range(10):
                e = int(I[r, k]); w = float(S[r, k])
                mass[e] = mass.get(e, 0.0) + w; cnt[e] = cnt.get(e, 0) + 1; maxw[e] = max(maxw.get(e, 0.0), w)
        U = len(mass); nU += U; nb += 1
        order = sorted(mass.items(), key=lambda kv: -kv[1]); tot = sum(mass.values())
        c = np.cumsum([kv[1] for kv in order]) / tot
        for j, t in enumerate([0.8, 0.9, 0.95]): cov[j] += int(np.searchsorted(c, t) + 1)
        B = int(np.ceil(0.5 * U)); keep = set(e for e, _ in order[:B])
        r0_out += sum(int(I[r, 0]) not in keep for r in range(R)); r1_out += sum(int(I[r, 1]) not in keep for r in range(R)); nrows += R
        for e in mass:
            (single_w if cnt[e] == 1 else shared_w).append(maxw[e])
            for j, t in enumerate([0.02, 0.05, 0.1]): light[j] += maxw[e] < t
        res_share += res[i][v].mean(); res_n += 1
    print(f"\n== {d}: blocks x layers {nb}; union per layer {nU/nb:.1f}; experts covering 80/90/95 % of the block's mass: {cov[0]/nb:.1f} / {cov[1]/nb:.1f} / {cov[2]/nb:.1f} (= x{cov[0]/nU:.2f} / x{cov[1]/nU:.2f} / x{cov[2]/nU:.2f} of the union)")
    print(f"   MoE-Spec budget B = 0.5 x union by summed mass: rows whose rank-0 expert is outside the budget {r0_out/nrows:.3f}, rank-1 outside {r1_out/nrows:.3f}")
    print(f"   singletons: mean max-weight {np.mean(single_w):.3f} (n {len(single_w)}), shared experts {np.mean(shared_w):.3f} (n {len(shared_w)}); union share with max row weight < 0.02 / 0.05 / 0.1: {light[0]/nU:.3f} / {light[1]/nU:.3f} / {light[2]/nU:.3f}")
    print(f"   mean renormalised weight by rank 0..9: {' '.join(f'{x:.3f}' for x in rankw/rankn)}; resident share of (row,rank) pairs at C147 after loads: {res_share/res_n:.3f}")
