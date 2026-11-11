"""Keep masks for the decision map in dm_oea_masks' format (keep [rows, 48, 10] bool, rank-sorted as DAY9's routing capture;
one <arm>/<window>.npz per window; the harness: PERT.update(rmask=<abs dir>, renorm=False)). Built from the MoonShots export
(ids, checks, dcf). Conventions copied from dm_oea_masks.py: live-cut checks (rows A .. A+L, L drafted rows + the bonus row);
rows after a check's first miss enter the union as the wrong-draft stand-in (row p + 97 mod N) with their own drafts' cf;
only the realized rows (A .. first miss, i.e. A .. A+kept-1) are masked; prompt rows and rows outside the walk keep all 10.
Rules (strict: no resident keep; every masked row also keeps any rank whose expert is in the check's loads = union members):
  c3_r2  = q1u_0.5_strict          : rows with cf >= 0.5 keep ranks 0-1 (+ members); the bonus row exact unless its next draft's cf >= 0.5
  c3_r1  = c3_r2 + unsure k6 L8-35 : rows with cf < 0.5 (and the bonus row without a qualifying draft) keep ranks 0-5 (+ members) in layers 8-35
  c3_t4  = unsure k8 outside, k5 in L8-35 (sure rows ranks 0-1 everywhere)
  c3_r4  = q1u_0_strict_bq         : every row with a draft (cf >= 0, incl. the bonus row with any next draft) keeps ranks 0-1 (+ members)
  c3_r5  = top1_strict_cf0_bq      : as c3_r4 with rank 0 only (+ members)
Self-check: the distinct count implied by the masks (loads union over the block) is printed per domain to compare with sim_lru.py."""
import numpy as np, os, sys, glob
D = "/root/c3mac/data/decision_map_moonshots"; OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/c3/cloud3/masks"
MID = np.array([8 <= l < 36 for l in range(48)]); OUTER = ~MID   # the bonus row without a next draft has cf -1 -> table[5] (exact)
# band tables (02:0x): k per cf band [<0.5, 0.5-0.9, 0.9-0.99, 0.99-0.999, >=0.999, no draft]; 'band' = (k, layers) applied to rows with cf < 0.9
EDGES = [0.0, 0.5, 0.9, 0.99, 0.999, 1.01]
def band_of(c, edges=None):
    e = edges or EDGES
    return int(np.searchsorted(e, c, side="right") - 1) if c >= 0 else len(e) - 1
EDGES2 = [0.0, 0.3, 0.9, 0.999, 1.01]        # c3_b2c: top-8 on [0.3, 0.9), top-5 on [0.9, 0.999), top-2 at >= 0.999
EDGES3 = [0.0, 0.3, 0.9, 0.99, 1.01]         # c3_b2d: top-8 on [0.3, 0.9), top-5 on [0.9, 0.99), top-2 at >= 0.99
ARMS = {
 "c3_b2d_reach07_u7": dict(table=[10, 7, 5, 2, 10], edges=EDGES3, reach=0.7),
 "c3_b2d_reach07": dict(table=[10, 8, 5, 2, 10], edges=EDGES3, reach=0.7),
 "c3_b2d": dict(table=[10, 8, 5, 2, 10], edges=EDGES3),
 "c3_b2c": dict(table=[10, 8, 5, 2, 10], edges=EDGES2),
 "c3_b2b": dict(table=[10, 8, 5, 2, 2, 10]),
 "c3_b2b_top1": dict(table=[10, 8, 5, 1, 1, 10]),
 "c3_b2b_band5": dict(table=[10, 8, 5, 2, 2, 10], band=(5, MID)),
 "c3_new1": dict(table=[10, 10, 8, 4, 2, 10]),
}
if len(sys.argv) > 2: ARMS = {k: v for k, v in ARMS.items() if k in sys.argv[2:]}
for arm, p in ARMS.items():
    od = os.path.join(OUT, arm); os.makedirs(od, exist_ok=True)
    stats = {}
    for f in sorted(glob.glob(os.path.join(D, "*.npz"))):
        z = np.load(f); nm = os.path.basename(f); dom = str(z["domain"]); ids = z["ids"].astype(np.int32); N = ids.shape[0]
        keep = np.ones(ids.shape, bool)
        st = stats.setdefault(dom, dict(pairs=0, kept=0, masked_rows=0, checks=0))
        for (A, R, kept), dc in zip(z["checks"], z["dcf"]):
            L = R - 1
            if A + R > N: continue
            rows = np.arange(A, A + R)
            for i in range(kept, R): rows[i] = (A + i + 97) % N      # stand-ins for rows after the first miss
            cf = np.full(R, -1.0); cf[:L] = dc[:L]
            if L < 15: cf[L] = dc[L]                                   # the bonus row's next draft (bq)
            kk = np.full((R, 48), 10, np.int64)
            reach = np.ones(R); run = np.cumprod(np.where(cf[:L] >= 0, cf[:L], 1.0)); reach[1:L + 1] = run[:L]
            for i in range(R):
                b = band_of(cf[i], p.get("edges")); kk[i, :] = p["table"][b]
                if "reach" in p and cf[i] >= 0 and reach[i] < p["reach"] and kk[i, 0] > 2: kk[i, :] = 2   # REACH-CUT: top-2 + members on rows unlikely to be reached
                if "band" in p and b <= 1 and cf[i] >= 0: kk[i, p["band"][1]] = np.minimum(kk[i, p["band"][1]], p["band"][0])
            I = ids[rows]
            for l in range(48):
                loads = set()
                for r in range(R): loads.update(I[r, l, :int(kk[r, l])].tolist())
                st["pairs"] += len(loads)
                for i in range(kept):                                  # realized rows only
                    row = A + i; k0 = int(kk[i, l])
                    km = np.zeros(10, bool); km[:k0] = True
                    km |= np.isin(ids[row, l], list(loads))
                    keep[row, l] = km
            st["kept"] += int(kept); st["masked_rows"] += int(kept); st["checks"] += 1
        np.savez_compressed(os.path.join(od, nm), keep=keep)
    print(arm + ": " + "; ".join(f"{d}: distinct per kept token {v['pairs']/v['kept']:.1f}, masked rows {v['masked_rows']}, checks {v['checks']}" for d, v in stats.items()), flush=True)
