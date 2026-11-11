"""Per-layer tile savings of 'unsure rows at k 5' on top of strict T1 (Q1 0.5 + BQ), live-cut checks, per domain: which layers
save the most tiles (to rank the band's layers against the map's per-layer sensitivity when it arrives)."""
import numpy as np, sys
sys.path.insert(0, "/tmp/claude-0/-home-user-OjoX/d0b96668-0fdb-5652-b283-b9e2b9efeb75/scratchpad/s")
from sim_union import load, checks_of
W = load()
for dom in ["code", "ifeval", "multi"]:
    base = np.zeros(48); cut = np.zeros(48); n = 0; K = 0
    for w in W:
        if w["dom"] != dom: continue
        for ch in checks_of(w):
            I = w["ids"][ch["rows"]]; R = ch["R"]; cf = ch["cf"]; has = cf == cf; c = np.nan_to_num(cf, nan=-1); sure = has & (c >= 0.5)
            k1 = np.where(sure, 2, 10); k5 = np.where(sure, 2, 5); K += ch["kept"]
            for l in range(48):
                s1 = set(); s5 = set()
                for r in range(R):
                    s1.update(I[r, l, :k1[r]].tolist()); s5.update(I[r, l, :k5[r]].tolist())
                base[l] += len(s1); cut[l] += len(s5)
    sav = (base - cut) / K
    print(f"{dom}: T1 tiles per kept token by layer (sum {base.sum()/K:.1f}); saving from unsure k5 per layer (sum {sav.sum():.1f} = x{1-sav.sum()/(base.sum()/K):.3f}):")
    print("   T1  :", " ".join(f"{x/K:4.1f}" for x in base))
    print("   save:", " ".join(f"{x:4.1f}" for x in sav))
    bands = {"0-7": range(0, 8), "8-35": range(8, 36), "36-47": range(36, 48), "0-2": range(0, 3), "3-7": range(3, 8)}
    print("   band shares of the saving:", {b: round(sav[list(r)].sum() / sav.sum(), 3) for b, r in bands.items()}, " per-layer mean saving:", {b: round(sav[list(r)].mean(), 2) for b, r in bands.items()})
