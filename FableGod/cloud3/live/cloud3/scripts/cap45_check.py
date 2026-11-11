"""cap45 replay pickle: 6 prompts x 5 real engine 16-row checks (ids per layer (16,10), kept). Union all 16 rows vs kept rows."""
import pickle, numpy as np
a, _ = pickle.load(open("/root/c3mac/data/band45_cap45.pkl", "rb"))
U = []; UK = []; KEPT = []; names = []
for nm, checks, res0 in a:
    for c in checks:
        rid = c["rid"]; k = int(c["kept"])
        u = np.mean([len(set(rid[l].ravel().tolist())) for l in range(48)])
        uk = np.mean([len(set(rid[l][:k].ravel().tolist())) for l in range(48)])
        U.append(u); UK.append(uk); KEPT.append(k); names.append(nm)
U = np.array(U); UK = np.array(UK); KEPT = np.array(KEPT)
print(f"cap45: {len(U)} checks; kept per check {KEPT.mean():.1f} (values {sorted(KEPT.tolist())}); union per layer 16 rows {U.mean():.1f}; kept rows only {UK.mean():.1f} (x{UK.sum()/U.sum():.3f}); distinct per kept token {48*U.sum()/KEPT.sum():.0f} vs kept-only {48*UK.sum()/KEPT.sum():.0f}")
for nm in dict.fromkeys(names):
    m = np.array([n == nm for n in names]); print(f"  {nm:34s} kept {KEPT[m].mean():5.1f} union {U[m].mean():5.1f} kept-only {UK[m].mean():5.1f}")
print("res0 sizes per layer (prompt 0):", sorted(set(len(s) for s in a[0][2])))
