"""P2 anatomy of live-cut checks on the 36 windows: where each distinct (expert, layer) comes from.
Per check and layer, each distinct expert is attributed to the FIRST category that holds (priority order):
  A: needed at rank 0-1 by a kept row (accepted draft rows + the first-miss / bonus row);
  B: needed at rank 2-9 by a kept row (and by no kept row at rank 0-1);
  C: needed only by rejected / past-cut rows (rows after the kept ones).
Also: multiplicity (how many rows of the check use the expert), the share of singletons, by layer band, by domain;
overlap of consecutive rows (same layer): mean |top10(j) & top10(j-1)| / 10 and vs a random baseline 10/512;
the union growth with depth inside the check: distinct after rows 0..i."""
import numpy as np, sys, json
sys.path.insert(0, "/tmp/claude-0/-home-user-OjoX/d0b96668-0fdb-5652-b283-b9e2b9efeb75/scratchpad/s")
from sim_union import load, checks_of
W = load()
BANDS = {"L0-7": range(0, 8), "L8-35": range(8, 36), "L36-47": range(36, 48)}
out = {}
for dom in ["code", "ifeval", "multi"]:
    cat = {b: np.zeros(3) for b in BANDS}; mult = {b: np.zeros(17) for b in BANDS}; tot = {b: 0 for b in BANDS}
    kept_rank01 = 0; kept_rank29 = 0; rej = 0; bonus_only = 0; nU = 0
    ovl = []; ovl_rank2 = []
    growth = np.zeros(17); growth_n = np.zeros(17)
    growth_kept = np.zeros(17); growth_kept_n = np.zeros(17)
    for w in W:
        if w["dom"] != dom: continue
        ids = w["ids"]
        for ch in checks_of(w):
            rows = ch["rows"]; kept = ch["kept"]; R = ch["R"]
            for l in range(48):
                band = [b for b, rg in BANDS.items() if l in rg][0]
                I = ids[rows, l, :]
                cnt = {}
                for r in range(R):
                    for e in I[r]: cnt[int(e)] = cnt.get(int(e), 0) + 1
                a01 = set(I[:kept, :2].ravel().tolist()); a29 = set(I[:kept, 2:].ravel().tolist()) - a01
                allk = a01 | a29
                for e, c in cnt.items():
                    tot[band] += 1; mult[band][min(c, 16)] += 1
                    if e in a01: cat[band][0] += 1
                    elif e in a29: cat[band][1] += 1
                    else: cat[band][2] += 1
                nU += len(cnt)
                # bonus-only: experts used only by the last kept row (the bonus / first-miss row) and no other kept row
                if kept >= 2:
                    others = set(I[:kept - 1].ravel().tolist()); last = set(I[kept - 1].ravel().tolist())
                    bonus_only += len(last - others - set(I[kept:].ravel().tolist()))
                # consecutive-row overlap within the check
                for r in range(1, R):
                    ovl.append(len(set(I[r].tolist()) & set(I[r - 1].tolist())) / 10)
                    ovl_rank2.append(len(set(I[r, :2].tolist()) & set(I[r - 1].tolist())) / 2)
                # growth with depth
                s = set()
                for r in range(R):
                    s.update(I[r].tolist()); growth[r] += len(s); growth_n[r] += 1
    res = {}
    for b in BANDS:
        n = tot[b]; res[b] = dict(A_kept_rank01=cat[b][0] / n, B_kept_rank29=cat[b][1] / n, C_rejected_only=cat[b][2] / n,
                                 singleton=mult[b][1] / n, used_by_2=mult[b][2] / n, used_by_3plus=mult[b][3:].sum() / n)
    nall = sum(tot.values())
    res["all"] = {k: sum(res[b][k] * tot[b] for b in BANDS) / nall for k in res["L0-7"]}
    res["bonus_only_share"] = bonus_only / nU
    res["consecutive_overlap_top10"] = float(np.mean(ovl)); res["consecutive_overlap_rank01_in_prev"] = float(np.mean(ovl_rank2))
    res["growth"] = (growth / np.maximum(growth_n, 1)).round(1).tolist()
    out[dom] = res
    print(f"\n== {dom}: distinct (expert,layer) attribution, share of the union (live-cut checks)")
    print(f"{'band':8s} {'A kept r0-1':>12s} {'B kept r2-9':>12s} {'C rejected':>11s} | {'single':>7s} {'by2':>6s} {'by3+':>6s}")
    for b in list(BANDS) + ["all"]:
        r = res[b]; print(f"{b:8s} {r['A_kept_rank01']:12.3f} {r['B_kept_rank29']:12.3f} {r['C_rejected_only']:11.3f} | {r['singleton']:7.3f} {r['used_by_2']:6.3f} {r['used_by_3plus']:6.3f}")
    print(f"bonus/first-miss row's private experts: {res['bonus_only_share']:.3f} of the union; consecutive-row overlap of top-10 {res['consecutive_overlap_top10']:.3f} (random 0.020); prev row holds this row's rank 0-1: {res['consecutive_overlap_rank01_in_prev']:.3f}")
    print("union growth with rows 1..16 (mean distinct per layer after r rows):", " ".join(f"{x:.0f}" for x in res["growth"][:16]))
json.dump(out, open("/tmp/claude-0/-home-user-OjoX/d0b96668-0fdb-5652-b283-b9e2b9efeb75/scratchpad/s/anatomy_out.json", "w"), indent=1)
