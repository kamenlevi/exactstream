"""C3-R6 massB_0.5 masks (Cloud Fable 3; the orchestrator 07:2x: form (a), from the baseline's router scores, with the exact
all-rows scores capture). Rule (REQUESTS.md C3-R6 + REPORT_P2 section 4, MoE-Spec's budget): per (check, layer), the union U
= every distinct expert in the check rows' top-10; B = ceil(0.5 x |U|); the kept set K = the B experts of U with the largest
summed renormalised router weight over the check's rows (ties broken by the lower expert id); every realized row keeps its
ranks 0-1 (Fable C's ruling 3, AUDIT_1105 section 7; --as-written turns this floor off) and its top-10 experts that are in K
(strict: nothing kept for being resident). A row without router scores takes the mean renormalised weight by rank as its mass
(ruling 2; a fallback only: the all-rows capture scores every row). Checks and rows follow cloud3's mask conventions
(the MoonShots export's live-cut checks; rows after the first miss enter as stand-ins (A + i + 97) mod N; only realized rows
A .. A+kept-1 are masked; prompt rows and rows outside every check keep all 10; checks past the window's end skipped).
Scores: <scores dir>/<window>.npz (scores [n - 1, 48, 10] float16 beside ids; the scapture). A row without scores (the
answer-row-only xcap source) makes its check "unscored": with --partial such checks keep all 10 on their realized rows (a
test mode only); the real build needs every row scored.
Out: <out dir>/<window>.npz keep [n - 1, 48, 10] bool (cloud3 / dm_oea_masks format). Prints the rank-0 / rank-1 drop rates
(REPORT_P2: 0.000 / 0.000-0.002) and the kept share.
Usage (BigMoE venv): python build_massb.py <scores dir> <out dir> [--xcap] [--partial]"""
import math, os, sys
import numpy as np
MS = os.path.expanduser("~/FableGod/cloud3/macpub/data/decision_map_moonshots")
ROUT = os.path.expanduser("~/FableGod/decision_map/out/routing")
args = [a for a in sys.argv[1:] if not a.startswith("--")]; XCAP = "--xcap" in sys.argv; PARTIAL = "--partial" in sys.argv
NOSTAND = "--no-standins" in sys.argv   # variant (test): the block = the realized rows only
FLOOR = "--as-written" not in sys.argv   # Fable C's ruling 3 (default): ranks 0-1 kept per row, then the budget
PROFILE = np.array([0.227, 0.150, 0.117, 0.097, 0.084, 0.075, 0.069, 0.064, 0.060, 0.057], np.float32)   # the engine's mean renormalised top-10 weight by rank (the orchestrator Thu 18:5x; was cloud3 P2 section 4's 0.294 ... 0.010)
src, out = os.path.expanduser(args[0]), os.path.expanduser(args[1]); os.makedirs(out, exist_ok=True)
names = sorted(f for f in os.listdir(MS) if f.endswith(".npz"))
st = {}
for nm in names:
    z = np.load(os.path.join(MS, nm)); ids = np.load(os.path.join(ROUT, nm))["ids"].astype(np.int64); N = ids.shape[0]
    assert np.array_equal(z["ids"], ids), nm
    s = np.load(os.path.join(src, nm)); sc = np.full((N, 48, 10), np.nan, np.float32)
    if XCAP:                                                   # answer rows only (row a + i)
        a = int(s["a"]); sc[a:a + s["w"].shape[0]] = s["w"].astype(np.float32); assert np.array_equal(s["ids"], ids[a:a + s["w"].shape[0]]), nm
    else:
        assert s["scores"].shape == (N, 48, 10) and np.array_equal(s["ids"], ids), (nm, s["scores"].shape, N)
        sc = s["scores"].astype(np.float32)
    keep = np.ones((N, 48, 10), bool); d = st.setdefault(str(z["domain"]), np.zeros(7, np.int64))   # rows, r0 out, r1 out, kept, cells, unscored checks, floor adds
    for (A, R, kept) in z["checks"].tolist():
        A, R, kept = int(A), int(R), int(kept)
        if A + R > N: continue
        rows = [A + i if i < kept else (A + i + 97) % N for i in range(R)]
        if NOSTAND: rows = rows[:kept]
        if np.isnan(sc[rows]).any():
            d[5] += 1
            if PARTIAL: continue                                # test mode: skip the check
            # Fable C's ruling 2 (AUDIT_1105 section 7): a row without router scores takes the mean renormalised weight by
            # rank as its mass vector, its ids unchanged (only a fallback: the all-rows capture scores every row)
            for i, r in enumerate(rows):
                if np.isnan(sc[r]).any(): sc[r] = PROFILE[None, :]
        for l in range(48):
            I, W = ids[rows, l], sc[rows, l]                    # [R, 10] each
            U, inv = np.unique(I.reshape(-1), return_inverse=True)
            mass = np.bincount(inv, weights=W.reshape(-1).astype(np.float64), minlength=len(U))
            B = math.ceil(0.5 * len(U))
            order = np.lexsort((U, -mass))                      # descending mass, then the lower id
            K = U[order[:B]]
            for i in range(kept):
                kb = np.isin(ids[A + i, l], K); k = kb.copy()   # kb: the budget's own set
                if FLOOR: k[:2] = True                          # Fable C's ruling 3: ranks 0-1 always, then the budget
                keep[A + i, l] = k
                d[0] += 1; d[1] += not kb[0]; d[2] += not kb[1]; d[3] += int(k.sum()); d[4] += 10; d[6] += int((k & ~kb).sum())
    np.savez_compressed(os.path.join(out, nm), keep=keep)
for dom, (n, r0, r1, kp, cells, un, fa) in st.items():
    print(f"[massB] {dom}: masked row-layers {n}; the budget leaves out rank 0 on {r0 / max(n, 1):.4f}, rank 1 on {r1 / max(n, 1):.4f} "
          f"(the floor adds back {fa} cells, {fa / max(cells, 1):.4f}: {'on' if FLOOR else 'off'}); kept share {kp / max(cells, 1):.3f}; "
          f"checks with scoreless rows {un} ({'skipped' if PARTIAL else 'rank profile'})", flush=True)
