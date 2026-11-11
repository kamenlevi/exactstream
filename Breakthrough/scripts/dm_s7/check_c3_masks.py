"""Bit checks of Cloud Fable 3's decision-map masks (Breakthrough, Thu 2026-11-05 07:xx; the orchestrator 06:5x: "tell me if any
mask fails your bit checks"). CPU only; reads cloud3's masks, the MoonShots export and the map's routing; writes nothing else.
(1) An independent rebuild of each arm from its SPEC (~/FableGod/cloud3/live/cloud3/REQUESTS.md C3-R11..R17), with the data
    conventions stated in cloud3's builder docstrings: the MoonShots export's live-cut checks (A, R, kept) and dcf per window;
    rows after a check's first miss enter the union as stand-ins (row (A + i + 97) mod N) with their own drafts' cf; only the
    realized rows A .. A+kept-1 are masked; a check's last row (the bonus row) takes its next draft's cf when the check has one
    (dc[L], L < 15), else it has no draft; checks running past the window's end are skipped. Strict: a masked row keeps ranks
    < k(row, layer) plus any rank whose expert is in the check's loads at that layer (every row's ranks < k), no resident keep.
    Bands (k by the cf of the draft the row verifies; no draft = 10):
      c3_b2d   >= 0.99: 2, [0.9, 0.99): 5, [0.3, 0.9): 8, < 0.3: 10        c3_b2c  as b2d with 0.999 for 0.99
      c3_b2b   >= 0.99: 2, [0.9, 0.99): 5, [0.5, 0.9): 8, < 0.5: 10        c3_b2b_top1  b2b with the >= 0.99 band at 1
      c3_b2b_band5  b2b + DRAFTED rows with cf < 0.9 at 5 in layers 8-35 (the spec text is silent on no-draft rows; with
                    them at 5 too, 14/36 windows differ: cloud3's build keeps a no-draft row exact)
      c3_new1  >= 0.999: 2, [0.99, 0.999): 4, [0.9, 0.99): 8, < 0.9: 10
      c3_b2d_reach07  b2d + REACH-CUT: a row with a draft, reach < 0.7 and k > 2 takes k 2 (reach = the product of the cf of
                      the check's drafts before the row)                   c3_b2d_reach07_u7  as reach07 with [0.3, 0.9) at 7
(2) Format: 36 windows per arm; keep [N, 48, 10] bool with N == the map's routing rows; rank 0 kept on every row; rows outside
    every check (prompt rows) all kept. (3) Nesting: b2d within b2c, reach07 within b2d, reach07_u7 within reach07, b2b_top1 and
    b2b_band5 within b2b. Usage (BigMoE venv): python check_c3_masks.py [arm ...]"""
import os, sys
import numpy as np
C3 = os.path.expanduser("~/FableGod/cloud3/live/cloud3/masks")
MS = os.path.expanduser("~/FableGod/cloud3/macpub/data/decision_map_moonshots")
ROUT = os.path.expanduser("~/FableGod/decision_map/out/routing")
MID = np.array([8 <= l < 36 for l in range(48)])

def k_of(arm, c):
    """k for a row whose draft has confidence c (c < 0: no draft), before layer bands and REACH-CUT."""
    if c < 0: return 10
    if arm in ("c3_b2d", "c3_b2d_reach07", "c3_b2d_reach07_u7", "c3_b2c"):
        top = 0.999 if arm == "c3_b2c" else 0.99
        return 2 if c >= top else 5 if c >= 0.9 else (7 if arm.endswith("_u7") else 8) if c >= 0.3 else 10
    if arm in ("c3_b2b", "c3_b2b_top1", "c3_b2b_band5"):
        return (1 if arm == "c3_b2b_top1" else 2) if c >= 0.99 else 5 if c >= 0.9 else 8 if c >= 0.5 else 10
    if arm == "c3_new1":
        return 2 if c >= 0.999 else 4 if c >= 0.99 else 8 if c >= 0.9 else 10
    raise KeyError(arm)

def rebuild(arm, z, N):
    ids = z["ids"].astype(np.int64); keep = np.ones((N, 48, 10), bool); n_chk = 0
    for (A, R, kept), dc in zip(z["checks"].tolist(), z["dcf"]):
        A, R, kept = int(A), int(R), int(kept); L = R - 1
        if A + R > N: continue
        n_chk += 1
        rows = [A + i if i < kept else (A + i + 97) % N for i in range(R)]
        cf = [float(dc[i]) if i < L else -1.0 for i in range(R)]
        if L < 15: cf[L] = float(dc[L])                       # the bonus row verifies the check's next draft
        reach, run = [], 1.0
        for i in range(R):
            reach.append(run)
            if i < L: run *= cf[i] if cf[i] >= 0 else 1.0
        k = np.empty((R, 48), np.int64)
        for i in range(R):
            ki = k_of(arm, cf[i])
            if arm.startswith("c3_b2d_reach07") and cf[i] >= 0 and reach[i] < 0.7 and ki > 2: ki = 2
            k[i, :] = ki
            if arm == "c3_b2b_band5" and 0 <= cf[i] < 0.9: k[i, MID] = 5   # drafted rows only (a no-draft row stays exact)
        I = ids[rows]                                          # [R, 48, 10]
        for l in range(48):
            loads = np.unique(np.concatenate([I[i, l, :k[i, l]] for i in range(R)]))
            for i in range(kept):
                keep[A + i, l] = (np.arange(10) < k[i, l]) | np.isin(ids[A + i, l], loads)
    return keep, n_chk

ARMS = sys.argv[1:] or ["c3_b2d", "c3_b2c", "c3_b2d_reach07", "c3_b2b_band5", "c3_b2b_top1", "c3_b2d_reach07_u7", "c3_b2b", "c3_new1"]
names = sorted(f for f in os.listdir(MS) if f.endswith(".npz"))
bad = []
print(f"[c3] {len(names)} windows in the MoonShots export; arms {ARMS}", flush=True)
mine = {}
for arm in ARMS:
    d = os.path.join(C3, arm); ok_fmt = ok_r0 = ok_out = eq = 0; ndiff = 0; first = None; chk = 0; rows = 0
    for nm in names:
        z = np.load(os.path.join(MS, nm)); r = np.load(os.path.join(ROUT, nm))["ids"]; N = r.shape[0]
        t = np.load(os.path.join(d, nm))["keep"] if os.path.exists(os.path.join(d, nm)) else None
        if t is None or t.shape != (N, 48, 10) or t.dtype != bool or z["ids"].shape != r.shape or not np.array_equal(z["ids"], r):
            continue
        ok_fmt += 1; ok_r0 += bool(t[..., 0].all())
        m, n_chk = rebuild(arm, z, N); chk += n_chk; rows += N; mine[(arm, nm)] = m
        masked = np.zeros(N, bool)
        for (A, R, kept) in z["checks"].tolist():
            if A + R <= N: masked[A:A + kept] = True
        ok_out += bool(t[~masked].all())
        same = np.array_equal(m, t); eq += same
        if not same:
            dd = np.argwhere(m != t); ndiff += len(dd)
            if first is None: first = (nm, dd[0].tolist(), "mine", m[tuple(dd[0][:2])].astype(int).tolist(), "theirs", t[tuple(dd[0][:2])].astype(int).tolist())
    line = (f"{arm}: format {ok_fmt}/36, rank 0 kept {ok_r0}/36, rows outside checks all kept {ok_out}/36, "
            f"REBUILD bit-equal {eq}/36 ({chk} checks, {rows} rows)")
    if eq < 36: line += f"; {ndiff} differing (row, layer, rank) cells; first {first}"
    print(line, flush=True)
    if not (ok_fmt == ok_r0 == ok_out == eq == 36): bad.append(arm)
for small, big in (("c3_b2d", "c3_b2c"), ("c3_b2d_reach07", "c3_b2d"), ("c3_b2d_reach07_u7", "c3_b2d_reach07"),
                   ("c3_b2b_top1", "c3_b2b"), ("c3_b2b_band5", "c3_b2b")):
    if small not in ARMS or big not in ARMS: continue
    nest = sum(not (np.load(os.path.join(C3, small, nm))["keep"] & ~np.load(os.path.join(C3, big, nm))["keep"]).any() for nm in names)
    print(f"[c3] nesting: {small} within {big} in {nest}/36 windows", flush=True)
    if nest != 36: bad.append(f"{small}<={big}")
print("ALL OK" if not bad else f"CHECK FAILED: {bad}", flush=True)
sys.exit(1 if bad else 0)
