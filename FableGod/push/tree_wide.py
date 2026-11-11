"""tree_wide.py (Fable B, PUSH-B): what the drafter's top-3 is worth when selection rows are cheap. CPU, on Breakthrough's
DFlash2 lattices (~/Breakthrough/data/trees/lattice_{code,chat}.npz: per block 15 depths x 16 candidates, the selector's
first-order transition P(child | parent) 'trans', the selector's pick, the answer's true tokens; code = 664 blocks of the 24
HumanEval windows, chat = 6,134 blocks of IFEval + multi). The k-best tree is tree_price.py's (first-order, each popped
node pushes its best child and its next sibling); kept = the longest true prefix in the tree + 1, capped at the sample's end.
New here: (1) trees past 16 rows (32 / 48 / 64 nodes), the ceiling of 'exact tree nodes' at any price; (2) the ANCHOR TREE:
siblings at the two rows the drafter's own first-miss probability ranks first (ANCHORCOV's anchors), with their first-order
continuations, and what a CHEAP (not exact) verdict on them is worth one cycle later: the next check's chain starts with the
continuation already verified at precision a per row instead of fresh drafts; (3) the cycle model -> tok/s per design.
Run: nice -n 15 ~/BigMoE/.venv/bin/python tree_wide.py > tree_wide.out (~1-2 min)."""
import os, heapq, numpy as np

D = os.path.expanduser("~/Breakthrough/data/trees")
ROWS_T = (16, 32, 48, 64)
A_LIST = (0.98, 0.95, 0.90)                        # cheap-verdict precision per row [K M-16 rows agree: code 0.97-0.99 skip, chat 0.85-0.98]
CMAX = (4, 8)


def kbest(first, ordr, srt, rows, DEP, K):
    lim = rows - 1
    out, heap = set(), [(-srt[0, 0, 0], (int(ordr[0, 0, 0]),), 0, 1.0)]
    while heap and len(out) < lim:
        nq, path, r, pp = heapq.heappop(heap)
        out.add(path)
        d = len(path)
        if d < DEP:
            i = first(d, path)
            heapq.heappush(heap, (nq * srt[d, i, 0], path + (int(ordr[d, i, 0]),), 0, -nq))
        if r + 1 < K:
            i = first(d - 1, path[:-1])
            heapq.heappush(heap, (-pp * srt[d - 1, i, r + 1], path[:-1] + (int(ordr[d - 1, i, r + 1]),), r + 1, pp))
    return out


def walk(T, J, nt, K):
    d, path = 0, ()
    while d < nt and J[d] < K and path + (int(J[d]),) in T:
        path += (int(J[d]),); d += 1
    return d


for dom in ("code", "chat"):
    z = np.load(f"{D}/lattice_{dom}.npz")
    CAND, TRANS, PICK, TRUE, NT = z["cand"], z["trans"], z["pick"], z["true"], z["ntrue"]
    SCAND, SPROB = z["sel_cand"], z["sel_prob"].astype(np.float64)
    B, DEP, K = CAND.shape
    CAP = np.where(NT == DEP, DEP + 1, NT)
    chain = np.zeros(B); tree = {R: np.zeros(B) for R in ROWS_T}; off = {R: np.zeros(B) for R in ROWS_T}
    fm_row = np.full(B, -1); cov3 = np.zeros(B, bool); cov2 = np.zeros(B, bool); cov16 = np.zeros(B, bool)
    anchor = np.zeros(B, bool); cont = np.zeros(B); full = np.zeros(B, bool)
    reach = np.zeros(DEP); hit = np.zeros(DEP)
    for b in range(B):
        nt, cap = int(NT[b]), int(CAP[b])
        Jr = np.full(DEP, K)
        for d in range(nt):
            w = np.nonzero(CAND[b, d] == TRUE[b, d])[0]; Jr[d] = w[0] if len(w) else K
        h = PICK[b, :nt] == TRUE[b, :nt]
        kc = int(np.argmin(h)) if not h.all() else nt
        for d in range(min(kc + 1, nt)):
            reach[d] += 1; hit[d] += h[d]
        chain[b] = min(kc + 1, cap)
        full[b] = kc >= nt
        tr = TRANS[b].astype(np.float64)
        ordr = np.argsort(-tr, axis=-1, kind="stable"); srt = np.take_along_axis(tr, ordr, -1)
        first1 = lambda d, path: path[-1] if d >= 1 and path else 0
        for R in ROWS_T:
            T = kbest(first1, ordr, srt, R, DEP, K)
            tree[R][b] = min(walk(T, Jr, nt, K) + 1, cap)
            ch = {tuple(int(np.nonzero(CAND[b, i] == PICK[b, i])[0][0]) for i in range(j + 1)) for j in range(DEP)}
            off[R][b] = len(T - ch)
        if kc < nt:                                    # a miss at row r = kc
            r = kc; fm_row[b] = r
            cov2[b] = TRUE[b, r] in SCAND[b, r, :2]; cov3[b] = TRUE[b, r] in SCAND[b, r, :3]; cov16[b] = Jr[r] < K
            cf = SPROB[b, :, 0]
            fmp = np.array([np.prod(cf[:j]) * (1 - cf[j]) for j in range(DEP)])
            anchor[b] = r in np.argsort(-fmp, kind="stable")[:2]
            if cov16[b]:                                # the continuation from the right sibling, along the first-order lattice
                i, c = int(Jr[r]), 0
                for d in range(r + 1, nt):
                    j = int(ordr[d, i, 0])
                    if CAND[b, d, j] != TRUE[b, d]:
                        break
                    c += 1; i = j
                cont[b] = c
    q = hit / np.maximum(reach, 1)
    kept_fresh = chain.mean()
    miss = fm_row >= 0
    print(f"==== {dom}: {B} blocks; chain 16 kept {kept_fresh:.2f}; all 15 drafts accepted {full.mean():.3f} of blocks; "
          f"per-depth accept rate q_d (depths 1-8): {' '.join('%.2f' % v for v in q[:8])} ====")
    print("  exact-node trees (every node at full price), kept per 16-row-anchored block and off-chain nodes:")
    for R in ROWS_T:
        print(f"    tree {R:2d} nodes: kept {tree[R].mean():.2f} ({100 * (tree[R].mean() / kept_fresh - 1):+.1f} % vs the chain), off-chain nodes {off[R].mean():.1f}")
    print(f"  at the first miss (blocks with one: {miss.mean():.3f}): the true token is in the drafter's top-2 {cov2[miss].mean():.3f}, top-3 {cov3[miss].mean():.3f}, top-16 {cov16[miss].mean():.3f}; "
          f"the miss row is one of the 2 first-miss-probability anchors {anchor[miss].mean():.3f}; both (anchor and top-3) {(anchor & cov3)[miss].mean():.3f}")
    print(f"  first-miss row: mean {fm_row[miss].mean():.2f}; continuation after the right sibling along the lattice (when in top-16): mean {cont[miss & cov16].mean():.2f} rows, "
          f"P(>= 1) {(cont[miss & cov16] >= 1).mean():.2f}, P(>= 4) {(cont[miss & cov16] >= 4).mean():.2f}")
    # the anchor tree's value one cycle later: the next check starts with the continuation pre-verified at precision a
    print("  ANCHOR TREE (2 anchors x 2 extra siblings + continuations): extra kept per cycle = P(anchor & top-3) x E[gain(c, a)],"
          " gain = sum_{k<=c} a^k - (1 - a^c) x kept_fresh (a wrong cheap verdict on the carried prefix costs the fresh check's kept):")
    sel = miss & anchor & cov3
    for cmax in CMAX:
        for a in A_LIST:
            c = np.minimum(cont[sel], cmax)
            g = np.array([sum(a ** k for k in range(1, int(ci) + 1)) - (1 - a ** int(ci)) * kept_fresh for ci in c])
            extra = g.sum() / B
            print(f"    cmax {cmax}, a {a:.2f}: extra kept per cycle {extra:+.2f} ({100 * extra / kept_fresh:+.1f} %); cheap rows per cycle {4 + 4 * cmax} (siblings 4 + continuations {4 * cmax})")
    # the deep form: all 16 cheap rows run ahead at depths 17-32 (useful only after a full accept)
    print(f"  DEEP FORM (16 cheap rows at depths 17-32, useful after a full accept, P {full.mean():.3f}): at a = 0.98 the pre-verified next chain keeps "
          f"{sum(0.98 ** k for k in range(1, 17)) + 1:.1f}, and the share of cycles with one, x = p_ff / (1 - p_fp + p_ff) = "
          f"{full.mean() / (1 - 0.98 ** 16 + full.mean()):.2f} -> kept per cycle {full.mean() / (1 - 0.98 ** 16 + full.mean()) * (sum(0.98 ** k for k in range(1, 17)) + 1) + (1 - full.mean() / (1 - 0.98 ** 16 + full.mean())) * kept_fresh:.2f}")
    print()

# ---------------------------------------------------------------- the cycle model [P on K]
print("==== cycle model: ms per cycle and tok/s; today = C 256 day (push brief: check 84.2 = dense 34 + 3.0 x rows + 0.157 x misses at the mix's 7.6 rows / 171 misses), "
      "floor = B2 / B6's floor engine (dense 23 EXACT, experts 3.9 + 1.54 x rows [C2 bytes], SSD 70 % hidden, draft 5, commit 4.7, rest 5) ====")
ROW_GPU, MS_MISS, DENSE, DRAFT, REST = 3.0, 0.157, 34.0, 12.0, 5.0


def today(rows, misses, cheap=0, cheap_ms=0.6, dense_extra=0.0, draft_extra=0.0):
    return DRAFT + draft_extra + DENSE + dense_extra + ROW_GPU * rows + MS_MISS * misses + cheap * cheap_ms + REST


def floor(rows, ssd_exposed, dense=23.0, cheap=0, cheap_ms=0.35, dense_extra=0.0, draft=5.0):
    return draft + dense + dense_extra + 3.9 + 1.54 * rows + ssd_exposed + cheap * cheap_ms + 4.7 + 5.0


# code: live rows 9.84, kept 10.54 [K D-23d], misses on the first 10 rows at C 256 ~ 245 (row_split.out: code cum 36.1 ms at 0.149 over rows 0-9);
# chat: live rows 4 (MTP chain, D-23d), kept 2.53, misses ~ 90 [K M-16 ifeval 74-106 at C 224, G at C 256]
for name, rows, kept, misses, ssd_floor in (("code", 9.84, 10.54, 245, 13.0), ("chat", 4.0, 2.53, 90, 3.0)):
    base_t, base_f = today(rows, misses), floor(rows, ssd_floor)
    print(f"\n-- {name}: rows {rows}, kept {kept}, misses {misses} --")
    print(f"  today: cycle {base_t:.0f} ms -> {1e3 * kept / base_t:.1f} tok/s; floor: cycle {base_f:.0f} ms -> {1e3 * kept / base_f:.1f} tok/s")
    # D1 anchor tree, same pass: cheap rows ride the 16-row padding on chat (dense extra 0 up to 12 rows), the 32-row forms on code (+9.8 today, +8 floor)
    for cheap, extra_kept_pct, label in ((12, 0.20, "12 cheap rows, kept +20 %"), (20, 0.30, "20 cheap rows, kept +30 %"), (36, 0.40, "36 cheap rows, kept +40 %")):
        pad = max(0, 16 - rows)
        dx_t = 9.8 * max(0, cheap - pad) / 16; dx_f = 8.0 * max(0, cheap - pad) / 16
        k2 = kept * (1 + extra_kept_pct)
        for drx, dl in ((0.0, "ANE draft hidden"), (12.0, "GPU draft +12")):
            ct = today(rows, misses, cheap, 0.6, dx_t, drx); cf = floor(rows, ssd_floor, 23.0, cheap, 0.35, dx_f, 5.0 + drx * 0.4)
            print(f"  D1 same-pass {label:28s} [{dl:16s}]: today {ct:.0f} ms, {1e3 * k2 / ct:.1f} tok/s ({100 * (k2 / ct / (kept / base_t) - 1):+.0f} %); "
                  f"floor {cf:.0f} ms, {1e3 * k2 / cf:.1f} tok/s ({100 * (k2 / cf / (kept / base_f) - 1):+.0f} %)")
    # D1 two-pass: a cheap pass (dense + cheap rows) then the exact pass
    for cheap, extra_kept_pct in ((20, 0.30), (36, 0.40)):
        k2 = kept * (1 + extra_kept_pct)
        ct = today(rows, misses) + DENSE + cheap * 0.6; cf = floor(rows, ssd_floor) + 23.0 + cheap * 0.35
        print(f"  D1 two-pass {cheap} cheap rows, kept +{100 * extra_kept_pct:.0f} %: today {ct:.0f} ms, {1e3 * k2 / ct:.1f} tok/s ({100 * (k2 / ct / (kept / base_t) - 1):+.0f} %); "
              f"floor {cf:.0f} ms, {1e3 * k2 / cf:.1f} tok/s ({100 * (k2 / cf / (kept / base_f) - 1):+.0f} %)")
    # D2 the all-resident cheap chain one cycle ahead (M-16 within 32 rows): its routes prefetch the next exact pass's misses (U1 recall 0.71-0.84)
    for recall, cheap_ms, lab in ((0.75, 2.64, "all-resident rows (2.64 ms), recall 0.75"), (0.50, 0.6, "top-2 rows (0.6 ms), recall 0.50")):
        ct = today(rows, misses * (1 - recall), 16, cheap_ms, 9.8); k2 = kept * 1.15
        print(f"  D2 runahead chain, {lab}, kept +15 %: today {ct:.0f} ms, {1e3 * k2 / ct:.1f} tok/s ({100 * (k2 / ct / (kept / base_t) - 1):+.0f} %)")
