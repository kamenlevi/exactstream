#!/usr/bin/env python3 -I
"""floor_bytes.py (FableGod, mission 13, Mon 2 Nov 2026). CPU only, no numpy (parses .npy from the .npz by hand).
Bytes side of the floor, from Breakthrough's DAY9 routing capture (~/FableGod/decision_map/out/routing/<window>.npz:
ids [n-1, 48, 10] rank-sorted top-10 expert ids per position and layer; tokens [n]) joined with floor_rows.json
(per answer row: class, drafter band, oracle level):
  1. expert records per 16-row block (the union over rows per layer) under each scheme -> per kept token;
  2. an LRU replay per layer at C slots (12 / 24 / 45 / 51 GB lines) -> misses per kept token -> SSD cap;
  3. hot-set concentration of ranks 0-1 (what a small cache can hold);
  4. a probe: does routing agreement with the previous position predict the 125B's sureness?
Schemes (k per row; 'R' = the row reuses the previous row's MoE output on layers 8-35, so it posts no experts there):
  exact: top-10 everywhere; q1_oracle: top-2 on 125B-sure rows; q1_999: top-2 on drafter cf >= 0.999;
  q1_99: top-2 on cf >= 0.99; new1: bands (>= 0.999: 2, 0.99-0.999: 4, 0.9-0.99: 8, else 10);
  real1: the realizable 1 %-bar floor (>= 0.999: 2, 0.99-0.999: 5, else 10); oracle: floor_rows' per-row level;
  top2_all: top-2 on every row (the floor of floors).
"""
import zipfile, struct, array, re, json, os, sys
from collections import OrderedDict, Counter, defaultdict

OUT = os.path.expanduser("~/Breakthrough/data/floor1")
rows = json.load(open(os.path.expanduser("~/Breakthrough/data/floor1/floor_rows.json")))
base = json.load(open(f"{OUT}/baseline.json"))
R = defaultdict(dict)
for r in rows: R[r["nm"]][r["j"]] = r
import math
for cut in ("topk2", "topk5"):                       # the cheap pass's own p1 under the cut (the self-certificate)
    Pc = json.load(open(f"{OUT}/{cut}.json"))
    for nm, w in Pc.items():
        for j, lp in enumerate(w["lp"]): R[nm][j]["p1_" + cut] = math.exp(lp[0])
MB4, MBT0, MBT = 2.8, 2.8, 2.1          # 4-bit record; tiers: ranks 0-1 at 2.8, ranks 2-9 at 2.1 (TRIED: 48 x 10 x 2.2 = 1.06 GB)
def rec_mb(k): return min(k, 2) * MBT0 + max(k - 2, 0) * MBT

def load_npy(z, name):
    d = z.read(name); hl = struct.unpack("<H", d[8:10])[0]; hdr = d[10:10 + hl].decode("latin1"); body = d[10 + hl:]
    descr = re.search(r"'descr': '([^']+)'", hdr).group(1)
    shape = tuple(int(x) for x in re.findall(r"\d+", re.search(r"'shape': \(([^)]*)\)", hdr).group(1)))
    a = array.array({"<i2": "h", "<i4": "i", "<i8": "q"}[descr]); a.frombytes(body); return a, shape

W = {}
for f in sorted(os.listdir(f"{OUT}/routing")):
    nm = f
    if nm not in base: continue
    z = zipfile.ZipFile(f"{OUT}/routing/{f}")
    ids, sh = load_npy(z, "ids.npy"); tok, _ = load_npy(z, "tokens.npy"); gen, _ = load_npy(z, "gen.npy")
    n1, L, K = sh; assert (L, K) == (48, 10), sh
    b = base[nm]; a = b["a"]; na = len(b["tk"])
    assert a + na == n1, (nm, a, na, n1)
    W[nm] = dict(dom=b["dom"], a=a, na=na, ids=ids, n1=n1, gen=gen[0])
print("windows with routing:", len(W), Counter(w["dom"] for w in W.values()), "; gen of first =", next(iter(W.values()))["gen"])

def row_k(scheme, r):
    """(k, reuse) for an answer row under a scheme."""
    cf = r["cf"]
    if scheme == "exact": return 10, 0
    if scheme == "top2_all": return 2, 0
    if scheme == "q1_oracle": return (2 if r["cls"] == "sure" else 10), 0
    if scheme == "q1_999": return (2 if cf is not None and cf >= 0.999 else 10), 0
    if scheme == "q1_99": return (2 if cf is not None and cf >= 0.99 else 10), 0
    if scheme == "new1":
        if cf is None: return 10, 0
        return (2 if cf >= 0.999 else 4 if cf >= 0.99 else 8 if cf >= 0.9 else 10), 0
    if scheme == "real1":
        if cf is None: return 10, 0
        return (2 if cf >= 0.999 else 5 if cf >= 0.99 else 10), 0
    if scheme == "oracle": return tuple(r["oracle"])
    if scheme == "top5_all": return 5, 0
    if scheme == "cert5": return (5 if r["p1_topk5"] >= 0.9 else 10), 0      # two-pass: top-5 everywhere, exact where the cheap pass is unsure
    if scheme == "cert2": return (2 if r["p1_topk2"] >= 0.99 else 10), 0
    raise ValueError(scheme)

SCHEMES = ["exact", "q1_oracle", "q1_999", "q1_99", "new1", "real1", "oracle", "top2_all", "top5_all", "cert5", "cert2"]
DOMS = ["code", "ifeval", "multi"]

def expert_sets(w, pos, k, reuse):
    """list over layers of the row's expert ids (rank-sorted top-k; empty on reused layers)."""
    ids = w["ids"]; out = []
    for l in range(48):
        if reuse and 8 <= l <= 35: out.append(()); continue
        o = (pos * 48 + l) * 10
        out.append(tuple(ids[o:o + k]))
    return out

# 1. union per 16-row block -------------------------------------------------------------------------------------------
print("\n### 1. expert records per 16-row block (union over rows per layer, summed over 48 layers), per domain and scheme")
print("| dom | scheme | blocks | records/block (mean) | MB/block tiers | per kept token at 16 kept (MB) | at today's kept (MB) | at the route's kept (MB) | union/layer at top-10 law 92 |")
print("|---|---|---|---|---|---|---|---|---|")
KEPT_TODAY = {"code": 10.54, "ifeval": 2.03, "multi": 2.03}
KEPT_ROUTE = {"code": 12.5, "ifeval": 3.3, "multi": 3.3}
block_stats = {}
for dom in DOMS:
    ws = [w for w in W.values() if w["dom"] == dom]
    if not ws: continue
    for sc in SCHEMES:
        nb = 0; recs = 0.0; mb = 0.0; uni10 = 0.0
        for nm, w in W.items():
            if w["dom"] != dom: continue
            for j0 in range(0, w["na"] - 15, 16):
                nb += 1
                for l in range(48):
                    U = {}                                   # expert id -> best rank among rows (for the tier bytes)
                    for j in range(j0, j0 + 16):
                        r = R[nm][j]; k, re_ = row_k(sc, r)
                        if re_ and 8 <= l <= 35: continue
                        o = ((w["a"] + j) * 48 + l) * 10
                        for rk in range(k):
                            e = w["ids"][o + rk]
                            if e not in U or rk < U[e]: U[e] = rk
                    recs += len(U); mb += sum(MBT0 if rk < 2 else MBT for rk in U.values())
                    if sc == "exact": uni10 += len(U)
        if nb == 0: continue
        block_stats[(dom, sc)] = (recs / nb, mb / nb)
        print(f"| {dom} | {sc} | {nb} | {recs / nb:.0f} | {mb / nb:.0f} | {mb / nb / 16:.0f} | {mb / nb / KEPT_TODAY[dom]:.0f} | {mb / nb / KEPT_ROUTE[dom]:.0f} | {uni10 / nb / 48:.1f} |" if sc == "exact"
              else f"| {dom} | {sc} | {nb} | {recs / nb:.0f} | {mb / nb:.0f} | {mb / nb / 16:.0f} | {mb / nb / KEPT_TODAY[dom]:.0f} | {mb / nb / KEPT_ROUTE[dom]:.0f} | |")

# 2. LRU replay ------------------------------------------------------------------------------------------------------
print("\n### 2. LRU replay per layer (C slots per layer; prompt rows exact and warming; 16-row blocks on answer rows): misses per kept token (at today's kept), MB, SSD cap tok/s = 13000 / MB")
CS = [(51, "12 GB (C 51)"), (160, "24 GB (C 160)"), (256, "45 GB (C 256)"), (320, "51 GB (C 320)")]
print("| dom | C | " + " | ".join(SCHEMES) + " |")
print("|---|---|" + "---|" * len(SCHEMES))
for dom in DOMS:
    ws = [(nm, w) for nm, w in W.items() if w["dom"] == dom]
    if not ws: continue
    kept = KEPT_TODAY[dom]
    for C, lab in CS:
        cells = []
        for sc in SCHEMES:
            cache = [OrderedDict() for _ in range(48)]
            miss = 0; kept_tot = 0.0
            for nm, w in ws:
                ids = w["ids"]
                for pos in range(w["a"]):                   # prompt rows: exact, position by position
                    for l in range(48):
                        c = cache[l]; o = (pos * 48 + l) * 10
                        for rk in range(10):
                            e = ids[o + rk]
                            if e in c: c.move_to_end(e)
                            else:
                                c[e] = 1
                                if len(c) > C: c.popitem(last=False)
                for j0 in range(0, w["na"] - 15, 16):
                    kept_tot += kept
                    for l in range(48):
                        c = cache[l]; U = set()
                        for j in range(j0, j0 + 16):
                            r = R[nm][j]; k, re_ = row_k(sc, r)
                            if re_ and 8 <= l <= 35: continue
                            o = ((w["a"] + j) * 48 + l) * 10
                            U.update(ids[o:o + k])
                        for e in U:
                            if e in c: c.move_to_end(e)
                            else:
                                miss += 1; c[e] = 1
                                if len(c) > C: c.popitem(last=False)
            mpk = miss / kept_tot
            cells.append(f"{mpk:.1f} / {mpk * 2.2:.0f} MB / {13000 / max(mpk * 2.2, 1e-9):.0f}")
        print(f"| {dom} | {lab} | " + " | ".join(cells) + " |")

# 3. hot-set concentration -------------------------------------------------------------------------------------------
print("\n### 3. concentration: share of (row, layer) rank-0/1 slots and of all top-10 slots covered by the N most frequent experts per layer (answer rows)")
for dom in DOMS:
    ws = [(nm, w) for nm, w in W.items() if w["dom"] == dom]
    if not ws: continue
    f01 = [Counter() for _ in range(48)]; fall = [Counter() for _ in range(48)]
    for nm, w in ws:
        ids = w["ids"]
        for j in range(w["na"]):
            pos = w["a"] + j
            for l in range(48):
                o = (pos * 48 + l) * 10
                for rk in range(10):
                    e = ids[o + rk]; fall[l][e] += 1
                    if rk < 2: f01[l][e] += 1
    line = [dom]
    for N in (51, 102, 160, 256):
        c01 = sum(sum(v for _, v in f01[l].most_common(N)) for l in range(48)) / sum(sum(f01[l].values()) for l in range(48))
        call = sum(sum(v for _, v in fall[l].most_common(N)) for l in range(48)) / sum(sum(fall[l].values()) for l in range(48))
        line.append(f"N={N}: ranks 0-1 {100 * c01:.0f} %, all ranks {100 * call:.0f} %")
    uniq = sum(len(fall[l]) for l in range(48)) / 48
    print(f"- {dom}: " + "; ".join(line[1:]) + f"; distinct experts touched per layer {uniq:.0f} of 512")

# 4. routing-agreement probe -----------------------------------------------------------------------------------------
print("\n### 4. probe: P(125B-sure | the row's top-1 expert equals the previous position's top-1 in >= T of the 48 layers), and coverage")
for dom in DOMS:
    ws = [(nm, w) for nm, w in W.items() if w["dom"] == dom]
    if not ws: continue
    agree = []
    for nm, w in ws:
        ids = w["ids"]
        for j in range(w["na"]):
            pos = w["a"] + j
            if pos == 0: continue
            n_same = sum(1 for l in range(48) if ids[(pos * 48 + l) * 10] == ids[((pos - 1) * 48 + l) * 10])
            n_early = sum(1 for l in range(8) if ids[(pos * 48 + l) * 10] == ids[((pos - 1) * 48 + l) * 10])
            agree.append((n_same, n_early, R[nm][j]["cls"] == "sure", R[nm][j]["cf"]))
    base_rate = sum(s for _, _, s, _ in agree) / len(agree)
    out = [f"base {100 * base_rate:.0f} %"]
    for T in (12, 20, 28, 36):
        S = [s for n, _, s, _ in agree if n >= T]
        out.append(f"T>={T}: P(sure) {100 * (sum(S) / len(S) if S else 0):.0f} % on {100 * len(S) / len(agree):.0f} % of rows")
    for T in (4, 6, 8):
        S = [s for _, ne, s, _ in agree if ne >= T]
        out.append(f"early(0-7)>={T}: P(sure) {100 * (sum(S) / len(S) if S else 0):.0f} % on {100 * len(S) / len(agree):.0f} %")
    # beyond the drafter: among drafter-unsure rows (cf < 0.99), does agreement find the 125B-sure ones?
    U = [(n, s) for n, _, s, cf in agree if cf is None or cf < 0.99]
    ub = sum(s for _, s in U) / len(U)
    S = [s for n, s in U if n >= 28]
    out.append(f"drafter-unsure rows: base P(sure) {100 * ub:.0f} %, with T>=28 {100 * (sum(S) / len(S) if S else 0):.0f} % on {100 * len(S) / len(U):.0f} % of them")
    print(f"- {dom}: " + "; ".join(out))
