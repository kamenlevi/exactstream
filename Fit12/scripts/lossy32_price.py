"""lossy32_price.py (Fit12, Mon 16 Nov 2026 08:3x; WEEK98 PILLAR 2 LOSSY, NewIdeas' assignment; pre-registered in
/Users/kamen/Fit12/PLAN_12GB_90_fit12.md "WEEK98 PILLAR 2 LOSSY" before this ran). Desk only, CPU, python -I.

Prices candidate LOSSY 32-row checks at 45 GB: GB per check, floor ms, realistic ms (per class, from the measured engine
profile and the FAST stack's measured 16-row dense prices grown by Leap's per-tile factors), the CPU lower bound, and the
45 GB SSD-miss term. Labels: [K] measured record, [P] priced from measured parts, [est] assumption (a range).

Sources (read-only):
  - map code windows: ~/FableGod/decision_map/out/xcap/<w>.npz (answer rows: top-10 ids + weights, a = answer start),
    ~/FableGod/decision_map/out/bt/tiers_1c/<w>.npz (1c masks: keep = 4-bit tier, keep3 = MSB-3 tier), baseline.json (dom)
  - engine profile [K]: ~/Breakthrough/docs/CHECK55_PROFILE.md "Engine RESULT" (a082f8c): pass T / pass K per class
  - FAST 16-row dense prices [K]: ~/Leap/TABLE_350.md "PLAN 220 RE-PRICE SLOT, part 2" (Sat 11-07 14:37-14:42)
  - per-tile growth [est]: ~/Leap/scripts/week98/treek_price.py GROW
  - lossy-45 16-row HE checks [K]: ~/BigMoE/data/r57/rlh_256_checks.json (C 256, real decoding)
  - dense bytes [K]: ~/FableGod/deepdraft/LOSSY_CHECK_PLAN.md addendum (weight-file headers); SVH subset [K] TABLE_350
  - lossy K* kept [K]: ~/FableGod/deepdraft/TARGETS_450_900.md (tree-32 + close-call d1: 16.91 HE+ / 9.50 MBPP)
"""
import json
import os
import statistics as st

import numpy as np

OUT = "/Users/kamen/Fit12/data/lossy32"
M = os.path.expanduser("~/FableGod/decision_map/out")
BW = 280.0                      # GB/s, the floor bandwidth every WEEK98 floor uses
E4, E3 = 2.135e-3, 2.135e-3 * 0.778   # GB per expert: 4-bit, MSB-3 plane only (records3)
NL, ROWS = 48, 32
os.makedirs(OUT, exist_ok=True)
L = []


def p(s=""):
    print(s)
    L.append(s)


# ---------------------------------------------------------------- A. expert bytes per 32-row block on the code windows
base = json.load(open(os.path.join(M, "baseline.json")))
code = sorted(f for f in os.listdir(os.path.join(M, "xcap")) if base[f]["dom"] == "code")
allw = sorted(os.listdir(os.path.join(M, "xcap")))
acc = {k: [] for k in ("union", "ref2t_lsb", "c1_kept", "c1_lsb", "c1_map_e4", "c1_map_e3", "top2u")}
u16_code = []
union_all = []
for f in allw:
    z = np.load(os.path.join(M, "xcap", f))
    ids, a = z["ids"].astype(np.int32), int(z["a"])
    t = np.load(os.path.join(M, "bt", "tiers_1c", f))
    k4 = t["keep"].astype(bool)[a:a + ids.shape[0]]
    k3 = t["keep3"].astype(bool)[a:a + ids.shape[0]]
    assert k4.shape == ids.shape, (f, k4.shape, ids.shape)
    rank = np.broadcast_to(np.arange(10), ids.shape)
    if base[f]["dom"] == "code":
        for bj in range(ids.shape[0] // 16):
            for ly in range(NL):
                u16_code.append(len(np.unique(ids[bj * 16:(bj + 1) * 16, ly])))
    for bi in range(ids.shape[0] // ROWS):
        sl = slice(bi * ROWS, (bi + 1) * ROWS)
        for ly in range(NL):
            I = ids[sl, ly].ravel()
            R = rank[sl, ly].ravel()
            u = np.unique(I)
            union_all.append(len(u))
            if base[f]["dom"] != "code":
                continue
            K4 = k4[sl, ly].ravel()
            K3 = k3[sl, ly].ravel()
            acc["union"].append(len(u))
            acc["top2u"].append(len(np.unique(I[R < 2])))
            acc["ref2t_lsb"].append(len(np.unique(I[R < 2])))                 # REF2-T: LSB read iff some row ranks it 0-1
            kept = K4 | K3
            acc["c1_kept"].append(len(np.unique(I[kept])))
            acc["c1_lsb"].append(len(np.unique(I[K4 & (R < 2)])))             # engine semantics: 4-bit tier keeps base precision
            e4 = np.unique(I[K4])
            acc["c1_map_e4"].append(len(e4))                                   # map semantics: 4-bit at any rank
            acc["c1_map_e3"].append(len(np.setdiff1d(np.unique(I[K3]), e4)))
mean = {k: float(np.mean(v)) for k, v in acc.items()}
nblk = len(acc["union"]) // NL
c1_map_eq = mean["c1_map_e4"] + 0.778 * mean["c1_map_e3"]
p(f"# LOSSY32 price (Fit12 lossy32_price.py), code windows {len(code)}, 32-row blocks {nblk}")
p(f"A. per layer per 32-row block (code): union {mean['union']:.1f} (all 36 windows {np.mean(union_all):.1f}); "
  f"distinct top-2 experts {mean['top2u']:.1f}")
p(f"   REF2-T reads: {mean['ref2t_lsb']:.1f} at 4-bit + {mean['union'] - mean['ref2t_lsb']:.1f} at MSB-3")
p(f"   1c (lambda 1.6e-2, s 0.98) ENGINE semantics: kept {mean['c1_kept']:.1f} distinct, of them {mean['c1_lsb']:.1f} at 4-bit "
  f"+ {mean['c1_kept'] - mean['c1_lsb']:.1f} at MSB-3")
p(f"   1c MAP semantics: 4-bit {mean['c1_map_e4']:.1f} + MSB-3 {mean['c1_map_e3']:.1f} = {c1_map_eq:.1f} 4-bit-eq "
  f"(Breakthrough: 36.4 of a 99.9 union)")


def exp_gb(n4, n3):
    return NL * (n4 * E4 + n3 * E3)


GB_EXP = {
    "all4": exp_gb(mean["union"], 0),
    "ref2t": exp_gb(mean["ref2t_lsb"], mean["union"] - mean["ref2t_lsb"]),
    "c1_eng": exp_gb(mean["c1_lsb"], mean["c1_kept"] - mean["c1_lsb"]),
    "c1_map": exp_gb(mean["c1_map_e4"], mean["c1_map_e3"]),
}
# 1c's ENGINE-semantics bytes relative to its map bytes, used to carry Breakthrough's other lambdas (map 4-bit-eq) over
ENG_OVER_MAP = GB_EXP["c1_eng"] / GB_EXP["c1_map"]
p(f"   expert GB per 32-row check: all-4-bit {GB_EXP['all4']:.2f}; REF2-T {GB_EXP['ref2t']:.2f}; 1c engine "
  f"{GB_EXP['c1_eng']:.2f}; 1c map {GB_EXP['c1_map']:.2f} (engine / map {ENG_OVER_MAP:.3f})")

# self-check (c); Breakthrough's 99.9 / 36.4 were checked post hoc to be the ALL-domain (36-window) means
sc_c1 = abs(np.mean(union_all) / 99.9 - 1) <= 0.10 or abs(mean["union"] / 109.7 - 1) <= 0.10
sc_c2 = abs(c1_map_eq / 36.4 - 1) <= 0.03
p(f"   SELF-CHECK (c): union {np.mean(union_all):.1f} vs 99.9 / code {mean['union']:.1f} vs 109.7 within 10 %: "
  f"{'PASS' if sc_c1 else 'FAIL'}; 1c map 4-bit-eq {c1_map_eq:.1f} vs 36.4 within 3 %: {'PASS' if sc_c2 else 'FAIL'}")

# ---------------------------------------------------------------- B. dense bytes [K] (GB per check, read once)
D = dict(hc=1.281, gdn=1.173, head=0.358, att=0.336, shared=0.133, router=0.067, ple=0.066, idx=0.021)
DENSE_EXACT = sum(D.values())                                         # 3.435
HC8_SAVE = 0.631                                                      # HC mixers + PLE bf16 -> 8-bit [K]
HC4_SAVE = 0.968                                                      # -> 4-bit [K]
V_ROWS = 248_320                                                      # vocab rows [est from 0.358 GB at 4.5 bits x 2560]
SVH_FRAC = (98_304 + 4_096) / V_ROWS                                  # lossy-45's screened head reads this share [K]
SL_FRAC = (16_384 + 64) / V_ROWS                                      # V1: 16K list + drafted / candidate ids [est]
dense = {
    "exact": DENSE_EXACT,
    "l45": DENSE_EXACT - HC8_SAVE - D["head"] * (1 - SVH_FRAC),
}
dense["l45_v1"] = dense["l45"] - D["head"] * (SVH_FRAC - SL_FRAC)
dense["l45_v1v3"] = dense["l45_v1"] - (HC4_SAVE - HC8_SAVE)
dense["l45_v1v3v4"] = dense["l45_v1v3"] - 0.335
p(f"B. dense GB: exact {dense['exact']:.3f}; lossy-45 (HC8 + SVH) {dense['l45']:.3f}; + V1 {dense['l45_v1']:.3f}; "
  f"+ V3 {dense['l45_v1v3']:.3f} (Breakthrough's plan 2.14 counts the full head as the base); + V4 {dense['l45_v1v3v4']:.3f}")

# ---------------------------------------------------------------- C. the engine profile [K] (a082f8c)
PT = dict(A16=59.75, C32=98.57, C32one=89.28)                          # pass T GPU busy
WALL = dict(A16=61.11, C32=102.09, C32one=90.46)
K_A16 = dict(exp=33.71, hc=9.38, gdnp=8.29, attp=3.03, shared=2.27, gdnc=2.23, head=1.67, router=1.31, attc=1.18, oth=1.27)
K_C32 = dict(exp=50.65, hc=11.51, gdnp=17.77, attp=6.14, shared=3.96, gdnc=4.56, head=3.53, router=2.02, attc=2.70, oth=2.18)
r16, r32 = sum(K_A16.values()) / PT["A16"], sum(K_C32.values()) / PT["C32"]
EXP16, EXP32 = K_A16["exp"] / r16, K_C32["exp"] / r32                 # pass-T scale
DENSE16_EXACT = PT["A16"] - EXP16
DENSE32ONE_EXACT = PT["C32one"] - EXP32
CPU32ONE = 12.16 + 14.10 + 1.68                                        # build + encode + read/commit [K]
UNION_ENG32 = 108.0                                                    # the profile's union at 32 rows [K slice]
# engine expert ms per REF2-T GB (its 3 code windows read like the map's code windows per union expert)
ref2t_per_union = GB_EXP["ref2t"] / mean["union"]
MS_PER_GB = EXP32 / (UNION_ENG32 * ref2t_per_union)
p(f"C. engine [K]: experts {EXP16:.2f} / {EXP32:.2f} ms (A16 / C32, pass-T scale); dense side A16 {DENSE16_EXACT:.2f}, "
  f"C32one {DENSE32ONE_EXACT:.2f}; CPU at C32one {CPU32ONE:.2f} ms")
p(f"   experts: {MS_PER_GB:.2f} ms per REF2-T GB = {MS_PER_GB * BW / 1e3:.2f} x the byte floor (REF2-T bytes); "
  f"{EXP32 / (UNION_ENG32 * NL * E4) / 1e3 * BW:.2f} x on an all-4-bit basis")

# ---------------------------------------------------------------- D. FAST dense side (lossy-45) [K at 16 rows]
F16 = dict(hc=3.77 + 0.08, attp=2.64 - 0.51, attc=0.51, gdnw=4.62 + (4.28 - 2.11), gdnc=2.11, moe=1.22, ple=0.52,
           head=0.78, unpriced=1.9)
GROW = dict(hc=0.25, attp=0.25, attc=(0.5, 1.0), gdnw=0.25, gdnc=1.0, moe=0.60, ple=0.37, head=0.26, unpriced=(0.25, 1.0))


def fast32(lo=True, hc_x=1.0, ple_x=1.0, head_x=1.0, gdn_x=1.0, extra=None):
    """the FAST dense side at 32 rows, one pass; extra = {class: multiplier} for the month ladder's kernel levers"""
    t = 0.0
    for k, v in F16.items():
        g = GROW[k]
        g = (g[0] if lo else g[1]) if isinstance(g, tuple) else g
        x = dict(hc=hc_x, ple=ple_x, head=head_x, gdnw=gdn_x, attp=gdn_x).get(k, 1.0)
        t += v * (1 + g) * x * ((extra or {}).get(k, 1.0))
    return t


F32_LO, F32_HI = fast32(True), fast32(False)
F32_X = DENSE32ONE_EXACT * sum(F16.values()) / DENSE16_EXACT           # cross-check: engine C32one x the FAST / exact ratio
p(f"D. FAST dense side: 16 rows {sum(F16.values()):.2f} ms [K]; 32 rows one pass {F32_LO:.1f}-{F32_HI:.1f} [P]; "
  f"cross-check (engine C32one x FAST/exact at 16) {F32_X:.1f}")
p(f"   two 16-row native calls instead (today's only > 16-row route without the DNR unpack) ~{2 * sum(F16.values()):.1f}")

# ---------------------------------------------------------------- E. 45 GB misses, lossy-45 code [K at 16 rows]
ck, mi, sv = [], [], []
for r in json.load(open(os.path.expanduser("~/BigMoE/data/r57/rlh_256_checks.json"))):
    for i in range(1, len(r["ck"])):                                   # check 0 (after the prefill) left out
        ck.append(r["ck"][i]); mi.append(r["miss"][i]); sv.append(r["svc"][i])
MISS16, SVC16, CK16 = st.median(mi), st.median(sv), st.median(ck)
MS_PER_MISS = SVC16 / MISS16
U16 = float(np.mean(u16_code))                                         # the same code windows at 16 rows [K]
p(f"   union per layer, code windows: 16 rows {U16:.1f}, 32 rows {mean['union']:.1f} (x{mean['union'] / U16:.2f})")
p(f"E. lossy-45 HE 16-row checks [K]: n {len(ck)}, check median {CK16:.1f} ms, misses {MISS16:.0f}, service "
  f"{SVC16:.1f} ms ({MS_PER_MISS:.3f} ms per miss)")


def miss32(read_share):
    m = MISS16 * mean["union"] / U16 * read_share
    return m, m * MS_PER_MISS


# self-check (a): the model at 16 rows for lossy-45: FAST dense + engine experts + misses + commit tail
COMMIT = 1.5                                                           # read + commit CPU at 16 rows [K 1.47]
m16 = sum(F16.values()) + EXP16 + SVC16 + COMMIT
sc_a = abs(m16 / CK16 - 1) <= 0.10
p(f"   SELF-CHECK (a): model at 16 rows {m16:.1f} ms vs measured {CK16:.1f}: {100 * (m16 / CK16 - 1):+.1f} % -> "
  f"{'PASS' if sc_a else 'FAIL'} (line +-10 %)")
m32x = DENSE32ONE_EXACT + EXP32
sc_b = abs(m32x / PT["C32one"] - 1) <= 0.05
p(f"   SELF-CHECK (b): exact-45 one pass at 32 rows {m32x:.2f} vs C32one GPU busy {PT['C32one']:.2f} (wall "
  f"{WALL['C32one']:.2f}): {'PASS' if sc_b else 'FAIL'} (calibration identity)")

# ---------------------------------------------------------------- F. the candidates
KEPT_HE, KEPT_MB, OVH = 16.91, 9.50, 4.0
# v2 (Mon 09:3x, Leap cf70494 dispatch_diet_price.md section 5): the 1c path adds ~34 MLX ops per MoE layer (BigMoE's
# tiers_mx + counter + km 28, Leap's 2-D dup pick 6; counted from the graph [K]) = ~1,630 per check, at ~1.1-2.1 us GPU
# [est] and ~6.5 us CPU [K] per dispatch: GPU +1.8-3.4 ms, CPU +10.6 ms, unless fused into router_fused (v1 had 0.3-1.0
# GPU and 0-2 CPU [est])
TIER_RULE = (1.8, 3.4)                                                 # GPU ms of the 1c path's glue per check
C1C_CPU = 1632 * 6.5e-3                                                # CPU ms of the 1c path's glue per check (10.6)
CPU_EXTRA = (0.0, 1.0)                                                 # [est] CPU ms of V1 / V3's few extra dispatches
HC4_X = (0.55, 0.75)                                                   # [est] HC time 4-bit / 8-bit at 32 rows
V1_HEAD_X = (0.15, 0.35)                                               # [est] head time with the 16K list / with SVH
V4_GDN_X = (0.80, 0.85)                                                # [est] GDN / attention projections at 3 vs 4 bits
c1_share = mean["c1_kept"] / mean["union"]


def cand(name, dgb, egb, read_share, v1=False, v2=False, v3=False, v4=False):
    """returns GB, floor, realistic (lo, hi) for GPU / CPU / misses / total, tok/s"""
    gb = dgb + egb
    flo = gb / BW * 1e3
    out = []
    for lo in (True, False):
        i = 0 if lo else 1
        d = fast32(lo, hc_x=HC4_X[i] if v3 else 1.0, ple_x=0.75 if v3 else 1.0, head_x=V1_HEAD_X[i] if v1 else 1.0,
                   gdn_x=V4_GDN_X[i] if v4 else 1.0)
        e = egb * MS_PER_GB + (TIER_RULE[i] if v2 else 0.0)
        gpu = d + e
        cpu = CPU32ONE + (C1C_CPU if v2 else 0.0) + (CPU_EXTRA[i] if (v1 or v3) else 0.0)
        _, ms_m = miss32(read_share)
        tot = max(gpu, cpu) + ms_m + 0.0
        d2 = d * 2 * sum(F16.values()) / fast32(lo)          # the same levers on two 16-row calls
        tot2 = max(d2 + e, cpu) + ms_m
        out.append((d, e, gpu, cpu, ms_m, tot, tot2))
    return dict(name=name, gb=gb, dgb=dgb, egb=egb, floor=flo, lo=out[0], hi=out[1])


C = []
C.append(dict(name="C0 exact-45 (REF2-T), 32 rows one pass, warm, C 160 [K]", gb=dense["exact"] + GB_EXP["ref2t"],
              dgb=dense["exact"], egb=GB_EXP["ref2t"], floor=(dense["exact"] + GB_EXP["ref2t"]) / BW * 1e3,
              lo=(DENSE32ONE_EXACT, EXP32, PT["C32one"], CPU32ONE, 0.0, WALL["C32one"], WALL["C32"]),
              hi=(DENSE32ONE_EXACT, EXP32, PT["C32one"], CPU32ONE, 0.0, WALL["C32one"], WALL["C32"])))
c0b = DENSE32ONE_EXACT + GB_EXP["ref2t"] * MS_PER_GB
C.append(dict(name="C0b exact-45 one pass on the map code windows' bytes [P]", gb=dense["exact"] + GB_EXP["ref2t"],
              dgb=dense["exact"], egb=GB_EXP["ref2t"], floor=(dense["exact"] + GB_EXP["ref2t"]) / BW * 1e3,
              lo=(DENSE32ONE_EXACT, GB_EXP["ref2t"] * MS_PER_GB, c0b, CPU32ONE, 0.0, c0b + 1.2, c0b + 1.2 + 11.6),
              hi=(DENSE32ONE_EXACT, GB_EXP["ref2t"] * MS_PER_GB, c0b, CPU32ONE, 0.0, c0b + 1.2, c0b + 1.2 + 11.6)))
C.append(cand("C1 lossy-45 as shipped, at 32 rows", dense["l45"], GB_EXP["ref2t"], 1.0))
C.append(cand("C2 C1 + V1 (16K code lm_head list)", dense["l45_v1"], GB_EXP["ref2t"], 1.0, v1=True))
C.append(cand("C3 C1 + V2 (1c tiers, lam 1.6e-2)", dense["l45"], GB_EXP["c1_eng"], c1_share, v2=True))
C.append(cand("C4 C1 + V1 + V2", dense["l45_v1"], GB_EXP["c1_eng"], c1_share, v1=True, v2=True))
C.append(cand("C5 C4 + V3 (HC + PLE 4-bit) = the 5.86 GB plan", dense["l45_v1v3"], GB_EXP["c1_eng"], c1_share, v1=True,
              v2=True, v3=True))
# other lambdas from Breakthrough's map 4-bit-eq, carried to engine bytes by ENG_OVER_MAP; read share scaled with bytes
for nm, eq, lam in (("C6 C5 at lam 1.8e-2 (frontier)", 33.6, "1.8e-2"), ("C7 1d pooled-safe (lam 8e-3) + V1 + V3", 54.8, "8e-3")):
    egb = NL * eq * E4 * ENG_OVER_MAP
    C.append(cand(nm, dense["l45_v1v3"], egb, c1_share * eq / c1_map_eq, v1=True, v2=True, v3=True))
C.append(cand("C8 C5 + V4 (GDN / attention 3-bit)", dense["l45_v1v3v4"], GB_EXP["c1_eng"], c1_share, v1=True, v2=True,
              v3=True, v4=True))

p()
p("| candidate | GB / 32-row check (dense + experts) | floor ms | FAST dense ms | experts ms | GPU ms | CPU ms | 45 GB misses ms | "
  "realistic ms | x floor | today's > 16-row route ms | tok/s HE+ / MBPP at K* (drafting hidden) |")
p("|---|---|---|---|---|---|---|---|---|---|---|---|")
rows = []
for c in C:
    lo, hi = c["lo"], c["hi"]

    def rng(i, f="{:.1f}"):
        a, b = sorted((lo[i], hi[i]))
        return f.format(a) if abs(a - b) < 0.05 else f.format(a) + "-" + f.format(b)
    tl, th = sorted((lo[5], hi[5]))
    tok = (f"{1e3 * KEPT_HE / (th + OVH):.0f}-{1e3 * KEPT_HE / (tl + OVH):.0f} / "
           f"{1e3 * KEPT_MB / (th + OVH):.0f}-{1e3 * KEPT_MB / (tl + OVH):.0f}")
    p(f"| {c['name']} | {c['gb']:.2f} ({c['dgb']:.2f} + {c['egb']:.2f}) | {c['floor']:.1f} | {rng(0)} | {rng(1)} | {rng(2)} | "
      f"{rng(3)} | {rng(4)} | **{rng(5)}** | {tl / c['floor']:.2f}-{th / c['floor']:.2f} | {rng(6)} | {tok} |")
    rows.append(dict(name=c["name"], gb=round(c["gb"], 3), dense_gb=round(c["dgb"], 3), exp_gb=round(c["egb"], 3),
                     floor_ms=round(c["floor"], 2), realistic_ms=[round(tl, 1), round(th, 1)],
                     gpu=[round(lo[2], 1), round(hi[2], 1)], cpu=[round(lo[3], 1), round(hi[3], 1)],
                     miss_ms=round(lo[4], 1), today_route_ms=[round(lo[6], 1), round(hi[6], 1)]))

# ---------------------------------------------------------------- G. what 25 ms would take (C5 decomposed)
c5 = [c for c in C if c["name"].startswith("C5")][0]
d5, e5, g5, cpu5, m5, t5, _t5b = c5["lo"]
dfl = c5["dgb"] / BW * 1e3
efl = c5["egb"] / BW * 1e3
p()
p(f"G. C5 (the 5.86 GB plan) at its low end, {t5:.1f} ms: FAST dense {d5:.1f} (its bytes' floor {dfl:.1f}: excess "
  f"{d5 - dfl:.1f}, of it cores {F16['attc'] * 1.5 + F16['gdnc'] * 2:.1f}); experts {e5:.1f} (floor {efl:.1f}); misses {m5:.1f}; "
  f"CPU {cpu5:.1f} (a bound: hidden while the GPU is longer)")
p(f"   25 ms needs, all together: misses -> 0 (a resident-only rule or no SSD), the dense side's weight passes at ~1.2x "
  f"their floor ({1.2 * dfl:.1f}) + cores ~{F16['attc'] * 1.5 + F16['gdnc'] * 2:.1f}, experts at ~1.1x ({1.1 * efl:.1f}), "
  f"and the CPU build + encode under ~25 ms (today {CPU32ONE:.1f})")
p(f"   -> {1.2 * dfl + F16['attc'] * 1.5 + F16['gdnc'] * 2 + 1.1 * efl:.1f} ms GPU at best for C5's bytes")

# ---------------------------------------------------------------- H. the month ladder (v2): recorded levers only
# Leap's FAST 1b list (TABLE_350 PLAN 220 part 2, 16-row prices [P]) as per-class cuts, carried to 32 rows by the class
# growth (a fractional cut of a class's time scales with it): chunked GDN step -0.8..-1.0 of the 2.11 ms GDN core; HC8
# restructure -1.05..-1.8 of the 3.85 ms HC (taken as the same share of a candidate's own HC time, so C5's HC4 is not
# double counted); router on a native 8-bit kernel -0.5 of the 1.22 ms MoE dense part; DN_KS=2 -0.3 of the 6.79 ms GDN
# weights; GDNB -0.13 on the GDN core [K FAST in place]
B1 = [dict(gdnc=1 - (0.8 + 0.13) / 2.11, hc=1 - 1.05 / 3.85, moe=1 - 0.5 / 1.22, gdnw=1 - 0.3 / 6.79),
      dict(gdnc=1 - (1.0 + 0.13) / 2.11, hc=1 - 1.8 / 3.85, moe=1 - 0.5 / 1.22, gdnw=1 - 0.3 / 6.79)]
RES_SHARE = mean["c1_lsb"] / mean["c1_kept"]          # resident-only MSB-3 rule: only 4-bit-tier (LSB) uses may miss [P]


def ladder(name, v1, v3, read_share):
    p()
    p(f"H. MONTH LADDER for {name} (realistic = max(GPU, CPU) + misses; ms per 32-row check; lo-hi):")
    p("| step | FAST dense | experts + 1c glue | GPU | CPU | misses | realistic | binds | tok/s HE+ at K* |")
    p("|---|---|---|---|---|---|---|---|---|")
    out = []
    for step in range(6):
        r = []
        for i, lo in enumerate((True, False)):
            kw = dict(hc_x=HC4_X[i] if v3 else 1.0, ple_x=0.75 if v3 else 1.0, head_x=V1_HEAD_X[i] if v1 else 1.0)
            d = fast32(lo, extra=B1[i] if step >= 2 else None, **kw)
            if step == 0:                                    # today's > 16-row route: two 16-row calls
                d = d * 2 * sum(F16.values()) / fast32(lo)
            glue = 0.0 if step >= 5 else TIER_RULE[i]
            e = GB_EXP["c1_eng"] * MS_PER_GB + glue
            cpu = CPU32ONE + (0.0 if step >= 5 else C1C_CPU) + (CPU_EXTRA[i] if (v1 or v3) else 0.0)
            share = read_share * (RES_SHARE if step >= 3 else 1.0)
            _, ms_m = miss32(share)
            if step == 4:                                    # no misses at all (a bound: needs no SSD in the check)
                ms_m = 0.0
            gpu = d + e
            r.append((d, e, gpu, cpu, ms_m, max(gpu, cpu) + ms_m, "CPU" if cpu > gpu else "GPU"))
        lab = ["0 today's route (two 16-row calls)", "1 + FAST one-pass forms > 16 rows (not built)",
               "2 + Leap's FAST 1b list (~3 days; P)", "3 + resident-only MSB-3 rule (not in TRIED; quality unpriced)",
               "4 (bound) misses 0", "5 + 1c glue fused into router_fused (Leap's lever A part)"][step]

        def rr(j):
            a, b = sorted((r[0][j], r[1][j]))
            return f"{a:.1f}" if abs(a - b) < 0.05 else f"{a:.1f}-{b:.1f}"
        tl, th = sorted((r[0][5], r[1][5]))
        binds = "/".join(sorted(set(x[6] for x in r)))
        p(f"| {lab} | {rr(0)} | {rr(1)} | {rr(2)} | {rr(3)} | {rr(4)} | **{rr(5)}** | {binds} | "
          f"{1e3 * KEPT_HE / (th + OVH):.0f}-{1e3 * KEPT_HE / (tl + OVH):.0f} |")
        out.append(dict(step=lab, realistic=[round(tl, 1), round(th, 1)], binds=binds))
    return out


LADDER = {"C3 1c-tiers": ladder("C3 = 1c-tiers (the gated variant)", False, False, c1_share),
          "C5 plan": ladder("C5 = the 5.86 GB plan (6.12 GB on code)", True, True, c1_share)}

json.dump(dict(version="v2 (1c glue from Leap cf70494)", mean=mean, gb_exp=GB_EXP, dense=dense, rows=rows, ladder=LADDER,
               selfchecks=dict(a=[round(m16, 2), round(CK16, 2), bool(sc_a)], b=[round(m32x, 2), PT["C32one"], bool(sc_b)],
                               c=[bool(sc_c1), bool(sc_c2)])),
          open(os.path.join(OUT, "lossy32_price.json"), "w"), indent=1)
open(os.path.join(OUT, "lossy32_price.txt"), "w").write("\n".join(L) + "\n")
