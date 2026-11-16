"""lossy32_board.py (Fit12, Mon 16 Nov 2026 13:1x): the LOSSY32 SCOREBOARD v3 (pre-registered in PLAN "RES-1c PRICE +
SCOREBOARD v3"; NewIdeas folds it into TARGETS_450_900 daily). Desk, CPU, python -I. v2 (lossy32_price.py) stays the
record; this is the board, refreshed as pieces land.

Per candidate, per 32-row check on code at 45 GB: GB (dense + experts) and the byte floor; Leap's bound at 100 % of every
peak (check25_bound.py's form: max(dense MACs, dense bytes) + expert bytes + cores 1.2 ms); the realistic GPU (warm,
one-pass forms) from v2's per-class model; the CPU; misses COUNTED from the map's residency masks (res_c256) instead of
v2's union-share scaling; LAUNCHES per check; tok/s at the lossy K* (kept 16.91, +4 ms, drafting hidden).
Changes vs v2:
- 1c bytes for every lambda from the code routing with the rule (tiers_np, engine semantics), not Breakthrough's pooled
  4-bit-eq (its catch e62c405: C6 33.6 -> code 35.7);
- misses: the 1c family 5.3 ms (v2, scaled) -> counted ~15 ms (1c replaces QOM, whose resident keep was what kept
  lossy-45's misses low); + RES rows (the resident-only MSB-3 rule, res1c_price.py);
- launches: C32one 4,022 [K] + the 1c glue 1,632 [K count] (or + 282 with Leap's fused t1cf: -1,350).
"""
import json
import os

import numpy as np

M = os.path.expanduser("~/FableGod/decision_map/out")
OUT = "/Users/kamen/Fit12/data/lossy32"
T4_K, T3_K, S = 2.775, 0.8457, 0.98
E4, E3, NL, BW = 2.784e-3, 2.064e-3, 48, 280.0        # v3.1 (Leap 13:45, check25_bound_real.txt): the engine's records3 pieces
                                                       # (4-bit 2.784 MB, MSB-3 2.064 MB), not 2.135 / 1.661
MS_PER_MISS = 0.250                         # lossy-45 rlh_256 [K] median service / median misses
LAMS = (8e-3, 1.6e-2, 1.8e-2, 2.0e-2)


def tiers_np(ids, w, lam, s=S):
    rw = s ** np.arange(ids.shape[0])
    u, inv = np.unique(ids.ravel(), return_inverse=True)
    imp = np.bincount(inv, weights=(w ** 2 * rw[:, None]).ravel(), minlength=len(u))
    return np.where(imp > T4_K * lam, 0, np.where(imp > T3_K * lam, 1, 2))[inv].reshape(ids.shape)


# ---------------------------------------------------------------- A. code routing: bytes and misses per lambda
base = json.load(open(os.path.join(M, "baseline.json")))
code = sorted(f for f in os.listdir(os.path.join(M, "xcap")) if base[f]["dom"] == "code")
acc = {lam: dict(lsb=[], kept=[], e4=[], e3=[], miss=[], miss_res=[]) for lam in LAMS}
ref = dict(union=[], lsb=[], miss_exact=[], miss_qom0=[])
mask_mism = 0
for f in code:
    z = np.load(os.path.join(M, "xcap", f))
    ids, w, a = z["ids"].astype(np.int64), z["w"].astype(np.float64), int(z["a"])
    res = np.load(os.path.join(M, "res_c256", f))["keep"].astype(bool)[a:a + ids.shape[0]]
    t = np.load(os.path.join(M, "bt", "tiers_1c", f))
    k4m, k3m = t["keep"].astype(bool)[a:a + ids.shape[0]], t["keep3"].astype(bool)[a:a + ids.shape[0]]
    rank = np.broadcast_to(np.arange(10), (32, 10))
    for bi in range(ids.shape[0] // 32):
        sl = slice(bi * 32, (bi + 1) * 32)
        blk = {lam: dict(lsb=0, kept=0, e4=0, e3=0, miss=0, miss_res=0) for lam in LAMS}
        rb = dict(union=0, lsb=0, miss_exact=0, miss_qom0=0)
        for ly in range(NL):
            I, W, Rs = ids[sl, ly], w[sl, ly], res[sl, ly]
            uniq = np.unique(I)
            rb["union"] += len(uniq)
            rb["lsb"] += len(np.unique(I[rank < 2]))
            nres = {int(e): not Rs[I == e].all() for e in uniq}
            for e in uniq:
                if nres[int(e)]:
                    rb["miss_exact"] += 1
                    rb["miss_qom0"] += bool((rank[I == e] < 2).any())
            for lam in LAMS:
                T = tiers_np(I, W, lam)
                if lam == 1.6e-2:
                    mask_mism += int((T != np.where(k4m[sl, ly], 0, np.where(k3m[sl, ly], 1, 2))).sum())
                b = blk[lam]
                b["lsb"] += len(np.unique(I[(T == 0) & (rank < 2)]))
                for e in uniq:
                    te = int(T[I == e].min())
                    if te == 2:
                        continue
                    b["kept"] += 1; b["e4"] += te == 0; b["e3"] += te == 1
                    if nres[int(e)]:
                        b["miss"] += 1; b["miss_res"] += te == 0
        for k in ref:
            ref[k].append(rb[k])
        for lam in LAMS:
            for k in acc[lam]:
                acc[lam][k].append(blk[lam][k])
nblk = len(ref["union"])
R = {k: float(np.mean(v)) for k, v in ref.items()}
A = {lam: {k: float(np.mean(v)) for k, v in d.items()} for lam, d in acc.items()}
EXP_GB = {"ref2t": (R["lsb"] * E4 + (R["union"] - R["lsb"]) * E3)}
for lam in LAMS:
    EXP_GB[lam] = A[lam]["lsb"] * E4 + (A[lam]["kept"] - A[lam]["lsb"]) * E3
L = []


def p(x=""):
    print(x); L.append(x)


p(f"# LOSSY32 SCOREBOARD v3.1 (Fit12 lossy32_board.py; real piece sizes), {len(code)} code windows, {nblk} 32-row blocks")
p(f"SELF-CHECK: the rule at lambda 1.6e-2 vs Breakthrough's tiers_1c masks: {mask_mism} differing slots "
  f"({'PASS' if mask_mism == 0 else 'FAIL'})")
p(f"per 32-row check: union {R['union'] / NL:.1f} per layer; REF2-T experts {EXP_GB['ref2t']:.2f} GB; misses at C 256 "
  f"(res_c256 masks): exact {R['miss_exact']:.1f}, QOM cf 0 (a lower bound of QOM) {R['miss_qom0']:.1f}")
for lam in LAMS:
    a_ = A[lam]
    p(f"  lambda {lam:.1e}: kept {a_['kept'] / NL:.1f} per layer (4-bit tier {a_['e4'] / NL:.1f}, MSB-3 {a_['e3'] / NL:.1f}), "
      f"map 4-bit-eq {(a_['e4'] + 0.778 * a_['e3']) / NL:.1f}; experts {EXP_GB[lam]:.2f} GB (engine); misses {a_['miss']:.1f}, "
      f"with RES {a_['miss_res']:.1f}")

# ---------------------------------------------------------------- B. the time model (v2's measured per-class inputs)
DENSE = dict(exact=3.435, l45=2.594, v1=2.470, v1v3=2.133, v1v3v4=1.798)
EXP32_ENG, UNION_ENG = 50.65 / (105.01 / 98.57), 108.0   # engine C32 experts (pass K scaled to pass T) [K a082f8c]
MS_PER_GB = EXP32_ENG / (UNION_ENG * NL * (R["lsb"] * E4 + (R["union"] - R["lsb"]) * E3) / R["union"])   # R: per-check sums over 48 layers   # v3.1: from the
                                 # measured engine time on the real piece sizes (v3 hard-coded 5.12 on the old ones)
p(f"ms per expert GB {MS_PER_GB:.2f} = {MS_PER_GB * BW / 1e3:.2f} x the byte floor (real piece sizes)")
FAST32 = dict(l45=(27.0, 28.6), v1=(26.1, 28.0), v1v3=(23.8, 26.6), v1v3v4=(21.6, 25.0))   # v2 [P], one pass
FAST16X2 = 39.6                  # two 16-row calls (today's > 16-row route), no levers [K 16-row x2]
CPU0, GLUE_CPU, GLUE_GPU = 27.94, 10.6, (1.8, 3.4)
L0, GLUE_L, T1CF_CUT = 4022, 1632, 1350
MAC_BF16, MAC_INT8, CORES = 8.53, 4.93, 1.2                # Leap check25_bound [K peak x count]
KEPT, OVH = 16.91, 4.0


def bound(dgb, egb, int8=False):
    return max(MAC_INT8 if int8 else MAC_BF16, dgb / BW * 1e3) + egb / BW * 1e3 + CORES


def cand(name, dk, egb, v2, miss, int8=False, launches=None):
    d = FAST32[dk]
    glue = GLUE_GPU if v2 else (0.0, 0.0)
    gpu = (d[0] + egb * MS_PER_GB + glue[0], d[1] + egb * MS_PER_GB + glue[1])
    cpu = CPU0 + (GLUE_CPU if v2 else 0.0)
    mm = miss * MS_PER_MISS
    real = (max(gpu[0], cpu) + mm, max(gpu[1], cpu) + mm)
    today = (max(gpu[0] - d[0] + FAST16X2 * d[0] / FAST32["l45"][0], cpu) + mm,
             max(gpu[1] - d[1] + FAST16X2 * d[1] / FAST32["l45"][1], cpu) + mm)
    gb = DENSE[dk] + egb
    lau = launches if launches is not None else L0 + (GLUE_L if v2 else 0)
    return dict(name=name, gb=gb, dgb=DENSE[dk], egb=egb, floor=gb / BW * 1e3, bound=bound(DENSE[dk], egb, int8),
                gpu=gpu, cpu=cpu, miss=miss, mm=mm, real=real, today=today, launches=lau,
                tok=(1e3 * KEPT / (real[1] + OVH), 1e3 * KEPT / (real[0] + OVH)))


l16, l18, l20, l08 = (A[x] for x in (1.6e-2, 1.8e-2, 2.0e-2, 8e-3))
C1_MISS = 52.2                   # v2: lossy-45's measured 35 per 16-row check x the union ratio (QOM at 32 rows)
C = [
    cand("C1 lossy-45 as shipped, 32 rows (QOM)", "l45", EXP_GB["ref2t"], False, C1_MISS),
    cand("C3 = 1c-tiers (lam 1.6e-2), being gated", "l45", EXP_GB[1.6e-2], True, l16["miss"]),
    cand("C3 + RES", "l45", EXP_GB[1.6e-2], True, l16["miss_res"]),
    cand("C5 = C3 + V1 + V3 (the plan)", "v1v3", EXP_GB[1.6e-2], True, l16["miss"]),
    cand("C5 + RES", "v1v3", EXP_GB[1.6e-2], True, l16["miss_res"]),
    cand("C5 + RES, int8 dense (bound only differs)", "v1v3", EXP_GB[1.6e-2], True, l16["miss_res"], int8=True),
    cand("C6 = C5 at lam 1.8e-2 (code)", "v1v3", EXP_GB[1.8e-2], True, l18["miss"]),
    cand("C6 + RES", "v1v3", EXP_GB[1.8e-2], True, l18["miss_res"]),
    cand("C6 + RES, int8 dense", "v1v3", EXP_GB[1.8e-2], True, l18["miss_res"], int8=True),
    cand("C6 + V4 + RES, int8 dense (V4: map FAIL)", "v1v3v4", EXP_GB[1.8e-2], True, l18["miss_res"], int8=True),
    cand("C5 at lam 2.0e-2 + RES (today's map arm)", "v1v3", EXP_GB[2.0e-2], True, l20["miss_res"]),
    cand("C7 1d (lam 8e-3) + V1 + V3 + RES", "v1v3", EXP_GB[8e-3], True, l08["miss_res"]),
]
p()
p("| candidate | GB (dense + experts) | floor ms | Leap bound at 100 % | GPU warm ms (one pass) | CPU ms | misses / ms | "
  "realistic ms | today's > 16-row route | launches per check | tok/s HE+ at K* |")
p("|---|---|---|---|---|---|---|---|---|---|---|")
rows = []
for c in C:
    f2 = lambda t: f"{t[0]:.1f}-{t[1]:.1f}"
    p(f"| {c['name']} | {c['gb']:.2f} ({c['dgb']:.2f} + {c['egb']:.2f}) | {c['floor']:.1f} | {c['bound']:.1f} | {f2(c['gpu'])} | "
      f"{c['cpu']:.1f} | {c['miss']:.1f} / {c['mm']:.1f} | **{f2(c['real'])}** | {f2(c['today'])} | {c['launches']:,} | "
      f"{c['tok'][0]:.0f}-{c['tok'][1]:.0f} |")
    rows.append({k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()})
p()
p("Launches: 4,022 = C32one [K]; + 1,632 for the 1c glue [K count, Leap cf70494]; Leap's fused t1cf cuts ~1,350 of it "
  "[K count]; Fable B's diet from built pieces ~2,500; 25 ms needs ~400 [Fable B]. The FAST one-pass route's own count is "
  "not measured yet (taken as C32one's).")
p("Misses: counted on the map's residency masks (C 256 base pool, exact-pass LRU replay). The engine anchor (lossy-45, "
  "16-row HE: measured median 35 vs the masks' QOM cf 0 lower bound 18-20) says the masks may undercount by up to ~1.7x.")

# ---------------------------------------------------------------- C. the month ladder for C5 (v3)
c5, c5r = C[3], C[4]
fast_b1 = (23.8 - 4.3 * 23.8 / 27.0, 26.6 - 5.7 * 26.6 / 28.6)   # v2's FAST 1b list on C5's dense side [P]
lad = []
g_e = EXP_GB[1.6e-2] * MS_PER_GB
for lab, dense_, glue, cpu_, miss_ in (
        ("0 today's route (two 16-row calls)", (FAST16X2 * 23.8 / 27.0, FAST16X2 * 26.6 / 28.6), GLUE_GPU, CPU0 + GLUE_CPU, l16["miss"]),
        ("1 + FAST one-pass forms > 16 rows (dn32 matmuls MEASURED -6.1 of it [K])", (23.8, 26.6), GLUE_GPU, CPU0 + GLUE_CPU, l16["miss"]),
        ("2 + Leap's FAST 1b list [P]", fast_b1, GLUE_GPU, CPU0 + GLUE_CPU, l16["miss"]),
        ("3 + RES (CHEAP on the map law; needs a map arm)", fast_b1, GLUE_GPU, CPU0 + GLUE_CPU, l16["miss_res"]),
        ("4 + the 1c glue fused (Leap's t1cf, built + equal on 384K slots)", fast_b1, (0.0, 0.0), CPU0, l16["miss_res"])):
    gl, gh = dense_[0] + g_e + glue[0], dense_[1] + g_e + glue[1]
    mm = miss_ * MS_PER_MISS
    rl, rh = max(gl, cpu_) + mm, max(gh, cpu_) + mm
    lad.append(dict(step=lab, gpu=[gl, gh], cpu=cpu_, miss_ms=mm, real=[rl, rh]))
p()
p("MONTH LADDER v3, C5 (ms per 32-row check; realistic = max(GPU, CPU) + misses):")
p("| step | GPU warm | CPU | misses ms | realistic | tok/s HE+ at K* |")
p("|---|---|---|---|---|---|")
for s_ in lad:
    p(f"| {s_['step']} | {s_['gpu'][0]:.1f}-{s_['gpu'][1]:.1f} | {s_['cpu']:.1f} | {s_['miss_ms']:.1f} | **{s_['real'][0]:.1f}-{s_['real'][1]:.1f}** | "
      f"{1e3 * KEPT / (s_['real'][1] + OVH):.0f}-{1e3 * KEPT / (s_['real'][0] + OVH):.0f} |")
p("Leap's own estimate for the measured C5 check by Friday: ~40-46 ms GPU warm (check25_plan 12:2x); its bound 23.9 ms at "
  "100 % of every peak. 900 needs ~15 ms at kept 16.91.")
os.makedirs(OUT, exist_ok=True)
json.dump(dict(version="v3.1 (records3 piece sizes, Leap 13:45)", ref=R, lam=A, exp_gb={str(k): v for k, v in EXP_GB.items()}, rows=rows, ladder=lad,
               selfcheck_mask_mismatch=mask_mism), open(os.path.join(OUT, "lossy32_board.json"), "w"), indent=1)
open(os.path.join(OUT, "lossy32_board.txt"), "w").write("\n".join(L) + "\n")
