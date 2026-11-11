"""DS report (Leap, Mon 2026-10-26; CPU only): drafter data slope from the per-sample replay JSONs (refine19.py R19_JSON).
T = tokens per pass on the 24 held-out HumanEval samples; primary T = the better of block 32 1-step / 2-step tail 0.9 per
drafter (as L1-20's 'best'). s_fix = T(L1-20, 100 %, 201 steps) / T(B, 50 %, 201 steps); s_prop = T(L1-20) / T(A, 50 %,
101 steps); c50 = T(B) / T(A). 90 % paired bootstrap CIs over the 24 samples (sample noise only, not which-half noise).
Exit 10 = run arm A next (s_fix < 1.12 and A not yet scored). Writes ~/Leap/data/ds_report.txt"""
import json, os
import numpy as np

D = os.path.expanduser("~/Leap/data")
MODES = ("32:1step", "32:tail_0.9", "16:1step")
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

def load(tag):
    p = f"{D}/ds_{tag}.json"
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    return {m: np.array(j["per"][m], float) for m in MODES if m in j["per"]}

def T(per, m):
    return per[m][:, 0].sum() / per[m][:, 1].sum()

def prim(per):
    return max(MODES[:2], key=lambda m: T(per, m))

rng = np.random.default_rng(0)
IDX = rng.integers(0, 24, (10000, 24))
def ratio_ci(num, mn, den, md):
    """paired bootstrap over samples of T(num, mn) / T(den, md)"""
    a, b = num[mn][IDX], den[md][IDX]
    r = (a[..., 0].sum(1) / a[..., 1].sum(1)) / (b[..., 0].sum(1) / b[..., 1].sum(1))
    return T(num, mn) / T(den, md), np.percentile(r, 5), np.percentile(r, 95)

def val_line(tag, stf):
    p = f"{D}/{stf}"
    if not os.path.exists(p):
        return
    v = json.load(open(p))["log"]["val"]
    cells = []
    for e in v:
        if e["tag"] in ("before",):
            continue
        s = e["selector"]
        cells.append(f"{e['tag']}: " + " ".join(f"d{d} {s[d - 1]:.2f}" for d in (8, 16, 20, 24, 31)))
    say(f"  val selector agreement {tag}: " + " | ".join(cells))

full, B, A = load("l120"), load("B"), load("A")
if full is None or B is None:
    raise SystemExit("ds_report: need ds_l120.json and ds_B.json")
say("DS: drafter data slope (block-32 DFlash2, L1-20 recipe), 24 held-out HumanEval samples, tokens per pass")
for tag, per in (("L1-20 100 % 201 steps", full), ("B 50 % 201 steps", B), ("A 50 % 101 steps", A)):
    if per is not None:
        say(f"  {tag:22s}: " + ", ".join(f"{m} {T(per, m):.2f}" for m in MODES) + f" (primary {prim(per)})")
val_line("L1-20", "l120_ft.state.json"); val_line("B", "ds_B.state.json"); val_line("A", "ds_A.state.json")
pf, pb = prim(full), prim(B)
s_fix, lo, hi = ratio_ci(full, pf, B, pb)
say(f"  s_fix = T(100 %) / T(B) = {s_fix:.3f} (90 % CI {lo:.3f}-{hi:.3f}); per mode: " + ", ".join(
    f"{m} {ratio_ci(full, m, B, m)[0]:.3f}" for m in MODES))
rc = 0
if A is None:
    if s_fix >= 1.12:
        say("  DS: GO (s_fix >= 1.12: more data helps at fixed compute) -> 4x on-policy windows")
    else:
        say("  DS: s_fix < 1.12 -> NEXT: arm A (50 % at 101 steps) to separate data from compute")
        rc = 10
else:
    pa = prim(A)
    s_prop, plo, phi = ratio_ci(full, pf, A, pa)
    c50, clo, chi = ratio_ci(B, pb, A, pa)
    say(f"  s_prop = T(100 %) / T(A) = {s_prop:.3f} (90 % CI {plo:.3f}-{phi:.3f}); c50 = T(B) / T(A) = {c50:.3f} "
        f"(90 % CI {clo:.3f}-{chi:.3f})")
    if s_fix >= 1.12:
        v = "GO (data helps at fixed compute)"
    elif s_fix < 1.05 and c50 < 1.05:
        v = "KILL (flat on data at fixed steps AND on compute at 50 %)"
    elif s_fix < 1.05:
        v = "compute-limited, not data-limited: next 100 % at 402 steps (~56 min, no new 125B data)"
    else:
        v = "BETWEEN: one 2x-data point decides" + (" (compute also moves: try 100 % at 402 steps first)" if c50 >= 1.05 else "")
    say(f"  DS verdict (pre-registered lines on s_fix: KILL < 1.05 with c50 < 1.05, GO >= 1.12): {v}")
C = load("C")   # DS2: 100 % at 401 steps (ds2_batch.sh)
if C is not None:
    pc = prim(C)
    say(f"  DS2: C 100 % 401 steps   : " + ", ".join(f"{m} {T(C, m):.2f}" for m in MODES) + f" (primary {pc})")
    val_line("C", "ds_C.state.json")
    c100, lo, hi = ratio_ci(C, pc, full, pf)
    say(f"  c100 = T(C) / T(100 %, 201 steps) = {c100:.3f} (90 % CI {lo:.3f}-{hi:.3f}); per mode: " + ", ".join(
        f"{m} {ratio_ci(C, m, full, m)[0]:.3f}" for m in MODES))
    v2 = ("ALIVE: price a long run" if c100 >= 1.12 else "KILL: training this drafter harder cannot reach ~17-19"
          if c100 < 1.07 else "BETWEEN: one more doubling (801 steps) decides")
    say(f"  DS2 verdict (pre-registered: ALIVE >= 1.12, KILL < 1.07): {v2}")
open(f"{D}/ds_report.txt", "w").write("\n".join(lines) + "\n")
raise SystemExit(rc)
