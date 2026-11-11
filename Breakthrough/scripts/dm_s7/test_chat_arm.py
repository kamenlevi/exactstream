"""Install test of NewIdeas' chat route arm (Breakthrough, Wed 2026-11-04 20:2x; the orchestrator 20:1x; NewIdeas QUEUE 20:1x
4fb08ca, masks 4324e09). Runs against the dm_ablations.py first on PYTHONPATH (install_chat.sh: the scratch copy). CPU:
  T1  ABLATIONS["qom_0.5_live_c147_bonus5g"] sets exactly PERT {rmask: the masks dir, renorm: False} (1(c)'s q1_union form,
      no t3: 4-bit everywhere on the stock path), and the harness's loader resolves the absolute path;
  T2  the masks: 36 windows, each the shape of 1(c)'s (~/FableGod/fablec/masks/qom_0.5_live_bonus5g) and of the routing;
  T3  against 1(c): drops only (no expert kept that 1(c) drops), ranks 0-1 identical, so the arm differs from 1(c) only by
      the 147-slot pool's rank 2-9 drops (NewIdeas' chat_arm_check.py: 7,774 such rows); kept share printed (NewIdeas 82.5 %).
Usage (BigMoE venv): PYTHONPATH=<dir with dm_ablations.py>:~/FableGod/decision_map python test_chat_arm.py"""
import os, sys
import numpy as np
import dm_ablations as A
import dm_common as C
ARM = "qom_0.5_live_c147_bonus5g"
MASKS = os.path.expanduser("~/NewIdeas/data/dm_masks/" + ARM)
ONE_C = os.path.expanduser("~/FableGod/fablec/masks/qom_0.5_live_bonus5g")
ROUT = os.path.join(os.path.expanduser("~/FableGod/decision_map/out"), "routing")
bad = []
def ok(c, m):
    print(("  ok   " if c else "  FAIL ") + m, flush=True)
    if not c: bad.append(m)
print(f"[t] dm_ablations from {A.__file__}", flush=True)
C.clear(); A.ABLATIONS[ARM](); p = dict(C.PERT); C.clear()
ok(p == {"rmask": MASKS, "renorm": False}, f"T1 PERT == {{rmask: masks dir, renorm: False}} ({p})")
ok(os.path.join(C.OUT, p["rmask"], "x") == os.path.join(MASKS, "x"), "T1 the loader's os.path.join(OUT, rmask, nm) resolves to the masks dir")
names = sorted(f for f in os.listdir(MASKS) if f.endswith(".npz"))
ok(len(names) == 36 and names == sorted(f for f in os.listdir(ONE_C) if f.endswith(".npz")), f"T2 {len(names)} windows, the same names as 1(c)'s")
shp = drops = r01 = rows = chg = kept = tot = 0
for nm in names:
    a, b = np.load(os.path.join(MASKS, nm))["keep"], np.load(os.path.join(ONE_C, nm))["keep"]
    ids = np.load(os.path.join(ROUT, nm))["ids"]
    shp += a.shape == b.shape == ids.shape
    if a.shape != b.shape: continue
    drops += not (a & ~b).any(); r01 += np.array_equal(a[..., :2], b[..., :2])
    rows += a.shape[0]; chg += int((a != b).any(axis=(1, 2)).sum()); kept += int(a.sum()); tot += a.size
ok(shp == len(names), f"T2 shapes == 1(c)'s == the routing's in {shp}/{len(names)} windows")
ok(drops == len(names), f"T3 drops only vs 1(c) in {drops}/{len(names)} windows (no expert kept that 1(c) drops)")
ok(r01 == len(names), f"T3 ranks 0-1 identical to 1(c) in {r01}/{len(names)} windows")
print(f"[t] rows {rows}, rows that differ from 1(c) {chg} (NewIdeas: 7,774); kept share {100 * kept / max(tot, 1):.1f} % "
      f"of row-layer-ranks (NewIdeas: 82.5 %)", flush=True)
print("ALL OK" if not bad else f"FAILED: {len(bad)}", flush=True)
sys.exit(1 if bad else 0)
