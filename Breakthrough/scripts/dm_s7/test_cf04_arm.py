"""Install test of NewIdeas' cf 0.4 chat arm (Breakthrough, Wed 2026-11-04 22:5x; NewIdeas 22:49, QUEUE 22:4x fb39925; the
orchestrator: a low-priority Thursday arm, after DENSE-3, only if a share slot is free). Runs against the dm_ablations.py
first on PYTHONPATH (install_cf04.sh: the scratch copy). CPU:
  T1  ABLATIONS["qom_0.3_live_c147_bonus4g"] sets exactly PERT {rmask: the masks dir, renorm: False} (the chat arms'
      q1_union form; no t3, 4-bit stock path), and the harness's loader resolves the absolute path;
  T2  the masks: 36 windows, each the shape of the cf 0 base's and of the routing;
  T3  against the cf 0 base qom_0_live_c147 (= qom_0.3_live_c147): drops only, ranks 0-1 identical, 313 rows differ (the
      bonus rows the 0.4 gate gates; NewIdeas);
  T4  nested in the 0.3 gate: every row that differs from the base also differs in qom_0.3_live_c147_bonus3g (528 rows).
Usage (BigMoE venv): PYTHONPATH=<dir with dm_ablations.py>:~/FableGod/decision_map python test_cf04_arm.py"""
import os, sys
import numpy as np
import dm_ablations as A
import dm_common as C
ARM = "qom_0.3_live_c147_bonus4g"
MASKS = os.path.expanduser("~/NewIdeas/data/dm_masks/" + ARM)
BASE = os.path.expanduser("~/NewIdeas/data/dm_masks/qom_0_live_c147")
CF03 = os.path.expanduser("~/NewIdeas/data/dm_masks/qom_0.3_live_c147_bonus3g")
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
ok(len(names) == 36 and names == sorted(f for f in os.listdir(BASE) if f.endswith(".npz")), f"T2 {len(names)} windows, the same names as the base's")
shp = drops = r01 = nest = rows = chg = chg3 = 0
for nm in names:
    a, b, c = (np.load(os.path.join(d, nm))["keep"] for d in (MASKS, BASE, CF03))
    ids = np.load(os.path.join(ROUT, nm))["ids"]
    shp += a.shape == b.shape == c.shape == ids.shape
    if not (a.shape == b.shape == c.shape): continue
    drops += not (a & ~b).any(); r01 += np.array_equal(a[..., :2], b[..., :2])
    d4, d3 = (a != b).any(axis=(1, 2)), (c != b).any(axis=(1, 2))
    nest += not (d4 & ~d3).any(); rows += a.shape[0]; chg += int(d4.sum()); chg3 += int(d3.sum())
ok(shp == len(names), f"T2 shapes == the base's == cf 0.3's == the routing's in {shp}/{len(names)} windows")
ok(drops == len(names), f"T3 drops only vs the cf 0 base in {drops}/{len(names)} windows")
ok(r01 == len(names), f"T3 ranks 0-1 identical to the base in {r01}/{len(names)} windows")
ok(chg == 313, f"T3 rows that differ from the base: {chg} (NewIdeas: 313)")
ok(nest == len(names), f"T4 every changed row is also changed at cf 0.3 in {nest}/{len(names)} windows ({chg} within {chg3}; NewIdeas: 528)")
print("ALL OK" if not bad else f"FAILED: {len(bad)}", flush=True)
sys.exit(1 if bad else 0)
