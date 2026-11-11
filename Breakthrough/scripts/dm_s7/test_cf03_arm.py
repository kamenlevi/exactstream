"""Install test of NewIdeas' cf 0.3 chat arm (Breakthrough, Wed 2026-11-04 22:0x; NewIdeas 22:02, QUEUE 22:0x 04ac744; the
orchestrator 22:0x: 4th in tonight's order, only if it can start by 22:50). Runs against the dm_ablations.py first on
PYTHONPATH (install_cf03.sh: the scratch copy). CPU:
  T1  ABLATIONS["qom_0.3_live_c147_bonus3g"] sets exactly PERT {rmask: the masks dir, renorm: False} (the chat arm's q1_union
      form; no t3, 4-bit stock path), and the harness's loader resolves the absolute path;
  T2  the masks: 36 windows, each the shape of the chat arm's (qom_0.5_live_c147_bonus5g) and of the routing;
  T3  against the cf 0 base qom_0_live_c147 (= qom_0.3_live_c147, NewIdeas check A): drops only (no expert added), ranks
      0-1 identical, rows that differ = the gated bonus rows (NewIdeas check B: 528). Against the chat arm (cf 0.5) it is
      NOT drops only: the pools evolve differently (cf 0.5 has more exact rows loading all 10).
Usage (BigMoE venv): PYTHONPATH=<dir with dm_ablations.py>:~/FableGod/decision_map python test_cf03_arm.py"""
import os, sys
import numpy as np
import dm_ablations as A
import dm_common as C
ARM = "qom_0.3_live_c147_bonus3g"
MASKS = os.path.expanduser("~/NewIdeas/data/dm_masks/" + ARM)
CHAT = os.path.expanduser("~/NewIdeas/data/dm_masks/qom_0.5_live_c147_bonus5g")
BASE = os.path.expanduser("~/NewIdeas/data/dm_masks/qom_0_live_c147")   # = qom_0.3_live_c147 bit for bit (NewIdeas check A)
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
ok(len(names) == 36 and names == sorted(f for f in os.listdir(CHAT) if f.endswith(".npz")), f"T2 {len(names)} windows, the same names as the chat arm's")
shp = drops = r01 = rows = chg = 0
for nm in names:
    a, b = np.load(os.path.join(MASKS, nm))["keep"], np.load(os.path.join(BASE, nm))["keep"]
    ids = np.load(os.path.join(ROUT, nm))["ids"]
    shp += a.shape == b.shape == ids.shape
    if a.shape != b.shape: continue
    drops += not (a & ~b).any(); r01 += np.array_equal(a[..., :2], b[..., :2])
    rows += a.shape[0]; chg += int((a != b).any(axis=(1, 2)).sum())
ok(shp == len(names), f"T2 shapes == the base's == the routing's in {shp}/{len(names)} windows")
ok(drops == len(names), f"T3 drops only vs the cf 0 base (qom_0_live_c147: no expert added) in {drops}/{len(names)} windows")
ok(r01 == len(names), f"T3 ranks 0-1 identical to the base in {r01}/{len(names)} windows")
print(f"[t] rows {rows}, rows that differ from the base {chg} (NewIdeas check B: 528, the gated bonus rows)", flush=True)
print("ALL OK" if not bad else f"FAILED: {len(bad)}", flush=True)
sys.exit(1 if bad else 0)
