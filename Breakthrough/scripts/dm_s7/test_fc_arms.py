"""test_fc_arms.py (Fable C, Thu 2026-11-05): a COPY of Breakthrough's scripts/dm_s7/test_c3_arms.py for build_1105's arms.
Changed lines carry "FC": the default source / snapshot dirs, ranks 0-1 required by default, and one exception: an arm with
"_z" in its name may hold rows that keep NO routed expert, in all of layers 8-35 and nowhere else (the middle-layer skip;
the harness multiplies the routing weights by the mask with renorm False, so such a row-layer posts a zero routed output,
the gate path's k 0 of gate_top0_cf0999_L08_35).
---- the original docstring follows ----
Install test of Cloud Fable 3's mask arms (install_c3.sh; Breakthrough, Thu 2026-11-05). Runs against the dm_ablations.py
first on PYTHONPATH (the scratch copy). CPU. Per arm: ABLATIONS[arm] sets exactly PERT {rmask: <snapshot dir>, renorm: False};
the harness's loader (os.path.join(OUT, rmask, nm)) resolves to the snapshot; the snapshot holds 36 windows whose files equal
cloud3's source byte for byte and whose keep arrays have the map's routing shapes.
Usage (BigMoE venv): PYTHONPATH=<dir with dm_ablations.py>:~/FableGod/decision_map python test_c3_arms.py <arm> ..."""
import filecmp, os, sys
import numpy as np
import dm_ablations as A
import dm_common as C
SRC = os.path.expanduser(os.environ.get("C3_SRC", "~/FableGod/fablec/build_1105/masks"))   # FC
SNAP = os.path.expanduser(os.environ.get("C3_SNAP", "~/FableGod/decision_map/out/fc"))   # FC
R01 = os.environ.get("TEST_R01", "1") == "1"   # FC (default on): ranks 0-1 kept on every row (QOM-form arms)
ROUT = os.path.expanduser("~/FableGod/decision_map/out/routing")
bad = []
def ok(c, m):
    print(("  ok   " if c else "  FAIL ") + m, flush=True)
    if not c: bad.append(m)
print(f"[t] dm_ablations from {A.__file__}", flush=True)
for arm in sys.argv[1:]:
    d = os.path.join(SNAP, arm)
    C.clear(); A.ABLATIONS[arm](); p = dict(C.PERT); C.clear()
    ok(p == {"rmask": d, "renorm": False}, f"{arm}: PERT == {{rmask: snapshot, renorm: False}}")
    ok(os.path.join(C.OUT, p["rmask"], "x") == os.path.join(d, "x"), f"{arm}: the loader resolves to the snapshot")
    names = sorted(f for f in os.listdir(d) if f.endswith(".npz"))
    same = sum(filecmp.cmp(os.path.join(d, n), os.path.join(SRC, arm, n), shallow=False) for n in names)
    shp = sum(np.load(os.path.join(d, n))["keep"].shape == np.load(os.path.join(ROUT, n))["ids"].shape for n in names)
    ok(len(names) == 36 and same == 36 and shp == 36, f"{arm}: 36 windows, {same}/36 byte-equal to the source, {shp}/36 routing shapes")
    if "_z" in arm:                                                                     # FC: the middle-layer skip
        r0 = z = 0
        for n in names:
            k = np.load(os.path.join(d, n))["keep"]; nothing = ~k.any(axis=2); zr = nothing.any(axis=1)
            r0 += bool((k[..., :2].all(axis=2) | nothing).all() and not nothing[:, :8].any() and not nothing[:, 36:].any()
                       and nothing[zr][:, 8:36].all()); z += int(zr.sum())
        ok(r0 == 36 and z > 0, f"{arm}: ranks 0-1 kept on every row-layer except {z} skipped rows (nothing kept in all of layers 8-35, ranks 0-1 elsewhere) in {r0}/36 windows")
    else:
        r0 = sum(bool(np.load(os.path.join(d, n))["keep"][..., :2 if R01 else 1].all()) for n in names)
        ok(r0 == 36, f"{arm}: rank 0{'-1' if R01 else ''} kept on every row in {r0}/36 windows")
print("ALL OK" if not bad else f"FAILED: {len(bad)}", flush=True)
sys.exit(1 if bad else 0)
