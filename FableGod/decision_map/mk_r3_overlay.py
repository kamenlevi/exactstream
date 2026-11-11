"""An overlay records3 store for the layout check (Breakthrough, Tue 2026-11-03; the orchestrator 18:0x): every layer a
symlink to records3's own file, except the layers BigMoE converted in its scratch (~/BigMoE/scratch/r3p11: L0 / L20 / L47
native+p11, 761cb3d's r3layout_ref.py), which link to those copies; index.json = records3's with those layers' entries
replaced by the scratch's. Zero bytes copied. dm_t3 reads it with DM_R3=<overlay> (and BIGMOE_EXT_BUILD=build_nat).
Usage (BigMoE venv): python mk_r3_overlay.py -> r3_overlay/"""
import json, os
R3 = os.path.expanduser("~/BigMoE/models/bare/records3"); SC = os.path.expanduser("~/BigMoE/scratch/r3p11")
OV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "r3_overlay")
ix, sx = json.load(open(os.path.join(R3, "index.json"))), json.load(open(os.path.join(SC, "index.json")))
assert all("layout" not in v for v in ix["layers"].values()), "records3 itself is no longer row-major"
os.makedirs(OV, exist_ok=True)
for L, e in ix["layers"].items():
    src = os.path.join(SC, sx["layers"][L]["file"]) if L in sx["layers"] else os.path.join(R3, e["file"])
    dst = os.path.join(OV, e["file"])
    if os.path.lexists(dst): os.remove(dst)
    os.symlink(src, dst)
    if L in sx["layers"]: ix["layers"][L] = sx["layers"][L]
json.dump(ix, open(os.path.join(OV, "index.json"), "w"))
print(f"r3_overlay: {len(ix['layers'])} layers, converted {sorted(sx['layers'], key=int)} -> {SC}")
