"""dm_t3 on converted records3 layers (the approved in-place conversion; BigMoE 761cb3d): BigMoE's scratch copies
(~/BigMoE/scratch/r3p11 = L0 / L20 / L47 native+p11, r3nat = L20 native; written by its r3layout_ref.py and checked there
== records3) read through dm_t3 must give the same pieces as records3 itself, bitwise: 64 random experts per layer, MSB
only and MSB + LSB. Also: the layout names (string and dict forms) map as the plan says, unknown ones refuse. CPU only.
Usage (BigMoE venv): python test_t3_layout.py   (T3_MODULE=<path> tests another copy of dm_t3.py before it is installed)"""
import importlib.util, json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/BigMoE/ext/build_nat"))      # BigMoE's layout-aware ext (read-only use)
import mlx.core as mx
import bigmoe_ext
HERE = os.path.dirname(os.path.abspath(__file__))
def load(r3):
    os.environ["DM_R3"] = r3
    spec = importlib.util.spec_from_file_location("dm_t3_" + str(abs(hash(r3))), os.environ.get("T3_MODULE", os.path.join(HERE, "dm_t3.py")))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
ok = True
A = load(os.path.expanduser("~/BigMoE/models/bare/records3"))
for name, want in (({"order": "native", "scales": "p11"}, 2), ({"order": "native"}, 1), ("native+p11", 2), ("native", 1),
                   ({"order": "rowmajor", "scales": "bf16"}, 0), (None, 0), ({"scales": "p11"}, None), ("weird", None)):
    try: got = A._layout(0, {} if name is None else {"layout": name})
    except RuntimeError: got = None
    ok &= got == want; print(f"layout {name!r} -> {got} (want {want})")
for D in ("~/BigMoE/scratch/r3p11", "~/BigMoE/scratch/r3nat"):
    D = os.path.expanduser(D); B = load(D)
    for L in sorted(json.load(open(os.path.join(D, "index.json")))["layers"], key=int):
        ids = np.sort(np.random.default_rng(int(L) + 7).choice(512, 64, replace=False))
        for lsb in (False, True):
            a, b = A.read(int(L), ids, lsb), B.read(int(L), ids, lsb)
            same = all(np.array_equal(np.array(a[p][f].view(mx.uint16) if a[p][f].dtype == mx.bfloat16 else a[p][f]),
                                      np.array(b[p][f].view(mx.uint16) if b[p][f].dtype == mx.bfloat16 else b[p][f]))
                       for p in a for f in a[p])
            ok &= same; print(f"{os.path.basename(D)} L{L} lsb {lsb}: {len(ids)} experts, every piece equal: {same}")
OV = os.path.join(HERE, "r3_overlay")                 # mk_r3_overlay.py's store (the Wed layout check), if built
if os.path.exists(OV):
    B = load(OV)
    for L in (0, 1, 20, 33, 47):
        ids = np.sort(np.random.default_rng(L + 11).choice(512, 32, replace=False))
        a, b = A.read(L, ids, True), B.read(L, ids, True)
        same = all(np.array_equal(np.array(a[p][f].view(mx.uint16) if a[p][f].dtype == mx.bfloat16 else a[p][f]),
                                  np.array(b[p][f].view(mx.uint16) if b[p][f].dtype == mx.bfloat16 else b[p][f])) for p in a for f in a[p])
        ok &= same; print(f"r3_overlay L{L} ({B._layout(L, B._index()[str(L)])}): 32 experts, every piece equal: {same}")
print("ALL OK" if ok else "FAILED")
