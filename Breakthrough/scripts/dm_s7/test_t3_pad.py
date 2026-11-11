"""dm_t3 on BigMoE's padded converted layout "native+p11+pad" (Breakthrough, Wed 2026-11-04 19:0x; Leap's 18:5x records3
readers triage: dm_t3's LAYOUT lacks the name, so after Thu's conversion (~10:55) every t3 arm would raise).
Through the PATCHED dm_t3 (T3_MODULE), BigMoE's padded scratch copy (~/BigMoE/scratch/r3p11pad: L0 / L20 / L47
"native+p11+pad", 11-03 22:52) must give the same pieces as today's row-major records3, bit for bit: 64 random experts
per layer plus the first and last ids, MSB only and MSB + LSB. Regression: the unpadded native+p11 copy (scratch/r3p11)
the same way. Names: "native+p11+pad" -> 2, as bigmoe.tier.LAYOUTS; the others unchanged; unknown names refuse.
Negative control: the INSTALLED dm_t3 refuses the padded copy (skipped once it has the name). CPU only; BigMoE's
layout-aware ext (ext/build_nat, read-only import).
Usage (BigMoE venv): T3_MODULE=<patched dm_t3.py> python test_t3_pad.py"""
import importlib.util, json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/BigMoE/ext/build_nat"))
import mlx.core as mx
mx.set_default_device(mx.cpu)
import bigmoe_ext
DM = os.path.expanduser("~/FableGod/decision_map")
NEW, OLD = os.environ["T3_MODULE"], os.path.join(DM, "dm_t3.py")
R3 = os.path.expanduser("~/BigMoE/models/bare/records3")
PAD, P11 = os.path.expanduser("~/BigMoE/scratch/r3p11pad"), os.path.expanduser("~/BigMoE/scratch/r3p11")
bad = []
def ok(c, m):
    print(("  ok   " if c else "  FAIL ") + m, flush=True)
    if not c: bad.append(m)
def load(path, r3):
    os.environ["DM_R3"] = r3
    spec = importlib.util.spec_from_file_location(f"dm_t3_{abs(hash((path, r3)))}", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
u16 = lambda x: np.array(x.view(mx.uint16) if x.dtype == mx.bfloat16 else x)
same = lambda a, b: a.keys() == b.keys() and all(a[p].keys() == b[p].keys() and np.array_equal(u16(a[p][f]), u16(b[p][f]))
                                                 for p in a for f in a[p])
print(f"[t] patched dm_t3: {NEW}; ext {bigmoe_ext.__file__.split('/BigMoE/')[-1]} (layout_max {bigmoe_ext.layout_max()})", flush=True)
A = load(NEW, R3)
for name, want in (("native+p11+pad", 2), ("native+p11", 2), ("native", 1), ("row-major", 0), ("rowmajor", 0),
                   ({"order": "native", "scales": "p11"}, 2), ({"order": "native"}, 1), ({"order": "rowmajor", "scales": "bf16"}, 0),
                   (None, 0), ({"scales": "p11"}, None), ("weird", None), ("native+p11+pad2", None)):
    try: got = A._layout(0, {} if name is None else {"layout": name})
    except RuntimeError: got = None
    ok(got == want, f"layout {name!r} -> {got} (want {want})")
for D in (PAD, P11):
    B = load(NEW, D); ix = json.load(open(os.path.join(D, "index.json")))["layers"]
    for L in sorted(ix, key=int):
        rng = np.random.default_rng(int(L) + 7)
        ids = np.unique(np.concatenate([rng.choice(512, 64, replace=False), [0, 1, 510, 511]]))
        for lsb in (False, True):
            a, b = A.read(int(L), ids, lsb), B.read(int(L), ids, lsb)
            ok(same(a, b), f"{os.path.basename(D)} L{L} ({ix[L].get('layout')}, recsz {ix[L]['recsz']}) "
                           f"{'MSB + LSB' if lsb else 'MSB'}: {len(ids)} experts == records3 bit for bit")
if '"native+p11+pad"' in open(OLD).read():
    print("  skip negative control: the installed dm_t3 already has the name", flush=True)
else:
    C = load(OLD, PAD)
    try: C.read(0, np.array([0, 1]), False); msg = "read it"
    except RuntimeError as e: msg = str(e)
    ok("unknown layout" in msg, f"the installed dm_t3 refuses the padded copy (Leap's blocker): {msg[:90]}")
print("ALL OK" if not bad else f"FAILED: {len(bad)}", flush=True)
sys.exit(1 if bad else 0)
