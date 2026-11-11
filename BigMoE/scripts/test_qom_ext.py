"""QOM CPU test (Wed 11-04): the loader's optional uses (bit 21 of a posted id, ext ensure()). No GPU kernels, no model
load: base pools on records3 layer 20 (models/bare), A = the layer under test, B = a plain twin for the bytes.
  (1) required misses are read; an optional-only miss is NOT read (slot -2, qom_skip +1, misses count only the required)
  (2) an optional use of a resident expert is served from the pool (its slot); bytes == B's
  (3) an expert with one optional and one required use is read (slot >= 0); bytes == B's
  (4) ids without bit 21: unchanged (qom counters still)
usage: .venv/bin/python scripts/test_qom_ext.py   (ext: TEST_EXT_BUILD, default build_qom)"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ext", os.environ.get("TEST_EXT_BUILD", "build_qom")))
sys.path.insert(0, ROOT)
import mlx.core as mx
import bigmoe_ext as ext
import bigmoe.tier as T

d, ix = T.load_index(os.path.join(ROOT, "models", "bare"))
r = ix["layers"]["20"]
SH4 = []
for o, i in ((640, 2560), (640, 2560), (2560, 640)):
    SH4 += [((o, i // 8), mx.uint32), ((o, i // 64), mx.bfloat16), ((o, i // 64), mx.bfloat16)]
E, C, OPT = 512, 8, 1 << 21
_, PA = T.make_pools(ext, 0, E, C, SH4, d, r)
_, PB = T.make_pools(ext, 1, E, C, SH4, d, r)
st = lambda: ext.stats()


def ens(L, ids):
    """ensure ids (ints, bit 21 allowed) on ext layer L; -> the slots ext.lookup gives for the plain ids"""
    ext.ensure(L, mx.array(np.array(ids, np.int32)))
    return ext.lookup(L, [x & ~OPT for x in ids])


def same(ids):
    ext.ensure(1, mx.array(np.array(ids, np.int32)))
    ha = [np.array(p.view(mx.uint8)).reshape(C, -1) for p in PA]
    hb = [np.array(p.view(mx.uint8)).reshape(C, -1) for p in PB]
    for x, sa, sb in zip(ids, ext.lookup(0, ids), ext.lookup(1, ids)):
        assert sa >= 0 and sb >= 0 and all(np.array_equal(a[sa], b[sb]) for a, b in zip(ha, hb)), (x, sa, sb)


s0 = st()
# (1) + (2): 10 and 11 required, 12 made resident first, then optional; 13 optional-only and missing
ens(0, [12])
m0 = st()["misses"]
sl = ens(0, [10, 11, 12 | OPT, 13 | OPT])
s1 = st()
assert sl[0] >= 0 and sl[1] >= 0 and sl[2] >= 0 and sl[3] < 0, sl
assert s1["qom_skip"] - s0["qom_skip"] == 1 and s1["qom_opt"] - s0["qom_opt"] == 2, (s1["qom_skip"], s1["qom_opt"])
assert s1["misses"] - m0 == 2, s1["misses"] - m0              # only 10 and 11 were read
same([10, 11, 12])
print(f"(1)+(2) required read, resident optional served (slot {sl[2]}), optional-only miss not read (lookup {sl[3]}); "
      f"bytes equal")
# (3) one optional and one required use of the same expert -> read
sl = ens(0, [20 | OPT, 20, 21])
assert sl[0] >= 0 and sl[2] >= 0, sl
same([20, 21])
print("(3) an optional + a required use of one expert: read; bytes equal")
# (4) no bit 21: no QOM counting
s3 = st()
ens(0, [30, 31])
s4 = st()
assert s4["qom_skip"] == s3["qom_skip"] and s4["qom_opt"] == s3["qom_opt"]
same([30, 31])
print("(4) plain ids: unchanged")
print(f"PASS: qom_skip {st()['qom_skip'] - s0['qom_skip']}, qom_opt {st()['qom_opt'] - s0['qom_opt']}")
