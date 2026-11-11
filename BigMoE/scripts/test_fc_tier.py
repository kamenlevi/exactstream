"""Lever 2 CPU test (Wed 11-04): forecast reads into the volatile tier (ext fc_issue, BIGMOE_FC_F / FS) give the same pool
bytes as plain demand loads. No GPU kernels, no model load: two ext layer pairs on the same records3 layer, A (base 0 +
sidecar 1) with a 6-slot tier and forecast reads, B (2 + 3) plain; every slot A fills after a forecast == B's slot for the
same expert, byte for byte, piece by piece.
  (1) landed: forecast 4 per part, wait, ensure them on A and B -> fc_hits +8, bytes equal
  (2) in flight: forecast 4, ensure at once -> the copy-backs wait for the reads (fc_late counts them), bytes equal
  (3) pressure: forecast 10 into a 6-slot tier -> at most 6 issued, none taken while being written; ensure 8 -> equal
  (4) skip: ids resident on A are not read again (fc_skip)
  (5) purge: landed + unpinned -> a purge drops the tier (vt_purged), new forecasts still land right; a purge while
      reads are in flight is refused (the slab stays pinned); bytes equal
usage: .venv/bin/python scripts/test_fc_tier.py [bare | mix]   (ext: TEST_EXT_BUILD, default build_l2; mix: the converted
       layer 20 of scratch/model_mix; run it with BIGMOE_RECORDS_DIRECT=1 too, for the direct and split read paths)"""
import os, sys
import numpy as np
os.environ.setdefault("BIGMOE_FC_F", "4")
os.environ.setdefault("BIGMOE_FC_FS", "4")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ext", os.environ.get("TEST_EXT_BUILD", "build_l2")))
sys.path.insert(0, ROOT)
import mlx.core as mx
import bigmoe_ext as ext
import bigmoe.tier as T

STORE = sys.argv[1] if len(sys.argv) > 1 else "bare"
if STORE == "mix":
    d, ix = T.load_index(os.path.join(ROOT, "scratch", "model_mix"), layout_ok=True)
else:
    d, ix = T.load_index(os.path.join(ROOT, "models", "bare"))
r = ix["layers"]["20"]
SH4 = []
for o, i in ((640, 2560), (640, 2560), (2560, 640)):
    SH4 += [((o, i // 8), mx.uint32), ((o, i // 64), mx.bfloat16), ((o, i // 64), mx.bfloat16)]
E, C, VT = 512, 8, 6
_, PA = T.make_pools(ext, 0, E, C, SH4, d, r)
LsA, _, PSA = T.make_sidecar(ext, E, C, SH4, d, r)
_, PB = T.make_pools(ext, 2, E, C, SH4, d, r)
LsB, _, PSB = T.make_sidecar(ext, E, C, SH4, d, r)
assert (LsA, LsB) == (1, 3), (LsA, LsB)
assert ext.set_vtier(0, VT) > 0 and ext.set_vtier(1, VT) > 0
POOLS = {0: PA, 1: PSA, 2: PB, 3: PSB}
st = lambda: ext.stats()
rng = np.random.default_rng(1104)
used = set()


def fresh(k):
    out = []
    while len(out) < k:
        x = int(rng.integers(E))
        if x not in used:
            used.add(x); out.append(x)
    return out


def same(LA, ids):
    """ensure ids on A's layer LA and on B's (LA + 2); every piece of every slot equal"""
    ext.ensure(LA, mx.array(np.array(ids, np.int32)))
    ext.ensure(LA + 2, mx.array(np.array(ids, np.int32)))
    ha = [np.array(p.view(mx.uint8)).reshape(C, -1) for p in POOLS[LA]]        # bf16 pieces: bytes via MLX (numpy
    hb = [np.array(p.view(mx.uint8)).reshape(C, -1) for p in POOLS[LA + 2]]    # has no bfloat16)
    for x, sa, sb in zip(ids, ext.lookup(LA, ids), ext.lookup(LA + 2, ids)):
        assert sa >= 0 and sb >= 0, (LA, x, sa, sb)
        for k, (a, b) in enumerate(zip(ha, hb)):
            assert np.array_equal(a[sa], b[sb]), f"layer {LA} expert {x} piece {k}: forecast-tier bytes != demand bytes"
    return len(ids)


s0, n_cmp = st(), 0
assert s0["fc_f"] == int(os.environ["BIGMOE_FC_F"]), s0["fc_f"]
# (1) landed
for LA in (0, 1):
    ids = fresh(4)
    assert ext.fc_test(LA, ids, 4) == 4
    ext.fc_wait()
    n_cmp += same(LA, ids)
s1 = st()
assert s1["fc_issued"] - s0["fc_issued"] == 8 and s1["fc_landed"] - s0["fc_landed"] == 8, s1
assert s1["fc_hits"] - s0["fc_hits"] == 8 and s1["fc_late"] == s0["fc_late"], s1
print(f"(1) landed: 8 forecast reads, 8 tier hits, bytes equal ({STORE})")
# (2) in flight: ensure right after issuing
late0 = s1["fc_late"]
for LA in (0, 1):
    ids = fresh(4)
    assert ext.fc_test(LA, ids, 4) == 4
    n_cmp += same(LA, ids)
ext.fc_wait()
s2 = st()
assert s2["fc_hits"] - s1["fc_hits"] == 8, s2
print(f"(2) in flight: 8 hits, {s2['fc_late'] - late0} of them still landing (waited {s2['fc_wait_s'] * 1e3:.2f} ms), "
      f"bytes equal")
# (3) pressure: more forecasts than tier slots, ensure while some are in flight
ids = fresh(10)
k = ext.fc_test(0, ids, 10)
assert 0 < k <= VT, k
n_cmp += same(0, ids[:C])
ext.fc_wait()
s3 = st()
print(f"(3) pressure: {k} of 10 issued into a {VT}-slot tier; tier drops {s3['vt_drops'] - s2['vt_drops']} "
      f"(forecasts never used {s3['fc_drops'] - s2['fc_drops']}); bytes equal")
# (4) skip: experts resident on A are not read again
res = ids[:3]
assert all(s >= 0 for s in ext.lookup(0, res))
sk0 = s3["fc_skip"]
assert ext.fc_test(0, res, 3) == 0
s4 = st()
assert s4["fc_skip"] - sk0 == 3, s4
print("(4) skip: 3 resident ids, 0 reads")
# (5) purge: landed + unpinned -> a purge drops the tier. (3)'s ensure ended while reads were in flight, so the slab stayed
# pinned (by design); one more call with nothing in flight unpins it.
pu0 = s4["vt_purged"]
ext.fc_wait()
ext.ensure(0, mx.array(np.array(res, np.int32)))
ext.vtier_purge(0)
ids = fresh(4)
assert ext.fc_test(0, ids, 4) == 4                             # the pin finds the slab purged: the index goes first
ext.fc_wait()
n_cmp += same(0, ids)
s5 = st()
assert s5["vt_purged"] - pu0 == 1, s5
ids = fresh(4)                                                 # a purge while reads are in flight: refused (pinned)
assert ext.fc_test(0, ids, 4) == 4
ext.vtier_purge(0)
ext.fc_wait()
n_cmp += same(0, ids)
s6 = st()
assert s6["vt_purged"] == s5["vt_purged"], s6
print(f"(5) purge: landed + unpinned -> purged and dropped ({s5['vt_purged'] - pu0}); in flight -> refused; bytes equal")
d = {k: s6[k] - s0[k] for k in ("fc_issued", "fc_landed", "fc_hits", "fc_late", "fc_drops", "fc_skip", "vt_hits")}
print(f"PASS ({STORE}, BIGMOE_RECORDS_DIRECT={os.environ.get('BIGMOE_RECORDS_DIRECT', '')}): {n_cmp} expert loads "
      f"compared, every piece equal; {d}")
