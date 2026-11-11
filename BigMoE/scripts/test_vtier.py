"""Unit test of the ext's BIGMOE_VTIER victim tier (ext/build_vt), no model: one layer of E = 16 experts whose records hold
known bytes, a pool of C = 4 slots and a tier of 3, driven through ext.ensure() like the loader does. Checks after every
call that each resident expert's slot holds exactly its record (so tier hits copy back the right bytes), and the counters:
an eviction goes into the tier when its slot is reused, a later miss on it is a tier hit (no SSD read), the tier's own LRU
drops its oldest when full. Run: .venv/bin/python scripts/test_vtier.py [BUILD=build_vt]"""
import os, sys, tempfile
import numpy as np
BUILD = sys.argv[1] if len(sys.argv) > 1 else "build_vt"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ext", BUILD))
import mlx.core as mx
import bigmoe_ext as ext

E, C, T = 16, 4, 3
PER = [16384, 32768, 16384]                            # three pieces, 16 KB-aligned like records3's parts
REC = sum(PER)
rng = np.random.default_rng(7)
recs = rng.integers(0, 256, size=(E, REC), dtype=np.uint8)
d = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
path = os.path.join(d, "records.bin"); recs.tofile(path)
pools = [mx.zeros((C, p), dtype=mx.uint8) for p in PER]
mx.eval(pools)
L = ext.add_layer(C, E, [-1] * 3, [0] * 3, PER, pools)
ext.set_records_part(L, path, REC, 0, REC)
nb = ext.set_vtier(L, T)
assert nb == T * REC, nb
ext.reset_stats()

def check(tag):
    tab = ext.lookup(L, list(range(E)))
    host = [np.array(p) for p in pools]
    for e, s in enumerate(tab):
        if s < 0:
            continue
        got = np.concatenate([h[s] for h in host])
        assert np.array_equal(got, recs[e]), f"{tag}: expert {e} in slot {s} holds the wrong bytes"
    return {e: s for e, s in enumerate(tab) if s >= 0}

def call(ids, tag):
    ext.ensure(L, mx.array(ids, dtype=mx.int32))
    res = check(tag); st = ext.stats()
    print(f"{tag:34s} resident {sorted(res)}  misses {st['misses']:2d}  tier hits {st['vt_hits']}  ins {st['vt_ins']}  "
          f"drops {st['vt_drops']}  purged {st['vt_purged']}")
    return st

st = call([0, 1, 2, 3], "fill 0-3")
assert st["misses"] == 4 and st["vt_hits"] == 0
st = call([4], "4 evicts 0 (ghost -> tier)")
assert st["vt_ins"] == 1
st = call([0], "0 = tier hit, evicts 1")
assert st["vt_hits"] == 1 and st["vt_ins"] == 2
st = call([5, 6], "5, 6 evict 2, 3 (tier 1, 2, 3)")
assert st["vt_ins"] == 4 and st["vt_drops"] == 0, st      # 0's hit freed its tier slot: 1, 2, 3 fit
st = call([1], "1 = tier hit; 4's copy-in drops 2")
assert st["vt_hits"] == 2 and st["vt_drops"] == 1, st     # 1's own tier slot is busy (copied back now): 2 goes
st = call([2, 3], "2 dropped = read, 3 = tier hit")
assert st["vt_hits"] == 3, st
for i in range(40):                                     # a long random walk: every resident slot must stay right
    call([int(x) for x in rng.choice(E, size=rng.integers(1, 4), replace=False)], f"random {i}")
tab = ext.lookup(L, list(range(E))); res = {e for e, s_ in enumerate(tab) if s_ >= 0}
st0 = ext.stats(); ext.vtier_purge(L)                 # macOS empties the slab: the tier's copies are gone
miss = [e for e in range(E) if e not in res][:2]
st = call(miss, "after a purge: no tier hits")
assert st["vt_purged"] == st0["vt_purged"] + 1 and st["vt_hits"] == st0["vt_hits"], (st0["vt_hits"], st["vt_hits"])
for i in range(10):
    call([int(x) for x in rng.choice(E, size=rng.integers(1, 4), replace=False)], f"random after purge {i}")
st = ext.stats()
assert sum(st["vt_hits_layer"]) == st["vt_hits"] and sum(st["vt_ins_layer"]) == st["vt_ins"], "per-layer counts"
print(f"PASS: {st['misses']} misses, {st['vt_hits']} tier hits, {st['vt_ins']} copies in, {st['vt_drops']} drops; "
      f"tier {st['vt_bytes'] / 1e6:.2f} MB, filled {st['vt_used_bytes'] / 1e6:.2f} MB")
