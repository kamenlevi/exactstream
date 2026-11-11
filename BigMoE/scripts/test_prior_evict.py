"""BIGMOE_EVICT_PRIOR smoke test (ext only, no model, seconds): one synthetic layer (E 512 experts of 64 B in a temp file
whose words encode (expert, word), C 8 slots) through the synchronous ensure() path. Experts 0-3 are used once, then 12
calls of 4 other experts each; then 0-3 again. Plain LRU must miss all 4 again; with BIGMOE_EVICT_PRIOR=30 and set_prior
giving 0-3 popularity 1 they must stay (0 misses). Every call's slots must hold its experts' bytes.
usage: BIGMOE_EXT_BUILD=build_prior [BIGMOE_EVICT_PRIOR=30] .venv/bin/python scripts/test_prior_evict.py"""
import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import mlx.core as mx
import bigmoe.gated as G

E, C, W = 512, 8, 16
data = (np.arange(E, dtype=np.int64)[:, None] * 65536 + np.arange(W)[None, :]).astype(np.int32)
path = os.path.join(tempfile.mkdtemp(dir=os.environ.get("TMPDIR")), "experts.bin")
data.tofile(path)
fd = os.open(path, os.O_RDONLY)
pool = mx.zeros((C, W), mx.int32); mx.eval(pool)
assert G.ext.add_layer(C, E, [fd], [0], [W * 4], [pool]) == 0
b = float(os.environ.get("BIGMOE_EVICT_PRIOR", "0") or 0)
if b:
    pr = np.zeros(E, np.float32); pr[:4] = 1.0
    G.ext.set_prior(0, pr.tolist())
bad = 0


def call(ids):
    global bad
    s0 = G.ext.stats()["misses"]
    slots = np.array(G.ext.ensure(0, mx.array(np.array(ids, np.int32))))
    got = np.array(mx.take(pool, mx.array(slots.astype(np.int32)), axis=0))
    bad += not np.array_equal(got, data[ids])
    return G.ext.stats()["misses"] - s0


call([0, 1, 2, 3])
for k in range(12):
    call([10 + 4 * k + j for j in range(4)])
m = call([0, 1, 2, 3])
want = 0 if b else 4
print(f"ext {G._BUILD}, EVICT_PRIOR {b}: re-use of the popular four -> {m} misses (want {want}); bytes wrong in {bad} calls")
print("PASS" if m == want and bad == 0 else "FAIL")
