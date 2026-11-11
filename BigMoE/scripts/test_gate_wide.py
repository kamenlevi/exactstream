"""R-55 GATE16 smoke test (ext only, no model, seconds): one synthetic layer (E 512 experts of 16 KB in a temp file whose
words encode (expert, word), C 192 slots) behind the sync-free spin gate with BIGMOE_MAXK=160. Each pass posts 16 rows x
top-10 = 160 ids (every third pass 4 rows = 40 ids) drawn from a working set that moves, so most passes have misses, and
gathers the pool rows at the returned slots on the GPU: every row must hold its expert's bytes. Any timeout fails.
usage: BIGMOE_MAXK=160 BIGMOE_EXT_BUILD=build_next .venv/bin/python scripts/test_gate_wide.py [passes=300]"""
import os, sys, tempfile, time
import numpy as np
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import mlx.core as mx
import bigmoe.gated as G

E, C, W = 512, 192, 4096                                  # experts, slots, int32 words per expert (16 KB)
NP = int(sys.argv[1]) if len(sys.argv) > 1 else 300
data = (np.arange(E, dtype=np.int64)[:, None] * 65536 + np.arange(W)[None, :]).astype(np.int32)
fd_dir = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
path = os.path.join(fd_dir, "experts.bin")
data.tofile(path)
fd = os.open(path, os.O_RDONLY)
gate = G.Gate(1)
pool = mx.zeros((C, W), mx.int32); mx.eval(pool)
assert G.ext.add_layer(C, E, [fd], [0], [W * 4], [pool]) == 0
print(f"gate MAXK {G.MAXK}, XLINES {G.XLINES}, ROW {G.ROW} ints, ext {G._BUILD}; {NP} passes", flush=True)
rng = np.random.default_rng(5)
bad, n_ids, t0 = 0, 0, time.perf_counter()
work = rng.choice(E, size=96, replace=False)
for p in range(NP):
    gate.new_pass()
    rows = min(16, G.MAXK // 10) if p % 3 else 4                                  # 16 rows at MAXK 160
    work = np.where(rng.random(96) < 0.3, rng.choice(E, size=96), work)          # ~30 % of the working set moves
    ids = np.stack([rng.choice(np.unique(work), 10, replace=False) for _ in range(rows)]).astype(np.int32)
    slots = gate.route_and_gate(0, mx.array(ids))
    got = np.array(mx.take(pool, slots.reshape(-1), axis=0))
    ok = np.array_equal(got, data[ids.reshape(-1)])
    bad += not ok
    n_ids += ids.size
    if not ok and bad <= 3:
        wrong = np.nonzero((got != data[ids.reshape(-1)]).any(axis=1))[0]
        print(f"  pass {p}: {len(wrong)} of {ids.size} rows wrong (first at id index {wrong[:5]})", flush=True)
gate.check()
s = G.ext.stats()
print(f"done in {time.perf_counter() - t0:.1f} s: {NP} passes, {n_ids} ids, wrong passes {bad}; misses {s['misses']}, "
      f"loader-served gates {s['gated']}, fast hits {gate.fast_hits()}", flush=True)
os.close(fd); os.remove(path); os.rmdir(fd_dir)
print("PASS" if bad == 0 else "FAIL")
sys.exit(1 if bad else 0)
