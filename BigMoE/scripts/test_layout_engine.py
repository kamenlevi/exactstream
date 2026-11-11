"""CPU test of the engine's records3 layout plumbing (bigmoe/tier.py + stream.py, docs/PLAN_records3_convert.md section 3)
on scratch/model_mix: a model directory of symlinks whose records3/ has layers 0 / 20 / 47 converted (native+p11, from
scripts/r3layout_ref.py) and the rest linked to models/bare/records3 (row-major). No GPU kernels, no model load.
  (1) load_index: a converted store raises unless layout_ok; an unknown layout raises; today's records3 loads as before.
  (2) pools: make_pools / make_sidecar on a converted layer allocate the packed s3 / b3 / b pieces (uint32 words of the
      index's sizes); 3 x 8 experts load through the ext and every slot == its record's part.
  (3) the prefill: _read_rebuilt(t) of 64 experts == ext.read_rebuilt on records3, on a converted layer and on a
      row-major one (the unchanged path).
  (4) the guards: tier_call raises on a converted layer without a layout router, and with 3b; M-01 / POM / mismatched
      pieces refuse.
usage: .venv/bin/python scripts/test_layout_engine.py [BIGMOE_PKG_DIR]   (default scratch/stage_layout: the staged patch)"""
import json, os, sys, tempfile
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, sys.argv[1] if len(sys.argv) > 1 else "scratch/stage_layout")
sys.path.insert(0, os.path.join(ROOT, "ext", os.environ.get("TEST_EXT_BUILD", "build_nat")))
sys.path.insert(0, PKG)
import mlx.core as mx
import bigmoe_ext as ext
import bigmoe.tier as T
assert os.path.realpath(T.__file__).startswith(os.path.realpath(PKG)), T.__file__

MIX = os.path.join(ROOT, "scratch", "model_mix")
BARE = os.path.join(ROOT, "models", "bare")
SH4 = []
for o, i in ((640, 2560), (640, 2560), (2560, 640)):
    SH4 += [((o, i // 8), mx.uint32), ((o, i // 64), mx.bfloat16), ((o, i // 64), mx.bfloat16)]
PERS4 = [int(np.prod(sh)) * (4 if dt == mx.uint32 else 2) for sh, dt in SH4]


def raises(f, *a, **kw):
    try:
        f(*a, **kw)
    except (RuntimeError, AssertionError) as e:
        return str(e)[:100] or type(e).__name__
    return None


# (1) load_index
msg = raises(T.load_index, MIX)
assert msg and "converted layers" in msg, msg
d, ix = T.load_index(MIX, layout_ok=True)
lay = {L: r.get("layout", "row-major") for L, r in ix["layers"].items()}
assert {L for L, v in lay.items() if v != "row-major"} == {"0", "20", "47"}, lay
d0, ix0 = T.load_index(BARE)                                  # today's store: no layout_ok needed
assert all("layout" not in r for r in ix0["layers"].values())
with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
    os.makedirs(os.path.join(tmp, "records3"))
    bad = json.loads(json.dumps(ix)); bad["layers"]["5"]["layout"] = "native+p12"
    json.dump(bad, open(os.path.join(tmp, "records3", "index.json"), "w"))
    msg2 = raises(T.load_index, tmp, layout_ok=True)
    assert msg2 and "unknown records3 layout" in msg2, msg2
print(f"(1) load_index: converted store refused without layout_ok ({msg[:60]}...); unknown layout refused; today's OK")

# (2) pools on a converted layer (ext layers 0 = base, 1 = sidecar)
r20 = ix["layers"]["20"]
msb, lsb = T._shapes(SH4, r20)
assert [T._nbytes(sh, dt) for sh, dt in msb] == list(r20["msb_piece_size"])
assert [T._nbytes(sh, dt) for sh, dt in lsb] == list(r20["lsb_piece_size"])
assert T._shapes(SH4) == T._shapes(SH4, ix0["layers"]["20"])   # row-major: today's shapes
for part in ("msb", "lsb"):                                     # today's records3 passes the same contiguity check
    T._check_part(ix0["layers"]["20"], list(ix0["layers"]["20"][f"{part}_piece_size"]), part)
base, P = T.make_pools(ext, 0, 512, 8, SH4, d, r20)
Ls, sideP, PS = T.make_sidecar(ext, 512, 8, SH4, d, r20)
assert Ls == 1 and P[1].dtype == mx.uint32 and P[1].shape == (8, r20["msb_piece_size"][1] // 4) and \
    PS[1].shape == (8, r20["lsb_piece_size"][1] // 4), (P[1].shape, PS[1].shape)   # packed (8804) or padded (9216) words
rng, n = np.random.default_rng(11), 0
with open(os.path.join(d, r20["file"]), "rb") as f:
    for _ in range(3):
        ids = rng.choice(512, 8, replace=False).astype(np.int32)
        for Lx, pools, offs, sizes in ((0, P, r20["msb_piece_off"], r20["msb_piece_size"]),
                                       (Ls, PS, r20["lsb_piece_off"], r20["lsb_piece_size"])):
            ext.ensure(Lx, mx.array(ids))
            host = [np.array(p).view(np.uint8).reshape(8, -1) for p in pools]
            for x, s in zip(ids.tolist(), ext.lookup(Lx, ids.tolist())):
                f.seek(x * r20["recsz"]); rec = np.frombuffer(f.read(r20["recsz"]), np.uint8)
                assert s >= 0 and all(np.array_equal(h[s], rec[o:o + z]) for h, o, z in zip(host, offs, sizes)), (Lx, x)
                n += 1
print(f"(2) pools: packed pieces {P[1].shape} uint32; {n} part loads, every slot == its record's part")

# (3) the prefill rebuild, converted and row-major layers, against today's reader on records3
def tdict(r, path):
    return dict(ext=ext, path=path, recsz=r["recsz"], msb_sz=r["msb_sz"], pers4=PERS4,
                layout=T.LAYOUTS[r.get("layout", "row-major")], rows3=[640, 640, 2560],
                offs=list(r.get("msb_piece_off", [])) + list(r.get("lsb_piece_off", [])))

for L in ("20", "5", "47"):
    r, r0 = ix["layers"][L], ix0["layers"][L]
    ids = np.random.default_rng(int(L)).choice(512, 64, replace=False).astype(np.int32).tolist()
    rows = list(range(64))[::-1]
    a = [mx.zeros((64, p), mx.uint8) for p in PERS4]; b = [mx.zeros((64, p), mx.uint8) for p in PERS4]
    mx.eval(*a, *b)
    T._read_rebuilt(tdict(r, os.path.join(d, r["file"])), ids, rows, a)
    ext.read_rebuilt(os.path.join(d0, r0["file"]), r0["recsz"], r0["msb_sz"], PERS4, ids, rows, b)
    assert all(np.array_equal(np.array(u), np.array(v)) for u, v in zip(a, b)), L
    print(f"(3) prefill L{L} ({r.get('layout', 'row-major')}): 64 experts' 4-bit pieces == records3's")

# (4) guards
bm = dict(L=20, tier=dict(layout=2))
m1 = raises(T.tier_call, None, None, None, bm, None)
T.LAYOUT_ROUTER = True
m2 = raises(T.tier_call, None, None, None, bm, None)            # TIER_GLU 3b (the default here) reads row-major only
T.LAYOUT_ROUTER = False
assert m1 and "layout router" in m1 and m2 and "layout router" in m2, (m1, m2)
class _M01Ext:                                                   # an ext that has the M-01 unpack (passes that check)
    m01_on = staticmethod(lambda: True)

os.environ["BIGMOE_M01"] = "1"
m3 = raises(T.make_pools, _M01Ext, 2, 512, 8, SH4, d, r20)
del os.environ["BIGMOE_M01"]
assert m3 and "row-major records only" in m3, m3
r_bad = json.loads(json.dumps(r20)); r_bad["msb_piece_size"][1] = 51200
m4 = raises(T._check_part, r_bad, [T._nbytes(sh, dt) for sh, dt in msb], "msb")
assert m4, "a piece size other than the index's must refuse"
print(f"(4) guards: tier_call refuses a converted layer without a router ({m1[:50]}...) and with 3b; M-01 and "
      f"mismatched pieces refuse")
print("PASS")
