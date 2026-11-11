"""Unit test of the records3 loader paths (DESIGN_3bit.md) on the real records3 files: load the same experts of one layer
(1) through the 9-piece safetensors path (the source bytes), (2) through BIGMOE_RECORDS=3 (records3 + CPU rebuild),
(3) the MSB-3 / LSB-1 parts through set_records_part (tier pools), (4) ext.read_rebuilt (tier-mode prefill), and
compare: (2) and (4) == (1) bitwise; (3) == the record bytes. Many experts, so the loader's GCD workers run (the first
version passed with 16 while a thread_local capture bug corrupted real runs).
Usage: test_records3.py [layer] [n_experts]   (CPU / SSD only, no GPU work)"""
import fcntl, json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("~/BigMoE/ext/" + os.environ.get("BIGMOE_EXT_BUILD", "build")))
import mlx.core as mx
import bigmoe_ext as ext
from bigmoe.stream import read_headers, PIECES, DT

MD = os.path.expanduser("~/BigMoE/models/bare")
L = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N = max(16, int(sys.argv[2]) if len(sys.argv) > 2 else 96)
r = json.load(open(os.path.join(MD, "records3", "index.json")))["layers"][str(L)]
assert r.get("layout", "row-major") == "row-major", "records3 is converted (docs/PLAN_records3_convert.md): this script reads row-major records only"
path3 = os.path.join(MD, "records3", r["file"])
hdr = read_headers(MD)
pre = f"language_model.model.layers.{L}.mlp.switch_mlp"
fds, offs, pers, shapes, dts = [], [], [], [], []
for proj, field in PIECES:
    p, off, dt, shape = hdr[f"{pre}.{proj}.{field}"]
    fd = os.open(p, os.O_RDONLY); fcntl.fcntl(fd, 48, 1)
    fds.append(fd); offs.append(off); shapes.append(tuple(shape[1:])); dts.append(DT[dt][1])
    pers.append(int(np.prod(shape[1:])) * np.dtype(DT[dt][0]).itemsize)
ids = [int(e) for e in np.random.default_rng(L).choice(r["E"], N, replace=False)]


def pools_for(fmt):
    P = [mx.zeros((N, *sh), dt) for sh, dt in zip(shapes, dts)]; mx.eval(*P)
    Lx = ext.add_layer(N, r["E"], fds, offs, pers, P)
    if fmt == 3:
        ext.set_records3(Lx, path3, r["recsz"], r["msb_sz"])
    sl = np.array(ext.ensure(Lx, mx.array(ids, dtype=mx.int32)))
    return [np.array(a.view(mx.uint16))[sl] for a in P]       # rows in ids order


ref = pools_for(0)
assert all(a.any() for a in ref), "pools not loaded"
ok = True
got = pools_for(3)
good = all(np.array_equal(a, b) for a, b in zip(ref, got)); ok &= good
print(f"BIGMOE_RECORDS=3 pools == safetensors pools ({N} experts, layer {L}):", good)
D = [mx.zeros((r["E"], *sh), dt) for sh, dt in zip(shapes, dts)]; mx.eval(*D)
ext.read_rebuilt(path3, r["recsz"], r["msb_sz"], pers, [int(e) for e in ids], [int(e) for e in ids], D)
good = all(np.array_equal(np.array(d.view(mx.uint16))[ids], a) for d, a in zip(D, ref)); ok &= good
print("read_rebuilt rows == safetensors pools:", good)
for name, off, ln, sizes in (("msb", 0, r["msb_sz"], r["msb_piece_size"]),
                             ("lsb", r["msb_sz"], r["recsz"] - r["msb_sz"], r["lsb_piece_size"])):
    P = [mx.zeros((N, n // 2), mx.uint16) for n in sizes]; mx.eval(*P)
    Lx = ext.add_layer(N, r["E"], [0] * len(P), [0] * len(P), list(sizes), P)
    ext.set_records_part(Lx, path3, r["recsz"], off, ln)
    sl = np.array(ext.ensure(Lx, mx.array(ids, dtype=mx.int32)))
    with open(path3, "rb") as g:
        for e, s in zip(ids, sl):
            g.seek(e * r["recsz"] + off); body = g.read(sum(sizes)); o = 0
            for a, n in zip(P, sizes):
                ok &= np.array(a[int(s)]).tobytes() == body[o:o + n]; o += n
    print(f"{name} part pool == record bytes:", ok)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
