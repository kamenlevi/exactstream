"""Unit test of the M-01 loader paths (R-44 a; run with BIGMOE_M01=1 and an ext built with them, e.g.
BIGMOE_EXT_BUILD=build_m01) on the real records3m files, as test_records3.py does for records3: for many experts of some
layers, (1) the 9 4-bit pieces from the safetensors (source bytes), (2) ext.read_rebuilt on records3m (tier-mode prefill:
full record, fields unpacked, rebuilt to 4 bits) must equal (1) bitwise; (3) the MSB-3 / LSB-1 parts of records3m through
set_records_part (tier pools: fields unpacked into the pool pieces) must equal records3's part bytes. Also prints the load
time per expert of the part pools (cold reads, F_NOCACHE), to compare with BIGMOE_M01 unset on records3 (--r3).
Usage: BIGMOE_M01=1 BIGMOE_EXT_BUILD=build_m01 test_records3m.py [layers=0,23,47] [n_experts=96]
       BIGMOE_EXT_BUILD=build_m01 test_records3m.py --r3 [layers] [n]   (timing only, records3 parts, M-01 off)
(CPU / SSD only, no GPU work)"""
import fcntl, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("~/BigMoE/ext/" + os.environ.get("BIGMOE_EXT_BUILD", "build")))
import mlx.core as mx
import bigmoe_ext as ext
from bigmoe.stream import read_headers, PIECES, DT

R3 = "--r3" in sys.argv
argv = [a for a in sys.argv[1:] if a != "--r3"]
assert R3 or os.environ.get("BIGMOE_M01"), "set BIGMOE_M01=1 (or pass --r3 for the records3 timing)"
MD = os.path.expanduser("~/BigMoE/models/bare")
LAYERS = [int(x) for x in (argv[0] if argv else "0,23,47").split(",")]
N = max(16, int(argv[1]) if len(argv) > 1 else 96)
ix3 = json.load(open(os.path.join(MD, "records3", "index.json")))["layers"]
assert all(r.get("layout", "row-major") == "row-major" for r in ix3.values()), "records3 is converted (docs/PLAN_records3_convert.md): this script reads row-major records only"
ixm = json.load(open(os.path.join(MD, "records3m", "index.json")))["layers"]
hdr = read_headers(MD)
ok = True
for L in LAYERS:
    r3, rm = ix3[str(L)], ixm[str(L)]
    path3, pathm = os.path.join(MD, "records3", r3["file"]), os.path.join(MD, "records3m", rm["file"])
    ids = [int(e) for e in np.random.default_rng(100 + L).choice(r3["E"], N, replace=False)]
    if R3:
        rr, pp = r3, path3
    else:
        rr, pp = rm, pathm
        pre = f"language_model.model.layers.{L}.mlp.switch_mlp"
        fds, offs, pers, shapes, dts = [], [], [], [], []
        for proj, field in PIECES:
            p, off, dt, shape = hdr[f"{pre}.{proj}.{field}"]
            fd = os.open(p, os.O_RDONLY); fcntl.fcntl(fd, 48, 1)
            fds.append(fd); offs.append(off); shapes.append(tuple(shape[1:])); dts.append(DT[dt][1])
            pers.append(int(np.prod(shape[1:])) * np.dtype(DT[dt][0]).itemsize)
        P = [mx.zeros((N, *sh), dt) for sh, dt in zip(shapes, dts)]; mx.eval(*P)
        Lx = ext.add_layer(N, r3["E"], fds, offs, pers, P)
        sl = np.array(ext.ensure(Lx, mx.array(ids, dtype=mx.int32)))
        ref = [np.array(a.view(mx.uint16))[sl] for a in P]
        assert all(a.any() for a in ref), "pools not loaded"
        D = [mx.zeros((N, *sh), dt) for sh, dt in zip(shapes, dts)]; mx.eval(*D)
        ext.read_rebuilt(pathm, rm["recsz"], rm["msb_sz"], pers, ids, list(range(N)), D)
        good = all(np.array_equal(np.array(d.view(mx.uint16)), a) for d, a in zip(D, ref)); ok &= good
        print(f"L{L}: read_rebuilt on records3m == safetensors 4-bit pieces ({N} experts): {good}", flush=True)
    for name, off, ln, sizes, poff in (("msb", 0, rr["msb_sz"], r3["msb_piece_size"], r3["msb_piece_off"]),
                                       ("lsb", rr["msb_sz"], rr["recsz"] - rr["msb_sz"], r3["lsb_piece_size"],
                                        r3["lsb_piece_off"])):
        P = [mx.zeros((N, n // 2), mx.uint16) for n in sizes]; mx.eval(*P)
        Lx = ext.add_layer(N, r3["E"], [0] * len(P), [0] * len(P), list(sizes), P)
        ext.set_records_part(Lx, pp, rr["recsz"], off, ln)
        t = time.perf_counter()
        sl = np.array(ext.ensure(Lx, mx.array(ids, dtype=mx.int32)))
        dt_ms = 1e3 * (time.perf_counter() - t) / N
        good = True
        if not R3:
            with open(path3, "rb") as g:
                fcntl.fcntl(g.fileno(), 48, 1)   # F_NOCACHE: cached records3 pages would favour the A arm of a later A/B
                for e, s in zip(ids, sl):
                    for a, n, o in zip(P, sizes, poff):
                        g.seek(e * r3["recsz"] + o)
                        good &= np.array(a[int(s)]).tobytes() == g.read(n)
            ok &= good
        print(f"L{L}: {name} part pool{'' if R3 else ' == records3 part bytes: ' + str(good)}; load "
              f"{dt_ms:.3f} ms / expert ({ln} B parts, {'records3, M-01 off' if R3 else 'records3m'})", flush=True)
if not R3:
    print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
