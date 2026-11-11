"""R-48 M01-12 loader test of engine v3n (= v3m + BigMoE's M-01 unpack), a copy of ~/BigMoE/scripts/test_records3m.py
(BigMoE 6753d83 / 2e126dc) pointed at ~/Breakthrough/v3n and its ext/build. On the real files, for many experts of some
layers: (1) the 9 4-bit pieces from the safetensors (source bytes); (2) ext.read_rebuilt on records3m (tier-mode prefill:
full record, fields unpacked, rebuilt to 4 bits) must equal (1) bitwise; (3) the MSB-3 / LSB-1 parts of records3m through
set_records_part (the decode tier pools: fields unpacked into the pool pieces) must equal records3's part bytes.
--r3 (BIGMOE_M01 unset): the same checks on records3 through v3n with M-01 off (v3m's paths: (2) and (3) on records3).
Prints the load time per expert of the part pools (cold reads, F_NOCACHE). CPU / SSD only, no GPU work.
Usage: [BigMoE venv] BIGMOE_M01=1 test_v3n_m01.py [layers=0,23,47] [n_experts=96]
       [BigMoE venv] test_v3n_m01.py --r3 [layers] [n]"""
import fcntl, json, os, sys, time
import numpy as np
V3N = os.path.expanduser("~/Breakthrough/v3n")
sys.path.insert(0, V3N)
sys.path.insert(0, os.path.join(V3N, "ext", os.environ.get("BIGMOE_EXT_BUILD", "build")))
import mlx.core as mx
import bigmoe_ext as ext
from bigmoe_v3f.stream import read_headers, PIECES, DT

R3 = "--r3" in sys.argv
argv = [a for a in sys.argv[1:] if a != "--r3"]
assert R3 != bool(os.environ.get("BIGMOE_M01")), "set BIGMOE_M01=1, or pass --r3 with BIGMOE_M01 unset"
assert ext.__file__.startswith(V3N) and ext.m01_on() == (not R3), (ext.__file__, ext.m01_on())
print(f"ext {ext.__file__}, m01_on {ext.m01_on()}", flush=True)
MD = os.path.expanduser("~/BigMoE/models/bare")
LAYERS = [int(x) for x in (argv[0] if argv else "0,23,47").split(",")]
N = max(16, int(argv[1]) if len(argv) > 1 else 96)
ix3 = json.load(open(os.path.join(MD, "records3", "index.json")))["layers"]
ixm = json.load(open(os.path.join(MD, "records3m", "index.json")))["layers"]
hdr = read_headers(MD)
ok = True
for L in LAYERS:
    r3, rm = ix3[str(L)], ixm[str(L)]
    path3, pathm = os.path.join(MD, "records3", r3["file"]), os.path.join(MD, "records3m", rm["file"])
    ids = [int(e) for e in np.random.default_rng(100 + L).choice(r3["E"], N, replace=False)]
    rr, pp = (r3, path3) if R3 else (rm, pathm)
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
    ext.read_rebuilt(pp, rr["recsz"], rr["msb_sz"], pers, ids, list(range(N)), D)
    good = all(np.array_equal(np.array(d.view(mx.uint16)), a) for d, a in zip(D, ref)); ok &= good
    print(f"L{L}: read_rebuilt on {'records3' if R3 else 'records3m'} == safetensors 4-bit pieces ({N} experts): {good}",
          flush=True)
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
        with open(path3, "rb") as g:
            fcntl.fcntl(g.fileno(), 48, 1)   # F_NOCACHE: cached records3 pages would favour the A arm of a later A/B
            for e, s in zip(ids, sl):
                for a, n, o in zip(P, sizes, poff):
                    g.seek(e * r3["recsz"] + o)
                    good &= np.array(a[int(s)]).tobytes() == g.read(n)
        ok &= good
        print(f"L{L}: {name} part pool == records3 part bytes: {good}; load {dt_ms:.3f} ms / expert ({ln} B parts, "
              f"{'records3, M-01 off' if R3 else 'records3m'})", flush=True)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
