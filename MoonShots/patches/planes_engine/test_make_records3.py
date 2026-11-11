"""Unit test of make_records3.py on a synthetic records/ layer (40 experts of random words; scales of both signs with a
few zeros; optional subnormal scales to exercise the halve_bf16 note): convert + verify must pass, and a corrupted
records3 byte must fail. usage: test_make_records3.py SCRATCH_DIR   (CPU only, ~120 MB of scratch)"""
import json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_records3 as M

D = sys.argv[1]
src, out = os.path.join(D, "src"), os.path.join(D, "r3")
os.makedirs(src, exist_ok=True)
rng = np.random.default_rng(3)
E = 40
po, ps, o = [], [], 0
for of, K in M.SHAPES:
    for n in (of * K // 2, of * K // M.G * 2, of * K // M.G * 2):
        po.append(o); ps.append(n); o += n
recsz = M.pad(o)
recs = np.zeros((E, recsz), np.uint8)
for e in range(E):
    for p in range(3):
        w = rng.integers(0, 2 ** 32, ps[3 * p] // 4, dtype=np.uint64).astype(np.uint32)
        s = (rng.standard_normal(ps[3 * p + 1] // 2) * 2e-3).astype(np.float32)
        s[rng.integers(0, s.size, 5)] = 0
        b = (rng.standard_normal(ps[3 * p + 2] // 2) * 2e-2).astype(np.float32)
        for k, a in enumerate((w.view(np.uint8), M.f32_to_bf16(s).view(np.uint8), M.f32_to_bf16(b).view(np.uint8))):
            recs[e, po[3 * p + k]:po[3 * p + k] + a.size] = a
recs.tofile(os.path.join(src, "L00.bin"))
json.dump({"align": 16384, "layers": {"0": {"file": "L00.bin", "E": E, "recsz": recsz, "piece_off": po, "piece_size": ps}}},
          open(os.path.join(src, "index.json"), "w"))
cmd = [sys.executable, os.path.join(HERE, "make_records3.py"), "--layers", "0", "--out", out, "--src", src, "--chunk", "16",
       "--min-free-gb", "0"]
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout, r.stderr)
assert r.returncode == 0, "conversion of the synthetic layer failed"
# a flipped bit anywhere (w3, an LSB bit, s3, b3, b, padding) must fail the verify-only pass
lay = M.layout()
for name, off in (("w3", 12345), ("s3", lay["msb_piece_off"][1] + 7), ("b3", lay["msb_piece_off"][5] + 3),
                  ("l1", lay["lsb_piece_off"][2] + 99), ("b", lay["lsb_piece_off"][5] + 1), ("pad", lay["msb_sz"] - 5)):
    path = os.path.join(out, "L00.bin")
    with open(path, "r+b") as f:
        f.seek(17 * lay["recsz"] + off); c = f.read(1); f.seek(-1, 1); f.write(bytes([c[0] ^ 4]))
    r = subprocess.run(cmd + ["--verify-only", "--no-mlx"], capture_output=True, text=True)
    print(f"corrupted {name}: exit {r.returncode}:", [l for l in r.stdout.splitlines() if l.startswith("L00")][0][:160])
    assert r.returncode != 0, f"corruption in {name} not detected"
    with open(path, "r+b") as f:
        f.seek(17 * lay["recsz"] + off); f.write(c)
r = subprocess.run(cmd + ["--verify-only", "--no-mlx"], capture_output=True, text=True)
assert r.returncode == 0, r.stdout
print("test_make_records3: PASS")
