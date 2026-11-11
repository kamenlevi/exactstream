"""pack_check.py (Fable B): CPU check of gdn_w.py's packed weight stream and its column maps. numpy only, ~1 GB, no GPU.
It replays phase A's addressing (which word a lane reads, which fragment element it fills, where an output lands) in
float64 on random 4-bit weights at the real shapes and compares with the plain product W x. It does not test Metal."""
import importlib.util, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("gdn_w", os.path.join(HERE, "gdn_w.py"))
W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
rng = np.random.default_rng(6)
N, K, KG = 16480, 2560, 40
codes = rng.integers(0, 16, (N, K), dtype=np.uint32)
w = np.zeros((N, K // 8), np.uint32)
for j in range(8):
    w |= codes[:, j::8] << np.uint32(4 * j)                  # MLX layout: nibble j of word ws = code of K = 8 ws + j
s = rng.normal(0, 0.01, (N, KG)); b = rng.normal(0, 0.05, (N, KG))
x = rng.normal(0, 0.5, (16, K))
Wfull = (s[:, :, None] * codes.reshape(N, KG, 64) + b[:, :, None]).reshape(N, K)
ref = x @ Wfull.T                                            # [16, 16480]
wn, sn, bn = W.pack_gdn(w, s, b)
assert wn.shape == (16, 33, 40, 32, 8) and sn.shape == (16, 33, 40, 32)
FM, FQ = W.FM, W.FQ
qkv = np.full((16, 10240), np.nan); z = np.full((16, 6144), np.nan); ab = np.full((16, 96), np.nan)
hits = np.zeros(16480, int)
for hk in range(16):
    for tile in range(33):
        acc = np.zeros((16, 32))
        for g in range(40):
            C = np.zeros((16, 32)); A = x[:, g * 64:(g + 1) * 64]
            for st in range(4):
                Al = np.zeros((16, 16)); Bl = np.zeros((16, 32))
                for l in range(32):
                    fm, fq = int(FM[l]), int(FQ[l])
                    for j in range(4):
                        Al[fm, 4 * fq + j] = A[fm, 16 * fq + 4 * st + j]; Al[fm + 8, 4 * fq + j] = A[fm + 8, 16 * fq + 4 * st + j]
                    for p in range(2):
                        word = int(wn[hk, tile, g, l, st * 2 + p])
                        for j in range(4):
                            two = ((word >> (4 * j)) & 0x000F000F)
                            Bl[4 * fq + j, 16 * p + fm] = 128 + (two & 0xFFFF)
                            Bl[4 * fq + j, 16 * p + fm + 8] = 128 + (two >> 16)
                C += Al @ Bl
            xs = A.sum(1)
            acc += sn[hk, tile, g][None, :] * C + (bn[hk, tile, g] - 128.0 * sn[hk, tile, g])[None, :] * xs[:, None]
        for n in range(32):
            col = tile * 32 + n
            if tile < 32:
                if col < 256:
                    qkv[:, (0 if col < 128 else 2048 - 128) + hk * 128 + col] = acc[:, n]; hits[(0 if col < 128 else 1920) + hk * 128 + col] += 1
                elif col < 640:
                    qkv[:, 4096 + hk * 384 + col - 256] = acc[:, n]; hits[4096 + hk * 384 + col - 256] += 1
                else:
                    z[:, hk * 384 + col - 640] = acc[:, n]; hits[10240 + hk * 384 + col - 640] += 1
            elif n < 6:
                ab[:, (0 if n < 3 else 48) + hk * 3 + n % 3] = acc[:, n]; hits[16384 + (0 if n < 3 else 48) + hk * 3 + n % 3] += 1
    print("head %2d done" % hk, flush=True)
got = np.concatenate([qkv, z, ab], 1)
err = np.abs(got - ref).max() / np.abs(ref).max()
print("every output written exactly once: %s; max rel error vs W x (fp64): %.2e" % (bool((hits == 1).all()), err))
ok = bool((hits == 1).all()) and err < 1e-9
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
