"""expread_bench.py (Leap, Mon 16 Nov 2026 14:1x; CHECK25, NewIdeas 13:4x: "a no-math read kernel over the same 44
experts' bytes, one launch. >= 250 GB/s -> kernel structure; ~140 -> access pattern"): where the expert one-call's
1.5-1.6x over its byte floor comes from.
Same setup as exp32_bench.py: layer 20's converted pools (BigMoE's loader, read-only; all 512 experts resident), the
decision map's code windows, 32-row blocks of real 1c routing (keep = 4-bit, keep3 = MSB-3). Per block, the bytes the
one-call reads: every kept expert's 9 MSB pieces (3 x w3, 3 x s3, 3 x b3) at its base slot, plus the 6 LSB pieces (3 x l1,
3 x b) of every expert with a rank 0-1 use at 4-bit, at its sidecar slot.
Kernels (one launch per block; no math: uint4 loads XOR-folded, one word per simdgroup written so no load is dead):
  real     those pieces at their real slots, split into chunks of CH bytes (one threadgroup of TPG threads per chunk)
  contig   the same byte count from ONE pool array's consecutive slots (the streaming best case on these buffers)
Timed as exp32: NB blocks' launches independent in one graph per round, 7 rounds, median of 6 -> us per call, GB/s, x the
280 GB/s floor. Variants: CH 16 / 64 / 256 KB, TPG 256 / 1024.
usage: [BigMoE venv] expread_bench.py   (MEM ~2 GB; quiet lock for clean numbers)"""
import json
import os
import sys
import time

os.environ["BIGMOE_MAXK"] = "320"
import numpy as np
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import mlx.core as mx
import bigmoe.gated as BG
import bigmoe.tier as T

OUT = os.path.expanduser("~/Leap/data/week98/check25/expread_bench.txt")
L_ = []


def say(s):
    print(s, flush=True)
    L_.append(s)


ext = BG.ext
PLAN, R4, LM = "4433333333", 2, 20
T.PLAN = PLAN
BARE = os.path.expanduser("~/BigMoE/models/bare")
SH4 = []
for o, i in ((640, 2560), (640, 2560), (2560, 640)):
    SH4 += [((o, i // 8), mx.uint32), ((o, i // 64), mx.bfloat16), ((o, i // 64), mx.bfloat16)]
d_, ix = T.load_index(BARE, layout_ok=True)
r = ix["layers"][str(LM)]
base, PB = T.make_pools(ext, 0, 512, 512, SH4, d_, r)
Ls, side, PS = T.make_sidecar(ext, 512, 512, SH4, d_, r)
allids = np.arange(512, dtype=np.int32)
for j in range(0, 512, 64):
    ext.ensure(0, mx.array(allids[j:j + 64])); ext.ensure(Ls, mx.array(allids[j:j + 64]))
sm_of = np.array(ext.lookup(0, allids.tolist()), np.int32)
sl_of = np.array(ext.lookup(Ls, allids.tolist()), np.int32)
assert (sm_of >= 0).all() and (sl_of >= 0).all()
ARR = list(PB) + list(PS)                                # 9 MSB piece arrays, then 6 LSB piece arrays (slot-major)
for a in ARR:
    assert a.dtype == mx.uint32, a.dtype
WPS = [int(np.prod(a.shape[1:])) for a in ARR]           # 32-bit words per slot of each piece array
assert all(w % 4 == 0 for w in WPS), WPS
MSB_B = 4 * sum(WPS[:9])
LSB_B = 4 * sum(WPS[9:])
say(f"\nEXPREAD BENCH {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; layer {LM}; pieces per expert MSB "
    f"{MSB_B / 1e6:.4f} MB ({len(PB)} arrays) + LSB {LSB_B / 1e6:.4f} MB ({len(PS)} arrays); words per slot {WPS}")

# real 1c blocks (as exp32_bench): kept experts + the 4-bit rank 0-1 experts per 32-row block
MAP = os.path.expanduser("~/FableGod/decision_map/out")
basej = json.load(open(os.path.join(MAP, "baseline.json")))
code = [f for f in sorted(os.listdir(os.path.join(MAP, "xcap"))) if basej[f]["dom"] == "code"]
BLK = []
for f in code:
    z = np.load(os.path.join(MAP, "xcap", f))
    tz = np.load(os.path.join(MAP, "bt", "tiers_1c", f))
    a0 = int(z["a"])
    n = z["ids"].shape[0]
    for b0 in range(0, n - 31, 32):
        ids = z["ids"][b0:b0 + 32, LM].astype(np.int64)
        k4 = tz["keep"][a0 + b0:a0 + b0 + 32, LM].astype(bool)
        k3 = tz["keep3"][a0 + b0:a0 + b0 + 32, LM].astype(bool)
        kept = k4 | k3
        ex = np.unique(ids[kept]) if kept.any() else np.unique(ids[:, :1])
        lsb = np.unique(ids[:, :R4][k4[:, :R4]])
        BLK.append((ex, lsb))
NB = min(24, len(BLK))
gb = [(len(e) * MSB_B + len(l) * LSB_B) / 1e9 for e, l in BLK[:NB]]
say(f"{len(code)} code windows -> {len(BLK)} 32-row blocks (timing {NB}); kept experts per block mean "
    f"{np.mean([len(e) for e, _ in BLK[:NB]]):.1f}, 4-bit rank 0-1 experts {np.mean([len(l) for _, l in BLK[:NB]]):.1f}; "
    f"GB per call mean {np.mean(gb):.4f} (floor {np.mean(gb) / 0.28 * 1e3:.1f} us at 280 GB/s)")


X4 = 36 * 1024 // 16                                       # one segment's x: ~7 rows x 2560 bf16 = 36 KB, in uint4


def segs_real(ex, lsb, ch_words, xr=0.0, wr=1):
    """(array, word offset, uint4 count, x base, x uint4 count, 0) per chunk; xr: x bytes re-read per weight byte (the
    one-call reads ~2x: each 32-column threadgroup reads 2 x 16 rows x K of x for ~65-89 KB of weights); wr 0: x only"""
    s = []
    for j, e in enumerate(ex):
        xb = (j % 64) * X4
        for a in range(9):
            o0 = int(sm_of[e]) * WPS[a]
            for c in range(0, WPS[a], ch_words):
                n4 = min(ch_words, WPS[a] - c) // 4
                s.append((a, o0 + c, n4 * wr, xb, int(n4 * xr), 0))
    for j, e in enumerate(lsb):
        xb = (j % 64) * X4
        for a in range(9, 15):
            o0 = int(sl_of[e]) * WPS[a]
            for c in range(0, WPS[a], ch_words):
                n4 = min(ch_words, WPS[a] - c) // 4
                s.append((a, o0 + c, n4 * wr, xb, int(n4 * xr), 0))
    return mx.array(np.array(s, np.uint32).reshape(-1))


def segs_contig(nbytes, ch_words):
    """the same byte count from base's first w3 array (slot-major, contiguous over slots), from slot 0"""
    s = []
    tot = nbytes // 4 // 4 * 4
    for c in range(0, tot, ch_words):
        s.append((0, c, min(ch_words, tot - c) // 4, 0, 0, 0))
    return mx.array(np.array(s, np.uint32).reshape(-1))


names = [f"a{i}" for i in range(len(ARR))]
cases = "\n".join(f"      case {i}: p = (const device uint4*)(a{i} + off); break;" for i in range(len(ARR)))
SRC = f"""
    uint tg = threadgroup_position_in_grid.x;
    uint t = thread_position_in_threadgroup.x;
    uint a = segs[6 * tg + 0];
    uint off = segs[6 * tg + 1];
    uint n4 = segs[6 * tg + 2];
    uint xb = segs[6 * tg + 3];
    uint nx = segs[6 * tg + 4];
    const device uint4* p = (const device uint4*)(a0);
    switch (a) {{
{cases}
      default: n4 = 0;
    }}
    uint4 acc = uint4(0);
    for (uint i = t; i < n4; i += TPG) {{
      acc ^= p[i];
    }}
    const device uint4* xp = (const device uint4*)(xbuf) + xb;
    for (uint i = t; i < nx; i += TPG) {{
      acc ^= xp[i % 2304];
    }}
    uint v = acc.x ^ acc.y ^ acc.z ^ acc.w;
    v = simd_sum(v);
    if (thread_index_in_simdgroup == 0) {{
      out[tg * (TPG / 32) + simdgroup_index_in_threadgroup] = v;
    }}
"""
K = mx.fast.metal_kernel(name="expread", input_names=names + ["segs", "xbuf"], output_names=["out"], source=SRC)
XBUF = mx.zeros((64 * X4 * 4,), mx.uint32) + 7                # 64 segments x 36 KB of "x" (L2-resident: 2.4 MB)
mx.eval(XBUF)


def call(sg, tpg):
    ns = sg.size // 6
    return K(inputs=ARR + [sg, XBUF], template=[("TPG", tpg)], grid=(ns * tpg, 1, 1), threadgroup=(tpg, 1, 1),
             output_shapes=[(ns * (tpg // 32),)], output_dtypes=[mx.uint32])[0]


def run(sgs, tpg, rounds=7):
    ts = []
    for _ in range(rounds):
        mx.synchronize()
        t0 = time.perf_counter()
        ys = [call(s, tpg) for s in sgs]
        mx.eval(ys)
        mx.synchronize()
        ts.append(time.perf_counter() - t0)
    return 1e6 * float(np.median(ts[1:])) / len(sgs)


# a correctness probe: XOR of a whole piece array region == numpy's (the kernel reads what it should)
e0 = int(BLK[0][0][0])
sg0 = mx.array(np.array([(0, int(sm_of[e0]) * WPS[0], WPS[0] // 4, 0, 0, 0)], np.uint32).reshape(-1))
o_ = call(sg0, 256)
mx.eval(o_)
ref = np.array(PB[0][int(sm_of[e0])]).reshape(-1).view(np.uint32)
# per-simdgroup sums of per-lane xors: recompute the same fold on the host
lanes = 256
w4 = ref.reshape(-1, 4)
acc = np.zeros((lanes, 4), np.uint32)
for i in range(lanes):
    acc[i] = np.bitwise_xor.reduce(w4[i::lanes], axis=0) if len(w4[i::lanes]) else 0
v = acc[:, 0] ^ acc[:, 1] ^ acc[:, 2] ^ acc[:, 3]
host = (v.reshape(-1, 32).astype(np.uint64).sum(axis=1) & 0xFFFFFFFF).astype(np.uint32)
ok = np.array_equal(np.array(o_).astype(np.uint32), host)
say(f"probe: the kernel's fold of expert {e0}'s first piece == numpy's: {ok}")
if "probe" in sys.argv[1:]:                             # correctness only (no timing)
    open(OUT, "a").write("\n".join(L_) + "\n")
    sys.exit(0 if ok else 1)
res = []
for ch_kb in (16, 64, 256):
    chw = ch_kb * 1024 // 4
    S_real = [segs_real(e, l, chw) for e, l in BLK[:NB]]
    S_cont = [segs_contig(int(g * 1e9), chw) for g in gb]
    mx.eval(S_real + S_cont)
    for tpg in (256, 1024):
        ur = run(S_real, tpg)
        uc = run(S_cont, tpg)
        g = float(np.mean(gb))
        say(f"  CH {ch_kb:3d} KB, TPG {tpg:4d}: real {ur:7.1f} us/call ({g / (ur * 1e-6):5.1f} GB/s, x{280 / (g / (ur * 1e-6)):.2f} the "
            f"280 floor) | contig {uc:7.1f} us ({g / (uc * 1e-6):5.1f} GB/s); threadgroups per call real "
            f"{np.mean([s.size // 6 for s in S_real]):.0f}")
        row = dict(ch_kb=ch_kb, tpg=tpg, us_real=ur, us_contig=uc, gb=g)
        if ch_kb == 64:                                    # the one-call's x re-reads beside its weights (L2 traffic)
            for xr in (2.0, 2.5):
                S_wx = [segs_real(e, l, chw, xr=xr) for e, l in BLK[:NB]]
                S_x = [segs_real(e, l, chw, xr=xr, wr=0) for e, l in BLK[:NB]]
                mx.eval(S_wx + S_x)
                uwx, ux = run(S_wx, tpg), run(S_x, tpg)
                say(f"      + x re-reads at {xr} x the weight bytes (from 2.4 MB, cache-resident): weights + x {uwx:7.1f} us | x alone "
                    f"{ux:7.1f} us | weights alone {ur:7.1f} us -> {'they compete (sum-like)' if uwx > 0.8 * (ur + ux) else 'they overlap'}")
                row[f"us_wx{xr}"], row[f"us_x{xr}"] = uwx, ux
        res.append(row)
say(f"(the one-call on the same blocks: exp32_bench.txt, 571-614 us at 32 rows; done; peak {mx.get_peak_memory() / 1e9:.1f} GB)")
open(OUT, "a").write("\n".join(L_) + "\n")
json.dump(res, open(OUT.replace(".txt", ".json"), "w"), indent=1)
