"""B10(b) (Leap, Sun 2026-10-25; NewIdeas 06:24 + 06:57): what does decoding QTIP-style bitshift-trellis experts cost inside
the routed expert GLU on the M5 Pro GPU, per (row, rank) pair, against MoonShots' 3c (today's fused 3 / 4-bit GLU)?
Format = Fit12's B3 = Breakthrough's D2 (fit_b3.py, trellis.py; read only):
  - gate / up: x' = Hblk512(sx * x) once per token per layer (per-layer signs); down: h' = Hblk128(sd_e * h) per pair
    (per-expert signs). W = W' R^T, so y = W' x'.
  - W' / per-row bf16 RMS scale, cut in 16 x 16 tiles (row-major), each a bitshift trellis: L = 12 bit state, k = 2 new
    bits per weight, free 12-bit start state; weight t = LUT[bits 2t .. 2t + 11 of the tile's stream] (MSB first), LUT =
    4,096 Gaussian quantiles in a fixed random order (make_lut(12, 0)), here in fp16.
  - Tile row r (16 weights) = bits [32r, 32r + 42) = words r, r + 1. Tiles are padded to 17 words (2.125 b) and stored
    word-interleaved: words[e][row block][w][tile column], so a simdgroup's loads are contiguous.
  - Ranks with the residual (RES mask) decode a second stream of the same shape with its own row scales and add it.
Kernels (this file): TA = gate + up decode-GEMV + SwiGLU per pair; TB = down decode-GEMV per pair. Decode CODE:
0 = LUT read from device memory (cached), 1 = LUT staged in threadgroup memory, 2 = QTIP's 3INST computed code
(no LUT; a speed reference only: Fit12's quality numbers are for the LUT code).
Pools: 512 experts of random streams (every bitstream is a valid bitshift-trellis path, and decode cost does not depend
on the values). 3c runs on the real records3 pools of the same layer. Routing: the real router's top-10 on Fit12's
captured MoE inputs of layer 24 (~/Fit12/data/q2/x_L24.npy), S consecutive tokens per pass.
Timing as MoonShots' wide/microbench.py: N calls chained by a data dependence, one eval, minus the bare chain; here
the chain cycles over 6 passes so the working set exceeds the SLC (as across 48 layers). Quiet research mode (RULES.md):
the arms run in interleaved rounds (paired A/B), PAUSE is checked before each round, and a paused round is redone.
Usage: [BigMoE venv] b10_trellis.py test | sweep | bench   -> ~/Leap/data/b10_{test,sweep,bench}.txt"""
import json, math, os, sys, time
from functools import lru_cache
from statistics import NormalDist
import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MODE = sys.argv[1] if len(sys.argv) > 1 else "test"
LAYER, E, R = 24, 512, 10
K, N, D, TW = 2560, 640, 2560, 17
N_CH = int(os.environ.get("B10_N", "30"))
MD = os.path.expanduser("~/BigMoE/models/bare")
OUT = os.path.expanduser(f"~/Leap/data/b10_{MODE}.txt")
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


# ---- the LUT (Breakthrough's make_lut(12, seed 0) = Fit12's LUT), fp16
_q = np.array([NormalDist().inv_cdf((i + 0.5) / 4096) for i in range(4096)], np.float32)
LUT32 = _q[np.random.default_rng(0).permutation(4096)]
LUT16 = LUT32.astype(np.float16)
LUT = mx.array(LUT16)

_HDR = r"""
inline float inst3(uint s) {                                // QTIP 3INST: an LCG, a mask, a sum of two fp16 halves
    const uint xq = 89226354u * s + 64248484u;
    const uint mq = (xq & 0x8FFF8FFFu) ^ 0x3B603B60u;
    const half2 hv = as_type<half2>(mq);
    return static_cast<float>(hv.x) + static_cast<float>(hv.y);
}
// 16 weights of one tile row: state c = bits [2c, 2c + 12) of (w0 : w1), MSB first; dot with xv[0..15]
#define TDOT16(LUTEXPR)                                                                                     \
    float a = 0.0f;                                                                                         \
    for (int c = 0; c < 16; ++c) {                                                                          \
        const uint s = (c <= 10 ? (w0 >> (20 - 2 * c)) : ((w0 << (2 * c - 20)) | (w1 >> (52 - 2 * c)))) & 0xFFFu; \
        a = fma(LUTEXPR, xv[c], a);                                                                         \
    }                                                                                                       \
    return a;
inline float tdot_dev(uint w0, uint w1, thread const float* xv, const device half* lut) { TDOT16(static_cast<float>(lut[s])) }
inline float tdot_tg(uint w0, uint w1, thread const float* xv, const threadgroup half* lut) { TDOT16(static_cast<float>(lut[s])) }
inline float tdot_i3(uint w0, uint w1, thread const float* xv) { TDOT16(inst3(s)) }
#define TDOT(W0, W1, XV) (CODE_C == 1 ? tdot_tg(W0, W1, XV, (const threadgroup half*)lut_s) : \
                          (CODE_C == 2 ? tdot_i3(W0, W1, XV) : tdot_dev(W0, W1, XV, lut)))
"""

_STAGE = r"""
    threadgroup uint4 lut_s[CODE_C == 1 ? 512 : 1];
    if (CODE_C == 1) {
        for (uint i = thread_position_in_threadgroup.x; i < 512; i += NSG_C * 32)
            lut_s[i] = ((const device uint4*)lut)[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

# TA: one threadgroup = NSG simdgroups; the first half run gate rows, the second half the same up rows (RA each);
# loops over PPT pairs. Lane l takes tile columns l, l + 32, ... (16 inputs each), x' loaded once per tile column.
_TA = r"""
    constexpr int K = K_C, N = N_C, RA = RA_C, NSG = NSG_C, PPT = PPT_C, NR = NR_C, TW = 17;
    constexpr int KT = K / 16, NT = N / 16, HS = NSG / 2, ROWS = HS * RA;
    constexpr uint EW = (uint)(NT * TW * KT);
    const uint lane = thread_index_in_simdgroup;
    const uint sgi = simdgroup_index_in_threadgroup;
    const bool up = sgi >= (uint)HS;
    const uint jl = (sgi % HS) * RA;
    const uint j0 = threadgroup_position_in_grid.x * ROWS + jl;
""" + _STAGE + r"""
    threadgroup float gv[ROWS];
    for (int pp = 0; pp < PPT; ++pp) {
        const uint p = threadgroup_position_in_grid.y * PPT + pp;
        if (p >= (uint)NP_C) break;
        const uint r = p % NR, m = p / NR;
        const uint e = (uint)slot[p];
        const bool res = ((RES_C >> r) & 1) != 0;
        const device uint* W = (up ? wu : wg) + (size_t)e * EW;
        const device uint* V = (up ? ru : rg) + (size_t)e * EW;
        const device float* xr = xq + (size_t)m * K;
        float acc[RA], acc2[RA];
        for (int i = 0; i < RA; ++i) { acc[i] = 0.0f; acc2[i] = 0.0f; }
        for (int b = 0; b < KT / 32; ++b) {
            const uint tc = b * 32 + lane;
            float xv[16];
            for (int c = 0; c < 4; ++c) {
                const float4 t = ((const device float4*)(xr + tc * 16))[c];
                xv[4 * c] = t.x; xv[4 * c + 1] = t.y; xv[4 * c + 2] = t.z; xv[4 * c + 3] = t.w;
            }
            for (int i = 0; i < RA; ++i) {
                const uint o = j0 + i;
                const uint base = ((o >> 4) * TW + (o & 15)) * KT + tc;
                acc[i] += TDOT(W[base], W[base + KT], xv);
                if (res) acc2[i] += TDOT(V[base], V[base + KT], xv);
            }
        }
        const device bfloat16_t* sc = (up ? su : sgt) + (size_t)e * N;
        const device bfloat16_t* rsc = (up ? rsu : rsg) + (size_t)e * N;
        float ov[RA];
        for (int i = 0; i < RA; ++i) {
            float s = simd_sum(acc[i]) * static_cast<float>(sc[j0 + i]);
            if (res) s += simd_sum(acc2[i]) * static_cast<float>(rsc[j0 + i]);
            ov[i] = s;
        }
        if (!up && lane == 0) for (int i = 0; i < RA; ++i) gv[jl + i] = ov[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (up && lane == 0) {
            for (int i = 0; i < RA; ++i) {
                const float g = gv[jl + i];
                h[(size_t)p * N + j0 + i] = g / (1.0f + metal::exp(-g)) * ov[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

# TB: each simdgroup runs RB down rows of one pair; its RB x N/16 tile rows are spread over the 32 lanes.
_TB = r"""
    constexpr int N = N_C, D = D_C, RB = RB_C, NSG = NSG_C, PPT = PPT_C, NR = NR_C, TW = 17;
    constexpr int NT = N / 16, DT = D / 16, Q = RB * NT;
    static_assert(Q % 32 == 0, "RB * N / 16 must be a multiple of 32");
    constexpr uint EW = (uint)(DT * TW * NT);
    const uint lane = thread_index_in_simdgroup;
    const uint sgi = simdgroup_index_in_threadgroup;
    const uint d0 = (threadgroup_position_in_grid.x * NSG + sgi) * RB;
""" + _STAGE + r"""
    for (int pp = 0; pp < PPT; ++pp) {
        const uint p = threadgroup_position_in_grid.y * PPT + pp;
        if (p >= (uint)NP_C) break;
        const uint r = p % NR;
        const uint e = (uint)slot[p];
        const bool res = ((RES_C >> r) & 1) != 0;
        const device uint* W = wd + (size_t)e * EW;
        const device uint* V = rd + (size_t)e * EW;
        const device float* hr = hq + (size_t)p * N;
        float acc[RB], acc2[RB];
        for (int k = 0; k < RB; ++k) { acc[k] = 0.0f; acc2[k] = 0.0f; }
        for (int i = 0; i < Q / 32; ++i) {
            const uint q = lane + 32 * i;
            const uint row = q / NT, tc = q % NT;
            const uint o = d0 + row;
            const uint base = ((o >> 4) * TW + (o & 15)) * NT + tc;
            float xv[16];
            for (int c = 0; c < 4; ++c) {
                const float4 t = ((const device float4*)(hr + tc * 16))[c];
                xv[4 * c] = t.x; xv[4 * c + 1] = t.y; xv[4 * c + 2] = t.z; xv[4 * c + 3] = t.w;
            }
            const float a1 = TDOT(W[base], W[base + NT], xv);
            const float a2 = res ? TDOT(V[base], V[base + NT], xv) : 0.0f;
            for (int k = 0; k < RB; ++k) {
                acc[k] += row == (uint)k ? a1 : 0.0f;
                acc2[k] += row == (uint)k ? a2 : 0.0f;
            }
        }
        const device bfloat16_t* sc = sd + (size_t)e * D;
        const device bfloat16_t* rsc = rsd + (size_t)e * D;
        for (int k = 0; k < RB; ++k) {
            float s = simd_sum(acc[k]) * static_cast<float>(sc[d0 + k]);
            if (res) s += simd_sum(acc2[k]) * static_cast<float>(rsc[d0 + k]);
            if (lane == 0) y[(size_t)p * D + d0 + k] = s;
        }
    }
"""


@lru_cache(None)
def _ka():
    return mx.fast.metal_kernel(name="trellis_a", input_names=["xq", "wg", "wu", "rg", "ru", "sgt", "su", "rsg", "rsu", "lut", "slot"],
                                output_names=["h"], source=_TA, header=_HDR)


@lru_cache(None)
def _kb():
    return mx.fast.metal_kernel(name="trellis_b", input_names=["hq", "wd", "rd", "sd", "rsd", "lut", "slot"],
                                output_names=["y"], source=_TB, header=_HDR)


def make_pools(seed=0):
    """random-stream trellis pools: words [E, O/16 * 17 * I/16] u32 per projection (base and residual), bf16 row scales,
    per-layer x signs and per-expert down signs"""
    mx.random.seed(seed)
    rng = np.random.default_rng(seed + 1)
    P = {}
    for nm, (O, I) in (("g", (N, K)), ("u", (N, K)), ("d", (D, N))):
        for st in ("", "r"):
            P[st + "w" + nm] = mx.array(rng.integers(0, 1 << 32, (E, (O // 16) * TW * (I // 16)), dtype=np.uint32))
            P[st + "s" + nm] = (mx.random.uniform(0.6, 1.4, (E, O)) * (0.02 if st == "" else 0.0054)).astype(mx.bfloat16)
    P["sx"] = mx.array(rng.choice([-1.0, 1.0], K).astype(np.float32))
    P["sdn"] = mx.array(rng.choice([-1.0, 1.0], (E, N)).astype(np.float32))
    mx.eval(list(P.values()))
    return P


def trellis_moe(x, ids, w, P, code=1, res=0, ra=4, nsg_a=8, rb=4, nsg_b=8, ppt_a=1, ppt_b=1, parts=False):
    """x [M, K] bf16, ids [M, R] int32 (rank order), w [M, R] scores -> y [M, D] bf16 (the block's weighted sum)"""
    M = x.shape[0]
    NP = M * R
    xs = x.astype(mx.float32) * P["sx"]
    xq = mx.hadamard_transform(xs.reshape(M, K // 512, 512), scale=1 / math.sqrt(512)).reshape(M, K)
    slot = ids.reshape(-1).astype(mx.int32)
    rows_a = nsg_a // 2 * ra
    tm = [("CODE_C", code), ("RES_C", res), ("NP_C", NP), ("NR_C", R)]
    (h,) = _ka()(inputs=[xq, P["wg"], P["wu"], P["rwg"], P["rwu"], P["sg"], P["su"], P["rsg"], P["rsu"], LUT, slot],
                 template=tm + [("K_C", K), ("N_C", N), ("RA_C", ra), ("NSG_C", nsg_a), ("PPT_C", ppt_a)],
                 grid=(N // rows_a * 32 * nsg_a, -(-NP // ppt_a), 1), threadgroup=(32 * nsg_a, 1, 1),
                 output_shapes=[(NP, N)], output_dtypes=[mx.float32])
    hs = h * P["sdn"][slot]
    hq = mx.hadamard_transform(hs.reshape(NP, N // 128, 128), scale=1 / math.sqrt(128)).reshape(NP, N)
    (y,) = _kb()(inputs=[hq, P["wd"], P["rwd"], P["sd"], P["rsd"], LUT, slot],
                 template=tm + [("N_C", N), ("D_C", D), ("RB_C", rb), ("NSG_C", nsg_b), ("PPT_C", ppt_b)],
                 grid=(D // (rb * nsg_b) * 32 * nsg_b, -(-NP // ppt_b), 1), threadgroup=(32 * nsg_b, 1, 1),
                 output_shapes=[(NP, D)], output_dtypes=[mx.float32])
    if parts:
        return xq, h, hq, y
    return (y.reshape(M, R, D) * w.astype(mx.float32)[..., None]).sum(axis=1).astype(x.dtype)


# ---- numpy reference decode (from the bitstream definition) and forward
def sylvester(n):
    H = np.ones((1, 1))
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / math.sqrt(n)


def np_decode(words, O, I, lut=LUT16.astype(np.float64)):
    """words [O/16 * 17 * I/16] u32 (interleaved) -> W' [O, I] float64 (without the row scales)"""
    wv = words.reshape(O // 16, TW, I // 16).astype(np.uint64)
    V = (wv[:, :16, :] << np.uint64(32)) | wv[:, 1:17, :]                     # [rb, r, tc] 64-bit windows
    sh = (np.uint64(52) - np.uint64(2) * np.arange(16, dtype=np.uint64))
    st = (V[..., None] >> sh) & np.uint64(0xFFF)                               # [rb, r, tc, c]
    return lut[st.astype(np.int64)].transpose(0, 1, 2, 3).reshape(O, I)


def bitstream_selftest(rng):
    """encoder convention: s_t = ((s_{t-1} << 2) | b_t) & 4095, stream = s_0 (12 bits) then b_1 .. b_255 (2 bits),
    MSB first, padded to 17 words; the window decode must give back every s_t"""
    s0, b = int(rng.integers(4096)), rng.integers(4, size=255)
    states = [s0]
    for t in range(255):
        states.append(((states[-1] << 2) | int(b[t])) & 4095)
    bits = [(s0 >> (11 - i)) & 1 for i in range(12)] + [int(v >> (1 - j)) & 1 for v in b for j in range(2)]
    bits += [0] * (TW * 32 - len(bits))
    words = np.array([int("".join(map(str, bits[32 * i:32 * i + 32])), 2) for i in range(TW)], np.uint64)
    V = (words[:16] << np.uint64(32)) | words[1:17]
    dec = [(int(V[r]) >> (52 - 2 * c)) & 0xFFF for r in range(16) for c in range(16)]
    return dec == states


def np_moe(x, ids, w, P, res_mask):
    """float64 reference of trellis_moe for one pass"""
    tonp = lambda a: np.array(a.astype(mx.float32)).astype(np.float64)
    xs = tonp(x) * tonp(P["sx"])
    H512, H128 = sylvester(512), sylvester(128)
    xq = (xs.reshape(-1, K // 512, 512) @ H512).reshape(-1, K)
    sdn = tonp(P["sdn"])
    cache = {}

    def Wfull(e, nm, O, I, with_res):
        key = (e, nm, with_res)
        if key not in cache:
            Wb = np_decode(np.array(P["w" + nm][e]), O, I) * tonp(P["s" + nm][e])[:, None]
            if with_res:
                Wb = Wb + np_decode(np.array(P["rw" + nm][e]), O, I) * tonp(P["rs" + nm][e])[:, None]
            cache.clear(); cache[key] = Wb                                      # keep memory bounded
        return cache[key]
    M = xq.shape[0]
    out = np.zeros((M, D))
    for m in range(M):
        for r in range(R):
            e, rr = int(ids[m, r]), bool((res_mask >> r) & 1)
            g = Wfull(e, "g", N, K, rr) @ xq[m]
            u = Wfull(e, "u", N, K, rr) @ xq[m]
            hh = g / (1 + np.exp(-g)) * u
            hq = ((hh * sdn[e]).reshape(N // 128, 128) @ H128).reshape(N)
            out[m] += float(w[m, r]) * (Wfull(e, "d", D, N, rr) @ hq)
    return out


# ---- real routing: the router's top-10 on Fit12's captured MoE inputs (layer 24)
def routing():
    idx = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
    pre = f"language_model.model.layers.{LAYER}.mlp.gate."
    sh = {}
    def tensor(n):
        f = idx[n]
        if f not in sh:
            sh[f] = mx.load(os.path.join(MD, f))
        return sh[f][n]
    wq, s, b = (tensor(pre + k) for k in ("weight", "scales", "biases"))
    bits = wq.shape[-1] * 32 // (s.shape[-1] * 64)                              # the router is 8-bit
    X = mx.array(np.load(os.path.expanduser(f"~/Fit12/data/q2/x_L{LAYER}.npy"))[:1024]).astype(mx.bfloat16)
    g = mx.softmax(mx.quantized_matmul(X, wq, s, b, transpose=True, group_size=64, bits=bits).astype(mx.float32), axis=-1, precise=True)
    order = mx.argsort(-g, axis=-1)[:, :R]
    wts = mx.take_along_axis(g, order, axis=-1)
    mx.eval(order, wts)
    return X, np.array(order).astype(np.int32), np.array(wts / wts.sum(-1, keepdims=True)).astype(np.float32)


def passes(ids, wts, S, n=6, seed=0):
    starts = np.random.default_rng(seed).choice(ids.shape[0] // S, n, replace=False) * S
    return [(mx.array(ids[a:a + S]), mx.array(wts[a:a + S]).astype(mx.bfloat16), len(np.unique(ids[a:a + S]))) for a in starts]


PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")              # quiet research mode (RULES.md): heat pause
QUIET = os.path.exists(os.path.expanduser("~/.local/share/quiet/FAN_CAP"))


def chain(g, x, ps):
    """N_CH calls chained through x, cycling over the passes ps"""
    y = x
    for i in range(N_CH):
        ii, ww, _ = ps[i % len(ps)]
        y = y + g(y, ii, ww) * 0
    return y


def one_round(f, x, ps):
    """ms per call for one chain (one eval, minus the bare chain); None when a heat pause overlapped it"""
    while os.path.exists(PAUSE):
        time.sleep(0.2)
    a = time.perf_counter(); mx.eval(chain(f, x, ps)); t = time.perf_counter() - a
    b = time.perf_counter(); mx.eval(chain(lambda y, i, w: y, x, ps)); t0 = time.perf_counter() - b
    return None if os.path.exists(PAUSE) or t + t0 > 5 else (t - t0) / N_CH * 1e3


def timed_arms(arms, x, ps, rounds=3):
    """median ms per call per arm: 2 warm chains each, then `rounds` rounds interleaved over all arms (paired A/B);
    a round that a pause overlapped (PAUSE after it, or a SIGSTOP jump) is redone"""
    for nm, f in arms:
        for _ in range(2):
            mx.eval(chain(f, x, ps))
    ms = {nm: [] for nm, _ in arms}
    for _ in range(rounds):
        for nm, f in arms:
            v = None
            while v is None:
                v = one_round(f, x, ps)
            ms[nm].append(v)
    return {nm: float(np.median(v)) for nm, v in ms.items()}


def load_3c():
    import tierglu_v3d as TG
    RD = os.path.join(MD, "records3")
    ix = json.load(open(f"{RD}/index.json"))["layers"][str(LAYER)]
    rec = ix["recsz"]
    raw = np.fromfile(f"{RD}/{ix['file']}", dtype=np.uint8).reshape(ix["E"], rec)
    SH = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}
    msb, lsb = {}, {}
    for p, pr in enumerate(TG.PROJ):
        o, i = SH[pr]
        ow, os_, ob = ix["msb_piece_off"][3 * p:3 * p + 3]
        msb[pr] = {"w3": mx.array(np.ascontiguousarray(raw[:, ow:ow + o * i * 3 // 8]).view(np.uint32).reshape(E, o, i * 3 // 32)),
                   "s3": mx.array(np.ascontiguousarray(raw[:, os_:os_ + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16),
                   "b3": mx.array(np.ascontiguousarray(raw[:, ob:ob + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16)}
        lw, lb = ix["lsb_piece_off"][2 * p:2 * p + 2]
        lsb[pr] = {"l1": mx.array(np.ascontiguousarray(raw[:, lw:lw + o * i // 8]).view(np.uint32).reshape(E, o, i // 32)),
                   "b": mx.array(np.ascontiguousarray(raw[:, lb:lb + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16)}
    del raw
    mx.eval([v for d in (msb, lsb) for pr in d for v in d[pr].values()])
    return TG, msb, lsb


def bytes_per_expert(P, res):
    b = sum(P[k][0].nbytes for k in ("wg", "wu", "wd", "sg", "su", "sd")) + N // 8                # + down signs (1 bit)
    return b + (sum(P[k][0].nbytes for k in ("rwg", "rwu", "rwd", "rsg", "rsu", "rsd")) if res else 0)


if __name__ == "__main__":
    t0 = time.time()
    rng = np.random.default_rng(0)
    say(f"B10(b) {MODE} ({time.strftime('%a %H:%M')}): trellis k=2 L=12 decode in the routed GLU, layer {LAYER}, MLX {mx.__version__}"
        + (" [QUIET MODE: fans capped at 3,000 rpm; absolute times are quiet-mode numbers, compare arms]" if QUIET else ""))
    assert all(bitstream_selftest(rng) for _ in range(20)), "window decode != encoder states"
    say("  bitstream self-test: window decode == encoder recurrence on 20 random tiles")
    P = make_pools()
    say(f"  pools: {E} experts, base {bytes_per_expert(P, False) / 1e6:.3f} MB / expert, base + residual "
        f"{bytes_per_expert(P, True) / 1e6:.3f} MB ({time.time() - t0:.0f} s)")
    X, ids, wts = routing()
    say(f"  routing: layer {LAYER} router top-{R} on {X.shape[0]} captured tokens")
    if MODE == "test":
        x = X[:4]
        ii, ww = mx.array(ids[:4]), mx.array(wts[:4]).astype(mx.bfloat16)
        for code in (0, 1, 2):
            for res in (0, 0x3):
                cfg = dict(code=code, res=res)
                yk = np.array(trellis_moe(x, ii, ww, P, **cfg).astype(mx.float32)).astype(np.float64)
                if code == 2:
                    say(f"  code 2 (3INST, speed reference) res {res:#x}: runs, |y| max {np.abs(yk).max():.3g}")
                    continue
                yr = np_moe(x, ids[:4], np.array(ww.astype(mx.float32)), P, res)
                rel = np.abs(yk - yr).max() / np.abs(yr).max()
                say(f"  code {code} res {res:#x}: max |kernel - numpy| / max |y| = {rel:.2e} (|y| max {np.abs(yr).max():.3g})")
                assert rel < 1e-2, rel
        for ra, nsg_a, rb, nsg_b, ppt in ((4, 2, 4, 2, 1), (8, 8, 8, 4, 4)):
            ya = np.array(trellis_moe(x, ii, ww, P, code=1, res=3, ra=ra, nsg_a=nsg_a, rb=rb, nsg_b=nsg_b, ppt_a=ppt, ppt_b=ppt).astype(mx.float32))
            yb = np.array(trellis_moe(x, ii, ww, P, code=1, res=3).astype(mx.float32))
            say(f"  geometry ra {ra} nsg_a {nsg_a} rb {rb} nsg_b {nsg_b} ppt {ppt}: max |diff| vs default {np.abs(ya - yb).max():.3g}")
    elif MODE in ("sweep", "bench", "bench2"):
        TG, msb, lsb = load_3c()
        say(f"  3c pools loaded (records3 layer {LAYER}) ({time.time() - t0:.0f} s)")
        x16 = X[:16]
        res_all = {}
        Ss = (16,) if MODE == "sweep" else (4, 16)
        for S in Ss:
            ps = passes(ids, wts, S)
            x = X[:S]
            dist = np.mean([p[2] for p in ps])
            arms = []
            if MODE == "sweep":
                for code in (0, 1, 2):
                    for (ra, nsg_a) in ((4, 2), (4, 8), (8, 8)):
                        for (rb, nsg_b) in ((4, 2), (4, 8), (8, 4)):
                            for ppt in (1, 4):
                                arms.append((f"T code {code} ra {ra}/{nsg_a} rb {rb}/{nsg_b} ppt {ppt}",
                                             lambda y, i, w, c=dict(code=code, ra=ra, nsg_a=nsg_a, rb=rb, nsg_b=nsg_b, ppt_a=ppt, ppt_b=ppt): trellis_moe(y, i, w, P, **c)))
                arms.append(("3c 3333333333", lambda y, i, w: TG.tier_glu(y, msb, lsb, i, None, plan="3333333333", scores=w)))
            elif MODE == "bench2":                              # repeat check: 3c twice, 3INST at its own best sweep geometry
                best = json.load(open(os.path.expanduser("~/Leap/data/b10_best.json")))
                c2 = dict(code=2, ra=4, nsg_a=8, rb=4, nsg_b=8, ppt_a=1, ppt_b=1)
                arms += [("3c 3333333333 (3-bit)", lambda y, i, w: TG.tier_glu(y, msb, lsb, i, None, plan="3333333333", scores=w)),
                         ("3c 4444444444 (4-bit)", lambda y, i, w: TG.tier_glu(y, msb, lsb, i, i, plan="4444444444", scores=w)),
                         ("3c 4433333333 (today)", lambda y, i, w: TG.tier_glu(y, msb, lsb, i, i[:, :2], plan="4433333333", scores=w)),
                         (f"T code {best['code']} base (LUT best)", lambda y, i, w: trellis_moe(y, i, w, P, res=0, **best))]
                for nm, res in (("base", 0), ("base + residual", 0x3FF), ("spec: residual on ranks 0-1", 0x3)):
                    arms.append((f"T code 2 ra 4/8 rb 4/8 {nm}", lambda y, i, w, res=res: trellis_moe(y, i, w, P, res=res, **c2)))
                arms.append(("3c 3333333333 (3-bit) again", lambda y, i, w: TG.tier_glu(y, msb, lsb, i, None, plan="3333333333", scores=w)))
            else:
                best = json.load(open(os.path.expanduser("~/Leap/data/b10_best.json")))
                arms += [("3c 3333333333 (3-bit)", lambda y, i, w: TG.tier_glu(y, msb, lsb, i, None, plan="3333333333", scores=w)),
                         ("3c 3333333333 + MLX combine", lambda y, i, w: (TG.tier_glu(y, msb, lsb, i, None, plan="3333333333")
                                                                          * w[..., None]).sum(axis=1)),
                         ("3c 4444444444 (4-bit)", lambda y, i, w: TG.tier_glu(y, msb, lsb, i, i, plan="4444444444", scores=w)),
                         ("3c 4433333333 (today)", lambda y, i, w: TG.tier_glu(y, msb, lsb, i, i[:, :2], plan="4433333333", scores=w))]
                for code in (best["code"], 2):
                    c = dict(best, code=code)
                    for nm, res in (("base", 0), ("base + residual", 0x3FF), ("spec: residual on ranks 0-1", 0x3)):
                        arms.append((f"T code {code} {nm}", lambda y, i, w, c=c, res=res: trellis_moe(y, i, w, P, res=res, **c)))
            for nm, ms in timed_arms(arms, x, ps, rounds=int(os.environ.get("B10_ROUNDS", "3"))).items():
                res_all[(S, nm)] = ms
                say(f"  S {S:2d} (distinct {dist:5.1f}) {nm:42s}: {ms:.3f} ms / layer, {ms / (S * R) * 1e3:.2f} us / pair")
        if MODE == "sweep":
            tk = {k: v for k, v in res_all.items() if k[1].startswith("T code") and " code 2 " not in k[1]}
            (S, nm), v = min(tk.items(), key=lambda kv: kv[1])
            f = nm.split()
            best = dict(code=int(f[2]), ra=int(f[4].split("/")[0]), nsg_a=int(f[4].split("/")[1]), rb=int(f[6].split("/")[0]),
                        nsg_b=int(f[6].split("/")[1]), ppt_a=int(f[8]), ppt_b=int(f[8]))
            json.dump(best, open(os.path.expanduser("~/Leap/data/b10_best.json"), "w"))
            say(f"  best LUT config: {nm} ({v:.3f} ms)")
    say(f"  done in {time.time() - t0:.0f} s")
    open(OUT, "w").write("\n".join(lines) + "\n")
