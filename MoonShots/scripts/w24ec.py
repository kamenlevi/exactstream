"""R-16 W24-ec screen (NewIdeas Tue 19:3x; exact, after W24-sp step 2): entropy-code the MSB-3 codes (3-bit m per weight,
pooled order-0 entropy 2.72 b, data/msb_entropy.txt) on the SSD copy and decode them on the GPU while it idles in a 24 GB
pass. Coder = interleaved rANS (32-bit state, 16-bit renormalization, 10-bit frequencies, one static table per layer):
one simdgroup per chunk of 65,536 symbols (25 chunks per projection), lane l decodes the 8-symbol groups l, l + 32, ...
and writes them in MLX's 3-bit packing (3 bytes per group), refills come from the chunk's one word stream in lane order
(simd_ballot + popcount). Chunk overhead: 32 initial states (uint32) + one word offset.
Data: real records3 code bytes (w3 pieces of gate / up / down) of NE random experts from layers 0, 24, 47.
PASS/KILL (NewIdeas, written first): PASS = decode >= 25 GB/s of codes (decoded bytes) on the M5 GPU AND <= 2.80 b per
weight (everything stored: words, states, offsets, tables) over the sampled layers; KILL = < 10 GB/s, or > 2.90 b.
Exactness: the decoded bytes must equal the records3 code bytes bit for bit (else KILL regardless).
usage: [BigMoE venv] w24ec.py [experts_per_layer=32]   -> stdout + ~/MoonShots/data/w24ec.txt"""
import fcntl, json, os, sys, time
import numpy as np
import mlx.core as mx

D3 = os.path.expanduser("~/BigMoE/models/bare/records3/")
IX = json.load(open(D3 + "index.json"))["layers"]
NE = int(sys.argv[1]) if len(sys.argv) > 1 else 32
LAYERS = tuple(int(v) for v in os.environ.get("W24EC_LAYERS", "0,24,47").split(","))
NB, M = 10, 1 << 10                                         # frequency precision (1 KB slot table per simdgroup)
CH = int(os.environ.get("W24EC_CH", "65536"))               # symbols per chunk (one simdgroup)
PIECE = 640 * 2560                                          # symbols per projection
W3 = PIECE * 3 // 8
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
OUT = os.path.expanduser("~/MoonShots/data/w24ec.txt" if CH == 65536 and len(LAYERS) == 3 else f"~/MoonShots/data/w24ec_ch{CH}.txt")
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def unpack3(p):                                             # (n, 3k) MLX 3-bit packing -> (n, 8k) symbols
    t = p.reshape(len(p), -1, 3).astype(np.uint32)
    w = t[:, :, 0] | (t[:, :, 1] << 8) | (t[:, :, 2] << 16)
    return ((w[:, :, None] >> (3 * np.arange(8, dtype=np.uint32))) & 7).astype(np.uint8).reshape(len(p), -1)


def read_codes(L, experts):                                 # (len(experts) * 3, W3) code bytes, F_NOCACHE preads
    ix = IX[str(L)]
    fd = os.open(D3 + ix["file"], os.O_RDONLY)
    fcntl.fcntl(fd, 48, 1)
    out = np.empty((len(experts) * 3, W3), np.uint8)
    for a, e in enumerate(experts):
        for p in range(3):
            assert ix["msb_piece_size"][3 * p] == W3
            out[3 * a + p] = np.frombuffer(os.pread(fd, W3, e * ix["recsz"] + ix["msb_piece_off"][3 * p]), np.uint8)
    os.close(fd)
    return out


def table(sym):                                             # quantized frequencies (sum M, each >= 1), cumulative, slots
    h = np.bincount(sym.ravel(), minlength=8).astype(np.float64)
    f = np.maximum(1, np.round(h / h.sum() * M)).astype(np.int64)
    f[np.argmax(f)] += M - f.sum()
    c = np.concatenate([[0], np.cumsum(f)[:-1]])
    slots = np.repeat(np.arange(8, dtype=np.uint8), f)
    H0 = -(h / h.sum() * np.log2(np.maximum(h / h.sum(), 1e-300))).sum()
    return f, c, slots, H0


def encode(sym, f, c):
    """sym (NC, CH) -> word stream (decode order: step asc, lane asc), per-chunk word offsets (NC + 1), states (NC, 32)"""
    NC = len(sym)
    S = sym.reshape(NC, CH // 256, 32, 8)                   # [chunk, t, lane, j]: symbol 8 * (32 t + lane) + j
    fq, cq = f.astype(np.uint64), c.astype(np.uint64)
    x = np.full((NC, 32), 1 << 16, np.uint64)
    em = []
    for k in range(CH // 32 - 1, -1, -1):
        t, j = divmod(k, 8)
        s = S[:, t, :, j]
        fs, cs = fq[s], cq[s]
        o = x >= (fs << np.uint64(32 - NB))                 # x_max = ((L >> NB) << 16) * f, L = 2^16
        ci = np.nonzero(o)
        em.append((ci[0].astype(np.int32), (x[o] & np.uint64(0xFFFF)).astype(np.uint16)))
        x = np.where(o, x >> np.uint64(16), x)
        x = (x // fs) * np.uint64(M) + x % fs + cs
    ch = np.concatenate([e[0] for e in em[::-1]])
    wd = np.concatenate([e[1] for e in em[::-1]])
    order = np.argsort(ch, kind="stable")
    off = np.concatenate([[0], np.cumsum(np.bincount(ch, minlength=NC))]).astype(np.uint32)
    return wd[order], off, x.astype(np.uint32)


def decode_cpu(words, off, states, f, c, slots, n=4):       # reference for the first n chunks
    out = []
    for ci in range(n):
        x = states[ci].astype(np.uint64)
        w = words[off[ci]:off[ci + 1]]
        pos = 0
        res = np.empty((CH // 256, 32, 8), np.uint8)
        for k in range(CH // 32):
            t, j = divmod(k, 8)
            slot = x & np.uint64(M - 1)
            s = slots[slot.astype(np.int64)]
            x = f[s].astype(np.uint64) * (x >> np.uint64(NB)) + slot - c[s].astype(np.uint64)
            need = x < (1 << 16)
            nn = int(need.sum())
            x[need] = (x[need] << np.uint64(16)) | w[pos:pos + nn].astype(np.uint64)
            pos += nn
            res[t, :, j] = s
        assert pos == len(w) and (x == (1 << 16)).all()
        out.append(res.reshape(-1))
    return out


_SRC = r"""
    const uint l = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint c = threadgroup_position_in_grid.x * SG_C + sg;
    threadgroup uint8_t tsym[SG_C * M_C];
    threadgroup uint tfc[SG_C * 8];
    const bool live = c < (uint)NC_C;
    const uint tb = live ? (uint)ctab[c] : 0u;
    for (uint i = l; i < (uint)M_C; i += 32) tsym[sg * M_C + i] = symtab[tb * M_C + i];
    if (l < 8) tfc[sg * 8 + l] = fc[tb * 8 + l];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!live) return;
    const threadgroup uint8_t* ts = tsym + sg * M_C;
    const threadgroup uint* tf = tfc + sg * 8;
    uint x = states[c * 32 + l];
    const device uint16_t* in = words + woff[c];
    uint pos = 0;
    device uint8_t* o = out + (size_t)c * (CH_C * 3 / 8);
    const ulong below = (1ul << l) - 1ul;
    for (uint t = 0; t < (uint)(CH_C / 256); ++t) {
        uint packed = 0;
        for (uint j = 0; j < 8; ++j) {
            const uint slot = x & (uint)(M_C - 1);
            const uint s = ts[slot];
            const uint v = tf[s];
            x = (v >> 16) * (x >> NB_C) + slot - (v & 0xFFFFu);
            const bool need = x < 65536u;
            const ulong mask = (ulong)simd_ballot(need);
            if (need) x = (x << 16) | (uint)in[pos + (uint)popcount(mask & below)];
            pos += (uint)popcount(mask);
            packed |= s << (3 * j);
        }
        const uint g = l + 32 * t;
        o[3 * g] = (uint8_t)(packed & 0xFFu); o[3 * g + 1] = (uint8_t)((packed >> 8) & 0xFFu); o[3 * g + 2] = (uint8_t)(packed >> 16);
    }
"""


def kernel():
    return mx.fast.metal_kernel(name="w24ec_rans", input_names=["words", "woff", "states", "symtab", "fc", "ctab"],
                                output_names=["out"], source=_SRC)


def gpu_decode(K, A, NC, sg):
    (y,) = K(inputs=A, template=[("SG_C", sg), ("M_C", M), ("NB_C", NB), ("NC_C", NC), ("CH_C", CH)],
             grid=(-(-NC // sg) * 32 * sg, 1, 1), threadgroup=(32 * sg, 1, 1),
             output_shapes=[(NC, CH * 3 // 8)], output_dtypes=[mx.uint8])
    return y


def main():
    t_all = time.time()
    say(f"R-16 W24-ec ({time.strftime('%a %d %b %H:%M:%S')}), MLX {mx.__version__}: interleaved rANS on MSB-3 codes, "
        f"layers {LAYERS}, {NE} experts each, chunk {CH} symbols, {NB}-bit table per layer")
    say("PASS = decode >= 25 GB/s of codes AND <= 2.80 b / weight (all stored bytes); KILL = < 10 GB/s or > 2.90 b; "
        "any decoded byte != records3 -> KILL")
    rng = np.random.default_rng(16)
    allw, allo, alls, allt, alltab, raw_all, nsym, store = [], [], [], [], [], [], 0, 0
    base = 0
    for li, L in enumerate(LAYERS):
        t0 = time.time()
        ex = sorted(rng.choice(512, NE, replace=False).tolist())
        raw = read_codes(L, ex)                             # (NE * 3, W3)
        sym = unpack3(raw)                                  # (NE * 3, PIECE)
        f, c, slots, H0 = table(sym)
        chunks = sym.reshape(-1, CH)
        words, off, st = encode(chunks, f, c)
        ref = decode_cpu(words, off, st, f, c, slots, n=2)
        assert all(np.array_equal(r, chunks[i]) for i, r in enumerate(ref)), "CPU reference decode"
        nb = words.nbytes + st.nbytes + 4 * len(chunks) + 16                    # words, states, offsets, 8 freqs
        nsym += chunks.size; store += nb
        say(f"L{L}: experts {ex[:6]}...; H0 {H0:.4f} b; freqs {f.tolist()}; coded {8 * words.nbytes / chunks.size:.4f} b "
            f"+ states {8 * st.nbytes / chunks.size:.4f} + offsets/table {8 * (nb - words.nbytes - st.nbytes) / chunks.size:.4f}"
            f" = {8 * nb / chunks.size:.4f} b / weight ({time.time() - t0:.1f} s, CPU ref decode of 2 chunks OK)")
        allw.append(words); allo.append(off[:-1].astype(np.int64) + base); base += len(words)
        alls.append(st); allt.append(np.full(len(chunks), li, np.uint32)); alltab.append((slots, f, c)); raw_all.append(raw)
    bpw = 8 * store / nsym
    words = mx.array(np.concatenate(allw))
    woff = mx.array(np.concatenate(allo).astype(np.uint32))
    states = mx.array(np.concatenate(alls).reshape(-1))
    symtab = mx.array(np.concatenate([t[0] for t in alltab]))
    fc = mx.array(np.concatenate([((t[1].astype(np.uint32) << 16) | t[2].astype(np.uint32)) for t in alltab]))
    ctab = mx.array(np.concatenate(allt))
    NC = int(ctab.shape[0])
    ref = np.concatenate([r.reshape(-1) for r in raw_all])
    A = [words, woff, states, symtab, fc, ctab]
    mx.eval(A)
    K = kernel()
    gb = NC * CH * 3 / 8 / 1e9
    best = None
    for sg in (1, 2, 4, 8):
        y = gpu_decode(K, A, NC, sg)
        mx.eval(y)
        bad = int((np.array(y).reshape(-1) != ref).sum())
        ts = []
        for r in range(7):
            while os.path.exists(PAUSE):
                time.sleep(0.5)
            a = time.perf_counter(); mx.eval(gpu_decode(K, A, NC, sg)); ts.append(time.perf_counter() - a)
        t = float(np.median(ts[2:]))
        say(f"  GPU decode, {sg} simdgroup(s) per threadgroup: {NC} chunks, {gb * 1e3:.1f} MB of codes in {t * 1e3:.2f} ms = "
            f"{gb / t:.1f} GB/s (min {gb / max(ts[2:]):.1f}, max {gb / min(ts[2:]):.1f}); mismatching bytes {bad}")
        if best is None or (bad == 0 and gb / t > best[0]):
            best = (gb / t if bad == 0 else 0.0, sg, bad)
    rate, sg, bad = best
    verdict = "KILL (not exact)" if bad else ("PASS" if rate >= 25 and bpw <= 2.80 else
                                              ("KILL" if rate < 10 or bpw > 2.90 else "BETWEEN"))
    say(f"VERDICT: {verdict}: {bpw:.4f} b / weight over {nsym / 1e6:.1f} M weights ({100 * (1 - bpw / 3):.2f} % fewer code "
        f"bytes than 3 b), best decode {rate:.1f} GB/s at {sg} simdgroup(s) / threadgroup; {time.time() - t_all:.0f} s")
    nx = 3 * PIECE // CH                                     # latency: one expert's chunks alone, one chunk alone
    for nm, n in (("one expert", nx), ("one chunk", 1), ("8 experts", 8 * nx)):
        B = [words, woff[:n], states[:n * 32], symtab, fc, ctab[:n]]
        mx.eval(gpu_decode(K, B, n, sg))
        ts = []
        for r in range(9):
            a = time.perf_counter(); mx.eval(gpu_decode(K, B, n, sg)); ts.append(time.perf_counter() - a)
        say(f"  latency, {nm} ({n} chunks of {CH}): median {1e3 * float(np.median(ts[2:])):.3f} ms, min {1e3 * min(ts[2:]):.3f} "
            f"(includes MLX dispatch + eval overhead)")
    open(OUT, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
