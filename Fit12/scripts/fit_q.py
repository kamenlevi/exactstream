"""Fit12 Q1 (2026-10-24, CPU only): bytes per streamed expert, on sampled records3 experts (MSB-3 | LSB-1 of the 4-bit
experts; layout ~/BigMoE/DESIGN_3bit.md). Three questions (pre-registered in ~/Fit12/PLAN.md, C1):
  O1  lossless: order-0 entropy / static Huffman length of the MSB-3 codes m, byte-wise entropy of s3 / b3 (bf16), lzma;
  O2  re-quantization: relative error ||W' - W4||^2 / ||W4||^2 vs the 4-bit weights W4 (the reference's own weights) of
      formats with and without a randomized block Hadamard on K (blocks 512 / 128), at their bits per weight;
  O3  shared base: cosine of each expert with the leave-one-out mean of its sampled layer-mates (0 = independent).
Layout checks first: q = 2m + l has 0 in every group (MLX affine: the edge maps to 0); b3 == bf16(b + s * mean(l)).
Reads 40 records with pread (117 MB), numpy only, 4 BLAS threads. Usage: [BigMoE venv] -I fit_q.py > data/c1/q1.txt"""
import os
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import heapq, json, lzma, time
import numpy as np

R = os.path.expanduser("~/BigMoE/models/bare/records3")
IDX = json.load(open(os.path.join(R, "index.json")))
LAYERS, NE = [2, 12, 24, 36, 46], 8
SH = {"gate": (640, 2560), "up": (640, 2560), "down": (2560, 640)}
PROJ = ("gate", "up", "down")
LM3 = np.array([-2.1520, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1520], np.float32)   # Lloyd-Max N(0,1)
LM2 = np.array([-1.5104, -0.4528, 0.4528, 1.5104], np.float32)


def bf16(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def rbf16(x):                                   # fp32 -> nearest-even bf16, back as fp32
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16).view(np.float32)


def read(L, e):
    li = IDX["layers"][str(L)]
    # (Wed 4 Nov 17:3x; Leap's records3 reader triage) a raw row-major reader: refuse a converted layer. Thu's conversion
    # (native+p11+pad, in place) adds a per-layer "layout" key; the same offsets would then hold permuted codes and
    # 11-bit-packed scales, decoded silently as garbage. Read a converted store through BigMoE's ext record_rowmajor.
    # An explicit raise, not an assert, so python -O cannot skip it.
    if li.get("layout", "row-major") != "row-major":
        raise RuntimeError(f"records3 L{L} is {li.get('layout')}: fit_q.read decodes row-major bytes only "
                           "(use BigMoE's ext record_rowmajor)")
    fd = os.open(os.path.join(R, li["file"]), os.O_RDONLY)
    buf = os.pread(fd, li["recsz"], e * li["recsz"])
    os.close(fd)
    mo, ms, lo, ls = li["msb_piece_off"], li["msb_piece_size"], li["lsb_piece_off"], li["lsb_piece_size"]
    out = {}
    for i, p in enumerate(PROJ):
        o, K = SH[p]
        w3 = np.frombuffer(buf, np.uint8, ms[3 * i], mo[3 * i])
        bits = np.unpackbits(w3, bitorder="little").reshape(o, K, 3)
        m = bits[..., 0] + 2 * bits[..., 1] + 4 * bits[..., 2]
        l = np.unpackbits(np.frombuffer(buf, np.uint8, ls[2 * i], lo[2 * i]), bitorder="little").reshape(o, K)
        g = lambda off, sz: np.frombuffer(buf, np.uint16, sz // 2, off).reshape(o, K // 64)
        out[p] = dict(m=m.astype(np.int16), l=l.astype(np.int16), s3u=g(mo[3 * i + 1], ms[3 * i + 1]),
                      b3u=g(mo[3 * i + 2], ms[3 * i + 2]), bu=g(lo[2 * i + 1], ls[2 * i + 1]))
    return out


def ent(x, n):
    c = np.bincount(x.reshape(-1), minlength=n).astype(np.float64)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum()), c


def huff_len(c):                                # average static-Huffman code length (bits / symbol)
    h = [(v, i, None) for i, v in enumerate(c) if v > 0]
    if len(h) < 2:
        return 1.0
    heapq.heapify(h)
    depth = {i: 0 for _, i, _ in h}
    kids, nid = {}, len(c)
    while len(h) > 1:
        a, b = heapq.heappop(h), heapq.heappop(h)
        kids[nid] = (a[1], b[1])
        heapq.heappush(h, (a[0] + b[0], nid, None))
        nid += 1
    def walk(n, d):
        if n in kids:
            walk(kids[n][0], d + 1); walk(kids[n][1], d + 1)
        else:
            depth[n] = d
    walk(h[0][1], 0)
    return float(sum(c[i] * depth[i] for i in depth) / c.sum())


def hadamard(n):
    H = np.array([[1.0]], np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


HB = {2560: hadamard(512), 640: hadamard(128)}


def rht(W, seed):                               # W [o, K] -> W (H D)^T per K-block (orthogonal, norms preserved)
    o, K = W.shape
    H = HB[K]
    B = H.shape[0]
    d = np.random.default_rng(seed).choice(np.array([-1.0, 1.0], np.float32), K)
    Wd = (W * d[None, :]).reshape(o, K // B, B)
    return np.einsum("ojb,cb->ojc", Wd, H, optimize=True).reshape(o, K)


def rtn(W, bits, g):                            # MLX-style affine min-max per group (bf16 scale and bias)
    o, K = W.shape
    X = W.reshape(o, K // g, g)
    lo, hi = X.min(-1, keepdims=True), X.max(-1, keepdims=True)
    n = (1 << bits) - 1
    s = rbf16(np.maximum((hi - lo) / n, 1e-12)); b = rbf16(lo)
    q = np.clip(np.round((X - b) / s), 0, n)
    return (q * s + b).reshape(o, K)


def lloyd(W, levels, g):                        # symmetric Lloyd-Max levels x per-group bf16 scale (best of 7 clip factors)
    o, K = W.shape
    X = W.reshape(o, K // g, g)
    rms = np.sqrt((X * X).mean(-1, keepdims=True)) + 1e-12
    best, berr = None, None
    mids = (levels[1:] + levels[:-1]) / 2
    for f in (0.8, 0.87, 0.93, 1.0, 1.07, 1.15, 1.25):
        s = rbf16(rms * f)
        Y = levels[np.searchsorted(mids, X / s)] * s
        err = ((Y - X) ** 2).sum(-1, keepdims=True)
        if best is None:
            best, berr = Y, err
        else:
            k = err < berr
            best, berr = np.where(k, Y, best), np.where(k, err, berr)
    return best.reshape(o, K)


FORMATS = [  # name, bits per weight (bf16 overheads included), rotated?, fn(W4 or W4', ctx)
    ("MSB-3 (v3 tier, today)", 3.5, False, None),
    ("MSB-2 truncation", 2.5, False, None),
    ("RTN-3 g64 fresh", 3.5, False, lambda W: rtn(W, 3, 64)),
    ("RTN-3 g128", 3.25, False, lambda W: rtn(W, 3, 128)),
    ("LM-3 g64 (symmetric)", 3.25, False, lambda W: lloyd(W, LM3, 64)),
    ("RHT + RTN-3 g64", 3.5, True, lambda W: rtn(W, 3, 64)),
    ("RHT + RTN-3 g128", 3.25, True, lambda W: rtn(W, 3, 128)),
    ("RHT + LM-3 g128", 3.125, True, lambda W: lloyd(W, LM3, 128)),
    ("RHT + LM-3 per row", None, True, None),     # bits filled per projection: 3 + 16 / K
    ("RHT + RTN-2 g32", 3.0, True, lambda W: rtn(W, 2, 32)),
    ("RHT + RTN-2 g64", 2.5, True, lambda W: rtn(W, 2, 64)),
    ("RHT + LM-2 g64", 2.25, True, lambda W: lloyd(W, LM2, 64)),
]


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    err = {f[0]: {p: [] for p in PROJ} for f in FORMATS}
    chk = dict(groups=0, q0=0, q15=0, b3ok=0)
    o1 = {p: dict(Hm=[], Hufm=[], Hl_m=[], s3=[], b3=[], lz_s3=[], lz_b3=[]) for p in PROJ}
    cosl = {p: [] for p in PROJ}
    for L in LAYERS:
        es = sorted(rng.choice(512, NE, replace=False).tolist())
        W4s = {p: [] for p in PROJ}
        for e in es:
            rec = read(L, e)
            for p in PROJ:
                r = rec[p]
                o, K = SH[p]
                m, l = r["m"], r["l"]
                q = 2 * m + l
                qg = q.reshape(o, K // 64, 64)
                chk["groups"] += qg.shape[0] * qg.shape[1]
                chk["q0"] += int((qg.min(-1) == 0).sum()); chk["q15"] += int((qg.max(-1) == 15).sum())
                s3, b3, b = bf16(r["s3u"]), bf16(r["b3u"]), bf16(r["bu"])
                s = s3 / 2
                b3p = rbf16(b + s * l.reshape(o, K // 64, 64).mean(-1, dtype=np.float32))
                chk["b3ok"] += int((b3p == b3).sum())
                sx = np.repeat(s, 64, axis=1); bx = np.repeat(b, 64, axis=1)
                W4 = sx * q + bx
                W4s[p].append(W4)
                n4 = float((W4 * W4).sum())
                # O1
                Hm, cm = ent(m, 8)
                o1[p]["Hm"].append(Hm); o1[p]["Hufm"].append(huff_len(cm))
                Hj, _ = ent(q, 16)
                o1[p]["Hl_m"].append(Hj - Hm)                       # H(l | m) = H(q) - H(m)
                for nm, u in (("s3", r["s3u"]), ("b3", r["b3u"])):
                    hi, _ = ent((u >> 8).astype(np.int64), 256); lo_, _ = ent((u & 255).astype(np.int64), 256)
                    o1[p][nm].append(hi + lo_)
                    o1[p]["lz_" + nm].append(len(lzma.compress(u.tobytes(), preset=6)) * 8 / u.size)
                # O2
                W3 = np.repeat(s3, 64, axis=1) * m + np.repeat(b3, 64, axis=1)
                err["MSB-3 (v3 tier, today)"][p].append(float(((W3 - W4) ** 2).sum()) / n4)
                m2 = q >> 2
                b2 = rbf16(b + s * (q & 3).reshape(o, K // 64, 64).mean(-1, dtype=np.float32))
                W2 = 4 * sx * m2 + np.repeat(b2, 64, axis=1)
                err["MSB-2 truncation"][p].append(float(((W2 - W4) ** 2).sum()) / n4)
                Wr = rht(W4, seed=1000 * L + 7 * PROJ.index(p))
                for name, bits, rot, fn in FORMATS[2:]:
                    X = Wr if rot else W4
                    if fn is None:                                  # per-row Lloyd-Max: one group = the whole row
                        Y = lloyd(X, LM3, K)
                    else:
                        Y = fn(X)
                    err[name][p].append(float(((Y - X) ** 2).sum()) / n4)
        for p in PROJ:                                              # O3: cosine with the leave-one-out mean
            A = np.stack([w.reshape(-1) for w in W4s[p]])
            tot = A.sum(0)
            for i in range(NE):
                mu = (tot - A[i]) / (NE - 1)
                cosl[p].append(float(A[i] @ mu / (np.linalg.norm(A[i]) * np.linalg.norm(mu) + 1e-30)))
        print(f"layer {L} done ({time.time() - t0:.0f} s): experts {es}", flush=True)

    g = chk["groups"]
    print(f"\nLAYOUT CHECKS over {g} groups: q min 0 in {chk['q0'] / g:.4%}, q max 15 in {chk['q15'] / g:.4%}, "
          f"b3 == bf16(b + s mean(l)) in {chk['b3ok'] / g:.4%}")
    print("\nO1 lossless (per tensor, mean over 40 experts; MSB-3 part = codes 3 bits + s3 16 + b3 16 per 64 weights)")
    print("| proj | H(m) bits | static Huffman m | H(l given m) | s3 byte-wise | b3 byte-wise | s3 lzma | b3 lzma | MSB part, entropy bound |")
    print("|---|---|---|---|---|---|---|---|---|")
    tot_now, tot_ent, tot_huf = 0.0, 0.0, 0.0
    for p in PROJ:
        d = {k: float(np.mean(v)) for k, v in o1[p].items()}
        now = 3 * 64 + 32
        best_sb = min(d["s3"], d["lz_s3"]) + min(d["b3"], d["lz_b3"])
        ent_b = d["Hm"] * 64 + best_sb
        huf_b = d["Hufm"] * 64 + best_sb
        tot_now += now; tot_ent += ent_b; tot_huf += huf_b
        print(f"| {p} | {d['Hm']:.3f} | {d['Hufm']:.3f} | {d['Hl_m']:.3f} | {d['s3']:.2f} | {d['b3']:.2f} | "
              f"{d['lz_s3']:.2f} | {d['lz_b3']:.2f} | {1 - ent_b / now:.1%} smaller (Huffman m: {1 - huf_b / now:.1%}) |")
    print(f"All three projections: entropy bound {1 - tot_ent / tot_now:.1%} smaller, static Huffman on m {1 - tot_huf / tot_now:.1%}")
    print("\nO2 re-quantization: relative squared error vs W4 (mean over 40 experts; lower = closer to the reference)")
    print("| format | bits / weight | gate | up | down | all | vs MSB-3 |")
    print("|---|---|---|---|---|---|---|")
    base = np.mean([np.mean(err["MSB-3 (v3 tier, today)"][p]) for p in PROJ])
    for name, bits, rot, fn in FORMATS:
        v = [float(np.mean(err[name][p])) for p in PROJ]
        a = float(np.mean(v))
        b_s = f"{bits}" if bits else "3.006 / 3.025 (gate+up / down)"
        print(f"| {name} | {b_s} | {v[0]:.4f} | {v[1]:.4f} | {v[2]:.4f} | {a:.4f} | {a / base:.2f}x |")
    print("\nO3 shared base: cosine of an expert with the leave-one-out mean of its 7 sampled layer-mates")
    for p in PROJ:
        c = np.array(cosl[p])
        print(f"  {p}: mean {c.mean():+.4f}, max {c.max():+.4f} (explained energy ~ cos^2 = {np.mean(c ** 2):.4f})")
    print(f"\ndone in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
