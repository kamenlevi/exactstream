"""M-27 LOSSLESS45 (MoonShots, Tue 2026-11-03 17:0x; BREAKTHROUGH PUSH bet, the orchestrator 17:0x; pre-registered in
~/NewIdeas/QUEUE.md M-27 before the run): lossless (exact) compression of records3's expert bytes, priced at the 45 GB line.
Sample: 4 experts x 48 layers (fixed seed), both parts of each record. MSB part per projection (gate / up / down, 640 x 2560):
3-bit codes (MLX packing, 8 values in 3 bytes), s3 / b3 bf16 per group of 64. LSB part: the 1-bit plane and b (bf16 per 64).
Per layer: order-0 entropy H0 and Huffman length of the codes; the LSB plane's entropy; the 16-bit entropy of s3 / b3 / b.
Shared base + delta: for every pair of the layer's sampled experts, H(codes e | codes e' at the same position) and
H(e - e' mod 8), against H0 (gain > 0 would mean experts share structure).
Formats (bytes per record part, 16 KB-aligned like records3): today; fixed-width (s3 / b3 / b at 11 bits: W24-sp's measured
lossless width, all pieces round-trip); entropy floor (rANS codes at H0 + 0.0161 b: W24-ec's measured states + tables;
scales at their 16-bit order-0 entropy). Pricing: band45's capture reads at C 256 + M-17 sizes (M-18: code 142.2 base +
118.2 sidecar reads per check, 47.0 MB per kept token; IFEval 25.1 + 25.9, 31.1; multi 20.6 + 16.8, 30.7).
Time balance: the GPU decodes a check's missed codes at W24-ec's measured 46 GB/s; the SSD time saved is at 12 GB/s.
Lines: bytes PASS >= 12 % saved per kept token on code (entropy floor), KILL < 8 %; time: decode ms < SSD ms saved, else
'no net gain'.
usage: [BigMoE venv] taskpolicy -b python lossless45.py -> data/lossless45.txt"""
import heapq, json, os, time
import numpy as np

D = os.path.expanduser("~/BigMoE/models/bare/records3/")
IX = json.load(open(D + "index.json"))["layers"]
_CONV = sorted(int(L) for L, r in IX.items() if str(r.get("layout", "row-major")) not in ("row-major", "rowmajor", "row_major", "0"))
if _CONV:                                                         # Leap 11-04: Thursday's in-place conversion
    raise SystemExit(f"records3 is converted (layers {_CONV[:6]}...): lossless45 decodes row-major pieces; "
                     "route it through BigMoE's load_index (layout_ok) first")
REC, MSB_SZ, LSB_SZ, PAGE = 2932736, 2162688, 770048, 16384
NPROJ, ROWS, K = 3, 640, 2560
W = ROWS * K                                                     # weights per projection
G = W // 64                                                      # groups per projection
READS = {"code": (142.2, 118.2, 47.0), "ifeval": (25.1, 25.9, 31.1), "multi": (20.6, 16.8, 30.7)}
RANS_OVH, FIX_BITS, GPU_GBS, SSD_GBS = 0.0161, 11, 46.0, 12.0
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def unpack3(b, n):
    t = b.reshape(-1, 3).astype(np.uint32)
    v = t[:, 0] | (t[:, 1] << 8) | (t[:, 2] << 16)
    return ((v[:, None] >> (3 * np.arange(8, dtype=np.uint32))) & 7).reshape(-1)[:n].astype(np.uint8)


def H(c):
    c = np.asarray(c, float)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())


def huffman_len(c):
    """average Huffman code length (bits per symbol) for the histogram c"""
    c = [int(x) for x in c if x > 0]
    if len(c) == 1:
        return 1.0
    h = [(x, i, (i,)) for i, x in enumerate(c)]
    heapq.heapify(h)
    depth = [0] * len(c)
    k = len(c)
    while len(h) > 1:
        a, _, sa = heapq.heappop(h)
        b, _, sb = heapq.heappop(h)
        for s in sa + sb:
            depth[s] += 1
        heapq.heappush(h, (a + b, k, sa + sb)); k += 1
    return sum(x * d for x, d in zip(c, depth)) / sum(c)


def pages(n):
    return int(-(-n // PAGE) * PAGE)


def main():
    t0 = time.time()
    rng = np.random.default_rng(20261103)
    per_layer, pair_gain, pair_diff = [], [], []
    tot = {k: np.zeros(65536) for k in ("s3", "b3", "b")}
    codes_all, lsb_ones, lsb_n = np.zeros(8), 0, 0
    for L in range(48):
        ly = IX[str(L)]
        ex = sorted(rng.choice(512, 4, replace=False).tolist())
        hist, sb = np.zeros(8), {k: np.zeros(65536) for k in ("s3", "b3", "b")}
        M = []
        with open(D + ly["file"], "rb") as f:
            for e in ex:
                f.seek(e * REC)
                rec = np.frombuffer(f.read(REC), np.uint8)
                mo, ms, lo, ls = ly["msb_piece_off"], ly["msb_piece_size"], ly["lsb_piece_off"], ly["lsb_piece_size"]
                mm = []
                for p in range(NPROJ):
                    m = unpack3(rec[mo[3 * p]:mo[3 * p] + ms[3 * p]], W)
                    hist += np.bincount(m, minlength=8); mm.append(m)
                    for q, nm in ((1, "s3"), (2, "b3")):
                        v = np.frombuffer(rec[mo[3 * p + q]:mo[3 * p + q] + ms[3 * p + q]].tobytes(), np.uint16)
                        sb[nm] += np.bincount(v, minlength=65536)
                    bits = np.unpackbits(rec[lo[2 * p]:lo[2 * p] + ls[2 * p]])
                    lsb_ones += int(bits.sum()); lsb_n += bits.size
                    v = np.frombuffer(rec[lo[2 * p + 1]:lo[2 * p + 1] + ls[2 * p + 1]].tobytes(), np.uint16)
                    sb["b"] += np.bincount(v, minlength=65536)
                M.append(np.concatenate(mm))
        h0 = H(hist)
        for i in range(4):
            for j in range(i + 1, 4):
                jt = np.bincount(M[i].astype(np.int32) * 8 + M[j], minlength=64).reshape(8, 8).astype(float)
                hc = sum(jt[a].sum() / jt.sum() * H(jt[a]) for a in range(8) if jt[a].sum() > 0)
                pair_gain.append(h0 - hc)
                pair_diff.append(H(np.bincount((M[i].astype(np.int16) - M[j]) % 8, minlength=8)) - h0)
        per_layer.append((L, h0, huffman_len(hist), H(sb["s3"]), H(sb["b3"]), H(sb["b"])))
        codes_all += hist
        for k in tot:
            tot[k] += sb[k]
        if L % 12 == 11:
            print(f"  layers to {L} ({time.time() - t0:.0f} s)", flush=True)
    h0, hf = H(codes_all), huffman_len(codes_all)
    hs = {k: H(tot[k]) for k in tot}
    lsb_h = H([lsb_ones, lsb_n - lsb_ones])
    hl = np.array([x[1] for x in per_layer])
    say(f"== M-27 LOSSLESS45 ({time.strftime('%a %H:%M')}): records3, 4 experts x 48 layers ({len(per_layer) * 4} records, "
        f"{len(per_layer) * 4 * REC / 1e9:.2f} GB read)")
    say(f"  3-bit codes: H0 {h0:.4f} b (per layer {hl.min():.4f}-{hl.max():.4f}), Huffman {hf:.4f} b, rANS ~{h0 + RANS_OVH:.4f} b, of 3")
    say(f"  LSB plane: P(1) {lsb_ones / lsb_n:.4f}, entropy {lsb_h:.5f} b of 1")
    say(f"  scales / biases 16-bit entropy: s3 {hs['s3']:.2f}, b3 {hs['b3']:.2f}, b {hs['b']:.2f} of 16 b (fixed-width lossless {FIX_BITS})")
    pg, pd = np.array(pair_gain), np.array(pair_diff)
    say(f"  shared base + delta ({len(pg)} expert pairs, same layer): H0 - H(e | e') mean {pg.mean():+.4f} b (max {pg.max():+.4f}); "
        f"H(e - e' mod 8) - H0 mean {pd.mean():+.3f} b (> 0: deltas cost more than the codes)")
    say("  per layer (H0 codes, Huffman, s3 / b3 / b entropy): " + "; ".join(
        f"L{L} {a:.3f}/{b:.3f}/{c:.1f}/{d:.1f}/{e:.1f}" for L, a, b, c, d, e in per_layer[::6]))
    codes_b = NPROJ * W * 3 / 8
    sc_b = NPROJ * G * 2 * 2                                     # s3 + b3 bf16
    lsbp_b, lsbb_b = NPROJ * W / 8, NPROJ * G * 2
    fmts = {"today": (MSB_SZ, LSB_SZ),
            "fixed-width scales (11 b)": (pages(codes_b + sc_b * FIX_BITS / 16), pages(lsbp_b + lsbb_b * FIX_BITS / 16)),
            "entropy floor (rANS + coded scales)": (pages(NPROJ * W * (h0 + RANS_OVH) / 8 + NPROJ * G * (hs["s3"] + hs["b3"]) / 8),
                                                    pages(NPROJ * W * min(1.0, lsb_h + RANS_OVH) / 8 + NPROJ * G * hs["b"] / 8))}
    say("  format: MSB / LSB part bytes; SSD MB per kept token at C 256 (code / IFEval / multi) and the saving")
    res = {}
    for name, (m, l) in fmts.items():
        v = {d: (bm * m + bl * l) / (bm * MSB_SZ + bl * LSB_SZ) * mbk for d, (bm, bl, mbk) in READS.items()}
        res[name] = v
        say(f"    {name:36s} {m:9d} / {l:7d} | {v['code']:5.1f} / {v['ifeval']:5.1f} / {v['multi']:5.1f} | "
            f"{100 * (1 - v['code'] / READS['code'][2]):5.1f} / {100 * (1 - v['ifeval'] / READS['ifeval'][2]):5.1f} / "
            f"{100 * (1 - v['multi'] / READS['multi'][2]):5.1f} %")
    bm, bl, mbk = READS["code"]
    m_e, l_e = fmts["entropy floor (rANS + coded scales)"]
    saved_mb = bm * (MSB_SZ - m_e) / 1e6 + bl * (LSB_SZ - l_e) / 1e6
    dec_ms = bm * NPROJ * W * (h0 + RANS_OVH) / 8 / 1e6 / GPU_GBS
    ssd_ms = saved_mb / SSD_GBS
    g = 1 - res["entropy floor (rANS + coded scales)"]["code"] / mbk
    say(f"  code, per check: {saved_mb:.1f} MB fewer SSD bytes = {ssd_ms:.2f} ms at {SSD_GBS:.0f} GB/s; GPU rANS decode of the missed "
        f"codes {dec_ms:.2f} ms at {GPU_GBS:.0f} GB/s -> net {ssd_ms - dec_ms:+.2f} ms per check if the decode sits on the GPU's path")
    say(f"  -> bytes: {100 * g:.1f} % saved on code -> " + ("PASS" if g >= 0.12 else "KILL" if g < 0.08 else "BETWEEN")
        + f"; time: " + ("net gain" if ssd_ms > dec_ms else "no net gain (decode >= SSD time saved)")
        + f"  ({time.time() - t0:.0f} s)")
    open(os.path.join(os.path.expanduser("~/MoonShots/data"), "lossless45.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
