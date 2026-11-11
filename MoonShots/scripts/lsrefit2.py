"""Does a per-group least-squares refit of the truncated tiers' scale and bias beat the stored truncation (scale 2^shift * s,
bias b + s * mean(q & low)) on the 125B's real experts? Weight-space squared error against the 4-bit expert w = s q + b,
per tier (3: m = q >> 1, 2: m = q >> 2), refit alpha, beta rounded to bf16 as they would be stored. Also the records4
cost: a refit P2 scale is no longer 4 s exactly, so tiers 3/4 would need their own stored scales.
usage: lsrefit2.py [layers, default 0,24,47] [experts per layer, default 32]   (CPU only, reads records3)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "patches", "planes_engine"))
import make_records3 as M

layers = [int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "0,24,47").split(",")]
NE = int(sys.argv[2]) if len(sys.argv) > 2 else 32
lay = M.layout()
R3 = lay["recsz"]
mo, ms, lo, ls = lay["msb_piece_off"], lay["msb_piece_size"], lay["lsb_piece_off"], lay["lsb_piece_size"]


def bf16(a):
    return M.bf16_to_f32(M.f32_to_bf16(a.astype(np.float32)))


tot = {n: np.zeros(4) for n in (3, 2)}                            # [trunc err, refit err, signal, groups with var 0]
for L in layers:
    f = M.open_nocache(os.path.expanduser(f"~/BigMoE/models/bare/records3/L{L:02d}.bin"), os.O_RDONLY)
    buf = M.aligned(NE * R3)
    M.pread_into(f, buf, 0)
    os.close(f)
    for i, (out, K) in enumerate(M.SHAPES):
        words = M.merge_codes(M.view(buf, NE, R3, mo[3 * i], ms[3 * i], np.uint8, inner=3), M.view(buf, NE, R3, lo[2 * i], ls[2 * i], np.uint8))
        q = ((words.reshape(NE, out, K // 8)[..., None] >> (4 * np.arange(8, dtype=np.uint32))) & 15).reshape(NE, out, K // 64, 64).astype(np.float32)
        s = (M.bf16_to_f32(M.view(buf, NE, R3, mo[3 * i + 1], ms[3 * i + 1], np.uint16)) * 0.5).reshape(NE, out, K // 64, 1)
        b = M.bf16_to_f32(M.view(buf, NE, R3, lo[2 * i + 1], ls[2 * i + 1], np.uint16)).reshape(NE, out, K // 64, 1)
        w = s * q + b
        sig = ((w - w.mean(-1, keepdims=True)) ** 2).sum()
        for n in (3, 2):
            sh = 4 - n
            m = np.floor(q / 2 ** sh)
            low = q - m * 2 ** sh
            wt = bf16(s * 2 ** sh) * m + bf16(b + s * low.mean(-1, keepdims=True))
            mm, wm = m.mean(-1, keepdims=True), w.mean(-1, keepdims=True)
            var = ((m - mm) ** 2).sum(-1, keepdims=True)
            alpha = np.where(var > 0, ((m - mm) * (w - wm)).sum(-1, keepdims=True) / np.maximum(var, 1e-30), s * 2 ** sh)
            alpha = bf16(alpha)
            beta = bf16(wm - alpha * mm)
            wl = alpha * m + beta
            et, el = ((wt - w) ** 2).sum(), ((wl - w) ** 2).sum()
            tot[n] += [et, el, sig, (var == 0).sum()]
            r = (alpha / (s * 2 ** sh))[var > 0]
            print(f"L{L:02d} {('gate', 'up', 'down')[i]:4s} tier {n}: rel err trunc {np.sqrt(et / sig):.4f} refit {np.sqrt(el / sig):.4f} "
                  f"(MSE x{el / et:.3f}); refit scale / stored scale median {np.median(r):.3f} [p10 {np.percentile(r, 10):.3f}, "
                  f"p90 {np.percentile(r, 90):.3f}]", flush=True)
for n in (3, 2):
    et, el, sig, z = tot[n]
    print(f"ALL tier {n}: rel err trunc {np.sqrt(et / sig):.4f} -> refit {np.sqrt(el / sig):.4f}: MSE x{el / et:.3f} "
          f"({100 * (1 - el / et):.1f} % less); constant-code groups {int(z)}", flush=True)
