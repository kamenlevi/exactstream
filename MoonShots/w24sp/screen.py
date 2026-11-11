"""W24-sp step (2) screen (NewIdeas GO Mon 08:33; Kamen's rapid screening: one run <= 10 min). 11-bit planar scales /
biases (w24sp/tg11.py) against today's 3c on real records3 experts of layers 24, 38, 47 (38 / 47 hold the span-7 pieces).
PASS/KILL (written first): PASS = 0 mismatching output bits at S = 1 / 4 / 16 rows on all 3 layers (today's plan
4433333333 with and without the fused combine, plus 4444444444 / 3333333333 passes that put every expert at a 4-bit and
a 3-bit rank) AND expert-kernel time (3c call, A + BC, layer 24, real routing) <= +2 % at 16 rows; KILL = any mismatch
or > +5 %; between = NewIdeas' call.
Converter (CPU): per piece assert no zero / subnormal / inf / nan and exponent span <= 7, pack, unpack, compare bitwise;
then the 3 layers are written as converted record files (scratch, deleted at the end) and the 11-bit pools are loaded
FROM those files; record sizes are reported from the files.
Timing as Leap's b10_trellis.py (read only): N_CH calls chained by a data dependence, cycling over 6 real-routing
passes (S consecutive tokens of the expA code trace), one eval, minus the bare chain; arms interleaved per round (today,
11-bit, today again = A/A noise), PAUSE checked before each round, a paused round is redone.
usage: [BigMoE venv] screen.py dev | full   -> stdout + ~/MoonShots/data/w24sp_screen.txt (full)"""
import json, os, sys, time
import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tg11
TG = tg11.TG

MODE = sys.argv[1] if len(sys.argv) > 1 else "dev"
D3 = os.path.expanduser("~/BigMoE/models/bare/records3/")
IX = json.load(open(D3 + "index.json"))["layers"]
SCR = os.environ.get("W24SP_SCRATCH", "/private/tmp/claude-501/-Users-kamen-MoonShots/4311ccb7-fdc5-4dac-ace8-b31665971f5f/scratchpad/w24sp")
OUT = os.path.expanduser("~/MoonShots/data/w24sp_screen.txt")
TR = os.path.expanduser("~/Breakthrough/data/expA/traces125/")
XL = {24: 24, 38: 36, 47: 46}                               # Fit12's captured MoE inputs of the nearest layer
LAYERS = (24, 38, 47) if MODE == "full" else (47,)
G, PB, SBW, SBL, SBB = tg11.G, tg11.PB, tg11.SBW, tg11.SBL, tg11.SBB
W3, L1 = 614400, 204800
BLK = 16384
MSB11 = -(-3 * (W3 + SBW) // BLK) * BLK                     # 2,064,384
LSB11 = -(-3 * (L1 + SBL) // BLK) * BLK                     # 720,896
REC11 = MSB11 + LSB11
SH = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
N_CH = 30
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def pack3(b):                                               # (P, n) offsets < 8 -> (P, 3n/8) MLX 3-bit packing
    w = (b.reshape(len(b), -1, 8).astype(np.uint32) << (3 * np.arange(8, dtype=np.uint32))).sum(2, dtype=np.uint32)
    return np.stack([w & 0xFF, (w >> 8) & 0xFF, w >> 16], 2).astype(np.uint8).reshape(len(b), -1)


def unpack3(p):
    t = p.reshape(len(p), -1, 3).astype(np.uint32)
    w = t[:, :, 0] | (t[:, :, 1] << 8) | (t[:, :, 2] << 16)
    return ((w[:, :, None] >> (3 * np.arange(8, dtype=np.uint32))) & 7).astype(np.uint8).reshape(len(p), -1)


def pack11(v):                                              # v (E, G) bf16 bits -> A (E, G), B (E, 3G/8), base (E,), span
    e = ((v >> 7) & 0xFF).astype(np.int16)
    assert ((v & 0x7FFF) != 0).all(), "zero scale / bias"
    assert ((e != 0) & (e != 255)).all(), "subnormal / inf / nan"
    e0 = e.min(1)
    span = e.max(1) - e0
    assert (span <= 7).all(), f"span {span.max()} > 7"
    a = (((v >> 8) & 0x80) | (v & 0x7F)).astype(np.uint8)
    b = pack3((e - e0[:, None]).astype(np.uint8))
    v2 = ((a.astype(np.uint16) & 0x80) << 8) | ((unpack3(b).astype(np.uint16) + e0[:, None].astype(np.uint16)) << 7) | \
        (a.astype(np.uint16) & 0x7F)
    assert np.array_equal(v2, v), "round trip"
    return a, b, e0.astype(np.uint8), span


def convert(L):
    """records3 layer L -> 11-bit records (E, REC11) uint8; returns (orig raw, new, span stats)"""
    ix = IX[str(L)]
    E, rec = ix["E"], ix["recsz"]
    mo, ms, lo, ls = ix["msb_piece_off"], ix["msb_piece_size"], ix["lsb_piece_off"], ix["lsb_piece_size"]
    raw = np.fromfile(D3 + ix["file"], dtype=np.uint8).reshape(E, rec)
    new = np.zeros((E, REC11), np.uint8)
    spans = {}
    sb16 = lambda o: np.ascontiguousarray(raw[:, o:o + 2 * G]).view(np.uint16)
    for p, pr in enumerate(TG.PROJ):
        assert ms[3 * p] == W3 and ms[3 * p + 1] == ms[3 * p + 2] == 2 * G and ls[2 * p] == L1 and ls[2 * p + 1] == 2 * G
        m0 = p * (W3 + SBW)
        new[:, m0:m0 + W3] = raw[:, mo[3 * p]:mo[3 * p] + W3]
        for t, nm in ((0, "s3"), (1, "b3")):
            a, b, e0, spans[(nm, pr)] = pack11(sb16(mo[3 * p + 1 + t]))
            q = m0 + W3 + t * PB
            new[:, q:q + G], new[:, q + G:q + PB], new[:, m0 + W3 + SBB + t] = a, b, e0
        l0 = MSB11 + p * (L1 + SBL)
        new[:, l0:l0 + L1] = raw[:, lo[2 * p]:lo[2 * p] + L1]
        a, b, e0, spans[("b", pr)] = pack11(sb16(lo[2 * p + 1]))
        new[:, l0 + L1:l0 + L1 + G], new[:, l0 + L1 + G:l0 + L1 + PB], new[:, l0 + L1 + PB] = a, b, e0
    return raw, new, spans


def pools_today(raw, ix):                                   # Leap's load_3c (b10_trellis.py), as BigMoE v3 lays the pools
    E = raw.shape[0]
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
    mx.eval([v for d in (msb, lsb) for pr in d for v in d[pr].values()])
    return msb, lsb


def pools_11(new):
    E = new.shape[0]
    msb, lsb = {}, {}
    for p, pr in enumerate(TG.PROJ):
        o, i = SH[pr]
        m0, l0 = p * (W3 + SBW), MSB11 + p * (L1 + SBL)
        msb[pr] = {"w3": mx.array(np.ascontiguousarray(new[:, m0:m0 + W3]).view(np.uint32).reshape(E, o, i * 3 // 32)),
                   "sb": mx.array(np.ascontiguousarray(new[:, m0 + W3:m0 + W3 + SBW]))}
        lsb[pr] = {"l1": mx.array(np.ascontiguousarray(new[:, l0:l0 + L1]).view(np.uint32).reshape(E, o, i // 32)),
                   "sb": mx.array(np.ascontiguousarray(new[:, l0 + L1:l0 + L1 + SBL]))}
    mx.eval([v for d in (msb, lsb) for pr in d for v in d[pr].values()])
    return msb, lsb


def trace(L):                                               # expA traces: top-10 of the rank-ordered top-20, 4 sets
    ids, ws = [], []
    for s in ("code", "chat", "math", "bulgarian"):
        z = np.load(TR + f"{s}_800_g.npz")
        i, w = z[f"L{L}"][:, :10].astype(np.int32), z[f"G{L}"][:, :10].astype(np.float32)
        ids.append(i); ws.append(w / w.sum(1, keepdims=True))
    return ids, ws


def r4_of(plan):
    return plan.count("4")


def bitwise(L, X, T, M, T11, M11, rng):
    """mismatching output elements (bf16 bit patterns) of tier_glu11 vs tier_glu, per S; and the number of calls"""
    ids_sets, ws_sets = trace(L)
    res = {}
    for S in (1, 4, 16):
        cases = []
        npass = -(-512 // (10 * S))
        for plan in ("4444444444", "3333333333", "4433333333"):
            perm = rng.permutation(512)
            for t in range(npass):
                ii = np.take(perm, np.arange(t * 10 * S, (t + 1) * 10 * S), mode="wrap").reshape(S, 10)
                ww = rng.dirichlet(np.ones(10), S).astype(np.float32)
                cases.append((plan, ii, ww))
        for k in range(8):                                  # real routing, today's plan
            st = rng.integers(0, 4)
            a = rng.integers(0, len(ids_sets[st]) - S)            # chat / math traces are shorter than 800
            cases.append(("4433333333", ids_sets[st][a:a + S], ws_sets[st][a:a + S]))
        bad, calls, pending = 0, 0, []
        for plan, ii, ww in cases:
            x = mx.array(X[rng.integers(0, len(X) - S) + np.arange(S)]).astype(mx.bfloat16)
            i = mx.array(ii.astype(np.int32))
            w = mx.array(ww).astype(mx.bfloat16)
            r4 = r4_of(plan)
            li = i[:, :r4] if r4 else None
            for sc in (None, w):
                kw = {} if sc is None else {"scores": sc}
                y0 = TG.tier_glu(x, T, M, i, li, plan=plan, **kw)
                y1 = tg11.tier_glu11(x, T11, M11, i, li, plan=plan, **kw)
                pending.append((y0.view(mx.uint16) != y1.view(mx.uint16)).sum())
                calls += 1
            if len(pending) >= 64:
                mx.eval(pending); bad += sum(int(v) for v in pending); pending = []
        mx.eval(pending); bad += sum(int(v) for v in pending)
        res[S] = (bad, calls)
    return res


def passes(ids, ws, S, n=6, seed=0):
    starts = np.random.default_rng(seed).choice(ids.shape[0] // S, n, replace=False) * S
    return [(mx.array(ids[a:a + S]), mx.array(ws[a:a + S]).astype(mx.bfloat16), len(np.unique(ids[a:a + S]))) for a in starts]


def chain(g, x, ps):
    y = x
    for i in range(N_CH):
        ii, ww, _ = ps[i % len(ps)]
        y = y + g(y, ii, ww) * 0
    return y


def one_round(f, x, ps):
    while os.path.exists(PAUSE):
        time.sleep(0.2)
    a = time.perf_counter(); mx.eval(chain(f, x, ps)); t = time.perf_counter() - a
    b = time.perf_counter(); mx.eval(chain(lambda y, i, w: y, x, ps)); t0 = time.perf_counter() - b
    return None if os.path.exists(PAUSE) or t + t0 > 5 else (t - t0) / N_CH * 1e3


def bench(L, X, T, M, T11, M11, rounds):
    ids_sets, ws_sets = trace(L)
    arms = [("today", lambda y, i, w: TG.tier_glu(y, T, M, i, i[:, :2], plan="4433333333", scores=w)),
            ("11-bit", lambda y, i, w: tg11.tier_glu11(y, T11, M11, i, i[:, :2], plan="4433333333", scores=w)),
            ("today again", lambda y, i, w: TG.tier_glu(y, T, M, i, i[:, :2], plan="4433333333", scores=w))]
    out = {}
    for S in (1, 4, 16):
        ps = passes(ids_sets[0], ws_sets[0], S)
        x = mx.array(X[:S]).astype(mx.bfloat16)
        for _, f in arms:
            for _ in range(2):
                mx.eval(chain(f, x, ps))
        ms = {nm: [] for nm, _ in arms}
        for r in range(rounds):
            order = arms if r % 2 == 0 else arms[::-1]       # alternate the order: no position bias
            for nm, f in order:
                v = None
                while v is None:
                    v = one_round(f, x, ps)
                ms[nm].append(v)
        med = {nm: float(np.median(v)) for nm, v in ms.items()}
        pr = np.array(ms["11-bit"]) / np.array(ms["today"])
        aa = np.array(ms["today again"]) / np.array(ms["today"])
        out[S] = (med, float(np.median(pr)), float(np.median(aa)), np.mean([p[2] for p in ps]))
        say(f"  S {S:2d} (distinct experts {out[S][3]:5.1f}): today {med['today']:.3f} ms, 11-bit {med['11-bit']:.3f} ms, "
            f"today again {med['today again']:.3f} ms per call | paired 11-bit / today median {out[S][1]:.4f} "
            f"({100 * (out[S][1] - 1):+.2f} %), A/A {aa.min():.4f}..{aa.max():.4f} (median {out[S][2]:.4f}), {rounds} rounds")
    return out


def main():
    t_all = time.time()
    rng = np.random.default_rng(0)
    say(f"W24-sp step (2) {MODE} ({time.strftime('%a %d %b %H:%M:%S')}), MLX {mx.__version__}, layers {LAYERS}; scratch {SCR}")
    say("PASS = 0 mismatching output bits at S = 1 / 4 / 16 on all layers AND 11-bit expert-kernel time <= +2 % at 16 rows; "
        "KILL = any mismatch or > +5 %; between = NewIdeas")
    os.makedirs(SCR, exist_ok=True)
    allbad, kept = 0, None
    try:
        for L in LAYERS:
            while os.path.exists(PAUSE):
                time.sleep(1)
            t0 = time.time()
            raw, new, spans = convert(L)
            f = os.path.join(SCR, f"L{L}_11bit.bin")
            new.tofile(f)
            E = raw.shape[0]
            del new
            sz = os.path.getsize(f)
            new = np.fromfile(f, dtype=np.uint8).reshape(E, -1)
            sp = {k: int(v.max()) for k, v in spans.items()}
            n7 = sum(int((v == 7).sum()) for v in spans.values())
            say(f"L{L}: converted {E} experts in {time.time() - t0:.1f} s (asserts + bitwise round trip on all {9 * E} pieces "
                f"passed); span max s3 {max(sp[('s3', p)] for p in TG.PROJ)} b3 {max(sp[('b3', p)] for p in TG.PROJ)} b "
                f"{max(sp[('b', p)] for p in TG.PROJ)}, span-7 pieces {n7} | file {sz:,} B = {sz // E:,} B per expert "
                f"(records3 {IX[str(L)]['recsz']:,}): MSB part {MSB11:,} (today {IX[str(L)]['msb_sz']:,}), LSB part "
                f"{LSB11:,} (today {IX[str(L)]['lsb_sz']:,})")
            T, M = pools_today(raw, IX[str(L)])
            T11, M11 = pools_11(new)
            del raw, new
            X = np.load(os.path.expanduser(f"~/Fit12/data/q2/x_L{XL[L]}.npy"), mmap_mode="r")
            t1 = time.time()
            res = bitwise(L, X, T, M, T11, M11, rng)
            for S, (bad, calls) in res.items():
                allbad += bad
            say(f"L{L}: bitwise " + ", ".join(f"S {S}: {bad} mismatching elements in {calls} calls" for S, (bad, calls) in res.items())
                + f" ({time.time() - t1:.1f} s)")
            if L == 24 and MODE == "full":
                kept = (X, T, M, T11, M11)
            else:
                del T, M, T11, M11
            os.remove(f)
        if MODE == "full" and kept is not None:
            say(f"timing, layer 24, plan 4433333333 + fused combine (A + BC), N_CH {N_CH}:")
            out = bench(24, *kept, rounds=int(os.environ.get("W24SP_ROUNDS", "15")))
            r16 = out[16][1]
            verdict = "KILL (mismatch)" if allbad else ("PASS" if r16 <= 1.02 else ("KILL" if r16 > 1.05 else "BETWEEN"))
            say(f"VERDICT: {verdict}: mismatches {allbad}, 16-row expert kernel {100 * (r16 - 1):+.2f} %")
        else:
            say(f"dev: mismatches {allbad}")
    finally:
        for fn in os.listdir(SCR):
            if fn.endswith("_11bit.bin"):
                os.remove(os.path.join(SCR, fn))
        say(f"scratch cleaned; total {time.time() - t_all:.0f} s; end {time.strftime('%H:%M:%S')}")
        if MODE == "full":
            open(OUT, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
