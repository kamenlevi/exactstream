# PUSH-A exposure replay (Fable A, CPU only): per-layer SSD bursts under an LRU at M-17's C 256 sizes on BigMoE's cap45
# captured ids (16-row checks as captured), then exposed = sum over layers of max(0, burst - cover) under the fitted burst
# model (0.28 ms per loader call with misses + MB / 12.7 GB/s) for today, D1 (one call per layer at 0.10 ms), D2 (issue one
# layer early at recall r, precision 0.8) and D1 + D2. Reads only ids / seq / L / rows / row_off from the npz (no x, no routers).
import zipfile, io, json, numpy as np, sys, time
CAP = "/Users/kamen/BigMoE/data/cap45/"
ORDER = ["humaneval_HumanEval_68", "math500_test_prealgebra_1640", "humaneval_HumanEval_91", "math500_test_algebra_694",
         "math500_test_algebra_1282", "gsm8k_708"]
SZ = json.load(open("/Users/kamen/BigMoE/data/seed45/laysz45_256.json"))
Cm, Cl = SZ["C_m"], SZ["C_l"]
MB_B, MB_S = 2162688 / 1e6, 770048 / 1e6
def npy(z, n):
    return np.load(io.BytesIO(z.open(n).read()), allow_pickle=False)
t0 = time.time()
checks_all = []                               # list of (prompt, seq, {L: ids[rows,10]})
for name in ORDER:
    z = zipfile.ZipFile(CAP + name + "_gate_0.npz")
    seq, Lz, rows, roff, ids = (npy(z, k) for k in ("seq.npy", "L.npy", "rows.npy", "row_off.npy", "ids.npy"))
    ch = {}
    for s, l, r, ro in zip(seq, Lz, rows, roff):
        c = ch.setdefault(int(s), {})
        if int(l) in c: continue              # the commit replay
        c[int(l)] = ids[int(ro):int(ro) + int(r)]
    NL = max(max(c) for c in ch.values()) + 1
    for s in sorted(ch):
        if len(ch[s]) == NL: checks_all.append((name, s, ch[s]))
print(f"checks {len(checks_all)} over {len(ORDER)} prompts, NL {NL}, sizes C_m sum {sum(Cm)} C_l sum {sum(Cl)}; load {time.time()-t0:.1f}s")
# LRU per layer per pool; chained over the whole run in prompt order (as cachesize45 / replay_c1 do); warm start: the first
# prompt's first 2 checks are skipped from the stats
from collections import OrderedDict
base = [OrderedDict() for _ in range(NL)]; side = [OrderedDict() for _ in range(NL)]
def touch(pool, cap, keys):
    miss = 0
    for k in keys:
        if k in pool: pool.move_to_end(k)
        else:
            miss += 1; pool[k] = 1
            if len(pool) > cap: pool.popitem(last=False)
    return miss
per_check = []                                # per check: arrays over layers of (base_miss, side_miss, rows)
for ci, (name, s, c) in enumerate(checks_all):
    bm = np.zeros(NL, int); sm = np.zeros(NL, int); rw = np.zeros(NL, int)
    for L in range(NL):
        I = c[L]; rw[L] = len(I)
        ub = np.unique(I).tolist(); us = np.unique(I[:, :2]).tolist()
        bm[L] = touch(base[L], Cm[L], ub); sm[L] = touch(side[L], Cl[L], us)
    per_check.append((name, bm, sm, rw))
warm = per_check[2:]
bmA = np.array([p[1] for p in warm]); smA = np.array([p[2] for p in warm])
mb = bmA * MB_B + smA * MB_S
print(f"LRU at M-17 C 256 (plain LRU, no prior): misses per check base {bmA.sum(1).mean():.0f} + side {smA.sum(1).mean():.0f} = "
      f"{(bmA+smA).sum(1).mean():.0f}; MB per check {mb.sum(1).mean():.0f}; calls with misses per check "
      f"{((bmA>0).sum(1)+(smA>0).sum(1)).mean():.1f} (layers with any miss {((bmA+smA)>0).sum(1).mean():.1f})")
print("per-layer mean misses (base+side), layers 0..47:", np.round((bmA+smA).mean(0), 1).tolist())
# scale to the live median check (171 misses, 274 MB at C 256 with the prior and the t 0.3 cut): one factor on bytes per layer
live = 171 / (bmA + smA).sum(1).mean()
print(f"scale to the live median check: x{live:.2f} on misses")
def exposed(fixed_ms, joint, r, cover, pieces=1.0):
    """per check: sum over layers of max(0, burst - cover); r = fraction of each layer's misses issued one layer early
    (the forecast, precision 0.8 -> x1.25 bytes), served in a window of cover + 1.0 ms; pieces: x on the lone-read part."""
    out = []; busy = []
    for bm_, sm_ in zip(bmA, smA):
        e = 0.0; b = 0.0
        for L in range(NL):
            nb, ns = bm_[L] * live, sm_[L] * live
            calls = (1 if nb > 0 else 0) + (1 if ns > 0 else 0)
            if joint and calls == 2: calls = 1
            byt = (nb * MB_B + ns * MB_S) / 12.7            # ms at 12.7 GB/s
            dem = (1 - r) * byt + fixed_ms * calls * pieces
            fc = r * byt * 1.25
            b += dem + fc
            # the forecast reads run in the previous layer's window (phase 2 + dense ~1.0 ms) on top of this layer's cover
            e += max(0.0, dem - cover) + max(0.0, fc + (0.1 if r > 0 else 0) - (cover + 1.0))
        out.append(e); busy.append(b)
    return np.mean(out), np.mean(busy)
for cover_ms, tag in ((0.75, "16-row cover 0.75 ms"), (0.50, "live ~10-row cover 0.50 ms")):
    print(f"\n== exposure per check, {tag} ==")
    for lab, kw in (("today (0.28/call, two rows)", dict(fixed_ms=0.28, joint=False, r=0.0)),
                    ("D1a JOINT only", dict(fixed_ms=0.28, joint=True, r=0.0)),
                    ("D1 all (JOINT + 0.10/call)", dict(fixed_ms=0.10, joint=True, r=0.0)),
                    ("D2 alone r 0.85", dict(fixed_ms=0.28, joint=False, r=0.85)),
                    ("D2 alone r 0.50 (L-input forecast)", dict(fixed_ms=0.28, joint=False, r=0.50)),
                    ("D1 + D2 r 0.85", dict(fixed_ms=0.10, joint=True, r=0.85)),
                    ("D1 + D2 r 0.50", dict(fixed_ms=0.10, joint=True, r=0.50)),
                    ("D1 + D2 r 1.0 (oracle)", dict(fixed_ms=0.10, joint=True, r=1.0))):
        e, b = exposed(cover=cover_ms, **kw)
        print(f"  {lab:38s} busy {b:5.1f} ms  exposed {e:5.1f} ms per check")
print(f"\ntotal {time.time()-t0:.1f}s")

# chat-like cover (4 rows ~0.25 ms per layer) and the night point (C 320: bytes x0.61 of C 256 [K cachesize45 37.9/62.4])
print("\n== chat-like cover 0.25 ms (bursts scaled to the code median: chat's bursts are smaller, so read the ratios) ==")
for lab, kw in (("today", dict(fixed_ms=0.28, joint=False, r=0.0)), ("D1 all", dict(fixed_ms=0.10, joint=True, r=0.0)),
                ("D1 + D2 r 0.85", dict(fixed_ms=0.10, joint=True, r=0.85)), ("D1 + D2 r 0.50", dict(fixed_ms=0.10, joint=True, r=0.50))):
    e, b = exposed(cover=0.25, **kw); print(f"  {lab:20s} busy {b:5.1f} exposed {e:5.1f} ms per check")
live_save = live; live = live * 37.9 / 62.4
print("\n== night C 320 (bytes x0.61), cover 0.50 ==")
for lab, kw in (("today", dict(fixed_ms=0.28, joint=False, r=0.0)), ("D1 all", dict(fixed_ms=0.10, joint=True, r=0.0)),
                ("D2 alone r 0.85", dict(fixed_ms=0.28, joint=False, r=0.85)), ("D1 + D2 r 0.85", dict(fixed_ms=0.10, joint=True, r=0.85))):
    e, b = exposed(cover=0.50, **kw); print(f"  {lab:20s} busy {b:5.1f} exposed {e:5.1f} ms per check")
live = live_save

# ---- rev 3 (Tue 22:5x): M-30 QUEUEMIX / M-29 SSDMICRO2 pricing [K ~/MoonShots/data/ssdmicro3.txt, ssdmicro2.txt] ----
# Forecasts are CHAINED: issued the moment L's demand burst returns, they read like a back-to-back call (their own first
# byte ~0.12 ms + bytes), and they must finish before L+1's row (window = max(dem, cover) + 1.0 ms); what does not fit
# lands on L+1's demand at fair share (M-30: +165 us per unfinished record), i.e. the excess is exposed. The demand burst
# keeps its first byte whatever is in flight (M-30); the wake is the keepalive's / ping's (inside fixed_ms). 'mixed' =
# the charter's wording (forecasts issued with the demand): the demand is extended by the forecast bytes (M-30 M arms).
def exposed2(fixed_ms, joint, r, cover, mixed=False, lead_fixed=0.12):
    out = []; busy = []; nb_ = []
    for bm_, sm_ in zip(bmA, smA):
        e = 0.0; b = 0.0; nburst = 0
        for L in range(NL):
            nb, ns = bm_[L] * live, sm_[L] * live
            calls = (1 if nb > 0 else 0) + (1 if ns > 0 else 0)
            if joint and calls == 2: calls = 1
            nburst += calls
            byt = (nb * MB_B + ns * MB_S) / 12.7
            dem = (1 - r) * byt + fixed_ms * calls
            fc = (r * byt * 1.25 + lead_fixed) if (r > 0 and byt > 0) else 0.0
            b += dem + fc
            if mixed:
                e += max(0.0, dem + fc - cover)
            else:
                window = max(dem, cover) + 1.0
                e += max(0.0, dem - cover) + max(0.0, dem + fc - window)
        out.append(e); busy.append(b); nb_.append(nburst)
    return np.mean(out), np.mean(busy), np.mean(nb_)
ARMS3 = (("today (0.283 per call, two rows)", dict(fixed_ms=0.283, joint=False, r=0.0)),
         ("S1 JOINT (2 x 0.283 - 0.120)", dict(fixed_ms=0.446, joint=True, r=0.0)),
         ("+ S2 keepalive / ping (-0.044)", dict(fixed_ms=0.402, joint=True, r=0.0)),
         ("+ S3 zero-copy A (-0.096: M-28's outside part)", dict(fixed_ms=0.306, joint=True, r=0.0)),
         ("+ S3 zero-copy B (-0.173: M-29's 2-record saving)", dict(fixed_ms=0.229, joint=True, r=0.0)),
         ("D1 floor (0.16)", dict(fixed_ms=0.16, joint=True, r=0.0)),
         ("D1 A (0.306) + D2 r 0.85 chained", dict(fixed_ms=0.306, joint=True, r=0.85)),
         ("D1 A (0.306) + D2 r 0.50 chained", dict(fixed_ms=0.306, joint=True, r=0.50)),
         ("D1 B (0.229) + D2 r 0.85 chained", dict(fixed_ms=0.229, joint=True, r=0.85)),
         ("D1 floor + D2 r 0.85 chained", dict(fixed_ms=0.16, joint=True, r=0.85)),
         ("D1 A + D2 r 1.0 chained (oracle)", dict(fixed_ms=0.306, joint=True, r=1.0)),
         ("D1 A + D2 r 0.85 MIXED (charter's 'join the open burst')", dict(fixed_ms=0.306, joint=True, r=0.85, mixed=True)),
         ("D1 A + D2 r 0.50 MIXED", dict(fixed_ms=0.306, joint=True, r=0.50, mixed=True)))
for cover_ms, tag in ((0.50, "live ~10-row cover 0.50 ms"), (0.25, "chat-like cover 0.25 ms")):
    print(f"\n== rev 3 (M-30 chained pricing), {tag} ==")
    for lab, kw in ARMS3:
        e, b, nbu = exposed2(cover=cover_ms, **kw)
        print(f"  {lab:58s} bursts {nbu:4.1f}  fixed {kw['fixed_ms']*nbu:5.1f}  busy {b:5.1f}  exposed {e:5.1f} ms per check")

# chained AND capped (the engine's sizing rule): L+1's forecast is cut to the window left after L's demand burst
# (max(dem, cover) + 1.0 - dem - its first byte 0.12 - a 0.10 ms margin, so the next demand never lands on unfinished
# forecasts, M-30); the records that do not fit stay demand misses of L+1 (read under L+1's cover, no precision overhead).
def exposed3(fixed_ms, joint, r, cover, margin=0.10, lead_fixed=0.12, prec=0.8):
    out = []; busy = []; ffit = []
    for bm_, sm_ in zip(bmA, smA):
        e = 0.0; b = 0.0; fsum = 0.0; fwant = 0.0
        for L in range(NL):
            nb, ns = bm_[L] * live, sm_[L] * live
            calls = (1 if nb > 0 else 0) + (1 if ns > 0 else 0)
            if joint and calls == 2: calls = 1
            byt = (nb * MB_B + ns * MB_S) / 12.7
            if byt <= 0: continue
            dem = (1 - r) * byt + fixed_ms * calls
            want = r * byt / prec
            avail = max(0.0, (max(dem, cover) + 1.0) - dem - lead_fixed - margin)
            fit = min(want, avail)
            dem += (want - fit) * prec
            fc = (fit + lead_fixed) if fit > 0 else 0.0
            b += dem + fc; fsum += fit; fwant += want
            e += max(0.0, dem - cover)
        out.append(e); busy.append(b); ffit.append(fsum / fwant if fwant > 0 else 1.0)
    return np.mean(out), np.mean(busy), np.mean(ffit)
ARMS4 = (("D1 A (0.306) + D2 r 0.85 capped", dict(fixed_ms=0.306, joint=True, r=0.85)),
         ("D1 A (0.306) + D2 r 0.50 capped", dict(fixed_ms=0.306, joint=True, r=0.50)),
         ("D1 B (0.229) + D2 r 0.85 capped", dict(fixed_ms=0.229, joint=True, r=0.85)),
         ("D1 B (0.229) + D2 r 0.50 capped", dict(fixed_ms=0.229, joint=True, r=0.50)),
         ("D1 floor (0.16) + D2 r 0.85 capped", dict(fixed_ms=0.16, joint=True, r=0.85)),
         ("D1 A + D2 r 1.0 capped (oracle)", dict(fixed_ms=0.306, joint=True, r=1.0)),
         ("D1 A + D2 r 0.85 capped, precision 1.0", dict(fixed_ms=0.306, joint=True, r=0.85, prec=1.0)),
         ("S2 only (0.402) + D2 r 0.85 capped (no zero-copy)", dict(fixed_ms=0.402, joint=True, r=0.85)))
for cover_ms, tag in ((0.50, "live ~10-row cover 0.50 ms"), (0.25, "chat-like cover 0.25 ms"), (0.75, "16-row cover 0.75 ms")):
    print(f"\n== rev 3 chained + capped (sizing rule), {tag} ==")
    for lab, kw in ARMS4:
        e, b, f = exposed3(cover=cover_ms, **kw)
        print(f"  {lab:52s} busy {b:5.1f}  exposed {e:5.1f} ms per check  forecast bytes that fit {100*f:4.0f} %")
