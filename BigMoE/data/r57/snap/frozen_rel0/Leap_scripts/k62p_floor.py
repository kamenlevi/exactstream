"""C2 + B3 (Leap, Mon 2026-11-02; DAY9 conditions 4 and 9; quiet GPU): the expert kernel against its byte floor at 16 rows
(C2) and the expert part of the per-row cost from 16 to 32 rows (B3's NAX32 beta), on real 125B layers (records3 pools,
slot = expert id) with real routing (Breakthrough's expA traces, consecutive tokens, top-10).
Per layer (6, 20, 40) x trace (code, chat, math, bulgarian) x S (16, 24, 32), 4 windows each:
- bytes the call must read: each distinct expert's MSB-3 pieces (2.15 MB) + the LSB-1 pieces (0.77 MB) of the experts
  that have a pair at a 4-bit rank (ranks 0-1 of REF2-T's 4433333333);
- ms of 3c (tier_glu with scores = today's one-call kernel + combine) and NAX6 (one call + combine_phase).
C2: per check (x 48 layers) NAX6 ms vs the byte floor (bytes / 300 and / 215 GB/s). Line (DAY9 #4, ROUTE_350 C2):
a kernel at the floor is worth building only if NAX6 is >= 1.3x above it.
B3: least-squares slope of ms vs S over 16 / 24 / 32 rows (x 48 = ms per added row per check) for 3c and NAX6; the
dense part adds D32's +4.23 ms per 16 rows (0.26 ms per row) and the GDN steps / attention rows ~0.3-0.4 (C0 measures).
Line (DAY9 #9): B3's runahead is KILL if the NAX32 beta is > 2.5 ms per row (M-16: 2.64 at <= 16 rows on 3c, live).
Usage: [BigMoE venv] k62p_floor.py -> ~/Leap/data/k62p_floor.txt"""
import json, os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/MoonShots/patches/planes_engine"))
import tierglu as TG
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel0/Leap_scripts"))
import tierglu_nax6 as N6
PROJ = TG.PROJ
PLAN, R, R4 = "4433333333", 10, 2
MSB_B, LSB_B = 3 * (640 * 2560 * 3 // 8 + 2 * 640 * 2560 // 64 * 2), 3 * (640 * 2560 // 8 + 640 * 2560 // 64 * 2)
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
RD = os.path.expanduser("~/BigMoE/models/bare/records3")
SH = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}
def load(layer):
    idx = json.load(open(f"{RD}/index.json"))["layers"][str(layer)]
    E, REC = idx["E"], idx["recsz"]
    raw = np.fromfile(f"{RD}/{idx['file']}", dtype=np.uint8).reshape(E, REC)
    msb, lsb = {}, {}
    for p, pr in enumerate(PROJ):
        o, i = SH[pr]
        ow, os_, ob = idx["msb_piece_off"][3 * p:3 * p + 3]
        msb[pr] = {"w3": mx.array(np.ascontiguousarray(raw[:, ow:ow + o * i * 3 // 8]).view(np.uint32).reshape(E, o, i * 3 // 32)),
                   "s3": mx.array(np.ascontiguousarray(raw[:, os_:os_ + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16),
                   "b3": mx.array(np.ascontiguousarray(raw[:, ob:ob + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16)}
        lw, lb = idx["lsb_piece_off"][2 * p:2 * p + 2]
        lsb[pr] = {"l1": mx.array(np.ascontiguousarray(raw[:, lw:lw + o * i // 8]).view(np.uint32).reshape(E, o, i // 32)),
                   "b": mx.array(np.ascontiguousarray(raw[:, lb:lb + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16)}
    del raw
    mx.eval([v for d in (msb, lsb) for pr in d for v in d[pr].values()])
    return msb, lsb
def tm(f, reps=8):
    for _ in range(3):
        mx.eval(*[f() for _ in range(reps)])
    ts = []
    for _ in range(7):
        mx.synchronize(); t1 = time.perf_counter(); mx.eval(*[f() for _ in range(reps)]); ts.append(time.perf_counter() - t1)
    return 1e3 * float(np.median(ts)) / reps
rng = np.random.default_rng(1)
mx.random.seed(1)
RES = {}                                                 # (layer, S) -> list of (bytes, ms 3c, ms nax6)
for layer in (6, 20, 40):
    msb, lsb = load(layer)
    for trace in ("code", "chat", "math", "bulgarian"):
        tr = np.load(os.path.expanduser(f"~/Breakthrough/data/expA/traces125/{trace}_800_g.npz"))[f"L{layer}"].astype(np.int32)
        for S in (16, 24, 32):
            for t in rng.integers(0, tr.shape[0] - S, 4):
                ids = tr[t:t + S, :R]
                b = MSB_B * len(np.unique(ids)) + LSB_B * len(np.unique(ids[:, :R4]))
                sm, sl = mx.array(ids), mx.array(ids[:, :R4].copy())
                x = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
                sc = mx.softmax(mx.random.normal((S, R)), axis=-1).astype(mx.bfloat16)
                def nax():
                    y = N6.tier_glu_nax6(x, msb, lsb, sm, sl, plan=PLAN, grp=N6.gpu_grp(sm, sl, PLAN)).reshape(S * R, -1)
                    return TG.combine_phase(y, y, mx.ones((S, R), mx.int32), sc)
                RES.setdefault((layer, S), []).append((b, tm(lambda: TG.tier_glu(x, msb, lsb, sm, sl, plan=PLAN, scores=sc)), tm(nax)))
    del msb, lsb
    mx.clear_cache()
    say(f"  layer {layer}: " + "; ".join(f"S {S}: {np.mean([r[0] for r in RES[(layer, S)]]) / 1e6:.0f} MB, 3c "
                                         f"{np.mean([r[1] for r in RES[(layer, S)]]):.3f} ms, NAX6 {np.mean([r[2] for r in RES[(layer, S)]]):.3f} ms"
                                         for S in (16, 24, 32)))
NL = 48
for S in (16, 24, 32):
    a = np.array([r for (l_, s_), v in RES.items() if s_ == S for r in v])
    b, c3, n6 = a[:, 0].mean() * NL, a[:, 1].mean() * NL, a[:, 2].mean() * NL
    say(f"S {S}: per check (x 48) {b / 1e9:.2f} GB; 3c {c3:.1f} ms ({b / c3 / 1e6:.0f} GB/s), NAX6 {n6:.1f} ms ({b / n6 / 1e6:.0f} "
        f"GB/s); floor {b / 300e6:.1f}-{b / 215e6:.1f} ms -> NAX6 / floor {n6 / (b / 300e6):.2f}-{n6 / (b / 215e6):.2f}x")
sl = {}
for k_, j in (("3c", 1), ("NAX6", 2)):
    Ss = np.array([16, 24, 32])
    ms = np.array([np.mean([r[j] for (l_, s_), v in RES.items() if s_ == S for r in v]) for S in Ss]) * NL
    sl[k_] = np.polyfit(Ss, ms, 1)[0]
say(f"B3: expert slope 16 -> 32 rows: 3c {sl['3c']:.2f}, NAX6 {sl['NAX6']:.2f} ms per added row per check; + dense matmuls "
    f"0.26 (D32) + GDN steps / attention ~0.3-0.4 -> NAX32 beta ~{sl['NAX6'] + 0.26 + 0.3:.2f}-{sl['NAX6'] + 0.26 + 0.4:.2f} "
    f"(line 2.5)")
open(os.path.expanduser("~/Leap/data/k62p_floor.txt"), "w").write("\n".join(lines) + "\n")
