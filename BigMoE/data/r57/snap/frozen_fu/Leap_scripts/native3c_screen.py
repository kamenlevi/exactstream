"""NATIVE-3C screen (Leap, Tue 2026-11-03 15:4x; pre-registered in ~/Leap/TABLE_350.md 'NATIVE-3C'): can the exact
reference's 3c serve experts from the native-layout records bit for bit, and at what speed?
Per layer (3, 20, 44): pools from records3 (row-major) and from data/native_sample (native; the disk format end to end,
memory-mapped), then on real routing (Breakthrough's expA traces), S 1 / 8 / 16:
  0. the sample file's pieces == native_map(records3's) for every expert;
  a. 3c-N(native) vs 3c(row-major), y bitwise: tier_glu; tier_glu with scores (BC); tier_glu_phase x 2 + combine_phase;
  b. NAX6-N(native) vs NAX6(row-major), y bitwise.
Speed on layer 20, dependent chains of 16: 3c + scores vs 3c-N + scores, NAX6 + combine vs NAX6-N + combine, S 16 / 8 / 1.
Lines: PASS = every comparison bitwise; FAIL = any mismatch. Speed: no line.
Usage: [BigMoE venv] native3c_screen.py -> ~/Leap/data/native3c_screen.txt (GPU, ~3-4 GB; share lock)"""
import json, os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_fu/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_fu/Leap_scripts"))
from tierglu import PROJ
import tierglu as TG
import tierglu_nax6 as N6
import tierglu_nax6n as NN
import tierglu_3cn as C

OUT = os.path.expanduser("~/Leap/data/native3c_screen.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"NATIVE-3C screen {time.strftime('%a %H:%M')} ({os.environ.get('N3_COND', 'shared GPU, rough')})")
RD = os.path.expanduser("~/BigMoE/models/bare/records3")
assert all(v.get("layout", "row-major") == "row-major" for v in json.load(open(f"{RD}/index.json"))["layers"].values()), \
    "records3 is converted (native+p11+pad, Thu 11-05): this script reads row-major bytes only"   # Leap 18:5x guard
ND = os.path.expanduser("~/Leap/data/native_sample")
SH = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}
PLAN = "4433333333"
R = len(PLAN)
idx = json.load(open(f"{RD}/index.json"))["layers"]
def pools(path, r):
    E, REC = r["E"], r["recsz"]
    raw = np.memmap(path, dtype=np.uint8, mode="r", shape=(E, REC))
    msb, lsb, npc = {}, {}, {}
    for p, pr in enumerate(PROJ):
        o, i = SH[pr]
        ow, os_, ob = r["msb_piece_off"][3 * p:3 * p + 3]
        lw, lb = r["lsb_piece_off"][2 * p:2 * p + 2]
        w3 = np.ascontiguousarray(raw[:, ow:ow + o * i * 3 // 8]).view(np.uint32).reshape(E, o, i * 3 // 32)
        s3 = np.ascontiguousarray(raw[:, os_:os_ + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)
        b3 = np.ascontiguousarray(raw[:, ob:ob + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)
        l1 = np.ascontiguousarray(raw[:, lw:lw + o * i // 8]).view(np.uint32).reshape(E, o, i // 32)
        bl = np.ascontiguousarray(raw[:, lb:lb + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)
        npc[pr] = dict(w3=w3, s3=s3, b3=b3, l1=l1, b=bl)
        msb[pr] = {"w3": mx.array(w3), "s3": mx.array(s3).view(mx.bfloat16), "b3": mx.array(b3).view(mx.bfloat16)}
        lsb[pr] = {"l1": mx.array(l1), "b": mx.array(bl).view(mx.bfloat16)}
    mx.eval([v for d in (msb, lsb) for pr in d for v in d[pr].values()])
    del raw
    return msb, lsb, npc
eqb = lambda a, b: bool(a.shape == b.shape and mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())
PER = {"w3": 6, "s3": 1, "b3": 1, "l1": 2, "b": 1}
all_ok = True
mx.random.seed(0)
speed = {}
for L in (3, 20, 44):
    r = idx[str(L)]
    msb, lsb, npr = pools(f"{RD}/{r['file']}", r)
    msb_n, lsb_n, npn = pools(f"{ND}/{r['file']}", r)
    ok0 = True
    for pr in PROJ:
        o, i = SH[pr]
        for f, per in PER.items():
            a = npr[pr][f]
            ok0 &= bool(np.array_equal(npn[pr][f].reshape(a.shape[0], -1), a.reshape(a.shape[0], -1)[:, NN.native_map(o, a.shape[2] // per, per)]))
    del npr, npn
    TR = {t: np.load(os.path.expanduser(f"~/Breakthrough/data/expA/traces125/{t}_800_g.npz"))[f"L{L}"].astype(np.int32)
          for t in ("code", "chat", "math", "bulgarian")}
    cnt = {k: [0, 0] for k in ("3c", "3c+scores", "3c phases", "NAX6")}
    for t, tr in TR.items():
        for S in (1, 8, 16):
            for t0 in np.linspace(0, tr.shape[0] - S, 3).astype(int):
                ids = tr[t0:t0 + S, :10].copy()
                ms_, ls_ = mx.array(ids), mx.array(ids[:, :2].copy())
                x = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
                sc = mx.softmax(mx.random.normal((S, R)), axis=-1).astype(mx.bfloat16)
                ph = mx.array(np.random.default_rng(int(t0) + S).integers(1, 3, (S, R)).astype(np.int32))
                pairs = {
                    "3c": (TG.tier_glu(x, msb, lsb, ms_, ls_, plan=PLAN), C.tier_glu_n(x, msb_n, lsb_n, ms_, ls_, plan=PLAN)),
                    "3c+scores": (TG.tier_glu(x, msb, lsb, ms_, ls_, plan=PLAN, scores=sc),
                                  C.tier_glu_n(x, msb_n, lsb_n, ms_, ls_, plan=PLAN, scores=sc)),
                    "3c phases": (TG.combine_phase(TG.tier_glu_phase(x, msb, lsb, ms_, ls_, ph, 1, plan=PLAN),
                                                   TG.tier_glu_phase(x, msb, lsb, ms_, ls_, ph, 2, plan=PLAN), ph, sc),
                                  TG.combine_phase(C.tier_glu_phase_n(x, msb_n, lsb_n, ms_, ls_, ph, 1, plan=PLAN),
                                                   C.tier_glu_phase_n(x, msb_n, lsb_n, ms_, ls_, ph, 2, plan=PLAN), ph, sc)),
                    "NAX6": (N6.tier_glu_nax6(x, msb, lsb, ms_, ls_, plan=PLAN, grp=N6.gpu_grp(ms_, ls_, PLAN)),
                             NN.tier_glu_nax6n(x, msb_n, lsb_n, ms_, ls_, plan=PLAN, grp=N6.gpu_grp(ms_, ls_, PLAN)))}
                mx.eval([v for pq in pairs.values() for v in pq])
                for k, (a, b) in pairs.items():
                    cnt[k][0] += eqb(a, b)
                    cnt[k][1] += 1
    ok = ok0 and all(c[0] == c[1] for c in cnt.values())
    all_ok &= ok
    say(f"layer {L}: sample file == native_map(records3) {ok0}; bitwise " +
        ", ".join(f"{k} {c[0]} / {c[1]}" for k, c in cnt.items()) + f" -> {'PASS' if ok else 'FAIL'}")
    if L == 20:
        NCH = 16
        tr = TR["code"]
        for S in (16, 8, 1):
            acc = {}
            for t0 in np.linspace(0, tr.shape[0] - S, 4).astype(int):
                ids = tr[t0:t0 + S, :10].copy()
                ms_, ls_ = mx.array(ids), mx.array(ids[:, :2].copy())
                x0 = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
                sc = mx.softmax(mx.random.normal((S, R)), axis=-1).astype(mx.bfloat16)
                one = mx.ones((S, R), mx.int32)
                mx.eval(x0, sc, one)
                def nax(f, m_, l_):
                    def g(x):
                        y = f(x, m_, l_, ms_, ls_, plan=PLAN, grp=N6.gpu_grp(ms_, ls_, PLAN)).reshape(-1, 2560)
                        return TG.combine_phase(y, y, one, sc)
                    return g
                arms = {"3c": lambda x: TG.tier_glu(x, msb, lsb, ms_, ls_, plan=PLAN, scores=sc),
                        "3c-N": lambda x: C.tier_glu_n(x, msb_n, lsb_n, ms_, ls_, plan=PLAN, scores=sc),
                        "NAX6": nax(N6.tier_glu_nax6, msb, lsb), "NAX6-N": nax(NN.tier_glu_nax6n, msb_n, lsb_n)}
                def chain(f):
                    out = None
                    for _ in range(NCH):
                        out = f(x0 if out is None else mx.depends(x0, out))
                    return out
                for f in arms.values():
                    mx.eval(chain(f))
                ts = {k: [] for k in arms}
                for _ in range(9):
                    for k, f in arms.items():
                        mx.synchronize(); t1 = time.perf_counter(); mx.eval(chain(f)); ts[k].append((time.perf_counter() - t1) / NCH)
                for k, v in ts.items():
                    acc.setdefault(k, []).append(float(np.median(v)))
            m = {k: 1e3 * float(np.mean(v)) for k, v in acc.items()}
            speed[S] = m
            say(f"  speed layer 20, S {S}: 3c {m['3c']:.3f} -> 3c-N {m['3c-N']:.3f} ms ({100 * (m['3c'] / m['3c-N'] - 1):+.1f} %); "
                f"NAX6 {m['NAX6']:.3f} -> NAX6-N {m['NAX6-N']:.3f} ms ({100 * (m['NAX6'] / m['NAX6-N'] - 1):+.1f} %)")
    del msb, lsb, msb_n, lsb_n
    mx.clear_cache()
say(f"NATIVE-3C verdict: {'PASS' if all_ok else 'FAIL'} (3c-N and NAX6-N on the native files == 3c and NAX6 on records3, "
    f"bitwise, layers 3 / 20 / 44, S 1 / 8 / 16)")
