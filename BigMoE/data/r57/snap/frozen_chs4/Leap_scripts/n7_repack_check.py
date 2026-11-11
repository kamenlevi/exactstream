"""N7-15 self-test (Leap, Tue 2026-11-03 00:0x; TABLE_350 'N7-15'): tierglu_nax6n.repack_slots, the in-place GPU repack of
dirty slots (BigMoE's in-service demand loads) to the native layout. All checks are bitwise, on real layer-20 records:
A. claim + rewrite: slots named in sm / sl with the flag set == repack(row-major) (all 15 pieces); every other slot
   unchanged; exactly those flags cleared; negative slots (-1 / LP1's -2) skipped; the returned slot arrays == the inputs.
B. a second call on the same arrays changes nothing (flags clear).
C. one dirty slot named by all 160 pairs is rewritten once (not twice).
D. two calls chained lazily in one graph (the second takes the first's returned slots, overlapping slots, the same flags).
E. end-to-end on real routing, 4 traces x S 16 / 8 / 1: pools half native (clean) and half row-major (dirty); after
   repack_slots, tier_glu_nax6n == tier_glu_nax6 on the row-major pools (y).
F. OOO's pattern: phase 1 (clean pairs) on NAX6-N, then repack_slots, then phase 2 (dirty pairs), combine_phase
   == NAX6 + combine_phase over all pairs.
Usage: [BigMoE venv] n7_repack_check.py -> ~/Leap/data/n7_repack_check.txt (GPU, small; share lock)"""
import json, os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs4/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs4/Leap_scripts"))
from tierglu import PROJ
import tierglu as TG
import tierglu_nax6 as N6
import tierglu_nax6n as NN

OUT = os.path.expanduser("~/Leap/data/n7_repack_check.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
LAYER, PLAN = 20, "4433333333"
RD = os.path.expanduser("~/BigMoE/models/bare/records3")
idx = json.load(open(f"{RD}/index.json"))["layers"][str(LAYER)]
E, REC = idx["E"], idx["recsz"]
SH = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}
TR = {t: np.load(os.path.expanduser(f"~/Breakthrough/data/expA/traces125/{t}_800_g.npz"))[f"L{LAYER}"].astype(np.int32)
      for t in ("code", "chat", "math", "bulgarian")}
WIN = {t: [TR[t][100:116, :10]] for t in TR}                              # the windows used by E / F
sel = np.unique(np.concatenate([w.reshape(-1) for ws in WIN.values() for w in ws] + [np.arange(8)]))
C = len(sel)
pos = {int(e): i for i, e in enumerate(sel)}                               # expert -> slot
mm = np.memmap(f"{RD}/{idx['file']}", dtype=np.uint8, mode="r", shape=(E, REC))
raw = np.ascontiguousarray(mm[sel])
del mm
# numpy pools [C, rows, words]: base (w3 u32, s3 / b3 u16) and sidecar (l1 u32, b u16), row-major
RM = {}
for p, pr in enumerate(PROJ):
    o, i = SH[pr]
    ow, os_, ob = idx["msb_piece_off"][3 * p:3 * p + 3]
    lw, lb = idx["lsb_piece_off"][2 * p:2 * p + 2]
    RM[(pr, "w3")] = np.ascontiguousarray(raw[:, ow:ow + o * i * 3 // 8]).view(np.uint32).reshape(C, o, i * 3 // 32)
    RM[(pr, "s3")] = np.ascontiguousarray(raw[:, os_:os_ + o * i // 32]).view(np.uint16).reshape(C, o, i // 64)
    RM[(pr, "b3")] = np.ascontiguousarray(raw[:, ob:ob + o * i // 32]).view(np.uint16).reshape(C, o, i // 64)
    RM[(pr, "l1")] = np.ascontiguousarray(raw[:, lw:lw + o * i // 8]).view(np.uint32).reshape(C, o, i // 32)
    RM[(pr, "b")] = np.ascontiguousarray(raw[:, lb:lb + o * i // 32]).view(np.uint16).reshape(C, o, i // 64)
del raw
PER = {"w3": 6, "s3": 1, "b3": 1, "l1": 2, "b": 1}
BASE, SIDE = ("w3", "s3", "b3"), ("l1", "b")
NAT = {}
for (pr, f), a in RM.items():
    o = a.shape[1]
    m = NN.native_map(o, a.shape[2] // PER[f], PER[f])
    NAT[(pr, f)] = np.ascontiguousarray(a.reshape(C, -1)[:, m].reshape(a.shape))
def pools(src):
    """MLX pools (fresh buffers) from {(proj, field): numpy [C, ...]}"""
    mk = lambda a: mx.array(a) if a.dtype == np.uint32 else mx.array(a).view(mx.bfloat16)
    m = {pr: {f: mk(src[(pr, f)]) for f in BASE} for pr in PROJ}
    l = {pr: {f: mk(src[(pr, f)]) for f in SIDE} for pr in PROJ}
    mx.eval([v for d in (m, l) for pr in d for v in d[pr].values()])
    return m, l
def back(m, l):
    out = {}
    for pr in PROJ:
        for f in BASE + SIDE:
            a = (m if f in BASE else l)[pr][f]
            out[(pr, f)] = np.array(a.view(mx.uint16) if a.dtype == mx.bfloat16 else a)
    return out
def mixed(native_slots):
    return {k: np.where(np.isin(np.arange(C), list(native_slots))[:, None, None], NAT[k], v) for k, v in RM.items()}
def expect(conv_m, conv_l, base_src=None):
    """per piece: the slots in conv_* native, the rest as base_src (default row-major)"""
    src = base_src or RM
    out = {}
    for k, v in src.items():
        conv = conv_m if k[1] in BASE else conv_l
        out[k] = np.where(np.isin(np.arange(C), list(conv))[:, None, None], NAT[k], v)
    return out
def same(a, b):
    return all(np.array_equal(a[k], b[k]) for k in a)
eqb = lambda a, b: bool(a.shape == b.shape and mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())
say(f"N7-15 repack_slots self-test {time.strftime('%a %H:%M')}; layer {LAYER}, {C} slots")
rng = np.random.default_rng(0)
ok_all = True
def report(name, ok, extra=""):
    global ok_all
    ok_all &= bool(ok)
    say(f"  {name}: {'PASS' if ok else 'FAIL'}{extra}")
# A
m, l = pools(RM)
sm_np = rng.integers(0, C, (16, 10)).astype(np.int32); sm_np[3, 4] = sm_np[9, 0] = -1
sl_np = rng.integers(0, C, (16, 2)).astype(np.int32); sl_np[5, 1] = -2; sl_np[0, 0] = -1
dm_np = (rng.random(C) < 0.5).astype(np.int32); dl_np = (rng.random(C) < 0.5).astype(np.int32)
dm, dl = mx.array(dm_np), mx.array(dl_np)
sm2, sl2 = NN.repack_slots(m, l, mx.array(sm_np), mx.array(sl_np), dm, dl)
mx.eval(sm2, sl2)
ref_m = {int(s) for s in sm_np.reshape(-1) if s >= 0 and dm_np[s]}
ref_l = {int(s) for s in sl_np.reshape(-1) if s >= 0 and dl_np[s]}
got = back(m, l)
want_dm = dm_np.copy(); want_dm[list(ref_m)] = 0
want_dl = dl_np.copy(); want_dl[list(ref_l)] = 0
report("A pools", same(got, expect(ref_m, ref_l)), f" ({len(ref_m)} base + {len(ref_l)} sidecar slots rewritten)")
report("A flags", np.array_equal(np.array(dm), want_dm) and np.array_equal(np.array(dl), want_dl))
report("A slots out == in", np.array_equal(np.array(sm2), sm_np) and np.array_equal(np.array(sl2), sl_np))
# B
sm3, sl3 = NN.repack_slots(m, l, mx.array(sm_np), mx.array(sl_np), dm, dl)
mx.eval(sm3, sl3)
report("B second call changes nothing", same(back(m, l), got))
# C
m, l = pools(RM)
dm_np = np.zeros(C, np.int32); dm_np[7] = 1
dl_np = np.zeros(C, np.int32); dl_np[7] = 1
dm, dl = mx.array(dm_np), mx.array(dl_np)
s2, t2 = NN.repack_slots(m, l, mx.full((16, 10), 7, mx.int32), mx.full((16, 2), 7, mx.int32), dm, dl)
mx.eval(s2, t2)
report("C 160 pairs, one slot: rewritten once", same(back(m, l), expect({7}, {7})))
# D
m, l = pools(RM)
dm_np = np.ones(C, np.int32); dl_np = np.ones(C, np.int32)
dm, dl = mx.array(dm_np), mx.array(dl_np)
a_np = rng.integers(0, C // 2, (16, 10)).astype(np.int32)
b_np = rng.integers(C // 4, 3 * C // 4, (16, 10)).astype(np.int32)
s1, t1 = NN.repack_slots(m, l, mx.array(a_np), mx.array(a_np[:, :2].copy()), dm, dl)
s2, t2 = NN.repack_slots(m, l, mx.depends(mx.array(b_np), s1), mx.depends(mx.array(b_np[:, :2].copy()), t1), dm, dl)
mx.eval(s2, t2)
conv_m = set(a_np.reshape(-1).tolist()) | set(b_np.reshape(-1).tolist())
conv_l = set(a_np[:, :2].reshape(-1).tolist()) | set(b_np[:, :2].reshape(-1).tolist())
report("D two chained calls", same(back(m, l), expect(conv_m, conv_l)))
# E / F
mx.random.seed(0)
mrm, lrm = pools(RM)                                                          # read-only row-major reference pools
ne = nf = n = 0
for t, ws in WIN.items():
    for w in ws:
        for S in (16, 8, 1):
            ids = np.vectorize(pos.get)(w[:S]).astype(np.int32)
            ms_, ls_ = mx.array(ids), mx.array(ids[:, :2].copy())
            x = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
            sc = mx.softmax(mx.random.normal((S, 10)), axis=-1).astype(mx.bfloat16)
            ref = N6.tier_glu_nax6(x, mrm, lrm, ms_, ls_, plan=PLAN, grp=N6.gpu_grp(ms_, ls_, PLAN))
            refc = TG.combine_phase(ref.reshape(S * 10, -1), ref.reshape(S * 10, -1), mx.ones((S, 10), mx.int32), sc)
            uniq = np.unique(ids)
            clean = set(uniq[::2].tolist())                                  # half the call's slots already native
            dm_np = np.array([0 if s in clean else 1 for s in range(C)], np.int32)
            # E: one call
            m, l = pools(mixed(clean))
            dm, dl = mx.array(dm_np), mx.array(dm_np.copy())
            s2, t2 = NN.repack_slots(m, l, ms_, ls_, dm, dl)
            y = NN.tier_glu_nax6n(x, m, l, s2, t2, plan=PLAN, grp=N6.gpu_grp(s2, t2, PLAN))
            mx.eval(ref, y)
            ne += eqb(ref, y)
            # F: OOO's two phases (phase 1 = pairs whose base and sidecar slots are clean)
            m, l = pools(mixed(clean))
            dm, dl = mx.array(dm_np), mx.array(dm_np.copy())
            ph_np = np.where(np.isin(ids, list(clean)), 1, 2).astype(np.int32)
            ph_np[:, :2] = np.where(np.isin(ids[:, :2], list(clean)), ph_np[:, :2], 2)
            ph = mx.array(ph_np)
            y1 = NN.tier_glu_nax6n(x, m, l, ms_, ls_, plan=PLAN,
                                   grp=N6.gpu_grp_phase(ms_, ls_, PLAN, ph, 1)).reshape(S * 10, -1)
            s2, t2 = NN.repack_slots(m, l, mx.depends(ms_, y1), ls_, dm, dl)
            y2 = NN.tier_glu_nax6n(x, m, l, s2, t2, plan=PLAN,
                                   grp=N6.gpu_grp_phase(s2, t2, PLAN, ph, 2)).reshape(S * 10, -1)
            yc = TG.combine_phase(y1, y2, ph, sc)
            mx.eval(refc, yc)
            nf += eqb(refc, yc)
            n += 1
            del m, l
            mx.clear_cache()
report("E NAX6-N after repack_slots == NAX6 (y)", ne == n, f" {ne} / {n}")
report("F OOO two phases + repack == NAX6 + combine", nf == n, f" {nf} / {n}")
say(f"N7-15 self-test: {'PASS' if ok_all else 'FAIL'}")
