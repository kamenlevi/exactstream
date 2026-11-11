"""QOM-OOO native check (Leap, Wed 2026-11-04 17:2x; TABLE_350 'QOM-OOO'): the QOM form of the 3c-N one-call
(ooo1c.tier_glu_3cn_1c(qom=True)) on Friday's converted store: layer 20's records3 converted (p11.to_p11, pad=True) into
bf16-native and native+p11+pad pools; the QOM scenario of ooo1c_qom_check.py; reference = BigMoE's QOM two phases on the
row-major pools (3c-N == 3c bitwise, NATIVE-3C). Regions released before the call, 3 scenarios per pool form.
Usage: [BigMoE venv] ooo1c_qom_native_check.py -> ~/Leap/data/ooo1c_qom_check.txt (appends; GPU, MEM ~5 GB; share lock)"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p4s/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p4s/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p4s/Breakthrough_keepset"))
import mlx.core as mx
import ooo1c as OC
import p11 as P11
import bigmoe.tier as BT
import qom_keep as QK
TG = BT._tier_glu_mod()
BT._lp1_patch()
OUT = os.path.expanduser("~/Leap/data/ooo1c_qom_check.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nQOM-OOO native check {time.strftime('%a %H:%M:%S')}")
PLAN, M, K, R4, L, NL = "4433333333", 16, 10, 2, 20, 96
RD = os.path.expanduser("~/BigMoE/models/bare/records3")
assert all(v.get("layout", "row-major") == "row-major" for v in json.load(open(f"{RD}/index.json"))["layers"].values()), \
    "records3 is converted (native+p11+pad, Thu 11-05): this script reads row-major bytes only"   # Leap 18:5x guard
r = json.load(open(f"{RD}/index.json"))["layers"][str(L)]
SH = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}
E = r["E"]
raw = np.memmap(f"{RD}/{r['file']}", dtype=np.uint8, mode="r", shape=(E, r["recsz"]))
rm_msb, rm_lsb = {}, {}
for p, pr in enumerate(("gate_proj", "up_proj", "down_proj")):        # row-major pools (today's)
    o, i = SH[pr]
    ow, os_, ob = r["msb_piece_off"][3 * p:3 * p + 3]
    lw, lb = r["lsb_piece_off"][2 * p:2 * p + 2]
    rm_msb[pr] = {"w3": mx.array(np.ascontiguousarray(raw[:, ow:ow + o * i * 3 // 8]).view(np.uint32).reshape(E, o, i * 3 // 32)),
                  "s3": mx.array(np.ascontiguousarray(raw[:, os_:os_ + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16),
                  "b3": mx.array(np.ascontiguousarray(raw[:, ob:ob + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16)}
    rm_lsb[pr] = {"l1": mx.array(np.ascontiguousarray(raw[:, lw:lw + o * i // 8]).view(np.uint32).reshape(E, o, i // 32)),
                  "b": mx.array(np.ascontiguousarray(raw[:, lb:lb + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16)}
q1 = P11.layout_p11(r, pad=True)
conv = np.empty((E, q1["recsz"]), np.uint8)
for e0 in range(0, E, 64):
    conv[e0:e0 + 64] = P11.to_p11(np.asarray(raw[e0:e0 + 64]), r, pad=True)[0]
del raw
nat, p1 = {"msb": {}, "lsb": {}}, {"msb": {}, "lsb": {}}
for nm, part, pr, off, sz, dt, per, packed in P11.pieces(q1):
    o, i = SH[pr]
    n = o * (i // 64) * per
    if packed:
        w = np.ascontiguousarray(conv[:, off:off + sz]).view(np.uint32)
        p1[part].setdefault(pr, {})[nm] = mx.array(w)
        nat[part].setdefault(pr, {})[nm] = mx.array(P11.unpack(w[:, :P11.piece_words(n)] if w.shape[1] > P11.piece_words(n) else w, n)
                                                     .reshape(E, o, i // 64)).view(mx.bfloat16)
    else:
        a = np.ascontiguousarray(conv[:, off:off + n * np.dtype(dt).itemsize]).view(dt)
        arr = mx.array(a.reshape((E, o, i * 3 // 32) if nm == "w3" else (E, o, i // 32)))
        nat[part].setdefault(pr, {})[nm] = arr
        p1[part].setdefault(pr, {})[nm] = arr
del conv
mx.eval([v for d in (rm_msb, rm_lsb) for pr in d.values() for v in pr.values()],
        [v for d in (nat, p1) for part in d.values() for pr in part.values() for v in pr.values()])
say(f"pools: row-major, bf16 native and native+p11+pad for L{L} (512 experts)")

NREG, PAIRS, RS = 128, 1, 192
region = mx.zeros((NREG * PAIRS * RS,), mx.int32)
status = mx.zeros((64,), mx.int32)
mx.eval(region, status)
REG = np.frombuffer(memoryview(region), dtype=np.int32); REG.setflags(write=True)
STA = np.frombuffer(memoryview(status), dtype=np.int32)
def release(off, slots, seq):
    w = OC.release_words(slots, seq)
    for i, v in enumerate(w):
        REG[off + 32 * (i // 31) + 1 + i % 31] = np.int32(np.uint32(v).view(np.int32))
    for j in range(RS // 32 - 1, -1, -1):
        REG[off + 32 * j] = seq
eqb = lambda a, b: bool(a.shape == b.shape and mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())
def scenario(rng, qom=True):
    ids = np.stack([rng.choice(512, K, replace=False) for _ in range(M)]).astype(np.int32)
    sure = rng.random(M) < 0.5 if qom else np.zeros(M, bool)
    mand = np.zeros((M, K), bool); mand[:, :2] = True; mand[~sure] = True
    union = set(ids[mand].tolist())
    opt = sure[:, None] & ~mand & ~np.isin(ids, list(union))
    uniq = np.unique(ids)
    res = set(rng.choice(uniq, int(0.7 * len(uniq)), replace=False).tolist())
    res_l = set(rng.choice(uniq, int(0.8 * len(uniq)), replace=False).tolist())
    inres = np.isin(ids, list(res))
    pre_m = np.where(inres, ids, -1).astype(np.int32)
    sm = np.where(opt & ~inres, -2, ids).astype(np.int32)                 # the loader: an optional miss is not read
    pre_l = np.where(np.isin(ids[:, :R4], list(res_l)), ids[:, :R4], -1).astype(np.int32)
    sl = ids[:, :R4].copy()
    if qom:
        pre_l = np.where(sure[:, None], -2, pre_l).astype(np.int32); sl = np.where(sure[:, None], -2, sl).astype(np.int32)
    return ids, sure, opt, pre_m, sm, pre_l, sl

def two_phase_qom(x, msb, lsb, ids, sure, pre_m, sm, pre_l, sl, sc):
    """r57_screen _ooo_call_t3's arithmetic with QOM (qm = (opt, sure)), the spin's answers = sm / sl"""
    nl2 = mx.array(sure.reshape(M, 1))
    ok = mx.array(pre_m >= 0) & mx.concatenate([mx.array(pre_l >= 0) | nl2, mx.ones((M, K - R4), mx.bool_)], axis=-1)
    phase = mx.where(ok, 1, 2).astype(mx.int32)
    l1 = mx.where(nl2, -2, mx.maximum(mx.array(pre_l), 0))
    y1 = TG.tier_glu_phase(x, msb, lsb, mx.array(pre_m), l1, phase, 1, plan=PLAN)
    smx = mx.array(sm)
    slx = mx.where(nl2, -2, mx.array(sl))
    cf_ = mx.where(mx.array(sure), 0.0, float("nan"))
    kq, _, _ = QK.qom_keep_mx(mx.array(ids) & 0xFFFFF, sc.reshape(M, K), cf_, smx >= 0)
    smx = mx.where(kq, smx, smx[:, :1])
    sc2 = mx.where(kq, sc.reshape(M, K), mx.zeros_like(sc.reshape(M, K)))
    y2 = TG.tier_glu_phase(x, msb, lsb, smx, slx, phase, 2, plan=PLAN)
    return TG.combine_phase(y1, y2, phase, sc2), kq

rng = np.random.default_rng(23)
seq, ok, nd = 700, True, 0
for nm, d in (("native", nat), ("native+p11+pad", p1)):
    for rep in range(3):
        seq += 1
        ids, sure, opt, pre_m, sm, pre_l, sl = scenario(rng)
        off_m, off_l = OC.region_offset(seq, NL, L, NREG, PAIRS, RS), OC.region_offset(seq, NL, NL // 2 + L, NREG, PAIRS, RS)
        offs = mx.array([off_m, off_l, seq], mx.int32)
        x = (mx.random.normal((M, 2560)) * 0.5).astype(mx.bfloat16)
        sc = mx.softmax(mx.random.normal((M, K)), axis=-1).astype(mx.bfloat16)
        mx.eval(x, sc)
        y_ref, kq = two_phase_qom(x, rm_msb, rm_lsb, ids, sure, pre_m, sm, pre_l, sl, sc)
        mx.eval(y_ref, kq)
        nd += int((~np.array(kq)).sum())
        release(off_m, sm.reshape(-1).tolist(), seq); release(off_l, sl.reshape(-1).tolist(), seq)
        y_1c, _, _ = OC.tier_glu_3cn_1c(x, d["msb"], d["lsb"], mx.array(pre_m), mx.array(pre_l), PLAN, region, offs, status, sc,
                                        qom=True)
        mx.eval(y_1c)
        same = eqb(y_1c, y_ref) and int(STA[0]) == 0
        ok &= same
        say(f"  {nm} rep {rep}: {int(opt.sum())} optional uses, {int((~np.array(kq)).sum())} dropped; 3c-N one-call qom == "
            f"the row-major QOM two phases {eqb(y_1c, y_ref)}; timeouts {int(STA[0])}")
say(f"QOM-OOO native: {'PASS' if ok else 'FAIL'} ({nd} dropped uses)")
