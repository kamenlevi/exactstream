"""QOM-OOO check (Leap, Wed 2026-11-04 17:1x; TABLE_350 'QOM-OOO'; the orchestrator's item 1 for tonight's 23:00 run):
one-call OOO on BigMoE's 3c path with QOM (ooo1c.tier_glu_3c_1c(qom=True); ref2._ooo_1c_call takes t["qom"]) vs BigMoE's
QOM two phases (r57_screen.py _ooo_call_t3 with qm, verbatim arithmetic: phase 1 on the posted slots, the spin's final
slots, Breakthrough's qom_keep_mx on (ids, scores, cf, resident = slot >= 0), dropped uses read rank 0's slot at weight
0, phase 2, combine_phase), on real records3 pools (layers 20 / 44, slot = expert id), 16 rows, top-10, plan 4433333333:
 - sure rows (half): their 4-bit ranks' sidecar -2 (U3's t3), their rank 2-9 uses outside the block's mandatory union
   are OPTIONAL: resident -> kept (its slot), not resident -> the loader returns -2 (dropped);
 - unsure rows and ranks 0-1: mandatory (final slot = id); ~70 % of experts resident at post time, the rest late.
 (1) region released BEFORE the call: one-call (qom) == the QOM two phases, bitwise; status no timeout;
 (2) a LATE release (written 20 ms after launch): the same bits;
 (3) the non-QOM one-call is unchanged: qom=False on a QOM-free scenario == BigMoE's two phases + combine, bitwise.
Usage: [BigMoE venv] ooo1c_qom_check.py -> ~/Leap/data/ooo1c_qom_check.txt (GPU, MEM ~3.5 GB; share lock)"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5p/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5p/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5p/Breakthrough_keepset"))
import mlx.core as mx
import ooo1c as OC
import bigmoe.tier as BT
import qom_keep as QK
TG = BT._tier_glu_mod()
BT._lp1_patch()                                          # QOM rides on U3's t3: LP1 on (as r57 for t3 arms)

OUT = os.path.expanduser("~/Leap/data/ooo1c_qom_check.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nQOM-OOO check {time.strftime('%a %H:%M:%S')}")
PLAN, M, K, R4 = "4433333333", 16, 10, 2
RD = os.path.expanduser("~/BigMoE/models/bare/records3")
assert all(v.get("layout", "row-major") == "row-major" for v in json.load(open(f"{RD}/index.json"))["layers"].values()), \
    "records3 is converted (native+p11+pad, Thu 11-05): this script reads row-major bytes only"   # Leap 18:5x guard
idx = json.load(open(f"{RD}/index.json"))["layers"]
SH = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}
def pools(r):
    E, REC = r["E"], r["recsz"]
    raw = np.memmap(f"{RD}/{r['file']}", dtype=np.uint8, mode="r", shape=(E, REC))
    msb, lsb = {}, {}
    for p, pr in enumerate(("gate_proj", "up_proj", "down_proj")):
        o, i = SH[pr]
        ow, os_, ob = r["msb_piece_off"][3 * p:3 * p + 3]
        lw, lb = r["lsb_piece_off"][2 * p:2 * p + 2]
        msb[pr] = {"w3": mx.array(np.ascontiguousarray(raw[:, ow:ow + o * i * 3 // 8]).view(np.uint32).reshape(E, o, i * 3 // 32)),
                   "s3": mx.array(np.ascontiguousarray(raw[:, os_:os_ + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16),
                   "b3": mx.array(np.ascontiguousarray(raw[:, ob:ob + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16)}
        lsb[pr] = {"l1": mx.array(np.ascontiguousarray(raw[:, lw:lw + o * i // 8]).view(np.uint32).reshape(E, o, i // 32)),
                   "b": mx.array(np.ascontiguousarray(raw[:, lb:lb + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16)}
    mx.eval([v for d in (msb, lsb) for pr in d for v in d[pr].values()])
    del raw
    return msb, lsb
P = {L: pools(idx[str(L)]) for L in (20, 44)}
NREG, PAIRS, RS, NL = 128, 1, 192, 96
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

rng = np.random.default_rng(11)
seq, ok1, ok2, ok3, ndrop = 300, True, True, True, 0
for L in (20, 44):
    msb, lsb = P[L]
    for rep in range(4):
        seq += 1
        ids, sure, opt, pre_m, sm, pre_l, sl = scenario(rng)
        off_m, off_l = OC.region_offset(seq, NL, L, NREG, PAIRS, RS), OC.region_offset(seq, NL, NL // 2 + L, NREG, PAIRS, RS)
        offs = mx.array([off_m, off_l, seq], mx.int32)
        x = (mx.random.normal((M, 2560)) * 0.5).astype(mx.bfloat16)
        sc = mx.softmax(mx.random.normal((M, K)), axis=-1).astype(mx.bfloat16)
        mx.eval(x, sc)
        y_2p, kq = two_phase_qom(x, msb, lsb, ids, sure, pre_m, sm, pre_l, sl, sc)
        mx.eval(y_2p, kq)
        drops = int((~np.array(kq)).sum()); ndrop += drops
        assert np.array_equal(~np.array(kq), opt & (sm < 0)), "the rule drops exactly the unread optional uses"
        if rep < 3:
            release(off_m, sm.reshape(-1).tolist(), seq); release(off_l, sl.reshape(-1).tolist(), seq)
            y_1c, rs_m, rs_l = OC.tier_glu_3c_1c(x, msb, lsb, mx.array(pre_m), mx.array(pre_l), PLAN, region, offs, status, sc,
                                                 qom=True)
            mx.eval(y_1c)
            same = eqb(y_1c, y_2p) and int(STA[0]) == 0
            ok1 &= same
            say(f"  L{L} rep {rep}: {int(sure.sum())} sure rows, {int(opt.sum())} optional uses, {drops} dropped; one-call qom == "
                f"QOM two phases {eqb(y_1c, y_2p)}; timeouts {int(STA[0])}")
        else:
            y_1c, rs_m, rs_l = OC.tier_glu_3c_1c(x, msb, lsb, mx.array(pre_m), mx.array(pre_l), PLAN, region, offs, status, sc,
                                                 qom=True)
            t0 = time.perf_counter()
            mx.async_eval(y_1c)
            while time.perf_counter() - t0 < 0.020:
                pass
            release(off_m, sm.reshape(-1).tolist(), seq); release(off_l, sl.reshape(-1).tolist(), seq)
            mx.eval(y_1c)
            dt = time.perf_counter() - t0
            same = eqb(y_1c, y_2p) and int(STA[0]) == 0 and dt >= 0.020
            ok2 &= same
            say(f"  L{L} late release: {drops} dropped; == QOM two phases {eqb(y_1c, y_2p)}; timeouts {int(STA[0])}; call {1e3 * dt:.1f} ms")
    # (3) the non-QOM form is unchanged
    seq += 1
    ids, sure, opt, pre_m, sm, pre_l, sl = scenario(rng, qom=False)
    off_m, off_l = OC.region_offset(seq, NL, L, NREG, PAIRS, RS), OC.region_offset(seq, NL, NL // 2 + L, NREG, PAIRS, RS)
    offs = mx.array([off_m, off_l, seq], mx.int32)
    x = (mx.random.normal((M, 2560)) * 0.5).astype(mx.bfloat16)
    sc = mx.softmax(mx.random.normal((M, K)), axis=-1).astype(mx.bfloat16)
    ok_ = (pre_m >= 0) & np.concatenate([pre_l >= 0, np.ones((M, K - R4), bool)], axis=1)
    phase = mx.array(np.where(ok_, 1, 2).astype(np.int32))
    y1 = TG.tier_glu_phase(x, msb, lsb, mx.array(pre_m), mx.array(np.maximum(pre_l, 0)), phase, 1, plan=PLAN)
    y2 = TG.tier_glu_phase(x, msb, lsb, mx.array(sm), mx.array(sl), phase, 2, plan=PLAN)
    y_ref = TG.combine_phase(y1, y2, phase, sc)
    release(off_m, sm.reshape(-1).tolist(), seq); release(off_l, sl.reshape(-1).tolist(), seq)
    y_1c, _, _ = OC.tier_glu_3c_1c(x, msb, lsb, mx.array(pre_m), mx.array(pre_l), PLAN, region, offs, status, sc)
    mx.eval(y_ref, y_1c)
    ok3 &= eqb(y_1c, y_ref) and int(STA[0]) == 0
say(f"(1) released before: one-call qom == QOM two phases, bitwise: {'PASS' if ok1 else 'FAIL'} ({ndrop} dropped uses in all)")
say(f"(2) late release: {'PASS' if ok2 else 'FAIL'}; (3) non-QOM one-call unchanged: {'PASS' if ok3 else 'FAIL'}")
say(f"QOM-OOO check: {'PASS' if ok1 and ok2 and ok3 else 'FAIL'}")
