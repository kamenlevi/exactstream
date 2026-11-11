"""OOO-1C native check (Leap, Wed 2026-11-04 15:3x; TABLE_350 'OOO-1C'): the native one-call forms for Friday's converted
store. Layer 20's records3, 512 experts, converted with scripts/p11.py (to_p11, pad=True) into bf16-native pools (the
11-bit scales decoded) and native+p11+pad pools; ~70 % base / ~80 % sidecar resident, regions released before the call
(and once 20 ms after the launch):
  NAX: tier_glu_nax6n_1c on native and on p11 pools == NAX6 on the row-major pools at the final slots (+ combine), bitwise;
  3c:  tier_glu_3cn_1c on native and on p11 pools == tier_glu(scores) on the row-major pools, bitwise;
  the resolved slots == the final slots; no timeout.
Usage: [BigMoE venv] ooo1c_native_check.py -> ~/Leap/data/ooo1c_check.txt (appends; GPU, MEM ~5 GB; share lock)"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_e5/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_e5/Leap_scripts"))
import mlx.core as mx
import ooo1c as OC
import tierglu_nax6 as N6
import p11 as P11
import bigmoe.tier as BT
TG = BT._tier_glu_mod()

OUT = os.path.expanduser("~/Leap/data/ooo1c_check.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nOOO-1C native check {time.strftime('%a %H:%M:%S')}")
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
    for i, v in enumerate(OC.release_words(slots, seq)):
        REG[off + 32 * (i // 31) + 1 + i % 31] = np.int32(np.uint32(v).view(np.int32))
    for j in range(RS // 32 - 1, -1, -1):
        REG[off + 32 * j] = seq
eqb = lambda a, b: bool(a.shape == b.shape and mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())
rng = np.random.default_rng(15)
ok, seq = True, 30
for rep in range(4):
    seq += 1
    ids = np.stack([rng.choice(512, K, replace=False) for _ in range(M)]).astype(np.int32)
    uniq = np.unique(ids)
    res_m = set(rng.choice(uniq, int(0.7 * len(uniq)), replace=False).tolist())
    res_l = set(rng.choice(uniq, int(0.8 * len(uniq)), replace=False).tolist())
    sm, sl = ids.copy(), ids[:, :R4].copy()
    pre_m = np.where(np.isin(ids, list(res_m)), ids, -1).astype(np.int32)
    pre_l = np.where(np.isin(sl, list(res_l)), sl, -1).astype(np.int32)
    off_m, off_l = OC.region_offset(seq, NL, L, NREG, PAIRS, RS), OC.region_offset(seq, NL, NL // 2 + L, NREG, PAIRS, RS)
    offs = mx.array([off_m, off_l, seq], mx.int32)
    x = (mx.random.normal((M, 2560)) * 0.5).astype(mx.bfloat16)
    sc = mx.softmax(mx.random.normal((M, K)), axis=-1).astype(mx.bfloat16)
    yn = N6.tier_glu_nax6(x, rm_msb, rm_lsb, mx.array(sm), mx.array(sl), plan=PLAN, grp=N6.gpu_grp(mx.array(sm), mx.array(sl), PLAN))
    ref_nax = TG.combine_phase(yn.reshape(M * K, -1), yn.reshape(M * K, -1), mx.ones((M, K), mx.int32), sc)
    ref_3c = TG.tier_glu(x, rm_msb, rm_lsb, mx.array(sm), mx.array(sl), plan=PLAN, scores=sc)
    mx.eval(ref_nax, ref_3c)
    late = rep == 3
    if not late:
        release(off_m, sm.reshape(-1).tolist(), seq); release(off_l, sl.reshape(-1).tolist(), seq)
    res = {}
    for nm, d in (("native", nat), ("p11", p1)):
        if late and nm == "native":
            continue
        y4, rm4, rl4 = OC.tier_glu_nax6n_1c(x, d["msb"], d["lsb"], mx.array(pre_m), mx.array(pre_l), mx.array(ids), PLAN,
                                            region, offs, status)
        y_nax = TG.combine_phase(y4.reshape(M * K, -1), y4.reshape(M * K, -1), mx.ones((M, K), mx.int32), sc)
        y_3c, r3m, r3l = OC.tier_glu_3cn_1c(x, d["msb"], d["lsb"], mx.array(pre_m), mx.array(pre_l), PLAN, region, offs,
                                            status, sc)
        if late:
            t0 = time.perf_counter()
            mx.async_eval(y_nax, y_3c, rm4, r3m)
            while time.perf_counter() - t0 < 0.020:
                pass
            release(off_m, sm.reshape(-1).tolist(), seq); release(off_l, sl.reshape(-1).tolist(), seq)
        mx.eval(y_nax, y_3c, rm4, rl4, r3m, r3l)
        res[nm] = (eqb(y_nax, ref_nax), eqb(y_3c, ref_3c),
                   np.array_equal(np.array(rm4), sm.reshape(-1)) and np.array_equal(np.array(r3m), sm.reshape(-1))
                   and np.array_equal(np.array(rl4), sl.reshape(-1)) and np.array_equal(np.array(r3l), sl.reshape(-1)))
    good = all(all(v) for v in res.values()) and int(STA[0]) == 0
    ok &= good
    say(f"  call {rep}{' (late release, 20 ms)' if late else ''}: " + "; ".join(
        f"{nm}: NAX6-N 1c == NAX6 {v[0]}, 3c-N 1c == 3c {v[1]}, slots {v[2]}" for nm, v in res.items()) + f"; timeouts {int(STA[0])}")
say(f"OOO-1C native forms: {'PASS' if ok else 'FAIL'}")

# ---------------------------------------------------------------- the same through ref2 (layout 2 = native+p11 records)
import ref2
import bigmoe.gated as BG
BT.PLAN = PLAN
ref2.apply()
RSg = 32 * (2 + BG.XLINES)
class Gate:
    def __init__(self):
        self.nl, self.nreg, self.seq = NL, 128, 41
        self.region = mx.zeros((self.nreg * BG.PAIRS * RSg,), mx.int32)
        self.status = mx.zeros((4096,), mx.int32)
        mx.eval(self.region, self.status)
        self.REG = np.frombuffer(memoryview(self.region), dtype=np.int32); self.REG.setflags(write=True)
    def release(self, Lx, slots):
        off = ((self.seq * self.nl + Lx) % self.nreg) * BG.PAIRS * RSg
        for i, v in enumerate(OC.release_words(slots, self.seq)):
            self.REG[off + 32 * (i // 31) + 1 + i % 31] = np.int32(np.uint32(v).view(np.int32))
        for j in range(RSg // 32 - 1, -1, -1):
            self.REG[off + 32 * j] = self.seq
gate = Gate()
gate.release(L, sm.reshape(-1).tolist()); gate.release(NL // 2 + L, sl.reshape(-1).tolist())
pre = {L: pre_m.reshape(-1), NL // 2 + L: pre_l.reshape(-1)}
post = lambda Lx, ids_, after=None: (mx.array(np.concatenate([[0], pre[Lx]]).astype(np.int32)), int(pre[Lx].size))
t = {"base": p1["msb"], "sideP": p1["lsb"], "segs": [(0, 2, 4), (2, 10, 3)], "side": NL // 2 + L, "layout": 2}
c0 = ref2.COUNTS["ooo_1c"]
ref2.CFG["ooo_1c"] = True
try:
    y3 = ref2._ooo_sel(x, mx.array(ids), {"gate": gate, "L": L}, dict(t), post, sc, None)
    ref2.CFG["tier_nax_ooo"], ref2.CFG["nax_native"] = True, True
    y4 = ref2._ooo_sel(x, mx.array(ids), {"gate": gate, "L": L}, dict(t), post, sc, None)
finally:
    ref2.CFG["ooo_1c"] = ref2.CFG["tier_nax_ooo"] = ref2.CFG["nax_native"] = False
mx.eval(y3, y4)
ref_ok = eqb(y3, ref_3c) and eqb(y4, ref_nax) and ref2.COUNTS["ooo_1c"] == c0 + 2
say(f"ref2 on native+p11 records (layout 2): 3c-N one call == 3c {eqb(y3, ref_3c)}, NAX6-N one call == NAX6 {eqb(y4, ref_nax)}, "
    f"2 one-calls counted {ref2.COUNTS['ooo_1c'] == c0 + 2} -> {'PASS' if ref_ok else 'FAIL'}")
