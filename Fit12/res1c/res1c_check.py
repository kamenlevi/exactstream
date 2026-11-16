"""res1c_check.py (Fit12, Mon 16 Nov 2026 14:0x; pre-registered in PLAN "RES1C INSTALLER"): the installer's GPU bit tests,
in Leap's ooo1c_2d_check form (its setup copied: BigMoE's loader on the CONVERTED records3 pools of layers 20 / 44, all
512 experts loaded, a stand-in gate with release regions, ref2._ooo_sel; read-only use of their files).
The stand-in post here also plays the loader's bit-21 rule: an id posted with bit 21 is optional; an expert whose every
posted use is optional and that is not resident is not read (its base slot -2, final); else as Leap's post.
 T1 OFF (F12_QOM2D False, F12_CNT None): the transformed _ooo_1c_call == the original (kept as _F12_ORIG_OOO1C) on 1c
    calls (2-D nolsb + km drops), 1-D U3 rows and plain calls: y bits, base and sidecar posts equal; and the guard still
    raises for a 2-D nolsb with t["qom"].
 T2 ON: random tagged rows (~30 %), random residency, RES off / on: masks() -> km substitution, 2-D nolsb, bit-21 opt;
    y == an explicit reference (each optional-only non-resident expert dropped: its row's rank-0 slot at weight 0;
    marked rank 0-1 uses sidecar -2), bit for bit; the post's drop decisions == the rule computed in numpy.
 T3 the counters: F12_CNT [opt posted, opt resolved -2] == (opt.sum(), the rule's dropped optional uses).
Usage (BigMoE venv; share lock, MEM ~4 GB): python -I res1c_check.py -> /Users/kamen/Fit12/data/res1c/res1c_check.txt
"""
import os, sys, time

OUT = "/Users/kamen/Fit12/data/res1c/res1c_check.txt"
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")


R2D = "/Users/kamen/BigMoE/data/r57/snap/frozen_t1c1/Leap_scripts"
import numpy as np
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, "/Users/kamen/BigMoE/data/r57/snap/frozen_t1c1/Leap_third_party/ms_gdn")
sys.path.insert(0, R2D)
sys.path.insert(0, "/Users/kamen/Fit12/res1c")
import mlx.core as mx
import ref2
assert os.path.dirname(os.path.abspath(ref2.__file__)) == R2D, ref2.__file__
ref2.apply()
import bigmoe.gated as BG
import bigmoe.tier as T
import bigmoe.tiers1c as T1C
import tierglu_nax6 as N6
import tierglu_nax6n as NN
import ooo1c as OC
import ooo1cq as OQ
import coh1cq
import res1c
ext = BG.ext
TG = T._tier_glu_mod()
say(f"\nres1c bit tests {time.strftime('%a %H:%M:%S')} (ref2 {R2D}; MLX {mx.__version__}; gated MODE {BG.MODE}, PAIRS {BG.PAIRS})")

# ---- the installer's ref2 transform only (the screen part is tested through masks(), the function it calls)
res1c._transform_ref2(ref2)
assert ref2._ooo_1c_call is not ref2._F12_ORIG_OOO1C and hasattr(ref2, "F12_QOM2D")
say(f"T0 transform installed: md5 guard {res1c.REF2_FN_MD5} matched; F12_QOM2D {ref2.F12_QOM2D}, F12_CNT {ref2.F12_CNT}")

PLAN, M, K, R4, NL, D = "4433333333", 16, 10, 2, 96, 2560
T.PLAN = PLAN
BARE = os.path.expanduser("~/BigMoE/models/bare")
SH4 = []
for o, i in ((640, 2560), (640, 2560), (2560, 640)):
    SH4 += [((o, i // 8), mx.uint32), ((o, i // 64), mx.bfloat16), ((o, i // 64), mx.bfloat16)]
C = 512
d_, ix = T.load_index(BARE, layout_ok=True)
nxt = [0]
P = {}
for Lm in (20, 44):
    r = ix["layers"][str(Lm)]
    assert r.get("layout") == "native+p11+pad", r.get("layout")
    L = nxt[0]
    base, _ = T.make_pools(ext, L, 512, C, SH4, d_, r)
    Ls, side, _ = T.make_sidecar(ext, 512, C, SH4, d_, r)
    nxt[0] = Ls + 1
    allids = np.arange(512, dtype=np.int32)
    for j in range(0, 512, 64):
        ext.ensure(L, mx.array(allids[j:j + 64])); ext.ensure(Ls, mx.array(allids[j:j + 64]))
    sm_of = np.array(ext.lookup(L, allids.tolist()), np.int32)
    sl_of = np.array(ext.lookup(Ls, allids.tolist()), np.int32)
    assert (sm_of >= 0).all() and (sl_of >= 0).all(), "an expert did not load"
    P[Lm] = (base, side, sm_of, sl_of)
say("pools built: layers 20 / 44, 512 experts each, native+p11+pad, C 512")
RS = 32 * (2 + BG.XLINES)


class Gate:
    def __init__(self):
        self.nl, self.nreg, self.seq = NL, 128, 9
        self.region = mx.zeros((self.nreg * BG.PAIRS * RS,), mx.int32)
        self.status = mx.zeros((4096,), mx.int32)
        mx.eval(self.region, self.status)
        self.REG = np.frombuffer(memoryview(self.region), dtype=np.int32); self.REG.setflags(write=True)
        self.STA = np.frombuffer(memoryview(self.status), dtype=np.int32)

    def release(self, Lx, slots):
        off = ((self.seq * self.nl + Lx) % self.nreg) * BG.PAIRS * RS
        for i, v in enumerate(OC.release_words(slots, self.seq)):
            self.REG[off + 32 * (i // 31) + 1 + i % 31] = np.int32(np.uint32(v).view(np.int32))
        for j in range(RS // 32 - 1, -1, -1):
            self.REG[off + 32 * j] = self.seq


GATE = Gate()
POSTS = []
B21 = 1 << 21


def loader_rule(posted, res_m):
    """BigMoE's ensure() decision on a base post: an expert whose every posted use carries bit 21 and that is not
    resident is not read -> (expert ids, per-use dropped)"""
    a = np.asarray(posted, np.int64).reshape(-1)
    opt = (a & B21) != 0
    e = a & (B21 - 1)
    req = set(e[~opt].tolist())
    drop = opt & ~np.isin(e, list(res_m)) & ~np.isin(e, list(req))
    return e, drop


def make_post(Lm, sm_of, sl_of, res_m, res_l):
    def post(Lx, ids_, after=None):
        a = np.array(ids_).astype(np.int64).reshape(-1)
        POSTS.append((Lx, a.copy()))
        if Lx == Lm:
            e, drop = loader_rule(a, res_m)
            s = np.where(np.isin(e, list(res_m)), sm_of[e], -1)
            s = np.where(drop, -2, s)                     # an optional-only cold expert: final at post (not read)
        else:
            s = np.where(np.isin(a, list(res_l)), sl_of[a], -1)
        return mx.array(np.concatenate([[0], s]).astype(np.int32)), int(a.size)
    return post


eqb = lambda a, b: bool(a.shape == b.shape and mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())


def dup_2d(idsu, mk):
    fl = np.flatnonzero(~mk.reshape(-1))
    d = idsu[:, :R4].reshape(-1)[fl[0]] if len(fl) else idsu[0, 0]
    return np.where(mk, d, idsu[:, :R4])


def call(Lm, idsu, scu, x, nolsb, res_m, res_l, qom=None, fn=None):
    """one MoE layer call through ref2._ooo_sel (fn: a temporary _ooo_1c_call); returns (y, base posts, sidecar posts, drop)"""
    msb, lsb, sm_of, sl_of = P[Lm]
    GATE.seq += 1
    t = {"base": msb, "sideP": lsb, "segs": [(0, 2, 4), (2, 10, 3)], "side": NL // 2 + Lm, "layout": 2}
    if nolsb is not None:
        t["nolsb"] = nolsb
    if qom is not None:
        t["qom"] = qom
    del POSTS[:]
    post = make_post(Lm, sm_of, sl_of, res_m, res_l)
    if nolsb is not None and getattr(nolsb[0], "ndim", 1) == 2:
        exp_s = dup_2d(idsu, np.array(nolsb[0]))
    elif nolsb is not None:
        sr = np.asarray(nolsb[1]); un = np.flatnonzero(~sr)
        exp_s = np.where(sr[:, None], idsu[un[0] if len(un) else 0, 0], idsu[:, :R4])
    else:
        exp_s = idsu[:, :R4]
    posted = np.where(np.array(qom[0]).reshape(M, K), idsu.astype(np.int64) | B21, idsu) if qom is not None else idsu
    _, drop = loader_rule(posted, res_m)
    fin_m = np.where(drop.reshape(-1), -2, sm_of[idsu].reshape(-1)).tolist()
    GATE.release(Lm, fin_m); GATE.release(NL // 2 + Lm, sl_of[exp_s].reshape(-1).tolist())
    keep = ref2._ooo_1c_call
    if fn is not None:
        ref2._ooo_1c_call = fn
    try:
        y = ref2._ooo_sel(x, mx.array(idsu.astype(np.uint32)), {"gate": GATE, "L": Lm}, t, post, scu, None)
        mx.eval(y)
    finally:
        ref2._ooo_1c_call = keep
    pb = [a for Lx, a in POSTS if Lx == Lm]
    ps = [a for Lx, a in POSTS if Lx != Lm]
    return y, (pb[0] if pb else None), (ps[0] if ps else None), drop.reshape(M, K)


def reference(Lm, idsu, scu, x, mk, drop):
    """explicit: a dropped use -> its row's rank-0 slot at weight 0; marked rank 0-1 uses sidecar -2; NAX6-N + combine"""
    msb, lsb, sm_of, sl_of = P[Lm]
    sm = sm_of[idsu]
    smr = mx.array(np.where(drop, sm[:, :1], sm).astype(np.int32))
    sl = sl_of[idsu[:, :R4]].copy(); sl[mk] = -2
    slr = mx.array(sl.astype(np.int32))
    y = NN.tier_glu_nax6n(x, msb, lsb, smr, slr, plan=PLAN, grp=N6.gpu_grp(smr, slr, PLAN))
    scr = mx.where(mx.array(drop), mx.zeros_like(scu), scu)
    return TG.combine_phase(y.reshape(M * K, -1), y.reshape(M * K, -1), mx.ones((M, K), mx.int32), scr)


def scen(rng):
    lg = (rng.standard_normal((M, 512)) * 2.0).astype(np.float32)
    top = np.argsort(-lg, axis=1)[:, :K]
    ids = top.astype(np.int64)
    w = np.take_along_axis(lg, top, 1); w = np.exp(w - w.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    uniq = np.unique(ids)
    res_m = set(rng.choice(uniq, int(0.7 * len(uniq)), replace=False).tolist())
    res_l = set(rng.choice(uniq, int(0.8 * len(uniq)), replace=False).tolist())
    x = mx.array((rng.standard_normal((M, D)) * 0.5).astype(np.float32)).astype(mx.bfloat16)
    sc = mx.array(w).astype(mx.bfloat16)
    tier = T1C.tiers_mx(mx.array(ids.astype(np.uint32)), sc, s=0.98, lam=0.016)
    mx.eval(x, sc, tier)
    return ids, w, res_m, res_l, x, sc, tier


ref2.CFG.update(ooo_1c=True, tier_nax_ooo=True, nax_1cq=True, nax_native=True)
rng = np.random.default_rng(1116_14)
ORIG = ref2._F12_ORIG_OOO1C

# ---------------- T1: OFF == the original
ok1, n1 = True, 0
for Lm in (20, 44):
    for coh in (True, False):
        ref2.CFG["nax_coh"] = coh
        for rep in range(3):
            ids, w, res_m, res_l, x, sc, tier = scen(rng)
            km = np.array(tier) != 2
            idsu = np.where(km, ids, ids[:, :1]); scu = mx.where(mx.array(km), sc, mx.zeros_like(sc))
            mk = np.array(tier)[:, :R4] != 0
            sure = np.zeros(M, bool); sure[rng.choice(M, 4, replace=False)] = True
            for lab, nl in (("1c", (mx.array(mk), None)), ("u3", (mx.array(sure), sure)), ("plain", None)):
                ya, pba, psa, _ = call(Lm, idsu, scu, x, nl, res_m, res_l)              # transformed (switch off)
                yb, pbb, psb, _ = call(Lm, idsu, scu, x, nl, res_m, res_l, fn=ORIG)     # the original
                c = (eqb(ya, yb) and np.array_equal(pba, pbb) and np.array_equal(psa, psb))
                ok1 &= c; n1 += 1
                if not c:
                    say(f"  T1 L{Lm} coh{int(coh)} rep {rep} {lab}: y {eqb(ya, yb)}, base posts {np.array_equal(pba, pbb)}, side posts {np.array_equal(psa, psb)}")
raised = False
try:
    ids, w, res_m, res_l, x, sc, tier = scen(rng)
    mk = np.array(tier)[:, :R4] != 0
    call(20, ids, sc, x, (mx.array(mk), None), res_m, res_l,
         qom=(mx.array(np.zeros((M, K), bool)), mx.ones((M,), mx.bool_), None))
except RuntimeError:
    raised = True
say(f"T1 OFF: transformed == original on 1c / 1-D U3 / plain calls, y bits + base and sidecar posts: "
    f"{'PASS' if ok1 else 'FAIL'} ({n1} pairs, COH on / off); the guard still raises for 2-D + qom: {'PASS' if raised else 'FAIL'}")

# ---------------- T2 + T3: ON == the explicit reference; the counters
ref2.F12_QOM2D[0] = True
ok2, ok3, n2, ndrop, nopt = True, True, 0, 0, 0
for Lm in (20, 44):
    for coh in (True, False):
        ref2.CFG["nax_coh"] = coh
        for res_on in (False, True):
            for rep in range(3):
                ids, w, res_m, res_l, x, sc, tier = scen(rng)
                com = rng.random(M) < 0.3
                km_, nl_, opt_ = res1c.masks(tier, mx.array(com), res_on)
                km, nl, opt = np.array(km_), np.array(nl_), np.array(opt_)
                idsu = np.where(km, ids, ids[:, :1]); scu = mx.where(mx.array(km), sc, mx.zeros_like(sc))
                ref2.F12_CNT = []
                y, pb, ps, drop = call(Lm, idsu, scu, x, (mx.array(nl), None), res_m, res_l,
                                       qom=(mx.array(opt), mx.ones((M,), mx.bool_), None))
                cnt = np.array(mx.stack(ref2.F12_CNT).sum(axis=0)) if ref2.F12_CNT else None
                ref2.F12_CNT = None
                # the rule in numpy, independently: an optional-only expert that is not resident is dropped
                e = idsu.astype(np.int64)
                req = set(e[~opt].tolist())
                drop_np = opt & ~np.isin(e, list(res_m)) & ~np.isin(e, list(req))
                yr = reference(Lm, idsu, scu, x, nl, drop_np)
                c2 = eqb(y, yr) and np.array_equal(drop, drop_np) and pb is not None and bool(((pb & B21) != 0).reshape(M, K).__eq__(opt).all())
                c3 = cnt is not None and int(cnt[0]) == int(opt.sum()) and int(cnt[1]) == int(drop_np.sum())
                ok2 &= c2; ok3 &= c3; n2 += 1; ndrop += int(drop_np.sum()); nopt += int(opt.sum())
                if not (c2 and c3):
                    say(f"  T2/T3 L{Lm} coh{int(coh)} res{int(res_on)} rep {rep}: y == ref {eqb(y, yr)}, drops == rule "
                        f"{np.array_equal(drop, drop_np)}, counters {None if cnt is None else cnt.tolist()} vs "
                        f"[{int(opt.sum())}, {int(drop_np.sum())}]")
ref2.F12_QOM2D[0] = False
say(f"T2 ON (tagged rows ~30 %, residency 70 %, RES off / on, COH on / off): y == the explicit reference bit for bit, "
    f"loader drops == the rule, bit 21 exactly on the optional uses: {'PASS' if ok2 else 'FAIL'} ({n2} calls; "
    f"{nopt} optional uses, {ndrop} dropped)")
say(f"T3 counters F12_CNT [optional posted, resolved -2] == the rule's counts: {'PASS' if ok3 else 'FAIL'} ({n2} calls)")
ALL = ok1 and raised and ok2 and ok3
say(f"RES1C BIT TESTS: {'ALL PASS' if ALL else 'FAIL'}")
sys.exit(0 if ALL else 1)
