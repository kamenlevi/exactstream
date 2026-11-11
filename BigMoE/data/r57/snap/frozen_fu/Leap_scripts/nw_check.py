"""NEVER-WAIT GPU check (Leap, Fri 2026-11-06 10:xx; data/nw/NW_DESIGN.md; BigMoE docs/NEVERWAIT_IFACE.md bit checks 1-3):
the candidate ref2's _ooo_1c_call + _hc_write on two consecutive one-calls (layer A = pools of layer 20 as k 0, layer B =
layer 44's as k 1), 3c-N 1C QOM and NAX6-N 1C-Q, M 16, top-10, PLAN 4433333333, QOM + U3 as the FAST arm (qomsplit_check's
harness: BigMoE's loader read-only, a fake post and release).
  (w) warm, nothing cold: NW on == NW off bitwise (y_A, the A write, y_B, the B write)
  (f) REF2_NW_FORCE '0:e1,e2' (two of A's routed experts late): y_A == A with those pairs at weight 0; the late call ==
      A with only those pairs; the B write == hc_write(h_B + delta x inj_A, y_B, inj_B)   (float equality)
  (c) cold: some demand experts of A not resident at the post (released before A's eval): as (f) with those pairs
-> ~/Leap/data/nw_check.txt"""
import importlib.util, os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_fu/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_fu/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_fu/Breakthrough_keepset"))
import mlx.core as mx
import ref2 as LIVE
LIVE.apply()                                             # as ooo1cq_check (its kernels ran under apply())
_p = os.environ.get("REF2_PATH", os.path.expanduser("~/Leap/data/nw/ref2_cand.py"))
_spec = importlib.util.spec_from_file_location("ref2_nw_cand", _p)
CAND = importlib.util.module_from_spec(_spec)
sys.modules["ref2_nw_cand"] = CAND
_spec.loader.exec_module(CAND)
import bigmoe.gated as BG
import bigmoe.tier as T
import tierglu_nax6 as N6
import tierglu_nax6n as NN
import ooo1c as OC
import qom_keep as QK
ext = BG.ext
TG = T._tier_glu_mod()

OUT = os.path.expanduser("~/Leap/data/nw_check.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
import hashlib
say(f"\nNEVER-WAIT check {time.strftime('%a %H:%M:%S')} ({os.environ.get('QSC_COND', 'share lock')}; MLX {mx.__version__}); "
    f"candidate {_p} md5 {hashlib.md5(open(_p, 'rb').read()).hexdigest()[:8]}, live ref2 md5 "
    f"{hashlib.md5(open(LIVE.__file__, 'rb').read()).hexdigest()[:8]}")
PLAN, M, K, R4, TS = "4433333333", 16, 10, 2, 0.9
T.PLAN = PLAN
assert getattr(BG, "MODE", "spin") == "spin"
if BG.XLINES < 4:
    BG.XLINES = 4                                        # 160 pairs per region need 6 lines (the engine's MAXK sets 4)
RS = 32 * (2 + BG.XLINES)
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
    P[Lm] = (L, Ls, base, side, sm_of, sl_of)
say(f"pools built: layers 20 / 44, 512 experts each, native+p11+pad, C {C}; XLINES {BG.XLINES}, PAIRS {BG.PAIRS}")


class Gate:
    pass
gate = Gate()
gate.nl, gate.nreg, gate.seq = 96, 64, 0
gate.region = mx.zeros((gate.nreg * BG.PAIRS * RS,), mx.int32)
gate.status = mx.zeros((64,), mx.int32)
mx.eval(gate.region, gate.status)
REG = np.frombuffer(memoryview(gate.region), dtype=np.int32)
REG.setflags(write=True)
STA = np.frombuffer(memoryview(gate.status), dtype=np.int32)

def release(off, slots, seq):
    w = OC.release_words(slots, seq)
    for i, v in enumerate(w):
        REG[off + 32 * (i // 31) + 1 + i % 31] = np.int32(np.uint32(v).view(np.int32))
    for j in range(RS // 32 - 1, -1, -1):
        REG[off + 32 * j] = seq

def scenario(rng):
    ids = np.stack([rng.choice(512, K, replace=False) for _ in range(M)]).astype(np.int32)
    sr = np.zeros(M, bool)
    srs = rng.choice(M, 8, replace=False)
    sr[srs] = True
    cf = np.full(M, 0.5, np.float32)
    cf[srs[:4]] = 0.95
    cf[srs[4:]] = 0.6
    top2 = np.arange(K)[None, :] < 2
    cand = (~sr)[:, None] | top2                          # the screen: unsure rows' ranks + sure rows' ranks 0-1
    inU = np.isin(ids, ids[cand])
    opt = sr[:, None] & ~(top2 | inU)
    uniq = np.unique(ids)
    res = set(rng.choice(uniq, int(0.7 * len(uniq)), replace=False).tolist())
    res_l = set(rng.choice(uniq, int(0.8 * len(uniq)), replace=False).tolist())
    return ids, sr, cf, opt, res, res_l

class Loader:
    """the fake post: base pairs (pair m * K + r) and sidecar pairs (m * R4 + r) in posting order"""
    def __init__(self, Lb, Ls, sm_of, sl_of, res, res_l):
        self.Lb, self.Ls, self.sm_of, self.sl_of, self.res, self.res_l = Lb, Ls, sm_of, sl_of, res, res_l
        self.fin_m = self.fin_l = self.posted = None
    def __call__(self, Lx, ids_p, after=None):
        a = np.array(ids_p).astype(np.int64).reshape(-1)
        e = (a & 0xFFFFF).astype(np.int32)
        if Lx == self.Lb:
            o = ((a >> 21) & 1).astype(bool)
            r_ = np.isin(e, list(self.res))
            pre = np.where(r_, self.sm_of[e], np.where(o, np.where(e % 2 == 0, -2, -1), -1)).astype(np.int32)
            self.fin_m = np.where(r_ | ~o, self.sm_of[e], -2).astype(np.int32)
            self.posted = (e.reshape(M, K), o.reshape(M, K))
        else:
            assert Lx == self.Ls, Lx
            r_ = np.isin(e, list(self.res_l))
            pre = np.where(r_, self.sl_of[e], -1).astype(np.int32)
            self.fin_l = self.sl_of[e].astype(np.int32)
        return mx.array(np.concatenate([[0], pre]).astype(np.int32)), len(pre)

def tdict(Lm, ids, sr, cf, opt, qom=True):
    Lb, Ls, base, side, _, _ = P[Lm]
    t = {"segs": [(0, R4, 4), (R4, K, 3)], "base": base, "sideP": side, "side": Ls, "layout": 2}
    if qom:
        t["qom"] = (mx.array(opt), mx.array(sr), mx.array(cf))
        t["nolsb"] = (mx.array(sr), sr.copy())
    return t

def run(mod, Lm, x, ids, sc, sr, cf, opt, res, res_l, nax, qom=True, late=False, seq=1):
    """one _ooo_1c_call of module mod; the release written before the launch, or 20 ms after it (late)"""
    Lb, Ls, _, _, sm_of, sl_of = P[Lm]
    gate.seq = seq
    bm = {"gate": gate, "L": Lb}
    t = tdict(Lm, ids, sr, cf, opt, qom)
    ld = Loader(Lb, Ls, sm_of, sl_of, res, res_l)
    offs = np.array(mod._ooo1c_offs(bm, t, R4))
    assert offs[0] != offs[1], offs
    y = mod._ooo_1c_call(x, mx.array(ids), bm, t, ld, sc, None, nax)
    assert y is not None, "the one-call declined"
    if not late:
        release(int(offs[0]), ld.fin_m.tolist(), seq); release(int(offs[1]), ld.fin_l.tolist(), seq)
        mx.eval(y)
        dt = 0.0
    else:
        t0 = time.perf_counter()
        mx.async_eval(y)
        while time.perf_counter() - t0 < 0.020:
            pass
        release(int(offs[0]), ld.fin_m.tolist(), seq); release(int(offs[1]), ld.fin_l.tolist(), seq)
        mx.eval(y)
        dt = time.perf_counter() - t0
    return y, ld, dt


eqb = lambda a, b: bool(a.size == b.size and mx.all(a.reshape(-1).view(mx.uint16) == b.reshape(-1).view(mx.uint16)).item())
eqf = lambda a, b: bool(a.size == b.size and mx.all(a.reshape(-1).astype(mx.float32) == b.reshape(-1).astype(mx.float32)).item())
CAND._ORIG["hc_write"] = CAND._hc_write_c                 # the plain write (the candidate is not applied here)
CAND._ON[0] = False
CAND.CFG["hc_pend_mlp"] = False
CAND.CFG["neverwait"] = True
rng = np.random.default_rng(611)
seq, ok = 300, {"w": True, "f": True, "c": True}


def hcw(h, y, inj):
    return CAND._hc_write(h, y, inj)


def two_layers(sA, sB, nax, nw, force="", resA=None):
    """A (pool 20, k 0) then B (pool 44, k 1): returns y_A, w_A, y_B, w_B and the late term (via the NW state)"""
    global seq
    CAND.CFG["nw_force"] = force
    CAND.NW_RESET()
    CAND.NW_ON = nw
    CAND.NWCNT = [] if nw else None
    out = {}
    for tag, Lm, s in (("A", 20, sA), ("B", 44, sB)):
        ids, sr, cf, opt, res, res_l, x, sc, h, inj = s
        if tag == "A" and resA is not None:
            res = resA
        seq += 1
        y, ld, _ = run(CAND, Lm, x, ids, sc, sr, cf, opt, res, res_l, nax, seq=seq)
        w = hcw(h, y.reshape(1, M, -1), inj)
        mx.eval(y, w)
        out[tag] = (y, w)
    CAND.NW_ON = False
    cnt = [np.array(c).tolist() for c in CAND.NWCNT] if nw else None
    return out, cnt


def layer(Lm):
    ids, sr, cf, opt, res, res_l = scenario(rng)
    x = (mx.random.normal((M, 2560)) * 0.5).astype(mx.bfloat16)
    sc = mx.softmax(mx.random.normal((M, K)), axis=-1).astype(mx.bfloat16)
    h = (mx.random.normal((1, M, 4 * 2560)) * 0.5).astype(mx.bfloat16)
    inj = mx.sigmoid(mx.random.normal((1, M, 4))).astype(mx.bfloat16)
    mx.eval(x, sc, h, inj)
    return [ids, sr, cf, opt, res, res_l, x, sc, h, inj]


for nax in (False, True):
    form = "NAX6-N 1C-Q" if nax else "3c-N 1C QOM"
    CAND.CFG["nax_1cq"] = nax
    CAND.CFG["nax_native"] = nax
    for rep in range(3):
        sA, sB = layer(20), layer(44)
        allA = set(np.unique(sA[0]).tolist())
        sA[4] = set(allA); sB[4] = set(np.unique(sB[0]).tolist())          # warm: every routed expert resident at the post
        sA[5] = set(allA); sB[5] = set(sB[4])
        off, _ = two_layers(sA, sB, nax, False)
        on, cnt = two_layers(sA, sB, nax, True)
        w_ok = all(eqb(on[t_][i], off[t_][i]) for t_ in "AB" for i in (0, 1))
        ok["w"] &= w_ok and all(c[1] == 0 for c in cnt)
        say(f"  (w) {form} rep {rep}: NW on == off bitwise {w_ok}; NWCNT {cnt}")
        # (f) force two demand experts of A late
        ids, sr, cf, opt = sA[0], sA[1], sA[2], sA[3]
        dem = ids[~opt]
        F = list(dict.fromkeys(dem.tolist()))[:2]
        fset = np.isin(ids, F)
        on_f, cnt_f = two_layers(sA, sB, nax, True, force=f"0:{F[0]},{F[1]}")
        pend_ok = CAND.NW["pend"] is not None and CAND.NW["pend"]["k"] == 1   # B (k 1 < 47) left its own pending
        sc = sA[7]
        sA0 = list(sA); sA0[7] = mx.where(mx.array(fset), mx.zeros_like(sc), sc)
        sAf = list(sA); sAf[7] = mx.where(mx.array(fset), sc, mx.zeros_like(sc))
        ref0, _ = two_layers(sA0, sB, nax, False)                 # A without the forced pairs
        refd, _ = two_layers(sAf, sB, nax, False)                 # A's forced pairs only = the late term
        dlt = refd["A"][0].reshape(1, M, -1)
        hB, iA, iB = sB[8], sA[9], sB[9]
        g = hB.reshape(1, M, 4, -1)
        t_ = dlt[..., None, :] * iA[..., :, None]
        hB2 = mx.where(t_ == 0, g, g + t_).reshape(hB.shape)
        wB_ref = CAND._hc_write_c(hB2, ref0["B"][0].reshape(1, M, -1), iB)
        f_ok = (eqf(on_f["A"][0], ref0["A"][0]) and eqf(on_f["A"][1], ref0["A"][1]) and eqf(on_f["B"][0], ref0["B"][0])
                and eqf(on_f["B"][1], wB_ref))
        ok["f"] &= f_ok and cnt_f[0][1] == int(fset.sum()) and pend_ok
        say(f"  (f) {form} rep {rep}: forced {F} ({int(fset.sum())} pairs): y_A {eqf(on_f['A'][0], ref0['A'][0])}, w_A "
            f"{eqf(on_f['A'][1], ref0['A'][1])}, y_B {eqf(on_f['B'][0], ref0['B'][0])}, w_B (late term) {eqf(on_f['B'][1], wB_ref)};"
            f" NWCNT A {cnt_f[0]}")
        # (c) cold demand experts at A's post, released before A's eval
        C = list(dict.fromkeys(dem.tolist()))[2:4]
        cset = np.isin(ids, C) & ~opt
        resA = set(allA) - set(C)
        on_c, cnt_c = two_layers(sA, sB, nax, True, resA=resA)
        sA0 = list(sA); sA0[7] = mx.where(mx.array(cset), mx.zeros_like(sc), sc)
        sAc = list(sA); sAc[7] = mx.where(mx.array(cset), sc, mx.zeros_like(sc))
        ref0, _ = two_layers(sA0, sB, nax, False)
        refd, _ = two_layers(sAc, sB, nax, False)
        dlt = refd["A"][0].reshape(1, M, -1)
        t_ = dlt[..., None, :] * iA[..., :, None]
        hB2 = mx.where(t_ == 0, g, g + t_).reshape(hB.shape)
        wB_ref = CAND._hc_write_c(hB2, ref0["B"][0].reshape(1, M, -1), iB)
        c_ok = (eqf(on_c["A"][0], ref0["A"][0]) and eqf(on_c["B"][0], ref0["B"][0]) and eqf(on_c["B"][1], wB_ref))
        ok["c"] &= c_ok and cnt_c[0][1] == int(cset.sum()) and int(STA[0]) == 0
        say(f"  (c) {form} rep {rep}: cold {C} ({int(cset.sum())} demand pairs): y_A {eqf(on_c['A'][0], ref0['A'][0])}, y_B "
            f"{eqf(on_c['B'][0], ref0['B'][0])}, w_B (late term) {eqf(on_c['B'][1], wB_ref)}; NWCNT A {cnt_c[0]}; "
            f"timeouts {int(STA[0])}")
    # (l) layer 47: A as the last MoE layer (k 47): forced rank-0 pairs stay (wait), the other forced pairs drop, no pending
    sA, sB = layer(20), layer(44)
    allA = set(np.unique(sA[0]).tolist()); sA[4] = set(allA); sA[5] = set(allA)
    ids = sA[0]
    F = list(dict.fromkeys(ids[:, 0].tolist()))[:1] + [e for e in dict.fromkeys(ids[:, 3].tolist()) if e not in ids[:, 0]][:1]
    fset = np.isin(ids, F)
    CAND.CFG["nw_force"] = f"47:{F[0]},{F[1]}"
    CAND.NW_RESET(); CAND.NW["k"] = 47; CAND.NW_ON = True; CAND.NWCNT = []
    seq += 1
    y47, _, _ = run(CAND, 20, sA[6], ids, sA[7], sA[1], sA[2], sA[3], sA[4], sA[5], nax, seq=seq)
    mx.eval(y47)
    CAND.NW_ON = False
    c47 = np.array(CAND.NWCNT[0]).tolist()
    pend47 = CAND.NW["pend"]
    drop = fset & (np.arange(K)[None, :] > 0)
    s47 = list(sA); s47[7] = mx.where(mx.array(drop), mx.zeros_like(sA[7]), sA[7])
    CAND.CFG["nw_force"] = ""
    CAND.NW_RESET()
    seq += 1
    y47r, _, _ = run(CAND, 20, s47[6], ids, s47[7], s47[1], s47[2], s47[3], s47[4], s47[5], nax, seq=seq)
    l_ok = eqf(y47, y47r) and pend47 is None and c47[3] == int(drop.sum()) and c47[4] == int((fset & (np.arange(K)[None, :] == 0)).sum())
    ok.setdefault("l", True)
    ok["l"] &= l_ok
    say(f"  (l) {form} layer 47: forced {F}: y == ref (rank-0 kept, others dropped) {eqf(y47, y47r)}, nothing pending "
        f"{pend47 is None}, NWCNT {c47} (drops {int(drop.sum())}, rank-0 stalls {int((fset & (np.arange(K)[None, :] == 0)).sum())})")
CAND.NW_RESET()
say(f"NEVER-WAIT check: (l) layer 47 {'PASS' if ok.get('l') else 'FAIL'}")
say(f"NEVER-WAIT check: (w) warm == off {'PASS' if ok['w'] else 'FAIL'}; (f) FORCE == reference {'PASS' if ok['f'] else 'FAIL'}; "
    f"(c) cold == reference {'PASS' if ok['c'] else 'FAIL'}")
print("RESULT", "PASS" if all(ok.values()) else "FAIL")
