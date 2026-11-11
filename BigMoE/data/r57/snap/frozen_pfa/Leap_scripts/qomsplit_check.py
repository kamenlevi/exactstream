"""QOM-SPLIT-1C GPU check (Leap, Fri 2026-11-06 01:4x; TABLE_350 'QOM-SPLIT-1C'; the orchestrator 01:3x: build after FLAT-2,
off by default, bitwise == FAST-NAX when off, post-drop work countable). Share lock, MEM ~4 GB.
Drives ref2's _ooo_1c_call ITSELF (the candidate REF2_PATH, default ~/Leap/data/split1c/ref2_split.py; the live
scripts/ref2.py for the off check) with a fake loader (post) and a fake release region, on the CONVERTED records3 pools
(native+p11+pad) of layers 20 / 44, all 512 experts (BigMoE's loader, read-only use), M 16, top-10, PLAN 4433333333.
Scenario: 8 QOM-sure rows (U3: rank 0-1 sidecars -2), 4 strict (cf 0.95 >= T 0.9) + 4 not (cf 0.6); optional uses = the
screen's opt (sure rows' ranks >= 2 outside the block's union); ~70 % of experts resident at post, the rest late; an
optional use the loader does not hold resolves to -2 (half at post, half late).
  reference: Breakthrough's qom_keep_mx(split=T) on the ORIGINAL post's final slots -> keep, then
    3c-N:   ooo1c.tier_glu_3cn_1c(qom=True) on the final slots with the rule's dropped pairs -2 (the kernel's QOM drop)
    NAX6-N: tierglu_nax6n.tier_glu_nax6n on the final slots (dropped -> the row's rank-0 slot) + combine_phase with
            scores x keep (ooo1cq_check's reference)
  (s) split on: y == reference, bitwise, released before (a) and 20 ms late (b); QCNT_1C == [kept, dropped, groups,
      split drops] from the reference; no poll timeout
  (o) split off: candidate y == live ref2 y, bitwise (QOM and non-QOM calls, both forms); QCNT_1C set == unset
-> ~/Leap/data/qomsplit_check.txt"""
import importlib.util, os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfa/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfa/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfa/Breakthrough_keepset"))
import mlx.core as mx
import ref2 as LIVE
LIVE.apply()                                             # as ooo1cq_check (its kernels ran under apply())
_p = os.environ.get("REF2_PATH", os.path.expanduser("~/Leap/data/split1c/ref2_split.py"))
_spec = importlib.util.spec_from_file_location("ref2_split_cand", _p)
CAND = importlib.util.module_from_spec(_spec)
sys.modules["ref2_split_cand"] = CAND
_spec.loader.exec_module(CAND)
import bigmoe.gated as BG
import bigmoe.tier as T
import tierglu_nax6 as N6
import tierglu_nax6n as NN
import ooo1c as OC
import qom_keep as QK
ext = BG.ext
TG = T._tier_glu_mod()

OUT = os.path.expanduser("~/Leap/data/qomsplit_check.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
import hashlib
say(f"\nQOM-SPLIT-1C check {time.strftime('%a %H:%M:%S')} ({os.environ.get('QSC_COND', 'share lock')}; MLX {mx.__version__}); "
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

def reference(Lm, x, ids, sc, sr, cf, opt, res, nax):
    """the split rule on the ORIGINAL post (optional uses with bit 21): Breakthrough's qom_keep_mx(split=TS)"""
    Lb, Ls, base, side, sm_of, sl_of = P[Lm]
    fin = np.where(opt & ~np.isin(ids, list(res)), -2, sm_of[ids]).astype(np.int32)
    cf_ = mx.array(np.where(sr, cf, np.nan).astype(np.float32))
    kp, _, _ = QK.qom_keep_mx(mx.array(ids), sc.astype(mx.float32), cf_, mx.array(fin >= 0), split=TS)
    keep = np.array(kp)
    assert keep[:, :2].all() and keep[~opt].all(), "the rule dropped a mandatory use"
    fl = np.where(sr[:, None], -2, sl_of[ids[:, :R4]]).astype(np.int32)
    if not nax:
        pre = np.where(keep, fin, -2).astype(np.int32)
        offs = mx.array([0, 0, 1], mx.int32)
        y, _, _ = OC.tier_glu_3cn_1c(x, base, side, mx.array(pre), mx.array(fl), PLAN, gate.region, offs, gate.status,
                                     sc.reshape(-1, K), qom=True)
    else:
        smr = mx.array(np.where(keep, fin, fin[:, :1]).astype(np.int32))
        slr = mx.array(fl)
        yy = NN.tier_glu_nax6n(x, base, side, smr, slr, plan=PLAN, grp=N6.gpu_grp(smr, slr, PLAN))
        scr = mx.where(mx.array(keep), sc, mx.zeros_like(sc))
        y = TG.combine_phase(yy.reshape(M * K, -1), yy.reshape(M * K, -1), mx.ones((M, K), mx.int32), scr)
    mx.eval(y)
    fd = np.where(keep, fin, fin[:, :1])
    groups = len(set(fd[fd >= 0].tolist()))
    strict = sr & (cf >= TS)
    return y, keep, groups, int((opt & strict[:, None]).sum()), int((opt & strict[:, None] & (fin >= 0)).sum())

eqb = lambda a, b: bool(a.size == b.size and mx.all(a.reshape(-1).view(mx.uint16) == b.reshape(-1).view(mx.uint16)).item())
rng = np.random.default_rng(1106)
ok_s, ok_o, ok_c, seq = True, True, True, 100
nst = nres = 0
for nax in (False, True):
    form = "NAX6-N 1C-Q" if nax else "3c-N 1C QOM"
    for mod in (LIVE, CAND):
        mod.CFG["nax_1cq"] = nax
        mod.CFG["nax_native"] = nax
    for Lm in (20, 44):
        for rep in range(4):
            ids, sr, cf, opt, res, res_l = scenario(rng)
            x = (mx.random.normal((M, 2560)) * 0.5).astype(mx.bfloat16)
            sc = mx.softmax(mx.random.normal((M, K)), axis=-1).astype(mx.bfloat16)
            mx.eval(x, sc)
            y_ref, keep, groups, nsd, nsr = reference(Lm, x, ids, sc, sr, cf, opt, res, nax)
            nst += nsd; nres += nsr
            # (s) the split on
            CAND.CFG["qom_split"] = TS
            CAND.QCNT_1C = []
            c0 = CAND.COUNTS["qom_split"]
            late = rep >= 2
            seq += 1
            y_s, ld, dt = run(CAND, Lm, x, ids, sc, sr, cf, opt, res, res_l, nax, late=late, seq=seq)
            q = np.array(CAND.QCNT_1C[0]).tolist() if len(CAND.QCNT_1C) == 1 else None
            want = [int(keep.sum()), M * K - int(keep.sum()), groups, nsd]
            strict = sr & (cf >= TS)
            e0 = np.repeat(ids[:, :1], K, axis=1)
            post_ok = bool(np.array_equal(ld.posted[0], np.where(opt & strict[:, None], e0, ids))
                           and np.array_equal(ld.posted[1], opt & ~strict[:, None]))
            same = (eqb(y_s, y_ref) and q == want and post_ok and CAND.COUNTS["qom_split"] == c0 + 1
                    and int(STA[0]) == 0 and (not late or dt >= 0.020))
            ok_s &= same
            say(f"  (s) {form} L{Lm} rep {rep}{' late' if late else ''}: == ref {eqb(y_s, y_ref)}, QCNT {q} vs {want}, "
                f"posted ok {post_ok}, timeouts {int(STA[0])}{f', {dt * 1e3:.1f} ms' if late else ''}"
                f" (split drops {nsd}, {nsr} of them resident)")
            # (o) the split off: candidate == live, QOM and non-QOM; QCNT set == unset
            CAND.CFG["qom_split"] = None
            for qom in (True, False):
                seq += 1
                CAND.QCNT_1C = None
                y_l, _, _ = run(LIVE, Lm, x, ids, sc, sr, cf, opt, res, res_l, nax, qom=qom, seq=seq)
                seq += 1
                y_c, _, _ = run(CAND, Lm, x, ids, sc, sr, cf, opt, res, res_l, nax, qom=qom, seq=seq)
                seq += 1
                CAND.QCNT_1C = []
                y_q, _, _ = run(CAND, Lm, x, ids, sc, sr, cf, opt, res, res_l, nax, qom=qom, seq=seq)
                qn_ = np.array(CAND.QCNT_1C[0]).tolist() if len(CAND.QCNT_1C) == 1 else None
                o1, o2 = eqb(y_c, y_l), eqb(y_q, y_l)
                ok_o &= o1 and o2 and int(STA[0]) == 0
                if qom:                                   # without the split the rule keeps resident optional uses
                    fin = np.where(opt & ~np.isin(ids, list(res)), -2, P[Lm][4][ids])
                    kq = (fin >= 0)
                    fd = np.where(kq, fin, fin[:, :1])
                    wq = [int(kq.sum()), M * K - int(kq.sum()), len(set(fd[fd >= 0].tolist())), 0]
                else:
                    sm = P[Lm][4][ids]
                    wq = [M * K, 0, len(set(sm.reshape(-1).tolist())), 0]
                ok_c &= qn_ == wq
                if not (o1 and o2 and qn_ == wq) or rep == 0:
                    say(f"  (o) {form} L{Lm} rep {rep} {'QOM' if qom else 'no QOM'}: cand == live {o1}, QCNT set == live {o2}, "
                        f"QCNT {qn_} vs {wq}")
            CAND.QCNT_1C = None
say(f"split drops exercised: {nst} ({nres} of them resident, i.e. kept without the split)")
say(f"QOM-SPLIT-1C check: (s) split == the rule {'PASS' if ok_s else 'FAIL'}; (o) off == live FAST-NAX "
    f"{'PASS' if ok_o else 'FAIL'}; QCNT_1C counts {'PASS' if ok_c else 'FAIL'}")
print("RESULT", "PASS" if (ok_s and ok_o and ok_c) else "FAIL")
