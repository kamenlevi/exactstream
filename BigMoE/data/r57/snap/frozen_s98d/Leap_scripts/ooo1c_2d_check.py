"""1c-tiers ref2 check (Leap, Mon 2026-11-16; Fit12 LOSSY_1C_TIERS_SPEC.md section 3; BigMoE r57_screen_fast _t1c).
ref2's FAST-NAX one-call (tier_nax_ooo + ooo_1c + nax_1cq + nax_native; COH on, as lossy-45, and off) with a 2-D
t["nolsb"] = (mask [M, R4] bool, None), driven through ref2._ooo_sel on the CONVERTED records3 pools of layers 20 / 44
(BigMoE's loader, read-only use; all 512 experts loaded), a stand-in gate (release regions in BigMoE's layout, as
ooo1c_wire_check) and a stand-in post that maps each POSTED id to its slot and records it. 16-row calls, top-10, PLAN
4433333333, R4 2. The reference: NN.tier_glu_nax6n on the final slots (sidecar -2 = tier 3) + combine_phase.
 (1) 1-D U3 sure rows (4 rows): == the reference (sidecar -2 on the sure rows), bitwise; sidecar posts == the 1-D rule.
 (2) 2-D marks equal to (1)'s rows: y, posts and the kernel's pre_l / rs_l == (1)'s, bit for bit.
 (3) 2-D per-use marks (random ~40 % of rank 0-1 uses, mixed within rows), COH on and off: == the reference with -2 at
     the marked uses; pre_l -2 exactly there; posts: unmarked = own id, marked = the first unmarked slot's id.
 (4) 2-D all marked: every sidecar post = ids[0, 0]; == the reference. (5) none marked: == the call without nolsb.
 (6) 1c as BigMoE runs it: bigmoe.tiers1c.tiers_mx on the scenario's ids + scores, drops through the km path (rank-0 id,
     score 0); marks tier[:, :2] != 0 and == 1: each == its reference; the LSB ids each posts that are not 4-bit.
 (7) late release (20 ms after launch, 2 LP1 sidecars resolving to -2) with 2-D marks: == the reference, no timeout.
 (8) guards (a 2-D mask with nax_1cq off / REF2 off / t["qom"] / a gate without regions -> RuntimeError); eprep and
     G2 bypassed for 2-D marks (== the reference, COUNTS eprep / g2_3cn unchanged); t["nolsb"] consumed;
     ref2.self_test; eprep_k.check.
 (9) LIVE vs CANDIDATE: REF2_DIR = the directory of the ref2.py under test (default /Users/kamen/BigMoE/data/r57/snap/frozen_s98d/Leap_scripts). --save X.npz
     stores (1)'s and the no-nolsb call's outputs (y bits, posts, pre_l, rs) for both layers and COH on / off;
     --compare A.npz B.npz checks them bit for bit. The 2-D parts run only on a ref2 that has them.
Usage: [BigMoE venv] ooo1c_2d_check.py [--save X.npz] | --compare A.npz B.npz
  -> ~/Leap/data/1ctiers/ooo1c_2d_check.txt (GPU, MEM ~4 GB; share lock)"""
import os, sys, time

OUT = os.path.expanduser("~/Leap/data/1ctiers/ooo1c_2d_check.txt")


def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")


if len(sys.argv) > 1 and sys.argv[1] == "--compare":
    import numpy as np
    A, B = np.load(sys.argv[2]), np.load(sys.argv[3])
    ks = sorted(set(A.files) | set(B.files))
    bad = [k for k in ks if k not in A.files or k not in B.files or A[k].shape != B[k].shape or not np.array_equal(A[k], B[k])]
    say(f"(9) live vs candidate, {len(ks)} arrays ({os.path.basename(sys.argv[2])} vs {os.path.basename(sys.argv[3])}): "
        f"{'PASS, bit for bit' if not bad else 'FAIL ' + str(bad[:6])}")
    sys.exit(1 if bad else 0)

SAVE = sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else None
R2D = os.path.abspath(os.path.expanduser(os.environ.get("REF2_DIR", "/Users/kamen/BigMoE/data/r57/snap/frozen_s98d/Leap_scripts")))
import numpy as np
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98d/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98d/Leap_scripts"))
sys.path.insert(0, R2D)
import mlx.core as mx
import ref2
assert os.path.dirname(os.path.abspath(ref2.__file__)) == R2D, (ref2.__file__, R2D)
import hashlib
R2MD5 = hashlib.md5(open(ref2.__file__, "rb").read()).hexdigest()
ref2.apply()
import bigmoe.gated as BG
import bigmoe.tier as T
import tierglu_nax6 as N6
import tierglu_nax6n as NN
import ooo1c as OC
import ooo1cq as OQ
import coh1cq
ext = BG.ext
TG = T._tier_glu_mod()
HAS2D = hasattr(ref2, "_ar_i32")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
say(f"\n1c-tiers ref2 check {time.strftime('%a %H:%M:%S')} (ref2 {R2D} md5 {R2MD5}; 2-D parts {'on' if HAS2D else 'OFF (live ref2)'}; "
    f"MLX {mx.__version__}; gated MODE {BG.MODE}, PAIRS {BG.PAIRS}, XLINES {BG.XLINES})")

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
    assert base["gate_proj"]["s3"].dtype == mx.uint32, "p11 pools expected on the converted store"
    P[Lm] = (base, side, sm_of, sl_of)
say(f"pools built: layers 20 / 44, 512 experts each, native+p11+pad, C {C}")

RS = 32 * (2 + BG.XLINES)


class Gate:                                                    # the parts of gated.Gate the one-call path reads
    def __init__(self, regions=True):
        self.nl, self.nreg, self.seq = NL, 128, 9
        if regions:
            self.region = mx.zeros((self.nreg * BG.PAIRS * RS,), mx.int32)
            self.status = mx.zeros((4096,), mx.int32)
            mx.eval(self.region, self.status)
            self.REG = np.frombuffer(memoryview(self.region), dtype=np.int32)
            self.REG.setflags(write=True)
            self.STA = np.frombuffer(memoryview(self.status), dtype=np.int32)

    def release(self, Lx, slots):
        off = ((self.seq * self.nl + Lx) % self.nreg) * BG.PAIRS * RS
        for i, v in enumerate(OC.release_words(slots, self.seq)):
            self.REG[off + 32 * (i // 31) + 1 + i % 31] = np.int32(np.uint32(v).view(np.int32))
        for j in range(RS // 32 - 1, -1, -1):
            self.REG[off + 32 * j] = self.seq


GATE = Gate()
POSTS = []                                                     # (Lx, posted ids) per post call
CAP = []                                                       # the one-call kernel's (pre_m, pre_l, rs_m, rs_l)
for mod, name in ((OQ, "tier_glu_nax6n_1cq"), (coh1cq, "tier_glu_nax6n_1cq_coh")):
    f0 = getattr(mod, name)

    def w(*a, _f0=f0, **k):
        out = _f0(*a, **k)
        CAP.append((a[3], a[4], out[1], out[2]))
        return out
    setattr(mod, name, w)


def make_post(Lm, sm_of, sl_of, res_m, res_l, lp1=()):
    def post(Lx, ids_, after=None):
        a = np.array(ids_).astype(np.int64).reshape(-1)        # a sync (the stand-in only; the real loader reads them async)
        POSTS.append((Lx, a.copy()))
        if Lx == Lm:
            s = np.where(np.isin(a, list(res_m)), sm_of[a], -1)
        else:
            s = np.where(np.isin(a, list(res_l)) & ~np.isin(a, list(lp1)), sl_of[a], -1)
        return mx.array(np.concatenate([[0], s]).astype(np.int32)), int(a.size)
    return post


def ref(x, msb, lsb, sm, sl, sc):
    smr, slr = mx.array(sm.astype(np.int32)), mx.array(sl.astype(np.int32))
    y = NN.tier_glu_nax6n(x, msb, lsb, smr, slr, plan=PLAN, grp=N6.gpu_grp(smr, slr, PLAN))
    return TG.combine_phase(y.reshape(M * K, -1), y.reshape(M * K, -1), mx.ones((M, K), mx.int32), sc)


eqb = lambda a, b: bool(a.shape == b.shape and mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())
bits = lambda a: np.array(a.view(mx.uint16))


def dup_2d(idsu, mk):
    """the expected sidecar posts for 2-D marks: the first unmarked slot's id (row-major), none -> ids[0, 0]"""
    fl = np.flatnonzero(~mk.reshape(-1))
    d = idsu[:, :R4].reshape(-1)[fl[0]] if len(fl) else idsu[0, 0]
    return np.where(mk, d, idsu[:, :R4])


def call(Lm, idsu, scu, x, nolsb, res_m, res_l, lp1=(), late=False, gate=None, extra=None, idt=np.int32):
    """one MoE layer call through ref2._ooo_sel; returns (y, base posts, sidecar posts, (pre_m, pre_l, rs_m, rs_l), dt)"""
    msb, lsb, sm_of, sl_of = P[Lm]
    g = gate or GATE
    g.seq += 1
    t = {"base": msb, "sideP": lsb, "segs": [(0, 2, 4), (2, 10, 3)], "side": NL // 2 + Lm, "layout": 2}
    if nolsb is not None:
        t["nolsb"] = nolsb
    if extra:
        t.update(extra)
    del POSTS[:], CAP[:]
    post = make_post(Lm, sm_of, sl_of, res_m, res_l, lp1)
    exp_s = None
    if nolsb is not None and getattr(nolsb[0], "ndim", 1) == 2:
        exp_s = dup_2d(idsu, np.array(nolsb[0]))
    elif nolsb is not None:
        sr = np.asarray(nolsb[1]); un = np.flatnonzero(~sr)
        exp_s = np.where(sr[:, None], idsu[un[0] if len(un) else 0, 0], idsu[:, :R4])
    else:
        exp_s = idsu[:, :R4]
    fin_m = sm_of[idsu].reshape(-1).tolist()
    fin_l = np.where(np.isin(exp_s, list(lp1)), -2, sl_of[exp_s]).reshape(-1).tolist()
    if not late and gate is None:                              # (a) released before the call
        g.release(Lm, fin_m); g.release(NL // 2 + Lm, fin_l)
    t0 = time.perf_counter()
    y = ref2._ooo_sel(x, mx.array(idsu.astype(idt)), {"gate": g, "L": Lm}, t, post, scu, None)
    left = "nolsb" in t
    if late:
        mx.async_eval(y)
        while time.perf_counter() - t0 < 0.020:
            pass
        g.release(Lm, fin_m); g.release(NL // 2 + Lm, fin_l)
    mx.eval(y)
    dt = time.perf_counter() - t0
    cap = tuple(np.array(a) for a in CAP[-1]) if CAP else None
    pb = [a for Lx, a in POSTS if Lx == Lm]
    ps = [a for Lx, a in POSTS if Lx != Lm]
    return y, (pb[0] if pb else None), (ps[0] if ps else None), cap, dt, exp_s, left


def scenario(rng, fm=0.7, fl=0.8):
    ids = np.stack([rng.choice(512, K, replace=False) for _ in range(M)]).astype(np.int64)
    uniq = np.unique(ids)
    res_m = set(rng.choice(uniq, int(fm * len(uniq)), replace=False).tolist())
    res_l = set(rng.choice(uniq, int(fl * len(uniq)), replace=False).tolist())
    x = mx.array((rng.standard_normal((M, D)) * 0.5).astype(np.float32)).astype(mx.bfloat16)
    lg = rng.standard_normal((M, K)).astype(np.float32)
    sc = mx.softmax(mx.array(lg), axis=-1).astype(mx.bfloat16)
    mx.eval(x, sc)
    return ids, res_m, res_l, x, sc


def ref_for(Lm, idsu, scu, x, mk, lp1=(), exp_s=None):
    msb, lsb, sm_of, sl_of = P[Lm]
    sl = sl_of[idsu[:, :R4]].copy()
    if exp_s is not None and len(lp1):
        sl[np.isin(exp_s, list(lp1)) & ~mk] = -2
    sl[mk] = -2
    return ref(x, msb, lsb, sm_of[idsu], sl, scu)


ref2.CFG.update(ooo_1c=True, tier_nax_ooo=True, nax_1cq=True, nax_native=True)
ok = True
SAV = {}
rng = np.random.default_rng(1116)                             # (1) and the no-nolsb call: the same draws on any ref2
rng2 = np.random.default_rng(1117)                            # the 2-D parts' own draws (so the saved arrays match)
n1 = n2 = n3 = n4 = n5 = 0
ok1 = ok2 = ok3 = ok4 = ok5 = True
for Lm in (20, 44):
    for coh in (True, False):
        ref2.CFG["nax_coh"] = coh
        for rep in range(3):
            ids, res_m, res_l, x, sc = scenario(rng)
            sure = np.zeros(M, bool); sure[rng.choice(M, 4, replace=False)] = True
            # (1) 1-D
            y1, pb1, ps1, cap1, _, exp1, left1 = call(Lm, ids, sc, x, (mx.array(sure), sure), res_m, res_l)
            mk1 = np.repeat(sure[:, None], R4, axis=1)
            yr = ref_for(Lm, ids, sc, x, mk1)
            c1 = (eqb(y1, yr) and np.array_equal(ps1, exp1.reshape(-1)) and np.array_equal(cap1[1].reshape(M, R4) == -2, mk1)
                  and not left1 and int(GATE.STA[0]) == 0)
            ok1 &= c1; n1 += 1
            # the no-nolsb call (for (5) and the live / candidate comparison)
            y0, pb0, ps0, cap0, _, _, _ = call(Lm, ids, sc, x, None, res_m, res_l)
            tag = f"L{Lm}_coh{int(coh)}_r{rep}"
            for nm, (yy, pb, ps, cp) in (("u3", (y1, pb1, ps1, cap1)), ("none", (y0, pb0, ps0, cap0))):
                SAV[f"{tag}_{nm}_y"] = bits(yy); SAV[f"{tag}_{nm}_pb"] = pb; SAV[f"{tag}_{nm}_ps"] = ps
                for ci, cn in enumerate(("pm", "pl", "rsm", "rsl")):
                    SAV[f"{tag}_{nm}_{cn}"] = cp[ci]
            if not HAS2D:
                continue
            # (2) 2-D marks equal to the sure rows
            y2, pb2, ps2, cap2, _, _, left2 = call(Lm, ids, sc, x, (mx.array(mk1), None), res_m, res_l)
            c2 = (eqb(y2, y1) and np.array_equal(pb2, pb1) and np.array_equal(ps2, ps1)
                  and all(np.array_equal(a, b) for a, b in zip(cap2, cap1)) and not left2)
            ok2 &= c2; n2 += 1
            # (3) 2-D per-use marks
            mk = rng2.random((M, R4)) < 0.4
            y3, pb3, ps3, cap3, _, exp3, left3 = call(Lm, ids, sc, x, (mx.array(mk), None), res_m, res_l)
            yr3 = ref_for(Lm, ids, sc, x, mk)
            c3 = (eqb(y3, yr3) and np.array_equal(ps3, exp3.reshape(-1)) and np.array_equal(cap3[1].reshape(M, R4) == -2, mk)
                  and np.array_equal(cap3[3].reshape(M, R4) == -2, mk) and not left3 and int(GATE.STA[0]) == 0)
            ok3 &= c3; n3 += 1
            # (4) all marked; (5) none marked
            mka = np.ones((M, R4), bool)
            y4, _, ps4, _, _, _, _ = call(Lm, ids, sc, x, (mx.array(mka), None), res_m, res_l)
            c4 = eqb(y4, ref_for(Lm, ids, sc, x, mka)) and bool((ps4 == ids[0, 0]).all())
            ok4 &= c4; n4 += 1
            y5, pb5, ps5, _, _, _, _ = call(Lm, ids, sc, x, (mx.array(np.zeros((M, R4), bool)), None), res_m, res_l)
            c5 = eqb(y5, y0) and np.array_equal(pb5, pb0) and np.array_equal(ps5, ps0)
            ok5 &= c5; n5 += 1
            if not (c1 and c2 and c3 and c4 and c5):
                say(f"  {tag}: (1) {c1} (2) {c2} (3) {c3} (4) {c4} (5) {c5}; timeouts {int(GATE.STA[0])}")
say(f"(1) 1-D U3 sure rows via ref2 == NAX6-N on the final slots + combine, posts == the 1-D rule, consumed: "
    f"{'PASS' if ok1 else 'FAIL'} ({n1} calls, COH on / off)")
ok &= ok1
if HAS2D:
    say(f"(2) 2-D marks == (1)'s rows: y, posts, pre_l, rs bit for bit: {'PASS' if ok2 else 'FAIL'} ({n2})")
    say(f"(3) 2-D per-use marks (~40 %): == the reference with -2 at the marked uses; pre_l / rs_l -2 exactly there; posts "
        f"== the first-unmarked rule: {'PASS' if ok3 else 'FAIL'} ({n3})")
    say(f"(4) all marked: posts all ids[0, 0], == the reference: {'PASS' if ok4 else 'FAIL'} ({n4}); "
        f"(5) none marked == no nolsb, bit for bit: {'PASS' if ok5 else 'FAIL'} ({n5})")
    ok &= ok2 and ok3 and ok4 and ok5

    # (6) 1c as BigMoE runs it
    import bigmoe.tiers1c as T1C
    ref2.CFG["nax_coh"] = True
    ok6, n6, xs = True, 0, {"!=0": [0, 0, 0], "==1": [0, 0, 0]}  # [non-4-bit LSB ids posted, distinct LSB ids posted, calls]
    shares, tmis = [], 0
    for Lm in (20, 44):
        for rep in range(6):
            ids, res_m, res_l, x, _ = scenario(rng2)
            lg = (rng2.standard_normal((M, 512)) * 2.0).astype(np.float32)   # router logits -> top-10, normalised over the 10
            top = np.argsort(-lg, axis=1)[:, :K]
            ids = top.astype(np.int64)
            w = np.take_along_axis(lg, top, 1); w = np.exp(w - w.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
            uniq = np.unique(ids)
            res_m = set(rng2.choice(uniq, int(0.7 * len(uniq)), replace=False).tolist())
            res_l = set(rng2.choice(uniq, int(0.8 * len(uniq)), replace=False).tolist())
            sc = mx.array(w).astype(mx.bfloat16)
            tier = np.array(T1C.tiers_mx(mx.array(ids.astype(np.uint32)), sc, s=0.98, lam=0.016)).astype(np.int64)
            tmis += int((tier != T1C.tiers_np(ids, np.array(sc.astype(mx.float32)), s=0.98, lam=0.016)).sum())   # info only
            km = tier != 2
            idsu = np.where(km, ids, ids[:, :1])
            scu = mx.where(mx.array(km), sc, mx.zeros_like(sc))
            shares.append([(tier == v).mean() for v in (0, 1, 2)])
            t4 = set(ids[tier == 0].tolist())
            for lab, mk in (("!=0", tier[:, :R4] != 0), ("==1", tier[:, :R4] == 1)):
                y6, _, ps6, cap6, _, exp6, _ = call(Lm, idsu, scu, x, (mx.array(mk), None), res_m, res_l, idt=np.uint32)
                c6 = eqb(y6, ref_for(Lm, idsu, scu, x, mk)) and np.array_equal(ps6, exp6.reshape(-1))
                ok6 &= c6; n6 += 1
                u = set(ps6.tolist())
                xs[lab][0] += len(u - t4); xs[lab][1] += len(u); xs[lab][2] += 1
                if not c6:
                    say(f"  (6) L{Lm} rep {rep} {lab}: == reference {eqb(y6, ref_for(Lm, idsu, scu, x, mk))}, posts {np.array_equal(ps6, exp6.reshape(-1))}")
    sh = np.mean(shares, axis=0)
    say(f"(6) 1c via tiers_mx (uint32 ids; {tmis} slots differ from tiers_np, info) + km drops, marks != 0 and == 1: each "
        f"== its reference, posts == the rule: "
        f"{'PASS' if ok6 else 'FAIL'} ({n6} calls; random router logits, slot shares 4-bit / MSB-3 / drop "
        f"{sh[0]:.2f} / {sh[1]:.2f} / {sh[2]:.2f}). LSB ids posted per call that are not 4-bit experts: != 0 "
        f"{xs['!=0'][0] / xs['!=0'][2]:.2f} of {xs['!=0'][1] / xs['!=0'][2]:.2f} distinct; == 1 {xs['==1'][0] / xs['==1'][2]:.2f} "
        f"of {xs['==1'][1] / xs['==1'][2]:.2f}")
    ok &= ok6

    # (7) late release with 2-D marks and 2 LP1 sidecars
    ok7, n7 = True, 0
    for Lm in (20, 44):
        for rep in range(3):
            ids, res_m, res_l, x, sc = scenario(rng2)
            mk = rng2.random((M, R4)) < 0.4
            nl_ids = sorted(set(ids[:, :R4][~mk].tolist()) - res_l)
            lp1 = set(rng2.choice(nl_ids, min(2, len(nl_ids)), replace=False).tolist()) if nl_ids else set()
            y7, _, ps7, cap7, dt, exp7, _ = call(Lm, ids, sc, x, (mx.array(mk), None), res_m, res_l, lp1=lp1, late=True)
            c7 = eqb(y7, ref_for(Lm, ids, sc, x, mk, lp1, exp7)) and dt >= 0.020 and int(GATE.STA[0]) == 0
            ok7 &= c7; n7 += 1
            if not c7:
                say(f"  (7) L{Lm} rep {rep}: == reference {eqb(y7, ref_for(Lm, ids, sc, x, mk, lp1, exp7))}, {1e3 * dt:.1f} ms, "
                    f"timeouts {int(GATE.STA[0])}")
    say(f"(7) late release (20 ms) + LP1 sidecars, 2-D marks: == the reference, no timeout: {'PASS' if ok7 else 'FAIL'} ({n7})")
    ok &= ok7

    # (8) guards, bypasses, self tests
    ids, res_m, res_l, x, sc = scenario(rng2)
    mk = rng2.random((M, R4)) < 0.4

    def raises(fn):
        try:
            fn()
        except RuntimeError as e:
            return "1c-tiers" in str(e)
        return False
    g = []
    ref2.CFG["nax_1cq"] = False
    g.append(raises(lambda: call(20, ids, sc, x, (mx.array(mk), None), res_m, res_l)))
    ref2.CFG["nax_1cq"] = True
    ref2._ON[0] = False
    try:
        g.append(raises(lambda: call(20, ids, sc, x, (mx.array(mk), None), res_m, res_l)))
    finally:
        ref2._ON[0] = True
    g.append(raises(lambda: call(20, ids, sc, x, (mx.array(mk), None), res_m, res_l,
                                 extra={"qom": (mx.zeros((M, K), mx.bool_), mx.zeros((M,), mx.bool_), None)})))
    g.append(raises(lambda: call(20, ids, sc, x, (mx.array(mk), None), res_m, res_l, gate=Gate(regions=False))))
    yr8 = ref_for(20, ids, sc, x, mk)
    e0, g0 = ref2.COUNTS["eprep"], ref2.COUNTS["g2_3cn"]
    ref2.CFG["eprep"] = True
    y8a, *_ = call(20, ids, sc, x, (mx.array(mk), None), res_m, res_l)
    ref2.CFG["eprep"] = False
    ref2.CFG["g2_rows"] = 16
    y8b, *_ = call(20, ids, sc, x, (mx.array(mk), None), res_m, res_l)
    ref2.CFG["g2_rows"] = 0
    byp = eqb(y8a, yr8) and eqb(y8b, yr8) and ref2.COUNTS["eprep"] == e0 and ref2.COUNTS["g2_3cn"] == g0
    st = ref2.self_test()
    st_ok = all(bool(v) for v in st.values())
    import eprep_k
    ek = eprep_k.check()                                       # the number of cases that differ
    ek_ok = ek == 0
    say(f"(8) guards raise (nax_1cq off / REF2 off / with t['qom'] / no regions): {g}; eprep + G2 bypassed for 2-D "
        f"(== the reference, COUNTS unchanged): {byp}; ref2.self_test: {'PASS' if st_ok else 'FAIL ' + str(st)}; "
        f"eprep_k.check: {ek} of 400 differ")
    ok &= all(g) and byp and st_ok and ek_ok
else:
    st = ref2.self_test()
    st_ok = all(bool(v) for v in st.values())
    say(f"ref2.self_test: {'PASS' if st_ok else 'FAIL ' + str(st)}")
    ok &= st_ok
if SAVE:
    np.savez(SAVE, **SAV)
    say(f"saved {len(SAV)} arrays -> {SAVE}")
say(f"1c-tiers ref2 check: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
