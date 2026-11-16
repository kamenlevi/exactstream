"""WEEK98 MODULE 2 (MoonShots): Leap's bit gate #1 for module 2 candidates: == ooo1cq bit for bit on
~/Leap/scripts/week98/ooo1c_m32_check.py's 24 scenarios. A copy of that check's setup (layers 20 / 44's converted pools
via BigMoE's loader, read-only; BIGMOE_MAXK 320; the release gate) and of its scenario draws (same seed, same draw
order: random routing, a popular expert on all 32 rows, two experts on > 16 rows, 2-D nolsb marks, late release; COH
on / off). Per scenario its three calls (32 rows; rows 0-15; rows 16-31) run through ref2._ooo_sel with ooo1cq (COH off)
or coh1cq (COH on) as the reference, then with each candidate form swapped in for ref2's f1c (m2_cand.tier_glu_1cq, its
COH twin for COH on): every output must equal the reference's bit for bit.
usage: ~/BigMoE/.venv/bin/python -I -B m2_m32.py [--forms C1,C1d2,C2,C1C2,C1d2C2]   (MEM ~3.5 GB, share lock)"""
import argparse
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--forms", default="C1,C1d2,C2,C1C2,C1d2C2")
ap.add_argument("--out", default=os.path.expanduser("~/MoonShots/week98/m2/m2_m32.txt"))
A_ = ap.parse_args()
os.environ["BIGMOE_MAXK"] = "320"
import numpy as np                                       # noqa: E402
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("~/Leap/third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx                                    # noqa: E402
import ref2                                              # noqa: E402
ref2.apply()
import bigmoe.gated as BG                                # noqa: E402
import bigmoe.tier as T                                  # noqa: E402
import ooo1c as OC                                       # noqa: E402
import ooo1cq as OQ                                      # noqa: E402
import coh1cq as CQ                                      # noqa: E402
import m2_cand as MC                                     # noqa: E402

L_ = []


def say(s):
    print(s, flush=True)
    L_.append(s)


# ---- Leap's ooo1c_m32_check.py setup (~/Leap/scripts/week98/ooo1c_m32_check.py), copied ----
assert BG.MAXK == 320 and BG.XLINES == 9, (BG.MAXK, BG.XLINES)
ext = BG.ext
PLAN, K, R4, NL, D = "4433333333", 10, 2, 96, 2560
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
RS = 32 * (2 + BG.XLINES)


class Gate:
    def __init__(self):
        self.nl, self.nreg, self.seq = NL, 128, 9
        self.region = mx.zeros((self.nreg * BG.PAIRS * RS,), mx.int32)
        self.status = mx.zeros((4096,), mx.int32)
        mx.eval(self.region, self.status)
        self.REG = np.frombuffer(memoryview(self.region), dtype=np.int32)
        self.REG.setflags(write=True)

    def release(self, Lx, slots):
        off = ((self.seq * self.nl + Lx) % self.nreg) * BG.PAIRS * RS
        w = OC.release_words(slots, self.seq)
        assert len(w) <= 31 * (RS // 32), (len(w), RS)
        for i, v in enumerate(w):
            self.REG[off + 32 * (i // 31) + 1 + i % 31] = np.int32(np.uint32(v).view(np.int32))
        for j in range(RS // 32 - 1, -1, -1):
            self.REG[off + 32 * j] = self.seq


GATE = Gate()
POSTS = []


def call(Lm, ids, sc, x, nolsb=None, late=False, lateset=()):
    msb, lsb, sm_of, sl_of = P[Lm]
    GATE.seq += 1
    t = {"base": msb, "sideP": lsb, "segs": [(0, 2, 4), (2, 10, 3)], "side": NL // 2 + Lm, "layout": 2}
    if nolsb is not None:
        t["nolsb"] = (mx.array(nolsb), None)
    del POSTS[:]

    def post(Lx, ids_, after=None):
        a = np.array(ids_).astype(np.int64).reshape(-1)
        POSTS.append((Lx, a.copy()))
        s = sm_of[a] if Lx == Lm else sl_of[a]
        s = np.where(np.isin(a, list(lateset)), -1, s)
        return mx.array(np.concatenate([[0], s]).astype(np.int32)), int(a.size)
    if nolsb is not None:
        fl = np.flatnonzero(~nolsb.reshape(-1))
        d = ids[:, :R4].reshape(-1)[fl[0]] if len(fl) else ids[0, 0]
        exp_s = np.where(nolsb, d, ids[:, :R4])
    else:
        exp_s = ids[:, :R4]
    fin_m, fin_l = sm_of[ids].reshape(-1).tolist(), sl_of[exp_s].reshape(-1).tolist()
    if not late:
        GATE.release(Lm, fin_m); GATE.release(NL // 2 + Lm, fin_l)
    t0 = time.perf_counter()
    y = ref2._ooo_sel(x, mx.array(ids.astype(np.int32)), {"gate": GATE, "L": Lm}, t, post, mx.array(sc), None)
    if late:
        mx.async_eval(y)
        while time.perf_counter() - t0 < 0.020:
            pass
        GATE.release(Lm, fin_m); GATE.release(NL // 2 + Lm, fin_l)
    mx.eval(y)
    return y


def scen(rng, kind, M=32):
    ids = np.stack([rng.choice(512, K, replace=False) for _ in range(M)]).astype(np.int64)
    if kind in ("pop1", "pop2", "pop2_2d"):
        ids[:, 0] = 7
        for m in range(M):
            ids[m, 1:] = rng.choice(np.setdiff1d(np.arange(512), [7, 11]), K - 1, replace=False)
    if kind in ("pop2", "pop2_2d"):
        ids[:20, 2] = 11
    lg = rng.standard_normal((M, K)).astype(np.float32)
    sc = np.array(mx.softmax(mx.array(lg), axis=-1).astype(mx.bfloat16).astype(mx.float32))
    x = (rng.standard_normal((M, D)) * 0.5).astype(np.float32)
    mk = (rng.random((M, R4)) < 0.6) if kind.endswith("_2d") else None
    return ids, sc, x, mk


FORMS = A_.forms.split(",")
F0, FC0 = OQ.tier_glu_nax6n_1cq, CQ.tier_glu_nax6n_1cq_coh
import tierglu as TGm                                    # noqa: E402
C0 = TGm.combine_phase
CAPD = {}


def _fcap(*a, **kw):
    out = F0(*a, **kw)
    CAPD["oq"] = (a, kw)
    return out


def _ccap(*a, **kw):
    out = C0(*a, **kw)
    CAPD["comb"] = (a, kw)
    return out


def with_form(form, fn):
    """run fn with ref2's f1c swapped: ooo1cq's entry (COH off) and coh1cq's (COH on) -> the candidate's"""
    if form is None:
        return fn()
    OQ.tier_glu_nax6n_1cq, CQ.tier_glu_nax6n_1cq_coh = MC.tier_glu_1cq(form, False), MC.tier_glu_1cq(form, True)
    try:
        return fn()
    finally:
        OQ.tier_glu_nax6n_1cq, CQ.tier_glu_nax6n_1cq_coh = F0, FC0


say(f"\nM2 M32 {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; Leap's 24 ooo1c_m32_check scenarios; forms "
    f"{FORMS} (grid cap {MC.GY_CAP} for {sorted(MC.CAPPED & set(FORMS))}) vs ooo1cq / coh1cq")
ref2.CFG.update(ooo_1c=True, tier_nax_ooo=True, nax_1cq=True, nax_native=True)
rng = np.random.default_rng(1132)
n = 0
bad = {f: 0 for f in FORMS}
for Lm in (20, 44):
    for coh in (True, False):
        ref2.CFG["nax_coh"] = coh
        for kind in ("rand", "rand_2d", "pop1", "pop2", "pop2_2d", "rand_late"):
            ids, sc, xf, mk = scen(rng, kind)
            x = mx.array(xf).astype(mx.bfloat16)
            scb = sc.astype(np.float32)
            late = kind.endswith("_late")
            ls_ = set(rng.choice(np.unique(ids), len(np.unique(ids)) // 3, replace=False).tolist()) if late else set()
            calls = [lambda: call(Lm, ids, mx.array(scb).astype(mx.bfloat16), x, mk, late=late, lateset=ls_),
                     lambda: call(Lm, ids[:16], mx.array(scb[:16]).astype(mx.bfloat16), x[:16], None if mk is None else mk[:16]),
                     lambda: call(Lm, ids[16:], mx.array(scb[16:]).astype(mx.bfloat16), x[16:], None if mk is None else mk[16:])]
            refs = [c() for c in calls]
            res = {}
            caps = []                                    # the C4 forms: replay each call's captured one-call + combine
            if any(f in MC.C4SET for f in FORMS):
                for c in calls:
                    CAPD.clear()
                    OQ.tier_glu_nax6n_1cq, TGm.combine_phase = _fcap, _ccap
                    try:
                        c()
                    finally:
                        OQ.tier_glu_nax6n_1cq, TGm.combine_phase = F0, C0
                    caps.append((CAPD["oq"], CAPD["comb"]))
            for f in FORMS:
                if f in MC.C4SET:
                    outs = []
                    for (a, kw), (ca, ckw) in caps:
                        o = MC.full_call(f, bool(ref2.CFG["nax_coh"]))(*a, ca[3], **{k: v for k, v in kw.items() if k in ("nsg_b", "ks", "ks_b")})
                        outs.append(o.reshape(-1, o.shape[-1]))
                    refs_ = [r_.reshape(-1, r_.shape[-1]) for r_ in refs]
                    res[f] = all(o.shape == r_.shape and bool(mx.all(o.view(mx.uint16) == r_.view(mx.uint16)).item())
                                 for o, r_ in zip(outs, refs_))
                    bad[f] += not res[f]
                    continue
                outs = [with_form(f, c) for c in calls]
                res[f] = all(o.shape == r_.shape and bool(mx.all(o.view(mx.uint16) == r_.view(mx.uint16)).item())
                             for o, r_ in zip(outs, refs))
                bad[f] += not res[f]
            n += 1
            say(f"  L{Lm} coh {int(coh)} {kind:9s}: max pairs per expert {np.bincount(ids.reshape(-1)).max()}; == ref "
                f"(32-row call + both 16-row calls): " + ", ".join(f"{f} {res[f]}" for f in FORMS))
say("M2 M32: " + "; ".join(f"{f} {n - bad[f]} of {n} scenarios == {'ooo1cq / coh1cq'} bit for bit" for f in FORMS))
open(A_.out, "a").write("\n".join(L_) + "\n")
sys.exit(1 if any(bad.values()) else 0)
