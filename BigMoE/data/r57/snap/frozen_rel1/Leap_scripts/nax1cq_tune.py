"""NAX1CQ-TUNE (Leap, Fri 2026-11-06 09:4x): FAST-NAX's expert one-call (ooo1cq.tier_glu_nax6n_1cq; ~25 ms per 16-row check,
the stack's expert path) at its tunables: kernel A's K split ks (default 4), kernel B's nsg_b (2) and ks_b (2). Real pools of
layers 20 / 44 (native+p11+pad, all 512 resident), M 16 / 2, top-10, PLAN 4433333333, every pair resident (no spin), U3
sure rows -2 as the FAST arm. Dependent chains alternating the two layers' pools (cold), round-robin, median of 7.
Routing: random (~140 groups per call) and skewed (Zipf-like popularity, ~90-100: closer to real blocks).
Bits vs the default config: max relative error (FAST: ks / ks_b change the K order). -> ~/Leap/data/nax1cq_tune.txt"""

import importlib.util, os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel1/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel1/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel1/Breakthrough_keepset"))
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

OUT = os.path.expanduser("~/Leap/data/nax1cq_tune.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
import hashlib
say(f"\nNAX1CQ-TUNE {time.strftime('%a %H:%M:%S')} ({os.environ.get('QSC_COND', 'share lock')}; MLX {mx.__version__}); "
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



import ooo1cq as OQ
rng = np.random.default_rng(77)
clock = time.perf_counter


def ids_for(Mx, skew):
    if not skew:
        return np.stack([rng.choice(512, K, replace=False) for _ in range(Mx)]).astype(np.int32)
    pop = 1.0 / np.arange(1, 513) ** 0.9
    pop = pop[rng.permutation(512)]
    pop /= pop.sum()
    return np.stack([rng.choice(512, K, replace=False, p=pop) for _ in range(Mx)]).astype(np.int32)


def case(Lm, Mx, skew):
    Lb, Ls, base, side, sm_of, sl_of = P[Lm]
    ids = ids_for(Mx, skew)
    sr = np.zeros(Mx, bool); sr[rng.choice(Mx, Mx // 2, replace=False)] = True
    pre_m = mx.array(sm_of[ids].astype(np.int32))
    pre_l = mx.array(np.where(sr[:, None], -2, sl_of[ids[:, :R4]]).astype(np.int32))
    x = (mx.random.normal((Mx, 2560)) * 0.5).astype(mx.bfloat16)
    mx.eval(pre_m, pre_l, x)
    offs = mx.array([0, 0, 1], mx.int32)
    return dict(x=x, base=base, side=side, pre_m=pre_m, pre_l=pre_l, ids=mx.array(ids), offs=offs,
                groups=len(set(sm_of[ids].reshape(-1).tolist())))


def call(c, ks, nsg_b, ks_b, x=None):
    y, _, _ = OQ.tier_glu_nax6n_1cq(c["x"] if x is None else x, c["base"], c["side"], c["pre_m"], c["pre_l"], c["ids"], PLAN,
                                   gate.region, c["offs"], gate.status, nsg_b=nsg_b, ks=ks, ks_b=ks_b)
    return y


def chain(C, cfg):
    tt = []
    for calls in (False, True):
        prev = mx.zeros((1,), mx.float32)
        for c in C:
            x = c["x"] + (prev * 0).astype(mx.bfloat16)
            if calls:
                prev = call(c, *cfg, x=x).reshape(-1)[:1].astype(mx.float32)
            else:
                prev = x.reshape(-1)[:1].astype(mx.float32)
        t0 = clock(); mx.eval(prev); tt.append(clock() - t0)
    return 1e6 * (tt[1] - tt[0]) / len(C)


DEF = (4, 2, 2)
CFGS = [DEF, (2, 2, 2), (5, 2, 2), (8, 2, 2), (10, 2, 2), (4, 1, 2), (4, 4, 2), (4, 2, 1), (4, 2, 5), (4, 4, 1), (8, 4, 1),
        (8, 2, 5), (5, 4, 2)]
for Mx in (16, 2):
    for skew in (True, False):
        C = [case(20 if i % 2 == 0 else 44, Mx, skew) for i in range(12)]
        y0 = call(C[0], *DEF).astype(mx.float32); mx.eval(y0)
        valid, err = [], {}
        for cfg in CFGS:
            ks, nsg_b, ks_b = cfg
            if 2560 % (64 * ks) or 640 % (64 * ks_b) or 64 * ks > 1024 or 32 * nsg_b * ks_b > 1024 or 2560 % (32 * nsg_b):
                continue
            try:
                y1 = call(C[0], *cfg).astype(mx.float32); mx.eval(y1)
            except Exception as e:
                say(f"  cfg {cfg}: {repr(e)[:120]}")
                continue
            err[cfg] = float(mx.max(mx.abs(y1 - y0)).item() / max(1e-30, mx.max(mx.abs(y0)).item()))
            valid.append(cfg)
            chain(C, cfg)
        acc = {c: [] for c in valid}
        for r in range(7):
            for c in (valid if r % 2 == 0 else valid[::-1]):
                acc[c].append(chain(C, c))
        t = {c: float(np.median(v)) for c, v in acc.items()}
        best = min(t, key=t.get)
        g = np.mean([c["groups"] for c in C])
        say(f"  M {Mx} {'skewed' if skew else 'random'} routing (~{g:.0f} groups per call): default {DEF} {t[DEF]:.1f} us; best {best} "
            f"{t[best]:.1f} us ({100 * (1 - t[best] / t[DEF]):+.1f} %, err {err[best]:.1e}); all: "
            + ", ".join(f"{c}: {t[c]:.1f}" for c in valid))
