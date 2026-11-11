"""G2-CHECK (Leap, Sun 2026-11-08 01:2x): ref2 candidate's _ooo_1c_call with REF2_G2_ROWS (3c-N's U3-capable GEMV one-call,
g2k) vs today's FAST-NAX one-call (ooo1cq) on real pools of layers 20 / 44 (native+p11+pad, all 512 resident), M 1-3, U3
sure rows (-2 sidecars), QOM optional uses, late pairs released before the launch. The two kernels sum in different orders
(FAST): the line is a max relative difference at the G2-PRICE level (~1-2e-2) on every case, the g2 count engaged, and no
non-finite output. -> stdout"""
import importlib.util, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_card/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_card/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_card/Breakthrough_keepset"))
assert os.environ.get("BIGMOE_POM_LP1") != "1"
import mlx.core as mx
_p = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Leap/data/g2/ref2_cand.py")
_spec = importlib.util.spec_from_file_location("ref2_g2_cand", _p)
CAND = importlib.util.module_from_spec(_spec)
sys.modules["ref2_g2_cand"] = CAND
_spec.loader.exec_module(CAND)
CAND.apply()
CAND.CFG["nax_1cq"] = True                               # L's FAST-NAX one-call (as nw_check sets it)
CAND.CFG["nax_native"] = True
import bigmoe.gated as BG
import bigmoe.tier as T
import ooo1c as OC
ext = BG.ext
PLAN, K, R4 = "4433333333", 10, 2
T.PLAN = PLAN
if BG.XLINES < 4:
    BG.XLINES = 4
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


class Gate:
    pass


gate = Gate()
gate.nl, gate.nreg, gate.seq = 96, 64, 0
gate.region = mx.zeros((gate.nreg * BG.PAIRS * RS,), mx.int32)
gate.status = mx.zeros((64,), mx.int32)
mx.eval(gate.region, gate.status)
REG = np.frombuffer(memoryview(gate.region), dtype=np.int32)
REG.setflags(write=True)


def release(off, slots, seq):
    w = OC.release_words(slots, seq)
    for i, v in enumerate(w):
        REG[off + 32 * (i // 31) + 1 + i % 31] = np.int32(np.uint32(v).view(np.int32))
    for j in range(RS // 32 - 1, -1, -1):
        REG[off + 32 * j] = seq


class Loader:
    def __init__(self, Lb, Ls, sm_of, sl_of, res, res_l, M):
        self.Lb, self.Ls, self.sm_of, self.sl_of, self.res, self.res_l, self.M = Lb, Ls, sm_of, sl_of, res, res_l, M
        self.fin_m = self.fin_l = None

    def __call__(self, Lx, ids_p, after=None):
        a = np.array(ids_p).astype(np.int64).reshape(-1)
        e = (a & 0xFFFFF).astype(np.int32)
        if Lx == self.Lb:
            o = ((a >> 21) & 1).astype(bool)
            r_ = np.isin(e, list(self.res))
            pre = np.where(r_, self.sm_of[e], np.where(o, np.where(e % 2 == 0, -2, -1), -1)).astype(np.int32)
            self.fin_m = np.where(r_ | ~o, self.sm_of[e], -2).astype(np.int32)
        else:
            r_ = np.isin(e, list(self.res_l))
            pre = np.where(r_, self.sl_of[e], -1).astype(np.int32)
            self.fin_l = self.sl_of[e].astype(np.int32)
        return mx.array(np.concatenate([[0], pre]).astype(np.int32)), len(pre)


# ---- D1C: determinism of the one-call (Leap, Sun 01:2x; BigMoE's C4 report: E + OOO_1C + NAX_1CQ differs run to run)
mode = os.environ.get("D1C_MODE", "all")
rng = np.random.default_rng(5)
CAND.CFG["g2_rows"] = 0
res_tab = []
seq = 1000
for kq in (True, False):                                  # FAST-NAX 1cq (L / BigMoE's arm) vs NAX6-N's plain 1c
    CAND.CFG["nax_1cq"] = kq
    for M in (1, 2, 4, 6):
        for late in (False, True):
            Lm = 20 if M % 2 else 44
            Lb, Ls, base, side, sm_of, sl_of = P[Lm]
            ids = np.stack([rng.choice(512, K, replace=False) for _ in range(M)]).astype(np.int32)
            uniq = np.unique(ids)
            res = set(uniq.tolist()) if not late else set(rng.choice(uniq, int(0.6 * len(uniq)), replace=False).tolist())
            res_l = set(uniq.tolist()) if not late else set(rng.choice(uniq, int(0.7 * len(uniq)), replace=False).tolist())
            x = (mx.random.normal((M, 2560)) * 0.5).astype(mx.bfloat16)
            sc = mx.array(rng.uniform(0.02, 0.3, (M, K)).astype(np.float32)).astype(mx.bfloat16)
            mx.eval(x, sc)
            ys = []
            for rep in range(8):
                seq += 1
                gate.seq = seq
                if os.environ.get("D1C_COH") == "1":      # COH-1CQ: odd reps through the coherent kernels (1cq only)
                    CAND.CFG["nax_coh"] = bool(rep % 2)
                bm = {"gate": gate, "L": Lb}
                t = {"segs": [(0, R4, 4), (R4, K, 3)], "base": base, "sideP": side, "side": Ls, "layout": 2}
                ld = Loader(Lb, Ls, sm_of, sl_of, res, res_l, M)
                offs = np.array(CAND._ooo1c_offs(bm, t, R4))
                y = CAND._ooo_1c_call(x, mx.array(ids), bm, t, ld, sc, None, True)
                assert y is not None
                if late and rep % 2:                      # odd reps: release 5 ms after the launch (the in-kernel poll spins)
                    import time as _t
                    t0 = _t.perf_counter(); mx.async_eval(y)
                    while _t.perf_counter() - t0 < 0.005:
                        pass
                release(int(offs[0]), ld.fin_m.tolist(), gate.seq); release(int(offs[1]), ld.fin_l.tolist(), gate.seq)
                mx.eval(y)
                ys.append(np.array(y.astype(mx.float32)))
            nd = sum(int(not np.array_equal(ys[0], v)) for v in ys[1:])
            res_tab.append((kq, M, late, nd))
            print(f"  {'1cq' if kq else '1c '} M {M} {'late (6 of 8 pre-released, odd reps spin 5 ms)' if late else 'all resident':45s}: "
                  f"{nd} of 7 repeats differ from rep 0", flush=True)
bad = sum(1 for r in res_tab if r[3])
print(f"D1C: {len(res_tab) - bad} of {len(res_tab)} configs deterministic over 8 repeats", flush=True)
