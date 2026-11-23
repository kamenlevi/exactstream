"""ooo1c_m32_check.py (Leap, Mon 16 Nov 2026 12:2x; CHECK25: the FAST-NAX one-call at 17-32 rows, the D8 bug class).
BigMoE's loop never sends > 16 rows to the one-call (R57_ROWS 16), so ref2._ooo_sel at M 32 has never run. The grouping
(tierglu_nax6._GRP, inherited by ooo1c / ooo1cq) starts a new group every 16 pairs of one key, so an expert picked by
more than 16 rows should take two tiles; this checks it on the GPU, on the converted pools of layers 20 / 44 (BigMoE's
loader, read-only), with BIGMOE_MAXK 320 (the 32-row routing row: XLINES 9, release regions of 341 slots).
Per scenario (random routing; a popular expert at rank 0 of every row; two experts on > 16 rows; 2-D nolsb masks;
late release), on COH on / off:
  32-row call == concat(the 16-row call on rows 0-15, the 16-row call on rows 16-31), bit for bit
(each pair's arithmetic depends only on its row, the expert's weights and the fixed 16-row tile), and the 32-row call's
expert posts / sidecar posts cover every used expert.
usage: [BigMoE venv] ooo1c_m32_check.py   (share lock ~3 GB)"""
import os
import sys
import time

os.environ["BIGMOE_MAXK"] = "320"
import numpy as np
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98c/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98c/Leap_scripts"))
import mlx.core as mx
import ref2
ref2.apply()
import bigmoe.gated as BG
import bigmoe.tier as T
import ooo1c as OC

OUT = os.path.expanduser("~/Leap/data/week98/check25/ooo1c_m32_check.txt")
L_ = []


def say(s):
    print(s, flush=True)
    L_.append(s)


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
say(f"\nOOO1C M32 check {time.strftime('%a %H:%M:%S')}: MLX {mx.__version__}; gated MAXK {BG.MAXK} XLINES {BG.XLINES} "
    f"PAIRS {BG.PAIRS}; pools of layers 20 / 44 (native+p11+pad, C {C}); ref2 md5 "
    f"{__import__('hashlib').md5(open(ref2.__file__, 'rb').read()).hexdigest()}")
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
    """one MoE layer through ref2._ooo_sel (all experts resident). Posts give every slot at once, except the experts in
    lateset (-1: the kernel polls the release region, which is written before the call, or 20 ms after with late)"""
    msb, lsb, sm_of, sl_of = P[Lm]
    M = ids.shape[0]
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
    pb = [a for Lx, a in POSTS if Lx == Lm]
    return y, (pb[0] if pb else None)


def scen(rng, kind, M=32):
    ids = np.stack([rng.choice(512, K, replace=False) for _ in range(M)]).astype(np.int64)
    if kind in ("pop1", "pop2", "pop2_2d"):
        ids[:, 0] = 7                                            # expert 7 at rank 0 of all 32 rows: 32 pairs
        for m in range(M):
            ids[m, 1:] = rng.choice(np.setdiff1d(np.arange(512), [7, 11]), K - 1, replace=False)
    if kind in ("pop2", "pop2_2d"):
        ids[:20, 2] = 11                                         # expert 11 at rank 2 of rows 0-19: 20 pairs
    lg = rng.standard_normal((M, K)).astype(np.float32)
    sc = np.array(mx.softmax(mx.array(lg), axis=-1).astype(mx.bfloat16).astype(mx.float32))
    x = (rng.standard_normal((M, D)) * 0.5).astype(np.float32)
    mk = (rng.random((M, R4)) < 0.6) if kind.endswith("_2d") else None
    return ids, sc, x, mk


ref2.CFG.update(ooo_1c=True, tier_nax_ooo=True, nax_1cq=True, nax_native=True)
rng = np.random.default_rng(1132)
n = bad = 0
for Lm in (20, 44):
    for coh in (True, False):
        ref2.CFG["nax_coh"] = coh
        for kind in ("rand", "rand_2d", "pop1", "pop2", "pop2_2d", "rand_late"):
            ids, sc, xf, mk = scen(rng, kind)
            x = mx.array(xf).astype(mx.bfloat16)
            scb = sc.astype(np.float32)
            late = kind.endswith("_late")
            ls_ = set(rng.choice(np.unique(ids), len(np.unique(ids)) // 3, replace=False).tolist()) if late else set()
            y32, pb = call(Lm, ids, mx.array(scb).astype(mx.bfloat16), x, mk, late=late, lateset=ls_)
            ya, _ = call(Lm, ids[:16], mx.array(scb[:16]).astype(mx.bfloat16), x[:16], None if mk is None else mk[:16])
            yb, _ = call(Lm, ids[16:], mx.array(scb[16:]).astype(mx.bfloat16), x[16:], None if mk is None else mk[16:])
            y16 = mx.concatenate([ya, yb], 0)
            mx.eval(y16)
            ok = y32.shape == y16.shape and bool(mx.all(y32.view(mx.uint16) == y16.view(mx.uint16)).item())
            if mk is not None:
                # 2-D marks: the 32-row call's dup pick may choose a different (row-major first unmarked) post than the
                # 16-row halves; a marked use reads MSB-3 only either way, so outputs match; the posts differ by design
                pass
            used = np.unique(ids)
            cov = pb is not None and set(np.unique(pb).tolist()) >= set(used.tolist())
            n += 1
            bad += not (ok and cov)
            say(f"  L{Lm} coh {int(coh)} {kind:9s}: 32-row == two 16-row calls bit for bit: {ok}; base posts cover the "
                f"{len(used)} used experts: {cov}; max pairs per expert {np.bincount(ids.reshape(-1)).max()}")
say(f"OOO1C M32: {n - bad} of {n} scenarios PASS")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "a").write("\n".join(L_) + "\n")
sys.exit(1 if bad else 0)
