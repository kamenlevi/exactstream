"""WEEK98 MODULE 2 (MoonShots): the shared setup for m2_*.py. A copy of Leap's exp32_bench.py setup (layer 20's
converted pools via BigMoE's loader, all 512 resident; the decision map's real 1c-tiered code windows in 32-row blocks;
km path; 2-D nolsb; BigMoE / Leap modules imported read-only, run with -I -B), the capture of ooo1cq's and the combine's
arguments per call, Staged (the one-call stage by stage on the captured inputs), gpu() (libmtlprof GPU us per eval) and
seg_bytes(). Import it first: it loads libmtlprof before mlx."""
import json
import os
import sys
import time

os.environ["BIGMOE_MAXK"] = "320"
sys.path.insert(0, os.path.expanduser("~/Leap/third_party/mtlprof"))
import mtlprof as MP                                     # noqa: E402  (before mlx: the hooks name pipelines)
MP.load(2)
import numpy as np                                       # noqa: E402
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("~/Leap/third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
import mlx.core as mx                                    # noqa: E402
import ref2                                              # noqa: E402
ref2.apply()
import bigmoe.gated as BG                                # noqa: E402
import bigmoe.tier as T                                  # noqa: E402
import ooo1cq as OQ                                      # noqa: E402
import tierglu as TGm                                    # noqa: E402
from tierglu import plan_masks, PROJ                     # noqa: E402
from tierglu_nax import rowsum64                         # noqa: E402

L_ = []
NB_MAX = int(os.environ.get("M2_BLOCKS", "24"))


def say(s=""):
    print(s, flush=True)
    L_.append(s)


# ---- exp32_bench's setup (Leap, ~/Leap/scripts/week98/exp32_bench.py at b5aa9e1), copied ----
ext = BG.ext
PLAN, K, R4, NL, D, LM = "4433333333", 10, 2, 96, 2560, 20
T.PLAN = PLAN
BARE = os.path.expanduser("~/BigMoE/models/bare")
SH4 = []
for o, i in ((640, 2560), (640, 2560), (2560, 640)):
    SH4 += [((o, i // 8), mx.uint32), ((o, i // 64), mx.bfloat16), ((o, i // 64), mx.bfloat16)]
d_, ix = T.load_index(BARE, layout_ok=True)
r = ix["layers"][str(LM)]
base, _ = T.make_pools(ext, 0, 512, 512, SH4, d_, r)
Ls, side, _ = T.make_sidecar(ext, 512, 512, SH4, d_, r)
allids = np.arange(512, dtype=np.int32)
for j in range(0, 512, 64):
    ext.ensure(0, mx.array(allids[j:j + 64])); ext.ensure(Ls, mx.array(allids[j:j + 64]))
sm_of = np.array(ext.lookup(0, allids.tolist()), np.int32)
sl_of = np.array(ext.lookup(Ls, allids.tolist()), np.int32)
assert (sm_of >= 0).all() and (sl_of >= 0).all()
RS = 32 * (2 + BG.XLINES)
MSB_MB = sum(r["msb_piece_size"]) / 1e6
LSB_MB = sum(r["lsb_piece_size"]) / 1e6


class Gate:
    def __init__(self):
        self.nl, self.nreg, self.seq = NL, 128, 9
        self.region = mx.zeros((self.nreg * BG.PAIRS * RS,), mx.int32)
        self.status = mx.zeros((4096,), mx.int32)
        mx.eval(self.region, self.status)
        self.REG = np.frombuffer(memoryview(self.region), dtype=np.int32)
        self.REG.setflags(write=True)


GATE = Gate()
CUR = [None]


def post(Lx, ids_, after=None):
    pb, ps, nb, ns = CUR[0]
    return (pb, nb) if Lx == 0 else (ps, ns)


def call(blk):
    x, ids, sc, nl, pb, ps, nb, ns = blk
    CUR[0] = (pb, ps, nb, ns)
    GATE.seq += 1
    t = {"base": base, "sideP": side, "segs": [(0, 2, 4), (2, 10, 3)], "side": NL // 2 + LM, "layout": 2,
         "nolsb": (nl, None)}
    return ref2._ooo_sel(x, ids, {"gate": GATE, "L": 0}, t, post, sc, None)


MAP = os.path.expanduser("~/FableGod/decision_map/out")
basej = json.load(open(os.path.join(MAP, "baseline.json")))
code = [f for f in sorted(os.listdir(os.path.join(MAP, "xcap"))) if basej[f]["dom"] == "code"]
B32, NK = [], []
for f in code:
    z = np.load(os.path.join(MAP, "xcap", f))
    tz = np.load(os.path.join(MAP, "bt", "tiers_1c", f))
    a = int(z["a"])
    n = z["ids"].shape[0]
    for b0 in range(0, n - 31, 32):
        rows = slice(b0, b0 + 32)
        ids = z["ids"][rows, LM].astype(np.int64)
        w = z["w"][rows, LM].astype(np.float32)
        k4 = tz["keep"][a + b0:a + b0 + 32, LM].astype(bool)
        k3 = tz["keep3"][a + b0:a + b0 + 32, LM].astype(bool)
        kept = k4 | k3
        idk = np.where(kept, ids, ids[:, :1])
        wk = np.where(kept, w, 0.0)
        nl = ~k4[:, :R4]
        m = 32
        nlm, idm = nl[:m], idk[:m]
        fl = np.flatnonzero(~nlm.reshape(-1))
        dpk = idm[:, :R4].reshape(-1)[fl[0]] if len(fl) else idm[0, 0]
        ids_s = np.where(nlm, dpk, idm[:, :R4])
        pb = mx.array(np.concatenate([[0], sm_of[idm.reshape(-1)]]).astype(np.int32))
        ps = mx.array(np.concatenate([[0], sl_of[ids_s.reshape(-1)]]).astype(np.int32))
        B32.append((mx.array(z["x"][rows, LM][:m].astype(np.float32)).astype(mx.bfloat16),
                    mx.array(idm.astype(np.int32)), mx.array(wk[:m]).astype(mx.bfloat16),
                    mx.array(nlm), pb, ps, int(idm.size), int(ids_s.size)))
        NK.append(len(np.unique(idm[kept[:m]])) if kept[:m].any() else 1)
mx.eval([t_ for blk in B32 for t_ in blk if isinstance(t_, mx.array)])
NB = min(NB_MAX, len(B32))
ref2.CFG.update(ooo_1c=True, tier_nax_ooo=True, nax_1cq=True, nax_native=True)
ref2.CFG["nax_coh"] = False
say(f"\nM2 SETUP {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; layer {LM}'s pools (all 512 resident); "
    f"{len(code)} code windows -> {len(B32)} 32-row blocks (profiling {NB}); MSB / LSB piece {MSB_MB:.3f} / "
    f"{LSB_MB:.3f} MB; kept experts per block mean {np.mean(NK[:NB]):.1f}")

# ---- capture: the one-call's and the combine's own arguments, per block ----
CAP = {}
_f0, _c0 = OQ.tier_glu_nax6n_1cq, TGm.combine_phase


def _fcap(*a, **kw):
    out = _f0(*a, **kw)
    CAP["oq"] = (a, kw)
    return out


def _ccap(*a, **kw):
    out = _c0(*a, **kw)
    CAP["comb"] = (a, kw)
    return out


OQ.tier_glu_nax6n_1cq, TGm.combine_phase = _fcap, _ccap


class Staged:
    """ooo1cq.tier_glu_nax6n_1cq + the combine, stage by stage on the captured inputs (same kernels, same templates)"""

    def __init__(self, a, kw, comb):
        import p11 as P11
        (self.x, self.msb_n, self.lsb_n, self.pre_m, self.pre_l, self.ids, self.plan, self.region, self.offs,
         self.status) = a
        self.ks, self.ks_b, self.nsg_b = kw.get("ks", 4), kw.get("ks_b", 2), kw.get("nsg_b", 2)
        assert kw.get("xs") is None and kw.get("grp") is None, kw.keys()
        M, Kd = self.x.shape
        R = len(self.plan)
        m4, m2, r4 = plan_masks(self.plan)
        self.M, self.Kd, self.R, self.r4 = M, Kd, R, r4
        self.p11 = self.msb_n["gate_proj"]["s3"].dtype == mx.uint32
        assert self.p11
        self.N, self.D = self.msb_n["gate_proj"]["w3"].shape[1], self.msb_n["down_proj"]["w3"].shape[1]
        pwm, pwl = self.msb_n["gate_proj"]["s3"].shape[-1], self.lsb_n["gate_proj"]["b"].shape[-1]
        self.tmpl = [("T", self.x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)),
                     ("DIAG_C", 0), ("GS_C", 1), ("PWM_C", pwm), ("PWL_C", pwl)]
        self.ls = self.pre_l.astype(mx.int32).reshape(-1) if r4 else mx.zeros((1,), mx.int32)
        self.ms = self.pre_m.astype(mx.int32).reshape(-1)
        self.comb_a, self.comb_kw = comb

    def grp(self):
        return OQ.gpu_grp_1cq(self.pre_m, self.pre_l, self.ids, self.plan)

    def xs(self):
        return rowsum64(self.x)

    def ka(self, grp, xs, ny=None, nseg_over=None):
        pidx, seg, nsg_, nseg = grp
        if nseg_over is not None:
            nsg_ = nseg_over
        g, u = self.msb_n["gate_proj"], self.msb_n["up_proj"]
        L = self.lsb_n
        bb = {p: self.msb_n[p]["b3"] for p in PROJ}
        return OQ._ka(True)(
            inputs=[self.x, xs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                    L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], self.ms, self.ls,
                    pidx, seg, nsg_, self.region, self.offs, self.status],
            template=self.tmpl + [("K_C", self.Kd), ("N_C", self.N), ("KS_C", self.ks)],
            grid=(self.N // 32 * 64 * self.ks, nseg if ny is None else ny, 1), threadgroup=(64 * self.ks, 1, 1),
            output_shapes=[(self.M * self.R, self.N), (self.M * self.R,), (max(self.M * self.r4, 1),)],
            output_dtypes=[self.x.dtype, mx.int32, mx.int32])

    def hs(self, h):
        return rowsum64(h)

    def kb(self, grp, h, hs, rs_m, rs_l, ny=None, nseg_over=None):
        pidx, seg, nsg_, nseg = grp
        if nseg_over is not None:
            nsg_ = nseg_over
        d = self.msb_n["down_proj"]
        L = self.lsb_n
        (y,) = OQ._kb(True)(
            inputs=[h, hs, d["w3"], d["s3"], d["b3"], self.msb_n["down_proj"]["b3"], L["down_proj"]["l1"],
                    L["down_proj"]["b"], rs_m, rs_l, pidx, seg, nsg_],
            template=self.tmpl + [("N_C", self.N), ("D_C", self.D), ("NSG_C", self.nsg_b), ("KS_C", self.ks_b)],
            grid=(self.D // (32 * self.nsg_b) * 32 * self.nsg_b * self.ks_b, nseg if ny is None else ny, 1),
            threadgroup=(32 * self.nsg_b * self.ks_b, 1, 1),
            output_shapes=[(self.M * self.R, self.D)], output_dtypes=[self.x.dtype])
        return y

    def comb(self, y):
        a = list(self.comb_a)
        yy = y.reshape(self.M * self.R, -1)
        return _c0(yy, yy, *a[2:], **self.comb_kw)

    def whole(self):
        grp = self.grp()
        xs = self.xs()
        h, rs_m, rs_l = self.ka(grp, xs)
        y = self.kb(grp, h, self.hs(h), rs_m, rs_l)
        return self.comb(y)


def _leaves(o):
    if isinstance(o, mx.array):
        return [o]
    if isinstance(o, (list, tuple)):
        return [a for v in o for a in _leaves(v)]
    return []


def gpu(fn):
    """run fn's graph alone (one eval), return (output, GPU us summed over its command buffers, n cbs, dispatches)"""
    mx.synchronize()
    c0 = MP.counts()
    out = fn()
    mx.eval(_leaves(out))
    mx.synchronize()
    c1 = MP.counts()
    cb = MP.cbs(c0["ncb"], c1["ncb"])
    us = float(sum(e - s for s, e, rd in zip(cb["start"], cb["end"], cb["ready"]) if rd)) * 1e6
    return out, us, int(c1["ncb"] - c0["ncb"]), int(c1["disp"] - c0["disp"])


def seg_bytes(st, grp):
    """per segment: its expert's pieces (A 2/3, B 1/3 of the MSB piece; + the LSB piece at tier 4), summed over
    segments, and the distinct-expert floor (MB)"""
    pidx, seg, nsg_, _ = grp
    mx.eval(pidx, seg, nsg_)
    pidx_, seg_, n = np.array(pidx), np.array(seg), int(np.array(nsg_)[0])
    ms, ls = np.array(st.ms), np.array(st.ls)
    m4, _, _ = plan_masks(st.plan)
    R = st.R
    a_mb = b_mb = 0.0
    uniq_m, uniq_l, live = set(), set(), 0
    for g in range(n):
        pf = int(pidx_[seg_[g]])
        r0 = pf % R
        e = int(ms[pf])
        if e < 0:
            continue
        live += 1
        tier4 = bool((m4 >> r0) & 1)
        el = -2
        if tier4:
            el = int(ls[(pf // R) * st.r4 + bin(m4 & ((1 << r0) - 1)).count("1")])
        a_mb += MSB_MB * 2 / 3 + (LSB_MB * 2 / 3 if tier4 and el >= 0 else 0.0)
        b_mb += MSB_MB / 3 + (LSB_MB / 3 if tier4 and el >= 0 else 0.0)
        uniq_m.add(e)
        if tier4 and el >= 0:
            uniq_l.add(el)
    return dict(nseg=n, live=live, a_mb=a_mb, b_mb=b_mb, grp_mb=a_mb + b_mb,
                uniq_mb=len(uniq_m) * MSB_MB + len(uniq_l) * LSB_MB, n_exp=len(uniq_m), n_lsb=len(uniq_l))


