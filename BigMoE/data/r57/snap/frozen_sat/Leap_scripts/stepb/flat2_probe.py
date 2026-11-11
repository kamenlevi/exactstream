"""FLAT-2 hooks for the live loop (zero session, Thu 5 Nov 2026). The harness is BigMoE's scripts/r57_screen.py at its
current HEAD, run unchanged except for two cycle stamps and four one-line calls into this module (flat2_patch.py
inserts them):
  after_prefill  once per run(), before the decode clock: restore the pools the pass before ended with
  stash          before the draft block is cut: keep the whole block and its gate masks (probe checks only)
  before_commit  the probe (probe checks only), then the commit's start time
  after_commit   wait for the commit replay and log its ms (stage and probe passes only)
The three per-check calls sit behind `if F2ON[0]:`, off in the measured passes: those run the engine's loop as it is.
The probe is Leap's C0 probe (~/Leap/scripts/c0_timeline.py, Mon 2 Nov) ported to today's loop: the check is rolled back
(mtplx rollback_after_verify) and re-run warm, then synced piece by piece, then at more rows of the same draft block (the
row sweep: a cold pass, then two warm), and last once more as it was, whose logits must equal the live check's bitwise.
New against C0: every re-run carries the FAST gate (FG) exactly as the loop sets it, and a sweep's masks are cut the way
the loop cuts a draft block (the rule is copied from r57_screen.py; flat2_patch.py refuses a HEAD where it changed).
Everything the engine owns comes through `ns` (the harness's globals), so dryrun_cpu.py can drive this file with a stub."""
import os
import re
import time
from collections import Counter

import numpy as np

import stepb_k as KB                                     # Leap's step 0 part B: the kernel-level synced forwards

SWEEP = tuple(int(x) for x in os.environ.get("FLAT2_SWEEP", "2,4,8,12,16").split(",") if x)
KREP = int(os.environ.get("STEPB_KREP", "3"))            # kernel-level forwards per probe, at 2 rows and at 16 rows
DPROBE = os.environ.get("STEPB_DRAFT") == "1"           # step B: the drafter's propose profiled once per probe (a cache copy)
DENSE = int(os.environ.get("STEPB_DENSE", "0"))         # step B gap hunt: N unsynced 16-row forwards with the experts stubbed
FULL = int(os.environ.get("STEPB_FULL", "0"))           # ... and N with the real experts (warm, after one loading forward)


def sync_ms(mx, clock=time.perf_counter, n=300):
    """one eval round trip in ms (C0's calibration): taken off every synced record"""
    a = mx.array([1.0]); mx.eval(a)
    ts = []
    for _ in range(n):
        t0 = clock(); b = a + 1; mx.eval(b); ts.append(clock() - t0)
    return 1e3 * float(np.median(ts))


class Hooks:
    def __init__(self, ns, clock=time.perf_counter):
        self.ns, self.clock = ns, clock
        self.mode, self.at = "meas", set()
        self.pool_prev = None                            # per layer (base ids, sidecar ids) resident at the last pass's end
        self.full = None                                 # the uncut draft block of a probe check
        self.t_commit0 = None
        self.SEG = {"on": False, "rec": [], "L": None}
        self.replayed = {}                               # step B: the replay runs once per row class
        self.dot = dict(on=False, L=set(), res=[], rows=0, dir=None)   # step B: the graph of each piece (kernel counts)
        self.rec = self._blank()

    @staticmethod
    def _blank():
        return dict(commit={}, probes=[], prewarm=None, errors=[])

    def new_pass(self, mode, at=()):
        assert mode in ("meas", "stage", "probe"), mode
        self.mode, self.at, self.full, self.t_commit0, self.rec = mode, set(at), None, None, self._blank()

    def take(self):
        r, self.rec = self.rec, self._blank()
        return r

    # ------------------------------------------------------------ pools: all-resident passes (FAR 4b's oracle form)
    def pool_snapshot(self):
        """the experts resident now, per layer: base pool and LSB sidecar (ext.lookup, as the harness's S10B dump)"""
        ns = self.ns
        ext, E = ns["BS"].ext, list(range(int(ns.get("N_EXPERTS", 512))))
        snap = []
        for sw in ns["SW_"]:
            bm = sw._bm
            if bm["full"]:
                snap.append((None, None))
                continue
            b = np.nonzero(np.asarray(ext.lookup(bm["L"], E)) >= 0)[0]
            s, side = None, (bm.get("tier") or {}).get("side")
            if side is not None:
                try:
                    s = np.nonzero(np.asarray(ext.lookup(side, E)) >= 0)[0]
                except Exception as e:                   # the base pools are what the gated rows need
                    self.rec["errors"].append(f"sidecar lookup: {e!r}")
            snap.append((b, s))
        self.pool_prev = snap
        return (sum(len(b) for b, _ in snap if b is not None), sum(len(s) for _, s in snap if s is not None))

    def after_prefill(self, L):
        """The prefill adopts the prompt's experts into the pools (stream.py's staged path), so a repeated pass would
        start as cold as the first. Put back what the pass before ended with: ext.ensure per pool, the harness's own A6
        warm call, capped at the pool's size. Outside the decode clock; its reads are logged."""
        if self.pool_prev is None or L.get("mtp"):
            return
        ns = self.ns
        mx, BG, stats = ns["mx"], ns["BG"], ns["stats"]
        t0, s0, n_ids = self.clock(), stats(), 0
        spare = int(os.environ.get("BIGMOE_SPARE", "0"))
        for sw, (base, side) in zip(ns["SW_"], self.pool_prev):
            bm = sw._bm
            if bm["full"]:
                continue
            t4 = bm.get("tier") or {}
            for L_, ids_, C_ in ((bm["L"], base, bm["C"]), (t4.get("side"), side, t4.get("C_l", 0))):
                if L_ is None or ids_ is None or C_ <= 0 or not len(ids_):
                    continue
                k_ = C_ - min(spare, C_ // 4)            # the ext's C_main (ensure aborts past it)
                BG.ext.ensure(L_, mx.array(np.asarray(ids_[-k_:], dtype=np.int32)))
                n_ids += min(len(ids_), k_)
        d = stats() - s0
        self.rec["prewarm"] = dict(ms=1e3 * (self.clock() - t0), ids=int(n_ids), miss=float(d[0]), io_ms=1e3 * float(d[1]))

    # ------------------------------------------------------------ per-check hooks (behind F2ON)
    def stash(self, L):
        if self.mode != "probe" or (len(L["kept"]) + 1) not in self.at:
            self.full = None
            return
        fz = L.get("fz")
        gated = fz is not None and (fz["thr"] is not None or fz["copy"])
        self.full = dict(dr=list(L["dr"]), fk=L.get("fk_"), sq=L.get("sq_"), cg=list(L["cg_"]) if gated else None,
                         c16=L.get("c16"), R_=L.get("R_", 16))

    def before_commit(self, L):
        n = len(L["kept"]) + 1                           # this check, 1-based
        if self.mode == "probe" and n in self.at and not L.get("mtp"):
            self.rec["probes"].append(self._probe(L, n))
        self.t_commit0 = self.clock()

    def after_commit(self, L):
        self.ns["mx"].eval(self.ns["states"](L["cache"]))
        if self.t_commit0 is not None:
            self.rec["commit"][len(L["kept"]) + 1] = 1e3 * (self.clock() - self.t_commit0)

    # ------------------------------------------------------------ the gate of a sweep's rows
    def masks_for(self, w, fz):
        """The first w rows of the stashed block as the loop would check them: r57_screen.py's cut of a draft block
        (the lines after `if maxd:`), verbatim but for the names. Returns (drafts, (keep mask, sure rows, row cf))."""
        f, ns = self.full, self.ns
        dr = f["dr"][:w - 1]
        fk = sq = qcf = None
        if f["fk"] is not None:
            fk = f["fk"][:len(dr) + 1].copy(); fk[len(dr)] = True
        if f["sq"] is not None:
            sq = f["sq"][:len(dr) + 1].copy()
            if ns.get("D17") and fz.get("qom"):
                if len(dr) == f["R_"] - 1:
                    sq[len(dr)] = f["c16"] is not None and f["c16"] >= fz["thr"]
            else:
                sq[len(dr)] = False
            if ns.get("QSPLIT") is not None:
                qcf = np.array((list(f["cg"]) + [f["c16"] if f["c16"] is not None else 1.0])[:len(dr) + 1], dtype=np.float32)
            if not sq.any() and not ns.get("U3FORCE"):
                sq = None
        return dr, (fk, sq, qcf)

    # ------------------------------------------------------------ the synced pass's wrappers (C0's seg_install)
    def _seg_install(self):
        ns, SEG, clock = self.ns, self.SEG, self.clock
        mx, Q, lay = ns["mx"], ns["Q"], ns["lay"]
        LIDX = {id(l): i for i, l in enumerate(lay)}
        HCK = {}
        for l in lay:
            HCK[id(l.attn_hyper_connection)] = "hc_a"
            HCK[id(l.mlp_hyper_connection)] = "hc_m"

        def arrs(o):
            if isinstance(o, mx.array):
                return [o]
            if isinstance(o, (list, tuple)):
                return [a for x in o for a in arrs(x)]
            return []

        def rec(kind, t0, t1, t2):
            SEG["rec"].append((kind, SEG["L"], 1e3 * (t1 - t0), 1e3 * (t2 - t1)))

        KK = KB.K
        TIXL = (getattr(ns.get("st"), "_tier_idx", None) or {}).get("layers") or {}

        DOT = self.dot
        if not DOT["L"]:
            gl = [i for i, l in enumerate(lay) if l.is_linear]
            al = [i for i, l in enumerate(lay) if not l.is_linear]
            DOT["L"] = set(gl[1:2] + al[:1])             # one GDN layer (the 2nd: layer 0 may carry extras) and one attention layer

        def dot(kind, outs):                             # piece-level synced pass only: the primitives built since the last eval
            if not DOT["on"] or KK["on"] or SEG["L"] not in DOT["L"]:
                return
            try:
                d_ = DOT["dir"] or os.path.join(os.environ.get("FLAT2_OUT", "."), "dot_" + os.environ.get("FLAT2_TAG", "x"))
                os.makedirs(d_, exist_ok=True)
                pth = os.path.join(d_, f"r{DOT['rows']}_L{SEG['L']}_{kind}.dot")
                mx.export_to_dot(pth, *outs)
                lab = re.findall(r'label ="([^"]+)"', open(pth).read())
                DOT["res"].append((kind, SEG["L"], DOT["rows"], dict(Counter(lab))))
            except Exception as e:
                DOT["res"].append((kind, SEG["L"], DOT["rows"], {"error": repr(e)[:160]}))

        def kend(kind, t1, t2):                          # kernel mode: the piece's last flush = its unwrapped glue
            if KK["on"]:
                KK["rec"].append((kind, SEG["L"], "glue_end", "", 0, "E", 1e3 * (t2 - t1), 0.0, 0.0))

        def mseg(kind, fn):                              # a method: build, then eval its outputs
            def w(self_, *a, **kw):
                if not SEG["on"]:
                    return fn(self_, *a, **kw)
                k_ = kind(self_) if callable(kind) else kind
                KK["ctx"].append(k_)
                try:
                    t0 = clock(); out = fn(self_, *a, **kw); t1 = clock()
                    dot(k_, arrs(out))
                    mx.eval(arrs(out))
                finally:
                    KK["ctx"].pop()
                t2 = clock()
                rec(k_, t0, t1, t2)
                kend(k_, t1, t2)
                return out
            return w

        def fseg(kind, fn):                              # a module function
            def w(*a, **kw):
                if not SEG["on"]:
                    return fn(*a, **kw)
                KK["ctx"].append(kind)
                try:
                    t0 = clock(); out = fn(*a, **kw); t1 = clock()
                    dot(kind, arrs(out))
                    mx.eval(arrs(out))
                finally:
                    KK["ctx"].pop()
                t2 = clock()
                rec(kind, t0, t1, t2)
                kend(kind, t1, t2)
                return out
            return w

        def lseg(fn):                                    # the whole layer (its pieces nest inside)
            def w(self_, *a, **kw):
                if not SEG["on"]:
                    return fn(self_, *a, **kw)
                SEG["L"] = KK["L"] = LIDX.get(id(self_))
                KK["ctx"].append("layer")
                try:
                    t0 = clock(); out = fn(self_, *a, **kw); t1 = clock()
                    dot("layer", arrs(out))
                    mx.eval(arrs(out))
                finally:
                    KK["ctx"].pop()
                t2 = clock()
                rec("layer", t0, t1, t2)
                kend("layer", t1, t2)
                SEG["L"] = KK["L"] = None
                return out
            return w

        def xseg(fn):                                    # the experts call: its inputs (router + shared) first
            def w(self_, x, indices, *a, **kw):
                if not SEG["on"]:
                    return fn(self_, x, indices, *a, **kw)
                t0 = clock()
                t_ = (getattr(self_, "_bm", None) or {}).get("tier") or {}
                dot("moe_pre", [x, indices] + [v for k_, v in t_.items() if k_ in ("scores", "shared") and isinstance(v, mx.array)])
                mx.eval([x, indices] + [v for k_, v in t_.items() if k_ in ("scores", "shared") and isinstance(v, mx.array)])
                t1 = clock()
                rec("moe_pre", t0, t0, t1)
                kon = KK["on"]
                if kon:                                  # kernel mode: the experts are one unit (inner calls counted)
                    KK["depth"] += 1; KK["cnt"] = {}
                try:
                    out = fn(self_, x, indices, *a, **kw)
                finally:
                    if kon:
                        KK["depth"] -= 1
                t2 = clock()
                dot("experts", arrs(out))
                mx.eval(arrs(out))
                t3 = clock()
                rec("experts", t1, t2, t3)
                if kon:
                    KB.experts_rec(SEG["L"], indices, TIXL, 1e3 * (t1 - t0), 1e3 * (t2 - t1), 1e3 * (t3 - t2))
                return out
            return w

        saved = []

        def put(obj, name, new):                         # own = the attribute sat on obj itself (else delete on restore)
            saved.append((obj, name, getattr(obj, name), name in vars(obj)))
            setattr(obj, name, new)

        put(Q.DecoderLayer, "__call__", lseg(Q.DecoderLayer.__call__))
        for T_ in {type(l.attn_hyper_connection) for l in lay}:
            put(T_, "__call__", mseg(lambda s_: HCK.get(id(s_), "hc_f"), T_.__call__))
        for T_ in {type(l.linear_attn) for l in lay if l.is_linear}:
            put(T_, "__call__", mseg("gdn", T_.__call__))
        for T_ in {type(l.self_attn) for l in lay if not l.is_linear}:
            put(T_, "__call__", mseg("att", T_.__call__))
        for T_ in {type(l.mlp) for l in lay}:
            put(T_, "__call__", mseg("moe", T_.__call__))
        for T_ in {type(l.mlp.switch_mlp) for l in lay}:
            put(T_, "__call__", xseg(T_.__call__))
        for T_ in {type(l.ple) for l in lay if "ple" in l}:
            put(T_, "__call__", mseg("ple", T_.__call__))
        put(Q, "_hyper_residual_write", fseg("hc_w", Q._hyper_residual_write))
        return saved

    @staticmethod
    def _seg_remove(saved):
        for obj, name, old, own in reversed(saved):
            if own:
                setattr(obj, name, old)
            else:
                delattr(obj, name)

    # ------------------------------------------------------------ step B: the drafter's propose
    def _draft_probe(self, L):
        """this cycle's propose again (same block, same context) on a copy of the drafter cache, with REF2 removed as the
        loop runs it: 4 unsynced walls, then one kernel-level pass with the chained replay of its kernel kinds"""
        import copy
        ns, clock = self.ns, self.clock
        mx, ref2, draft, propose, MASK = ns["mx"], ns["ref2"], ns["draft"], ns["propose"], ns["MASK"]
        out = {}

        def fresh():
            dc = []
            for c in L["dcache"]:
                c2 = copy.copy(c)
                for k_, v_ in list(vars(c).items()):
                    if isinstance(v_, mx.array):
                        setattr(c2, k_, mx.array(v_))
                dc.append(c2)
            mx.eval([v_ for c in dc for v_ in vars(c).values() if isinstance(v_, mx.array)])
            return dc

        try:
            R_ = int(L.get("R_", 16))
            blk = mx.array([[int(L["tok"])] + [MASK] * (R_ if L.get("_d17") else R_ - 1)])
            ctx = L["ctx"][None].astype(mx.bfloat16)
            mx.eval(blk, ctx)
            ref2.remove()
            try:
                walls = []
                for _ in range(4):
                    dc = fresh()
                    t0 = clock(); p_, c_ = propose(blk, ctx, dc); mx.eval(p_, c_); walls.append(1e3 * (clock() - t0))
                out["wall"] = walls
                dc = fresh()
                KB.K.update(rec=[], ctx=["draft"], L=None, depth=0, skip=0, cnt={}, cap=True, caps=[])
                KB.K["on"] = True
                try:
                    p_, c_ = propose(blk, ctx, dc); mx.eval(p_, c_)
                finally:
                    KB.K["on"] = False
                out["krec"] = [list(x) for x in KB.K["rec"]]
                out["replay"] = KB.replay(0.0)
            finally:
                KB.K["cap"], KB.K["caps"] = False, []
                ref2.apply()
        except Exception as e:
            out["error"] = repr(e)[:300]
        return out

    # ------------------------------------------------------------ step B: the dense pass alone (experts stubbed)
    def _dense_probe(self, fwd, tok, dr_, m_, stubbed=True, n=None):
        """the stashed block at 16 rows, DENSE times unsynced and once synced, with every MoE expert call returning zeros
        (no post, no reads, no expert kernels): what is left is the dense pass + glue. Wall stamps for the GPU-utilization
        sampler (gpu_util_sampler.sh)."""
        ns = self.ns
        mx, lay = ns["mx"], ns["lay"]
        T_ = {type(l.mlp.switch_mlp) for l in lay}
        saved = [(t_, t_.__call__) for t_ in T_]

        def stub(self_, x, indices, *a, **kw):
            return mx.zeros(x.shape[:-1] + (x.shape[-1],), x.dtype)

        out = {}
        try:
            if stubbed:
                for t_ in T_:
                    t_.__call__ = stub
            fwd([tok] + dr_, m_)                         # warm-up (with experts: loads the block's experts)
            fwd([tok] + dr_, m_)
            ws, bs = [], []
            out["t_start"] = time.time()                 # unix time: the sampler's clock
            for _ in range(n or DENSE):
                _, r_ = fwd([tok] + dr_, m_)
                ws.append(r_["wall"]); bs.append(r_["build"])
            out["t_end"] = time.time()
            out["wall"], out["build"] = ws, bs
            out["synced"] = fwd([tok] + dr_, m_, True)[1]
        except Exception as e:
            out["error"] = repr(e)[:300]
        finally:
            for t_, c_ in saved:
                t_.__call__ = c_
        return out

    # ------------------------------------------------------------ the probe
    def _probe(self, L, n):
        ns, clock, SEG = self.ns, self.clock, self.SEG
        mx, CS, stats, states = ns["mx"], ns["CS"], ns["stats"], ns["states"]
        model, ref2, FG, TAP = ns["model"], ns["ref2"], ns["FG"], ns["TAP"]
        cache, snap, lg_ref, fz = L["cache"], L["snap"], L["lg"], L.get("fz")
        real, rmask = [L["tok"]] + list(L["dr"]), (L.get("fk_"), L.get("sq_"), L.get("qcf_"))
        tA, sA = clock(), stats()
        last = [len(real)]

        def eq(a, b):
            return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a, b, equal_nan=True).item())

        def fwd(rows, masks, synced=False):
            fk, sq, qcf = masks
            CS.rollback_after_verify(cache, snap, last[0])
            mx.eval(states(cache))                       # the restored GDN states' copies, out of the timed window
            s0, saved = stats(), None
            if synced:
                saved = self._seg_install(); SEG["rec"] = []; SEG["on"] = True
                if synced == "k":                        # part B: every wrapped call timed alone
                    KB.K.update(rec=[], ctx=[], L=None, depth=0, skip=0, cnt={})
                    KB.K["on"] = True
            else:                                        # as live: the drafter's taps ride on the check
                ns["tap_reset"](); TAP["on"] = True
            FG.update(on=fk is not None or sq is not None, keep=mx.array(fk) if fk is not None else None,
                      sure=mx.array(sq) if sq is not None else None, sure_np=sq, t3=bool(fz and fz.get("t3")),
                      cold=bool(fz and fz.get("cold")) and sq is not None, qom=bool(fz and fz.get("qom")) and sq is not None,
                      qcf=mx.array(qcf) if qcf is not None else None)
            try:
                t0 = clock()
                with ref2.verify_scope(model):
                    lg_ = model(mx.array([rows]), cache=cache)
                t1 = clock()
                TAP["on"] = False
                am = mx.argmax(lg_[0], axis=-1)
                if synced:
                    mx.eval(am)
                else:
                    mx.eval(am, ns["taps"]())
                t2 = clock()
            finally:
                KB.K["on"] = False
                FG.update(on=False, keep=None, sure=None, sure_np=None, t3=False, cold=False, qom=False, qcf=None)
                TAP["on"] = False
                if synced:
                    SEG["on"] = False; self._seg_remove(saved)
            last[0] = len(rows)
            d = stats() - s0
            return lg_, dict(rows=len(rows), build=1e3 * (t1 - t0), wall=1e3 * (t2 - t0), miss=int(d[0]), io=1e3 * float(d[1]),
                             seg=[list(x) for x in SEG["rec"]] if synced else None,
                             krec=[list(x) for x in KB.K["rec"]] if synced == "k" else None,
                             kskip=KB.K["skip"] if synced == "k" else None)

        res = dict(check=n, rows=len(real), live_miss=float((sA - L["s0"])[0]), kept=int(L["k"]) + 1)
        try:
            w = []
            for _ in range(2):
                lg_, r_ = fwd(real, rmask)
                r_["equal"] = eq(lg_, lg_ref)
                w.append(r_)
            res["warm"] = w
            if not self.dot["res"] or self.dot["rows"] != len(real):
                self.dot.update(on=self.dot.get("done2") is None, rows=len(real))
            res["synced_real"] = fwd(real, rmask, True)[1]
            if self.dot["on"]:
                self.dot.update(on=False, done2=True); res["dot"] = list(self.dot["res"])
            res["k_real"] = []
            for j_ in range(KREP):                       # part B at the live check's rows (2)
                KB.K["cap"], KB.K["caps"] = (j_ == 0 and not self.replayed.get("2")), []
                lgk, rk = fwd(real, rmask, "k")
                rk["equal"] = eq(lgk, lg_ref)
                if KB.K["caps"]:                         # once per row class: the dependent-chain replay
                    rk["replay"] = KB.replay(0.0)
                    self.replayed["2"] = True
                KB.K["cap"], KB.K["caps"] = False, []
                res["k_real"].append(rk)
            sw, f = {}, self.full
            if (DENSE or FULL) and f is not None and len(f["dr"]) + 1 >= 16:   # the gap hunt, first (small pools: no sweep)
                if FULL:                                 # the real check at 16 rows, warm
                    res["full16"] = self._dense_probe(fwd, L["tok"], *self.masks_for(16, fz), stubbed=False, n=FULL)
                if DENSE:                                # the dense pass alone (experts stubbed)
                    res["dense16"] = self._dense_probe(fwd, L["tok"], *self.masks_for(16, fz))
                f = None
            if f is not None:
                tok = L["tok"]
                for r in SWEEP:
                    if r < 2 or r > len(f["dr"]) + 1:
                        continue
                    dr_, m_ = self.masks_for(r, fz)
                    sw[r] = [fwd([tok] + dr_, m_)[1] for _ in range(3)]   # cold (rows past the live check's), then two warm
                if len(f["dr"]) + 1 >= 16:
                    dr_, m_ = self.masks_for(16, fz)
                    self.dot.update(on=self.dot.get("done16") is None, rows=16, res=[])
                    res["synced_16"] = fwd([tok] + dr_, m_, True)[1]
                    if self.dot["on"]:
                        self.dot.update(on=False, done16=True); res["dot16"] = list(self.dot["res"])
                    res["k_16"] = []
                    for j_ in range(KREP):               # part B at 16 rows
                        KB.K["cap"], KB.K["caps"] = (j_ == 0 and not self.replayed.get("16")), []
                        rk = fwd([tok] + dr_, m_, "k")[1]
                        if KB.K["caps"]:
                            rk["replay"] = KB.replay(0.0)
                            self.replayed["16"] = True
                        KB.K["cap"], KB.K["caps"] = False, []
                        res["k_16"].append(rk)
            res["sweep"] = sw
            if DPROBE:
                res["draft"] = self._draft_probe(L)
        except Exception as e:                           # a failed probe must not cost the pass: put the check back
            res["error"] = repr(e)
            self.rec["errors"].append(f"probe at check {n}: {e!r}")
        lg_, r_ = fwd(real, rmask)                       # the cache as the live forward left it; the commit follows
        res["final_equal"] = eq(lg_, lg_ref)
        res["wall_s"] = clock() - tA
        res["stats"] = [float(x) for x in (stats() - sA)]
        return res
