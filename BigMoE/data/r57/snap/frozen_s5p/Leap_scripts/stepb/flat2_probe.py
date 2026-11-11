"""FLAT-2 hooks for the live loop (zero session, Thu 5 Nov 2026). The harness is BigMoE's scripts/r57_screen.py at its
current HEAD, run unchanged except for two cycle stamps and four one-line calls into this module (flat2_patch.py
inserts them):
  after_prefill  once per run(), before the decode clock: restore the pools the pass before ended with
  stash          before the draft block is cut: keep the whole block and its gate masks (probe checks only)
  before_commit  the probe (probe checks only), then the commit's start time
  after_commit   wait for the commit replay and log its ms (stage and probe passes only)
The three per-check calls sit behind `if F2ON[0]:`, off in the measured passes: those run the engine's loop as it is.
The probe is Leap's C0 probe (/Users/kamen/BigMoE/data/r57/snap/frozen_s5p/Leap_scripts/c0_timeline.py, Mon 2 Nov) ported to today's loop: the check is rolled back
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
ABL = os.environ.get("STEPB_ABL", "")                    # S2b step 1 (Sat 00:5x): in-situ unsynced price per dense piece: the
ABL_N = int(os.environ.get("STEPB_ABL_N", "12"))         # dense-only 16-row forward with one piece stubbed per variant
ABL_R = int(os.environ.get("STEPB_ABL_R", "5"))          # (hcr / hcra / hcrm / hcw / inproj / att / gdn / moe / ple), rounds
S2C = int(os.environ.get("STEPB_S2C", "0"))              # S2c step 1: N forwards per variant at a real check (own block alone,


class _SkipRest(Exception):                              # alternating with the previous checks' blocks)
    pass


def _flat(mx, o):
    """the mx arrays inside o (array / list / tuple / dict values), in order"""
    if isinstance(o, mx.array):
        return [o]
    if isinstance(o, (list, tuple)):
        return [a for x in o for a in _flat(mx, x)]
    if isinstance(o, dict):
        return [a for x in o.values() for a in _flat(mx, x)]
    return []


def _rewrap(mx, o, deps):
    """o's structure with every array made to depend on deps (mx.depends: no compute, an edge in the graph)"""
    if isinstance(o, mx.array):
        return mx.depends(o, deps) if deps else o
    if isinstance(o, tuple):
        return tuple(_rewrap(mx, x, deps) for x in o)
    if isinstance(o, list):
        return [_rewrap(mx, x, deps) for x in o]
    return o


def _zero_from(mx, ins):
    """a float32 scalar 0 computed FROM every input's buffer (a 1-element read of each): adding it to an output makes
    the output's kernel wait for the inputs on the GPU (MLX's encoder puts a barrier only on a real buffer read)"""
    return mx.concatenate([x.reshape(-1)[:1].astype(mx.float32) for x in ins]).sum() * 0.0


def _add_zero(mx, o, z):
    if isinstance(o, mx.array):
        return o + z.astype(o.dtype)
    if isinstance(o, tuple):
        return tuple(_add_zero(mx, x, z) for x in o)
    if isinstance(o, list):
        return [_add_zero(mx, x, z) for x in o]
    return o


def _memo(mx, fn, ids, store=None, mode="dep"):
    """S2b's stub: the first call per (instance, input shapes) runs fn and keeps its evaluated outputs. ids: the
    instances to stub when fn is a method (others pass through); None for a module function. Later calls, by mode:
      dep  the kept outputs through mx.depends on every input (Sat 00:5x; WRONG as a price: consumers read old buffers,
           so MLX's concurrent encoder drops the barrier and the chain overlaps; an upper bound only)
      bar  the kept outputs + a zero read from every input's buffer (_zero_from): the chain stays serial; the piece's
           own kernels are gone, a few tiny ones are added (priced by cal)
      cal  the REAL piece + the same zero term: what bar adds, so piece = wall(cal) - wall(bar)"""
    store = {} if store is None else store

    def w(*a, **kw):
        if ids is not None and (not a or id(a[0]) not in ids):
            return fn(*a, **kw)
        ins = _flat(mx, list(a[1:] if ids is not None else a)) + _flat(mx, kw)
        if mode == "cal":
            o = fn(*a, **kw)
            return _add_zero(mx, o, _zero_from(mx, ins)) if ins else o
        key = (id(a[0]) if ids is not None else 0, tuple(tuple(x.shape) for x in ins))
        if key not in store:
            o = fn(*a, **kw)
            mx.eval(_flat(mx, o))
            store[key] = o
            return o
        if mode == "bar":
            return _add_zero(mx, store[key], _zero_from(mx, ins)) if ins else store[key]
        return _rewrap(mx, store[key], ins)
    w.__wrapped__ = fn
    return w


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
        self.blocks = []                                 # S2C: the last checks' (rows, masks), for the alternating forwards

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
        if S2C and getattr(self, "real_pass", False) and not L.get("mtp"):   # S2c step 1b: the LIVE check synced per piece (the same
            mx_ = self.ns["mx"]                                  # S2c 1d: MLX's memory around the live check (GB)
            self.mem_live = dict(active0=mx_.get_active_memory() / 1e9, cache0=mx_.get_cache_memory() / 1e9)
            mx_.reset_peak_memory()
            self._live_saved = self._seg_install()               # wrappers as the probe's synced replay of its block)
            self.SEG["rec"] = []
            self.SEG["on"] = True

    def before_commit(self, L):
        if getattr(self, "_live_saved", None) is not None:   # S2c step 1b: the live check is done: keep its pieces
            self.SEG["on"] = False
            self._seg_remove(self._live_saved)
            self._live_saved = None
            self.live_seg = [list(x) for x in self.SEG["rec"]]
            self.live_cpu = [list(x) for x in self.SEG.get("cpu", [])]
            mx_ = self.ns["mx"]
            (getattr(self, "mem_live", None) or {}).update(active1=mx_.get_active_memory() / 1e9, cache1=mx_.get_cache_memory() / 1e9,
                                                          peak=mx_.get_peak_memory() / 1e9)
            self.SEG["rec"] = []
        n = len(L["kept"]) + 1                           # this check, 1-based
        if self.mode == "probe" and n in self.at and not L.get("mtp"):
            self.rec["probes"].append(self._probe(L, n))
        if S2C and self.mode == "probe" and not L.get("mtp"):   # after the probe: the probe check sees only earlier blocks
            cp = lambda a: None if a is None else np.array(a, copy=True)
            self.blocks = (getattr(self, "blocks", []) + [([L["tok"]] + list(L["dr"]),
                                                           (cp(L.get("fk_")), cp(L.get("sq_")), cp(L.get("qcf_"))))])[-3:]
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
        ns, SEG, clock0 = self.ns, self.SEG, self.clock
        mx, Q, lay = ns["mx"], ns["Q"], ns["lay"]
        LIDX = {id(l): i for i, l in enumerate(lay)}
        HCK = {}
        for l in lay:
            HCK[id(l.attn_hyper_connection)] = "hc_a"
            HCK[id(l.mlp_hyper_connection)] = "hc_m"
        CPUAT = {}                                       # S2c (Sat 01:2x): the thread's CPU time at each wrapper stamp
        SEG["cpu"] = []                                  # (kind, L, cpu ms of the build, cpu ms of the eval) per rec

        def clock():
            t = clock0()
            CPUAT[t] = time.thread_time()
            return t

        def arrs(o):
            if isinstance(o, mx.array):
                return [o]
            if isinstance(o, (list, tuple)):
                return [a for x in o for a in arrs(x)]
            return []

        def rec(kind, t0, t1, t2):
            SEG["rec"].append((kind, SEG["L"], 1e3 * (t1 - t0), 1e3 * (t2 - t1)))
            c0, c1, c2 = CPUAT.pop(t0, None), CPUAT.pop(t1, None), CPUAT.get(t2)
            CPUAT.pop(t2, None)
            if c0 is not None and c2 is not None:
                c1 = c0 if c1 is None else c1
                SEG["cpu"].append((kind, SEG["L"], 1e3 * (c1 - c0), 1e3 * (c2 - c1)))

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

    # ------------------------------------------------------------ S2c step 1: a real check's block, alone vs alternating
    def _s2c_probe(self, fwd, real, rmask):
        """At a real check (Sat 00:5x; NewIdeas' S2c line): S2C unsynced forwards (each ends in its eval, as the loop's
        check does) of (same) the check's own block, (alt2) alternating with the previous check's block, (alt3) cycling
        the last three checks' blocks, (same2) its own again (drift). Each forward rolls the cache back to this check's
        start, so only the rows (tokens, hence routed experts) differ. Unix stamps for the GPU sampler."""
        out = dict(rows=len(real), prev=[len(b[0]) for b in self.blocks])
        me = (real, rmask)

        def run(seq, label):
            ws, bs, who, ms = [], [], [], []
            t0 = time.time()
            for i in range(S2C):
                rows_, m_ = seq[i % len(seq)]
                _, r_ = fwd(rows_, m_)
                ws.append(r_["wall"]); bs.append(r_["build"]); who.append(i % len(seq)); ms.append(r_["miss"])
            out[label] = dict(wall=ws, build=bs, who=who, miss=ms, t_start=t0, t_end=time.time())

        run([me], "same")
        if self.blocks:
            run([self.blocks[-1], me], "alt2")
        if len(self.blocks) >= 2:
            run([self.blocks[-2], self.blocks[-1], me], "alt3")
        run([me], "same2")
        return out

    # ------------------------------------------------------------ S2b step 1: in-situ price per dense piece (ablation)
    def _abl_probe(self, fwd, tok, dr_, m_):
        """The stashed 16-row block, dense-only (every expert call returns zeros, as DENSE), ABL_R rounds; in each round
        every variant (base + the ABL list) runs 2 warm-up forwards and ABL_N unsynced timed ones. A variant stubs its
        pieces: the first call per (instance, input shapes) runs the real piece once and keeps its evaluated outputs;
        later calls return those outputs wrapped in mx.depends on every input array (so no upstream work is dropped),
        i.e. the piece's own kernels are gone and nothing else changes. The wall delta vs base = the piece's in-situ cost
        (valid while the forward stays GPU-bound: compare build vs wall). Unix stamps per window for the sampler."""
        ns = self.ns
        mx, lay, Q, ref2 = ns["mx"], ns["lay"], ns["Q"], ns["ref2"]

        def memo(fn, ids, mode):
            return _memo(mx, fn, ids, mode=mode)

        gl = [l for l in lay if l.is_linear]
        al = [l for l in lay if not l.is_linear]
        ipr = [l.linear_attn.in_proj_fused for l in gl if getattr(l.linear_attn, "in_proj_fused", None) is not None]
        pl = [l.ple for l in lay if "ple" in l]
        tm_ = ns["model"].language_model if hasattr(ns["model"], "language_model") else ns["model"]   # PLAN 220: the head
        hd = [o for o in (getattr(tm_, "lm_head", None), getattr(getattr(tm_, "model", None), "norm", None)) if o is not None]

        def cls(objs):
            return ("cls", {type(o) for o in objs}, {id(o) for o in objs})

        V = {"hcr": [cls([l.attn_hyper_connection for l in lay] + [l.mlp_hyper_connection for l in lay])],
             "hcra": [cls([l.attn_hyper_connection for l in lay])],
             "hcrm": [cls([l.mlp_hyper_connection for l in lay])],
             "hcw": [("mod", Q, "_hyper_residual_write")],
             "inproj": [cls(ipr)] if ipr else [],
             "att": [cls([l.self_attn for l in al])],
             "gdn": [cls([l.linear_attn for l in gl])],
             "moe": [cls([l.mlp for l in lay])],
             "ple": [cls(pl)] if pl else [],
             "head": [cls(hd)] if hd else []}   # PLAN 220: the final norm + lm_head
        SPEC = {}                                        # variant -> (mode, specs); "<mode>:<piece>", a bare piece = bar,
        for tok_ in [t_ for t_ in ABL.split(",") if t_]:  # "d5" = ref2's D5 fold on (CFG hc_pend_mlp) with no stub
            if tok_ == "d5":
                SPEC[tok_] = ("cfg", [("cfg", "hc_pend_mlp", True)])
                continue
            mode_, piece = tok_.split(":", 1) if ":" in tok_ else ("bar", tok_)
            if mode_ in ("bar", "cal", "dep") and V.get(piece):
                SPEC[tok_] = (mode_, V[piece])
        names = ["base"] + list(SPEC)

        def install(v):
            sv = []
            mode_, specs = SPEC.get(v, ("bar", []))
            for sp in specs:
                if sp[0] == "cls":
                    for T_ in sp[1]:
                        sv.append((T_, "__call__", T_.__call__, "__call__" in vars(T_)))
                        T_.__call__ = memo(T_.__call__, sp[2], mode_)
                elif sp[0] == "mod":
                    _, obj, name = sp
                    sv.append((obj, name, getattr(obj, name), name in vars(obj)))
                    setattr(obj, name, memo(getattr(obj, name), None, mode_))
                else:                                    # ("cfg", key, value): a ref2 switch for this variant only
                    _, key, val = sp
                    sv.append((ref2.CFG, key, ref2.CFG.get(key), "cfg"))
                    ref2.CFG[key] = val
            return sv

        def uninstall(sv):
            for obj, name, old, own in reversed(sv):
                if own == "cfg":
                    obj[name] = old
                elif own:
                    setattr(obj, name, old)
                else:
                    delattr(obj, name)

        T_x = {type(l.mlp.switch_mlp) for l in lay}
        saved_x = [(t_, t_.__call__) for t_ in T_x]

        def zstub(self_, x, indices, *a, **kw):
            return mx.zeros(x.shape[:-1] + (x.shape[-1],), x.dtype)

        out = {v: dict(wall=[], build=[], win=[]) for v in names}
        out["_meta"] = dict(names=names, n=ABL_N, rounds=ABL_R, rows=len(dr_) + 1, gdn=len(gl), att=len(al), inproj=len(ipr), head=len(hd),
                            ple=len(pl))
        try:
            for t_ in T_x:
                t_.__call__ = zstub
            for _ in range(ABL_R):
                for v in names:
                    sv = install(v)
                    try:
                        fwd([tok] + dr_, m_)                 # fills the variant's memo (real pieces, evaluated)
                        fwd([tok] + dr_, m_)
                        t0 = time.time()
                        for _ in range(ABL_N):
                            _, r_ = fwd([tok] + dr_, m_)
                            out[v]["wall"].append(r_["wall"]); out[v]["build"].append(r_["build"])
                        out[v]["win"].append((t0, time.time()))
                    finally:
                        uninstall(sv)
        except Exception as e:
            out["_meta"]["error"] = repr(e)[:300]
        finally:
            for t_, c_ in saved_x:
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
                             cpu=[list(x) for x in SEG.get("cpu", [])] if synced else None,
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
            if S2C and len(real) > 2:                    # S2c step 1 at a real check: its own block back to back, then
                live_ = getattr(self, "live_seg", None)  # alternating with the previous checks' blocks; step 1b: the
                livec_ = getattr(self, "live_cpu", None)  # live check per piece vs its block replayed synced (+ thread CPU)
                self.live_seg = self.live_cpu = None
                mem_ = dict(live=getattr(self, "mem_live", None),
                            rep=dict(active0=mx.get_active_memory() / 1e9, cache0=mx.get_cache_memory() / 1e9))
                mx.reset_peak_memory()
                r1_ = fwd(real, rmask, True)[1]
                mem_["rep"].update(active1=mx.get_active_memory() / 1e9, cache1=mx.get_cache_memory() / 1e9,
                                   peak=mx.get_peak_memory() / 1e9)
                res["s2c"] = self._s2c_probe(fwd, real, rmask)
                r2_ = fwd(real, rmask, True)[1]
                res["s2c"].update(live_seg=live_, replay_seg=r1_["seg"], replay_seg2=r2_["seg"], live_cpu=livec_,
                                  replay_cpu=r1_.get("cpu"), replay_cpu2=r2_.get("cpu"), mem=mem_)
                raise _SkipRest
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
            if (DENSE or FULL or ABL) and f is not None and len(f["dr"]) + 1 >= 16:   # the gap hunt, first (no sweep)
                if FULL:                                 # the real check at 16 rows, warm
                    res["full16"] = self._dense_probe(fwd, L["tok"], *self.masks_for(16, fz), stubbed=False, n=FULL)
                if DENSE:                                # the dense pass alone (experts stubbed)
                    res["dense16"] = self._dense_probe(fwd, L["tok"], *self.masks_for(16, fz))
                if ABL:                                  # S2b step 1: one dense piece stubbed per variant
                    res["abl"] = self._abl_probe(fwd, L["tok"], *self.masks_for(16, fz))
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
        except _SkipRest:
            pass
        except Exception as e:                           # a failed probe must not cost the pass: put the check back
            res["error"] = repr(e)
            self.rec["errors"].append(f"probe at check {n}: {e!r}")
        lg_, r_ = fwd(real, rmask)                       # the cache as the live forward left it; the commit follows
        res["final_equal"] = eq(lg_, lg_ref)
        res["wall_s"] = clock() - tA
        res["stats"] = [float(x) for x in (stats() - sA)]
        return res
