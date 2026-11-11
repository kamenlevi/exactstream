"""XPT: where the Python thread's time goes in a real 16-row check (Leap, Fri 2026-11-06 23:5x).

The orchestrator 23:5x asked to measure the expert path's Python ms per layer at SB 2 in the steady window, so that
NewIdeas can order S3 (the CPU-side lever) against S1. A warm 16-row forward builds in 40.2 ms vs 15.8 ms with the
experts stubbed (TABLE_350 'GPU IDLE AT 16 ROWS').

STEPB_XPT=1 turns it on, for the real-block passes only (flat2_driver.one_pass). It wraps three things on the calling
thread, each with wall (perf_counter), thread CPU (time.thread_time) and a perf_counter entry stamp (F2C's clock):
  lay  the trunk's DecoderLayer.__call__: one layer of a forward (48 calls per forward)
  xp   the expert call, type(l.mlp.switch_mlp).__call__: route / post / one-call graph construction (inside 'lay')
  ae   mx.async_eval: SB's midloop submits and the loop's own (Metal encoding of what was built)
attach() bins the calls into cycles by run()'s F2C stamps. In each cycle the layer calls form forwards of 48: the first
is the check (16 rows), the rest are the commit's. It adds per-check sums to pc:
  ck_lw / ck_lc  check forward: sum over its 48 layer calls, wall / CPU ms
  ck_xw / ck_xc  its expert calls, wall / CPU ms (ck_xn calls)
  ck_aw / ck_ac  async_eval calls between its first and last layer call + up to the next forward, wall / CPU ms
  cm_lw / cm_lc / cm_xw / cm_xc / cm_aw / cm_ac  the same for the commit's forwards (cm_f = how many)
  cy_aw / cy_ac  every async_eval in the cycle
CPU < wall inside a call = the thread waited there (GPU back-pressure, a host sync, the GIL).
Overhead: ~3-4 us per wrapped call, ~0.4 ms per 16-row forward (about 110 calls)."""
import time

LOG = []                                                 # (kind, perf_counter at entry, wall_ms, cpu_ms, rows)
ON = [False]
_SAVED = []
_pc, _tt = time.perf_counter, time.thread_time


def _rows(a):
    try:
        x = a[1]                                         # (self, x, ...): x [B, L, D] or [N, D]
        return int(x.shape[-2]) if len(x.shape) >= 2 else 1
    except Exception:
        return -1


def _wrap(kind, fn, ids=None):
    """ids: the instances to log (the trunk's layers / experts; a drafter sharing the class is logged as 'd' + kind)"""
    def w(*a, **k):
        if not ON[0]:
            return fn(*a, **k)
        t0, c0 = _pc(), _tt()                         # perf_counter: the clock of run()'s F2C cycle stamps (flat2_patch)
        try:
            return fn(*a, **k)
        finally:
            kd = kind if ids is None or id(a[0]) in ids else "d" + kind
            LOG.append((kd, t0, 1e3 * (_pc() - t0), 1e3 * (_tt() - c0), _rows(a) if ids is not None else 0))
    w.__wrapped__ = fn
    return w


CPUST = []                                               # the thread's CPU time (s) at each F2C append (cycle starts + the end)


class _TL(list):
    """run()'s F2C (a global of the harness namespace): each append also stamps the thread's CPU time"""
    def append(self, x):
        CPUST.append(_tt())
        list.append(self, x)


def install(ns):
    """wrap the trunk's layer class, the expert classes and mx.async_eval once; F2C stamps thread CPU too"""
    if _SAVED:
        return
    ns["F2C"] = _TL(ns["F2C"])
    mx, lay = ns["mx"], ns["lay"]
    lids = {id(l) for l in lay}
    xids = {id(l.mlp.switch_mlp) for l in lay}
    for T_ in {type(l) for l in lay}:
        _SAVED.append((T_, "__call__", T_.__call__))
        T_.__call__ = _wrap("lay", T_.__call__, lids)
    for T_ in {type(l.mlp.switch_mlp) for l in lay}:
        _SAVED.append((T_, "__call__", T_.__call__))
        T_.__call__ = _wrap("xp", T_.__call__, xids)
    _SAVED.append((mx, "async_eval", mx.async_eval))
    mx.async_eval = _wrap("ae", mx.async_eval)


def remove():
    for obj, name, old in reversed(_SAVED):
        setattr(obj, name, old)
    del _SAVED[:]


def reset():
    del LOG[:]
    del CPUST[:]


def _cycle(calls):
    """one cycle's calls in order -> the check's and the commit's sums"""
    fw, cur, nl = [], None, 0
    for kind, u, w, c, r in calls:
        if kind == "lay":
            if cur is None or nl == 48:
                cur = dict(lw=0.0, lc=0.0, xw=0.0, xc=0.0, xn=0, aw=0.0, ac=0.0, rows=r)
                fw.append(cur)
                nl = 0
            cur["lw"] += w
            cur["lc"] += c
            nl += 1
        elif kind == "xp" and cur is not None:
            cur["xw"] += w
            cur["xc"] += c
            cur["xn"] += 1
        elif kind == "ae" and cur is not None:
            cur["aw"] += w
            cur["ac"] += c
    ck = fw[0] if fw else None
    cm = fw[1:]
    out = dict(ck_rows=ck["rows"] if ck else -1, cm_f=len(cm))
    for k in ("lw", "lc", "xw", "xc", "xn", "aw", "ac"):
        out["ck_" + k] = ck[k] if ck else float("nan")
        out["cm_" + k] = sum(f[k] for f in cm) if cm else 0.0
    out["cy_aw"] = sum(w for kind, u, w, c, r in calls if kind == "ae")
    out["cy_ac"] = sum(c for kind, u, w, c, r in calls if kind == "ae")
    return out


def attach(pc, stamps):
    """add the per-check XPT sums to pc (check i = stamps[i] -> stamps[i + 1]); returns an error string or ''"""
    n = len(pc["ck"])
    if len(stamps) != n + 1:
        return f"xpt: {len(stamps)} stamps for {n} checks"
    cpu_ok = len(CPUST) == n + 1
    keys = None
    for i in range(n):
        a_, b_ = stamps[i], stamps[i + 1]
        o = _cycle(sorted((x for x in LOG if a_ <= x[1] < b_), key=lambda x: x[1]))   # entry order: a layer before its
                                                         # expert call (the log appends at exit, inner calls first)
        o["cy_wall"] = 1e3 * (b_ - a_)                   # the cycle (draft + check + commit + the loop's own work)
        o["cy_cpu"] = 1e3 * (CPUST[i + 1] - CPUST[i]) if cpu_ok else float("nan")   # the Python thread's CPU in it
        if keys is None:
            keys = list(o)
            for k in keys:
                pc[k] = []
        for k in keys:
            pc[k].append(o[k])
    return ""
