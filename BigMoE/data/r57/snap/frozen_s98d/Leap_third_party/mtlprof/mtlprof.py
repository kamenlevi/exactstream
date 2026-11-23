"""mtlprof.py (Breakthrough, Sun 15 Nov 2026; CHECK55 step 0): the Python side of libmtlprof (mtlprof.m).
load() must run BEFORE `import mlx.core` (the hooks name pipelines as MLX creates them). Then:
  P.counts()            -> dict of cumulative counters (disp, bar, fwait, fupd, enc, blit, commit, ncb)
  P.tag(i)              -> set the current tag (dispatches and command buffers committed while it is set carry it)
  P.cbs(i0, i1)         -> numpy record of command buffers [i0, i1): commit / start / end / done (s, CACurrentMediaTime base),
                           tag, ndisp, ready
  P.psos()              -> {slot: (name, cumulative dispatches)}
  P.now()               -> CACurrentMediaTime (s)
  busy(cb, t0, t1)      -> GPU busy seconds inside [t0, t1] (union of the command buffers' GPU intervals)
"""
import ctypes
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_L = [None]
NAMES = ("disp", "bar", "fwait", "fupd", "enc", "blit", "commit", "ncb")


def load(version=1):
    if _L[0] is None:
        L = ctypes.CDLL(os.path.join(HERE, f"libmtlprof_v{version}.dylib"))
        L.prof_now.restype = ctypes.c_double
        L.prof_cbs.restype = ctypes.c_long
        L.prof_psos.restype = ctypes.c_long
        _L[0] = L
    return _L[0]


def now():
    return _L[0].prof_now()


def on(v=True):
    _L[0].prof_on(1 if v else 0)


def tag(i):
    _L[0].prof_tag(int(i))


def counts():
    a = (ctypes.c_long * 8)()
    _L[0].prof_counts(a)
    return dict(zip(NAMES, list(a)))


def tag_counts():
    a = (ctypes.c_long * 64)()
    _L[0].prof_tag_counts(a)
    return np.array(list(a), np.int64)


def cbs(i0, i1):
    n = max(0, i1 - i0)
    D = [(ctypes.c_double * max(n, 1))() for _ in range(4)]
    I = [(ctypes.c_int * max(n, 1))() for _ in range(3)]
    k = _L[0].prof_cbs(ctypes.c_long(i0), ctypes.c_long(i1), *D, *I)
    r = np.zeros(k, dtype=[("commit", "f8"), ("start", "f8"), ("end", "f8"), ("done", "f8"), ("tag", "i4"),
                           ("ndisp", "i4"), ("ready", "i4")])
    for j, nm in enumerate(("commit", "start", "end", "done")):
        r[nm] = np.frombuffer(D[j], np.float64, k) if k else []
    for j, nm in enumerate(("tag", "ndisp", "ready")):
        r[nm] = np.frombuffer(I[j], np.int32, k) if k else []
    return r


def psos(maxn=8192):
    sl = (ctypes.c_long * maxn)()
    ct = (ctypes.c_long * maxn)()
    nm = ctypes.create_string_buffer(120 * maxn)
    k = _L[0].prof_psos(ctypes.c_long(maxn), sl, ct, nm)
    raw = nm.raw
    out = {}
    for j in range(k):
        s = raw[120 * j:120 * (j + 1)].split(b"\0", 1)[0].decode(errors="replace")
        out[int(sl[j])] = (s, int(ct[j]))
    return out


def pso_delta(a, b):
    """dispatches per pipeline name between two psos() snapshots"""
    d = {}
    for k, (nm, n) in b.items():
        n0 = a.get(k, (nm, 0))[1]
        if n > n0:
            d[nm] = d.get(nm, 0) + n - n0
    return d


def busy(cb, t0, t1):
    """GPU busy seconds in [t0, t1]: the union of [start, end] over ready command buffers, clipped"""
    iv = sorted((max(s, t0), min(e, t1)) for s, e, rd in zip(cb["start"], cb["end"], cb["ready"]) if rd and e > t0 and s < t1)
    tot, cs, ce = 0.0, None, None
    for s, e in iv:
        if e <= s:
            continue
        if cs is None:
            cs, ce = s, e
        elif s > ce:
            tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if cs is not None:
        tot += ce - cs
    return tot


def tag_pso(t):
    """cumulative dispatches per pipeline slot for tag t (numpy, HSIZE)"""
    n = _L[0].prof_hsize()
    a = (ctypes.c_long * n)()
    _L[0].prof_tag_pso(int(t), a)
    return np.array(list(a), np.int64)
