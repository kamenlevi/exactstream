"""convcache: keep a conversation's model state between requests, so a new turn prefills only its new tokens.

BigMoE docs/CONV_CACHE_DESIGN.md (B1 + the RAM tier of B2). `install(g)` swaps two globals of the decode loop (g = the
loop's __main__ globals, serve.py's G()): `prefill` and `MTPHead`. Both loops (r57_screen_rel1.py:986, r57_screen_w24.py:1013)
share one prefill(); the cached one below runs the same per-chunk calls.

- Checkpoints sit only at multiples of the prefill chunk and are taken during prefill. Each holds the GDN / PLE state (eager
  clones), the DFlash2 taps tail (SWIN rows) and the U3C footprint. The attention KV, the QSA indexer streams and the MTP
  head's cache stay in the live cache and are trimmed.
- A request restores the deepest checkpoint c <= (shared prefix - 1) and prefills [c, N) in the same chunks a cold prefill
  runs from c on: its logits and states are a cold prefill's, bit for bit. CONV_CACHE_CHECK=1 runs a cold prefill before
  every hit and compares (tests only: it doubles the prefill).
- Decode-written state is never reused: the previous reply is re-prefilled as part of the next turn.
- PF-A (R57_PFLM=1 / 2, layer-major over the same chunks; bitwise chunk-major, Leap 25/25 + 26/26 hashes) runs from c too,
  in ONE sweep: each layer's GDN / PLE state and U3C row are cloned as the sweep passes an edge (no extra expert reads).
  PF-B (R57_PFLM=3) and A6 "warm" prompts take the loop's own prefill (a miss, not stored).
- The taps: only the rows the drafter reads (the last SWIN) and the edges' tails are kept, not every position (exact; the
  loop's T held 25.6 KB per prompt token). CONV_CACHE_TAPS=all keeps every chunk (debug).
- Every copy is bitwise (_clone: an integer add of zero on the same-size unsigned view). An add of zero on the floats
  (cache_state._clone_tree's rule, used here up to 11-11) turns -0.0 into +0.0 and rewrites NaN bits: T1c's branch FAILs,
  row 0 of layer 35's QSA keys. So the GDN checkpoints are taken and restored here too (_snapshot_gdn, _restore_gdn).

serve.py:  import convcache as CC;  CC.install(G())  once;  CC.hint(sys_end=, lane=, side=, conv=)  before each item;
           CC.last()  after it (the info record).
Env: CONV_CACHE=0 (off: install() does nothing), CONV_CACHE_GB (RAM budget, default 2.5), CONV_CACHE_GRID (extra
checkpoints every this many tokens, default 8192, 0 = none), CONV_CACHE_CHECK=1, CONV_CACHE_LOG (a file for the lines).
"""
import hashlib
import os
import sys
import time

import mlx.core as mx
import numpy as np
from mtplx import cache_state as CS

ON = os.environ.get("CONV_CACHE", "1") != "0"
CHECK = os.environ.get("CONV_CACHE_CHECK") == "1"
BUDGET = float(os.environ.get("CONV_CACHE_GB", "2.5") or 2.5) * 1e9
GRID = int(os.environ.get("CONV_CACHE_GRID", "8192") or 0)
LOG = os.environ.get("CONV_CACHE_LOG", "")
TAPS_ALL = os.environ.get("CONV_CACHE_TAPS") == "all"
BRANCH = os.environ.get("CONV_CACHE_BRANCH", "1") != "0"   # 0: a request that would copy (branch) from an entry is a
                                                        # miss (cold prefill, new entry); the fallback for T1b's kind B

S = dict(g=None, orig=None, hint={}, last={}, entries=[], cur=None, pend_mtp=None, im_start=None, swin=2048)


def _say(msg):
    line = time.strftime("%H:%M:%S ") + "convcache: " + msg
    print(line, file=sys.stderr, flush=True)              # the engine log (serve.py points fd 1 and 2 there)
    if LOG:
        with open(LOG, "a") as f:
            f.write(line + "\n")


_UINT = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32, 8: mx.uint64}


def _clone(x):
    """a new buffer with x's exact bits. mx.array(x) can keep the storage (cache_state._clone_tree's note), and the add of
    zero used up to 11-11 is not a copy for floats: -0.0 + 0.0 = +0.0 and NaN bits are rewritten (T1c 11-11: row 0 of
    layer 35's QSA keys came out changed in both branch copies). An integer add of zero on the same-size unsigned view
    copies every bit, strided slices included."""
    if x is None:
        return None
    if mx.issubdtype(x.dtype, mx.inexact):
        u = _UINT[x.dtype.size]
        return (x.view(u) + mx.zeros((), dtype=u)).view(x.dtype)
    return x + mx.zeros((), dtype=x.dtype)                 # integers and bools: an add of zero is exact


def _clone_tree(v):
    """cache_state._clone_tree with _clone's bitwise copy"""
    if v is None:
        return None
    if isinstance(v, mx.array):
        return _clone(v)
    if isinstance(v, tuple):
        return tuple(_clone_tree(x) for x in v)
    if isinstance(v, list):
        return [_clone_tree(x) for x in v]
    if isinstance(v, dict):
        return {k: _clone_tree(x) for k, x in v.items()}
    return v


def _trimmable(c):
    return c is not None and hasattr(c, "is_trimmable") and c.is_trimmable()


def _snapshot_gdn(cache):
    """cache_state.snapshot_untrimmable_cache with the bitwise clone: the recurrent (non-trimmable) entries' state and
    meta state, None for the trimmable KV"""
    st = tuple(None if _trimmable(c) else _clone_tree(getattr(c, "state", None)) for c in cache)
    meta = tuple(None if _trimmable(c) else _clone_tree(getattr(c, "meta_state", None)) for c in cache)
    return CS.CacheSnapshot(states=st, meta_states=meta)


def _restore_gdn(cache, snap):
    """cache_state.restore_cache(cache, snap) (its _restore_state_preserving_container, clone=True) with the bitwise
    clone: the checkpoint stays intact for later restores"""
    for ent, st, meta in zip(cache, snap.states, snap.meta_states):
        if st is not None:
            cl = _clone_tree(st)
            cur = getattr(ent, "state", None)
            if hasattr(ent, "replace_state"):
                ent.replace_state(cl)
            elif isinstance(cur, list) and isinstance(cl, list) and len(cur) == len(cl):
                cur[:] = cl
            else:
                ent.state = cl
        if meta is not None:
            ent.meta_state = _clone_tree(meta)


def _leaves(entry):
    st = getattr(entry, "state", None)
    if st is None:
        return []
    if isinstance(st, (list, tuple)):
        return [x for x in st if x is not None]
    return [st]


def _nbytes(cache):
    n = 0
    for c in cache or []:
        try:
            b = c.nbytes
        except Exception:
            b = None
        n += int(b) if isinstance(b, (int, np.integer)) else sum(int(x.nbytes) for x in _leaves(c))
    return n


def _sha(x):
    if x.dtype == mx.bfloat16:
        x = x.view(mx.uint16)
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()[:16]


def _lcp(a, b):
    n = min(len(a), len(b))
    d = np.flatnonzero(a[:n] != b[:n])
    return int(d[0]) if len(d) else n


class TapView:
    """The taps of positions [base, base + len(arr)). T[a:b] slices by absolute position, which is all run() does
    (T[start:P], T[s0:P - 1], T[P - 1:P]); anything else raises."""

    def __init__(self, arr, base):
        self.arr, self.base = arr, int(base)
        self.shape = (self.base + int(arr.shape[0]),) + tuple(arr.shape[1:])
        self.dtype = arr.dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, k):
        if isinstance(k, slice) and k.step in (None, 1):
            a = 0 if k.start is None else int(k.start)
            b = self.shape[0] if k.stop is None else int(k.stop)
            if a < 0 or b < 0:
                raise IndexError(f"TapView: negative index {k!r}")
            if a < self.base:
                raise IndexError(f"TapView: rows from {a} asked, held from {self.base}")
            return self.arr[a - self.base:b - self.base]
        raise IndexError(f"TapView: only plain slices, got {k!r}")


class _TapSink:
    """per tap, the chunks of taps that the final rows [N - SWIN, N) and the edges' tails [e - SWIN, e) need; the rest is
    dropped as it is made (rows are evaluated at once, so a kept tap never holds its layer's whole output alive)"""

    def __init__(self, targets, c, swin, n_taps, tail, tail_base):
        self.spans = [(max(0, t - swin), t) for t in targets]
        self.keep = [dict() for _ in range(n_taps)]
        self.c, self.tail, self.tail_base = c, tail, tail_base

    def drain(self, TAP, a):
        """move every tap entry appended since the last drain (rows from absolute position a) into the sink"""
        for i, lst in enumerate(TAP["x"]):
            while lst:
                arr = lst.pop(0)
                if TAPS_ALL or any(a < t1 and a + arr.shape[0] > t0 for t0, t1 in self.spans):
                    mx.eval(arr)
                    self.keep[i][a] = arr

    def rows(self, lo, hi):
        """rows [lo, hi) in taps()'s layout [rows, n_taps * D]: the restored tail below c, the kept chunks above"""
        parts = []
        if lo < self.c:
            parts.append(self.tail[lo - self.tail_base:min(hi, self.c) - self.tail_base])
        if hi > self.c:
            per = []
            for k in self.keep:
                ch = [(a, x) for a, x in sorted(k.items()) if a < hi and a + x.shape[0] > max(lo, self.c)]
                assert ch, f"taps: rows [{max(lo, self.c)}, {hi}) not kept"
                x = mx.concatenate([x for _, x in ch]) if len(ch) > 1 else ch[0][1]
                a0 = ch[0][0]
                per.append(x[max(lo, self.c) - a0:hi - a0])
            parts.append(mx.stack(per, axis=1).reshape(per[0].shape[0], -1))
        return parts[0] if len(parts) == 1 else mx.concatenate(parts)


class Ckpt:
    __slots__ = ("edge", "gdn", "tail", "tail_base", "cap", "nbytes")

    def __init__(self, edge, gdn, cap):
        self.edge, self.gdn, self.cap = edge, gdn, cap
        self.tail, self.tail_base = None, 0
        self.nbytes = sum(int(x.nbytes) for st in gdn.states if st is not None for x in st if x is not None)
        self.nbytes += 0 if cap is None else int(cap.nbytes)

    def set_tail(self, tail, base):
        self.tail, self.tail_base = tail, base
        self.nbytes += int(tail.nbytes)


class Entry:
    def __init__(self, cache):
        self.cache, self.mtp, self.ck = cache, None, {}
        self.ids = np.zeros(0, np.int32)
        self.lane, self.conv, self.side, self.used = "main", None, False, time.time()

    def nbytes(self):
        return _nbytes(self.cache) + _nbytes(self.mtp) + sum(k.nbytes for k in self.ck.values())


def install(g):
    """swap the loop's prefill and MTPHead; returns False when off (CONV_CACHE=0) or already installed"""
    if not ON or S["g"] is not None:
        return False
    S["g"], S["orig"] = g, g["prefill"]
    tk = g.get("_TKZ")
    if tk is not None and hasattr(tk, "token_to_id"):
        S["im_start"] = tk.token_to_id("<|im_start|>")
    S["swin"] = int(g.get("SWIN") or 2048)
    g["prefill"] = prefill
    Base = g["MTPHead"]

    class MTPHeadCC(Base):
        """the chat path's head: on a hit it takes the live MTP cache trimmed to c (pairs (h_i, id_i+1), i < c, are the
        cold prefill's); either way its cache is registered on the request's entry for the next turn"""

        def __init__(self):
            pend, S["pend_mtp"] = S["pend_mtp"], None
            if pend is None:
                Base.__init__(self)
            else:
                self.cache = pend
                self.logits = self.h = None
            if S["cur"] is not None:
                S["cur"].mtp = self.cache

    g["MTPHead"] = MTPHeadCC
    _say(f"installed (budget {BUDGET / 1e9:.2f} GB, grid {GRID}, check {'on' if CHECK else 'off'}, SWIN {S['swin']})")
    return True


def hint(**kw):
    """the next request's hints (Leap's render meta): sys_end, lane, side, conv"""
    S["hint"] = dict(kw)


def last():
    return dict(S["last"])


def _edges(ids_np, h, c, chunk):
    """checkpoint edges for this prompt (chosen before its prefill): the prompt end (the last edge before the final
    <|im_start|>, where the next turn diverges), the end of system + tools, and every GRID tokens below the prompt end"""
    N = len(ids_np)
    end = N - 1
    if S["im_start"] is not None:
        pos = np.flatnonzero(ids_np == S["im_start"])
        if len(pos):
            end = min(end, int(pos[-1]))
    e_end = (end // chunk) * chunk
    out = {e_end}
    se = int(h.get("sys_end") or 0)
    if se >= chunk:
        out.add(min((se // chunk) * chunk, e_end))
    step = (GRID // chunk) * chunk
    if step >= chunk:
        out.update(range(step, e_end, step))
    return sorted(e for e in out if c < e <= N - 1 and e >= chunk)


def _lookup(ids_np, need_mtp, need_cap):
    best, bc = None, 0
    for e in S["entries"]:
        if need_mtp and e.mtp is None:
            continue
        lim = min(_lcp(e.ids, ids_np) - 1, len(ids_np) - 1)     # the MTP pair at c - 1 reads id c: share [0, c]
        ok = [k for k, ck in e.ck.items() if k <= lim and (ck.cap is not None or not need_cap) and ck.tail is not None]
        if ok and max(ok) > bc:
            best, bc = e, max(ok)
    return best, bc


def _prefix_state(src, c, ratio):
    k, v, raw, pooled = src.state
    nb = c // ratio
    return (_clone(k[..., :c, :]), _clone(v[..., :c, :]), _clone(raw[:, :c, :]),
            _clone(pooled[:, :nb, :]) if pooled is not None and nb > 0 else None)


def _restore(e, c, copy):
    g = S["g"]
    ck = e.ck[c]
    if copy:                                               # a branch: a new live cache holding [0, c) of e's
        cache = g["model"].make_cache()
        for dst, src in zip(cache, e.cache):
            if _trimmable(src):
                dst.state = _prefix_state(src, c, getattr(src, "ratio", 4))
        mtp = None
        if e.mtp is not None:
            mtp = g["model"].make_mtp_cache()
            for dst, src in zip(mtp, e.mtp):
                dst.state = _prefix_state(src, c, getattr(src, "ratio", 4))
    else:                                                  # the same conversation: trim e's live cache in place
        cache, mtp = e.cache, e.mtp
        for x in list(cache) + list(mtp or []):
            if _trimmable(x) and x.offset > c:
                x.trim(x.offset - c)
    _restore_gdn(cache, ck.gdn)                            # bitwise clones: the checkpoint stays intact for later restores
    mx.eval([x for ent in cache for x in _leaves(ent)], [x for ent in (mtp or []) for x in _leaves(ent)])
    return cache, mtp, ck


def _evict(reserve=0):
    ents = S["entries"]
    tot = sum(e.nbytes() for e in ents)
    for e in sorted((e for e in ents if e is not S["cur"]), key=lambda e: (not e.side, e.used)):
        if tot + reserve <= BUDGET:
            break
        tot -= e.nbytes()
        ents.remove(e)
    return tot


def _named(cache):
    """(label, leaf) per non-None leaf in _leaves' order; label = 'cache index:class[slot]' (the check's FAIL names)"""
    out = []
    for i, ent in enumerate(cache or []):
        st = getattr(ent, "state", None)
        if st is None:
            continue
        for j, x in enumerate(st if isinstance(st, (list, tuple)) else [st]):
            if x is not None:
                out.append((f"{i}:{type(ent).__name__}[{j}]", x))
    return out


def _rowfp(x, axis):
    """CONV_CACHE_CHECK drift diagnostic: a per-row print of one cache leaf, [rows, 2] uint32 = two integer sums of the
    row's raw bits (plain, position-weighted) over every other axis. Integer sums are exact, so equal rows give equal prints
    whatever the strides or capacity of the backing buffer."""
    if x.dtype in (mx.bfloat16, mx.float16):
        u = x.view(mx.uint16).astype(mx.uint32)
    elif x.dtype in (mx.float32, mx.int32, mx.uint32):
        u = x.view(mx.uint32)
    else:
        u = x.astype(mx.uint32)
    u = mx.moveaxis(u, axis, 0)
    u = u.reshape(u.shape[0], -1)
    w = mx.arange(u.shape[1], dtype=mx.uint32) * mx.array(2654435761, dtype=mx.uint32) + mx.array(1, dtype=mx.uint32)
    return mx.stack([u.sum(axis=1), (u * w).sum(axis=1)], axis=1)


def _fps(cache):
    """{(cache index, slot): [rows, 2] numpy} for every QSA leaf (keys, values, raw, pooled; rows = positions / blocks)"""
    out = {}
    for i, ent in enumerate(cache or []):
        if not _trimmable(ent):
            continue
        st = getattr(ent, "state", None)
        if not isinstance(st, (list, tuple)):
            continue
        for j, x in enumerate(st):
            if x is not None:
                out[(i, j)] = _rowfp(x, 2 if x.ndim == 4 else 1)
    mx.eval(list(out.values()))
    return {k: np.asarray(v) for k, v in out.items()}


def _fpcmp(old, new, upto=None):
    """rows where two _fps differ, over the rows both hold (and < upto): 'layer[slot] rows a..b (n of m)' per leaf"""
    msgs = []
    for key in sorted(old):
        a, b = old[key], new.get(key)
        if b is None:
            msgs.append(f"{key[0]}[{key[1]}] missing")
            continue
        n = min(len(a), len(b))
        if upto is not None:
            n = min(n, upto if key[1] != 3 else upto // 4)    # slot 3 (pooled) counts blocks of 4 positions
        d = np.flatnonzero((a[:n] != b[:n]).any(axis=1))
        if len(d):
            msgs.append(f"{key[0]}[{key[1]}] rows {d[0]}..{d[-1]} ({len(d)} of {n})")
    return msgs


def _x0msgs(cache=None, upto=None, gdn=None):
    """CONV_CACHE_CHECK census: the values an add of zero rewrites (-0.0, NaN bits), i.e. where the clone used up to 11-11
    changed bits. Per QSA leaf of `cache` below `upto`: 'L[slot] rows a..b (n rows, k values, first 0x....)'; per leaf of
    a GDN checkpoint `gdn` (a CacheSnapshot): 'gdn L[slot] k values (first 0x....)'."""
    def bad(x):
        u = _UINT[x.dtype.size]
        return x.view(u) != (x + mx.zeros((), dtype=x.dtype)).view(u)

    def first(x, b):
        u = _UINT[x.dtype.size]
        i = int(np.flatnonzero(np.asarray(b).reshape(-1))[0])
        return "0x%x" % int(np.asarray(x.view(u)).reshape(-1)[i])

    msgs = []
    for i, ent in enumerate(cache or []):
        st = getattr(ent, "state", None) if _trimmable(ent) else None
        for j, x in enumerate(st if isinstance(st, (list, tuple)) else []):
            if x is None or not mx.issubdtype(x.dtype, mx.inexact):
                continue
            ax = 2 if x.ndim == 4 else 1
            n = x.shape[ax] if upto is None else min(x.shape[ax], upto if j != 3 else upto // 4)
            xs = x[..., :n, :] if ax == 2 else x[:, :n, :]
            b = bad(xs)
            k = int(b.sum().item())
            if k:
                r = np.flatnonzero(np.asarray(mx.moveaxis(b, ax, 0).reshape(n, -1).any(axis=1)))
                msgs.append(f"{i}[{j}] rows {r[0]}..{r[-1]} ({len(r)} rows, {k} values, first {first(xs, b)})")
    for i, st in enumerate(gdn.states if gdn is not None else []):
        for j, x in enumerate(st if isinstance(st, (list, tuple)) else []):
            if x is None or not mx.issubdtype(x.dtype, mx.inexact):
                continue
            b = bad(x)
            k = int(b.sum().item())
            if k:
                msgs.append(f"gdn {i}[{j}] {k} values (first {first(x, b)})")
    return msgs


def _drift(tag, skip=None):
    """CONV_CACHE_CHECK: has any stored entry's QSA state changed since the end of its last prefill (rows below that
    prompt's end only: a decode appends after it and must never touch them)? One log line per changed entry."""
    for k, e in enumerate(S["entries"]):
        if e is skip or getattr(e, "fp", None) is None:
            continue
        msgs = _fpcmp(e.fp, _fps(e.cache))
        if msgs:
            _say(f"drift ({tag}): entry {k} (N {len(e.ids)}, {e.lane}): " + "; ".join(msgs[:12]))


def _cold_ref(ids, want_w, chunk):
    """CONV_CACHE_CHECK: the loop's own prefill on a fresh cache, before the warm one (U3C footprint kept aside)"""
    FG = S["g"]["FG"]
    cap = FG.get("cap")
    if cap is not None:
        FG["cap"] = np.zeros_like(cap)
    c2, last2, T2, ws2 = S["orig"](ids, want_w, chunk)
    N, sw = len(ids), S["swin"]
    ref = dict(last=_sha(last2), states=[_sha(x) for ent in c2 for x in _leaves(ent)], tail=_sha(T2[max(0, N - sw):N]),
               shapes=[tuple(x.shape) for _, x in _named(c2)], fps=_fps(c2),
               ws={a: _sha(w) for a, w in ws2}, cap=None if cap is None else FG["cap"].copy())
    if cap is not None:
        FG["cap"] = cap
    del c2, T2, ws2
    return ref


def prefill(ids, want_w=False, chunk=1024):
    g = S["g"]
    h, S["hint"] = S["hint"], {}
    N = len(ids)
    t0 = time.perf_counter()
    info = dict(n=N, hit=False, c=0, src="none", prefilled=N, miss="", restore_ms=0.0, prefill_ms=0.0, check="")
    S["last"], S["cur"], S["pend_mtp"] = info, None, None
    PFLM, A6 = g["PFLM"], g["A6"]
    if PFLM and N > chunk and (PFLM == 3 or g["PFLMM"].LMC["mode"] == 3):
        info["miss"] = "PF-B prefill (not cached until SEGALL)"
        return S["orig"](ids, want_w, chunk)
    if "warm" in A6 or (PFLM and A6):
        info["miss"] = "A6 warm / A6 with PF-A (not cached)"
        return S["orig"](ids, want_w, chunk)
    FG, TAP, EM, model, states = g["FG"], g["TAP"], g["EM"], g["model"], g["states"]
    need_cap = FG.get("cap") is not None
    ids_np = np.asarray(ids, dtype=np.int32)
    side = bool(h.get("side"))
    lane, conv = h.get("lane") or ("side" if side else "main"), h.get("conv")
    if CHECK:
        _drift("request start")
    e, c = _lookup(ids_np, want_w, need_cap) if N > chunk else (None, 0)
    copy = False
    if e is not None:                                      # take the entry over (same conversation, or its deepest edge) or copy
        same = (not side) and e.lane == lane and (conv is None or e.conv in (None, conv))
        copy = side or not (same or c >= max(e.ck))
        if copy and not BRANCH:                            # CONV_CACHE_BRANCH=0: a branch is a miss (cold prefill, new entry)
            e, c, copy = None, 0, False
            info["miss"] = "branch (CONV_CACHE_BRANCH=0)"
    if e is None and not info["miss"]:
        info["miss"] = "no stored prefix" if N > chunk else "one chunk"
    if CHECK and e is not None:
        tc = time.perf_counter()
        ref = _cold_ref(ids, want_w, chunk)
        info["check_ms"] = round(1e3 * (time.perf_counter() - tc), 1)
        t0 += time.perf_counter() - tc                     # prefill_ms below is the warm path alone
    else:
        ref = None
    tail, tail_base = None, 0
    if e is not None:
        tr = time.perf_counter()
        if copy:
            S["cur"] = e                                   # (kept while making room for the branch)
            if _evict(reserve=c * 30000) + c * 30000 > BUDGET and not side:
                copy = False                               # no room for a branch: take the entry over
            S["cur"] = None
        cache, mtp, ck = _restore(e, c, copy)
        if CHECK:
            _drift("after the cold check + restore")
            if getattr(e, "fp", None) is not None:
                msgs = _fpcmp(e.fp, _fps(cache), upto=c)
                if msgs:
                    _say(f"restore ({'copy' if copy else 'live'} at {c}): rows below c differ from the source's prefill: "
                         + "; ".join(msgs[:12]))
            try:                                           # a diagnostic: never the request's failure
                cen = _x0msgs(e.cache, upto=c, gdn=ck.gdn) + [f"MTP {m}" for m in _x0msgs(e.mtp, upto=c)]
                cen = "; ".join(cen[:12]) if cen else "none"
            except Exception as ex:
                cen = f"census error {type(ex).__name__}: {str(ex)[:160]}"
            _say(f"census ({'copy' if copy else 'live'} at {c}): values an add of zero rewrites (the pre-11-11 clone), "
                 f"source rows below c + the GDN checkpoint: {cen}")
        if need_cap:
            FG["cap"] = FG["cap"] | ck.cap
        tail, tail_base = ck.tail, ck.tail_base
        kept = {k: v for k, v in e.ck.items() if k <= c}
        if copy:
            cur = Entry(cache)
            S["entries"].append(cur)
        else:
            cur = e
        cur.ck, cur.mtp = kept, (mtp if want_w else None)
        S["pend_mtp"] = mtp if want_w else None
        info.update(hit=True, c=c, src="copy" if copy else "live", prefilled=N - c,
                    restore_ms=round(1e3 * (time.perf_counter() - tr), 2))
    else:
        _evict(reserve=N * 30000)
        cache, c = model.make_cache(), 0
        cur = Entry(cache)
        S["entries"].append(cur)
    edges = _edges(ids_np, h, c, chunk)
    sw = S["swin"]
    sink = _TapSink(edges + [N], c, sw, len(TAP["x"]), tail, tail_base)
    g["pf1"](True)
    g["tap_reset"]()
    TAP["on"] = True
    if PFLM and N > chunk:                                 # PF-A from c whenever the loop runs PF-A: the WHOLE prompt over one
                                                           # chunk (Leap T1 11-11 01:13: choosing by N - c ran short tails
                                                           # chunk-major, whose bits differ from PF-A's full-E scratch calls)
        lg, ws, caps = _run_pfa(g, ids, c, chunk, cache, want_w, set(edges), need_cap, sink)
    else:
        lg, ws, caps = _run_cm(g, ids, c, chunk, cache, want_w, set(edges), need_cap, sink)
    TAP["on"] = False
    g["pf1"](False)
    if not PFLM and EM.STATE.capture is not None and A6 & {"pools", "warm"}:
        g["A6CAP"].clear()
        g["A6CAP"].update(EM.STATE.capture)
        EM.STATE.capture = None
    base = max(0, N - sw) if not TAPS_ALL else 0
    if TAPS_ALL and c > 0:
        base = tail_base
    arr = sink.rows(base, N)
    T = arr if base == 0 else TapView(arr, base)
    for e_, ck in caps.items():
        b_ = max(0, e_ - sw)
        ck.set_tail(_clone(sink.rows(b_, e_)), b_)
    mx.eval(arr, [ck.tail for ck in caps.values()])
    del sink
    cur.ck.update(caps)
    cur.ids, cur.lane, cur.conv, cur.side, cur.used = ids_np, lane, conv, side, time.time()
    S["cur"] = cur
    if not cur.ck and cur in S["entries"]:                 # nothing restorable (a one-chunk prompt): not kept
        S["entries"].remove(cur)
    tot = _evict()
    info.update(prefill_ms=round(1e3 * (time.perf_counter() - t0), 1), entries=len(S["entries"]),
                gb=round(tot / 1e9, 3), edges=sorted(cur.ck))
    if CHECK:
        _drift("after this prefill", skip=cur)
        cur.fp = _fps(cache)                               # rows [0, N): the baseline the next request's drift check reads
    if ref is not None:
        bad = []
        if _sha(lg[0, -1]) != ref["last"]:
            bad.append("last logits")
        st = [_sha(x) for ent in cache for x in _leaves(ent)]
        if st != ref["states"]:
            nm = _named(cache)
            dk = [k for k, (a_, b_) in enumerate(zip(st, ref["states"])) if a_ != b_]
            lab = "; ".join(f"{nm[k][0]} {tuple(nm[k][1].shape)} vs {ref['shapes'][k]}" for k in dk[:8])
            more = "" if len(st) == len(ref["states"]) else f"; leaf count {len(st)} vs {len(ref['states'])}"
            rows = _fpcmp(ref["fps"], cur.fp if getattr(cur, "fp", None) is not None else _fps(cache))
            more += f"; QSA rows warm vs cold: {', '.join(rows[:8])}" if rows else ""
            bad.append(f"states ({len(dk)} of {len(st)}: {lab}{' ...' if len(dk) > 8 else ''}{more})")
        if _sha(T[max(0, N - sw):N]) != ref["tail"]:
            bad.append("taps tail")
        wd = [a for a, w in ws if ref["ws"].get(a) != _sha(w)]
        if wd:
            bad.append(f"widened rows at {wd}")
        if need_cap and not np.array_equal(FG["cap"], ref["cap"]):
            bad.append("U3C footprint")
        info["check"] = "PASS" if not bad else "FAIL: " + ", ".join(bad)
    _say(("hit" if info["hit"] else "miss") + f" N {N} c {info['c']} ({info['src']}{', ' + info['miss'] if info['miss'] else ''}) "
         f"prefilled {info['prefilled']}, restore {info['restore_ms']} ms, prefill {info['prefill_ms']} ms; entries "
         f"{info['entries']}, {info['gb']} GB; edges {info['edges']}" + (f"; check {info['check']}" if info["check"] else ""))
    return cache, lg[0, -1], T, ws


def _run_cm(g, ids, c, chunk, cache, want_w, edges, need_cap, sink):
    """the loop's chunk-major prefill body (r57_screen_rel1.py:1007-1021), verbatim, from c; a checkpoint after the chunk
    that ends at each edge"""
    FG, TAP, EM, model, states, A6 = g["FG"], g["TAP"], g["EM"], g["model"], g["states"], g["A6"]
    N, PFNOLOG, lg, ws, caps = len(ids), g["PFNOLOG"], None, [], {}
    for a in range(c, N, chunk):
        if ("pools" in A6 and a + chunk >= N) or ("warm" in A6 and a == 0):
            EM.STATE.capture = {}
        if PFNOLOG:
            last_ = a + chunk >= N
            lg = model(mx.array([ids[a:a + chunk]]), cache=cache, **({"logits_keep": 256} if last_ else {"emit_logits": False}))
            mx.eval(*([lg] if lg is not None else []), states(cache), [x[-1] for x in TAP["x"]])
        else:
            lg = model(mx.array([ids[a:a + chunk]]), cache=cache)
            mx.eval(lg, states(cache), [x[-1] for x in TAP["x"]])
        sink.drain(TAP, a)
        if want_w:
            w = g["TXT"]._last_widened
            assert w.shape[1] == min(chunk, N - a), w.shape
            mx.eval(w)
            ws.append((a, w))
        if a + chunk in edges:
            snap = _snapshot_gdn(cache)
            mx.eval([x for st in snap.states if st is not None for x in st if x is not None])
            caps[a + chunk] = Ckpt(a + chunk, snap, FG["cap"].copy() if need_cap else None)
    return lg, ws, caps


def _run_pfa(g, ids, c, chunk, cache, want_w, edges, need_cap, sink):
    """pflm.prefill_layer_major modes 1 / 2 (Leap's PF-A, engine/loop/pflm.py:217-264), verbatim, from c on `cache`. Each
    (layer, chunk) step is the chunk-major step, so a layer's GDN / PLE state and U3C row right after the chunk that ends
    at an edge are that edge's checkpoint for the layer: cloned as the sweep passes, one sweep, no extra expert reads"""
    import mtplx.models.qwen4_exp as Q
    import bigmoe.tier as BT
    PF, FG, TAP, model, states = g["PFLMM"], g["FG"], g["TAP"], g["model"], g["states"]
    lm = model.language_model
    tm = lm.model
    N = len(ids)
    starts = list(range(c, N, chunk))
    xs = [mx.array([ids[a:a + chunk]]) for a in starts]
    k_edge = {k: a + chunk for k, a in enumerate(starts) if a + chunk in edges}
    snaps = {e: [None] * len(cache) for e in k_edge.values()}
    metas = {e: [None] * len(cache) for e in k_edge.values()}
    rows = {e: [None] * len(cache) for e in k_edge.values()} if need_cap else {}

    def run():
        hs = []
        for x in xs:
            h = tm.embed_tokens(x)
            hs.append(mx.tile(h, (1, 1, tm.args.hc_count)))
        mx.eval(*hs)
        stage = tm._ple_stage_idx is not None and Q._COMPILED_VERIFY_PLE.get() is None
        BT._reuse_drop()
        PF.LMC.update(on=True, t=None, have=set())
        try:
            for L, (layer, cl) in enumerate(zip(tm.layers, cache)):
                for k, x in enumerate(xs):
                    if L == 0:
                        assert Q.create_ssm_mask(hs[k], cache[tm.ssm_idx]) is None, "PFLM: an ssm mask (padding?)"
                    if stage and L == tm._ple_stage_idx:
                        ple = tm.layers[tm._ple_stage_idx].ple
                        ple.ple_embedding.stage(x, cache[tm._ple_stage_idx], ple.NGRAM_IDX)
                    hs[k] = layer(hs[k], input_ids=x, ssm_mask=None, cache=cl)
                    mx.eval(hs[k], *states([cl]))
                    sink.drain(TAP, starts[k])
                    e = k_edge.get(k)
                    if e is not None:
                        if not _trimmable(cl):
                            st = cl.state
                            snaps[e][L] = [None if v is None else _clone(v) for v in st]
                            mx.eval([v for v in snaps[e][L] if v is not None])
                            metas[e][L] = getattr(cl, "meta_state", None)
                        if need_cap:
                            rows[e][L] = FG["cap"][L].copy()
        finally:
            PF.LMC.update(on=False, t=None, have=set(), sc=None)
            mx.clear_cache()
        ws = []
        if want_w:
            for k, h in enumerate(hs):
                assert h.shape[1] == min(chunk, N - starts[k]), h.shape
                mx.eval(h)
                ws.append((starts[k], h))
        tm._last_widened = hs[-1]
        out = tm.hyper_connection_mixer(hs[-1])[:, -256:]
        return lm._head_logits(out), ws

    if Q.qwen4_opdiet_enabled("rope"):
        with Q._rope_table_scope():
            lg, ws = run()
    else:
        lg, ws = run()
    mx.eval(lg, states(cache))
    caps = {}
    for e in snaps:
        snap = CS.CacheSnapshot(states=tuple(snaps[e]), meta_states=tuple(metas[e]))
        cap = None
        if need_cap:
            cap = FG["cap"].copy()
            for L, r in enumerate(rows[e]):
                cap[L] = r
        caps[e] = Ckpt(e, snap, cap)
    return lg, ws, caps
