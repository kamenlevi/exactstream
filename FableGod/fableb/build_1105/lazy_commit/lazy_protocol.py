"""lazy_protocol.py (Fable B, build_1105): a small exact model of the GDN commit protocol, today's ("stock") and the
lazy one, in numpy fp32. No MLX, no GPU. It exists to prove the BOOKKEEPING: with the same per-row arithmetic in the same
order, the lazy protocol's outputs, conv states and recurrent states equal today's bit for bit through any sequence of
checks, partial and full accepts, rollbacks, one-row AR steps, prefill chunks and stock-path forwards.
ref2_lazy.patch implements exactly the functions marked [ref2]; gdnbody_lazy.patch implements kernel_lazy.

Today (read from the code: mtplx qwen4_exp.py commit_verified_window 6616-6730, cache_state.py snapshot / restore, ref2's
_gdn_body_call at Leap cccfd42):
  forward(S rows)   reads entry[1], runs the S rows, writes the END state to entry[1]; stashes (qkv, q, k, v, a, b)
  commit, full      nothing (the forward left the post-window state)
  commit, partial   entry[1] = gated_delta_update(kept rows, from the SNAPSHOT's state): one more read + write of the
                    state per layer, in its own kernel, pulled by the next check's eval
  snapshot          views of the entry's leaves (new array objects); rollback = clones of them, arity from the snapshot
Lazy: a pure-GDN entry's state is [conv, S] (S the true state) or [conv, B, pack] (the true state = the pack's rows
replayed on B). The pack is one more leaf, so MTPLX's own snapshot and restore carry it: no identity, no hook there.
  forward(S rows)   the kernel replays the pack's rows on B (registers only), writes THAT state (the committed one) once,
                    runs the S rows, and emits their q | k | v | a | b as the new pack. state = [conv', committed, pack]
  commit, full      nothing (base + the S rows of the pack is the post-window state)
  commit, partial   MTPLX's commit runs as it is (it sets the conv state; its replay node for entry[1] is dropped
                    unevaluated); then state = [conv, committed, pack[:keep]] (a view). No kernel, no state traffic.
  anything else that reads entry[1] (a stock-path forward, a state reader): flush = today's replay, then proceed.
  a window whose forward took the stock path but whose snapshot is in the 3-leaf form: MTPLX's replay is handed the
  snapshot's TRUE state (its base with its pack replayed) instead of the bare base."""
import numpy as np

F = np.float32
H, DV, DK, C = 3, 4, 4, 6            # value heads, state rows / cols per head, conv channels (tiny, the logic is the same)


# ---------------------------------------------------------------- the arithmetic (shared by both protocols)
def gates(a, b):
    g = (F(1) / (F(1) + np.exp(-a, dtype=F))).astype(F)
    beta = (F(1) / (F(1) + np.exp(-b, dtype=F))).astype(F)
    return g, beta


def step(S, q, k, v, g, beta):
    """one row of the recurrence for every head (mlx_lm gated_delta_step's loop body): decay, erase / write, read"""
    S = (S * g[:, None, None]).astype(F)
    kv = (S * k[:, None, :]).sum(-1, dtype=F)
    delta = ((v - kv) * beta[:, None]).astype(F)
    S = (S + k[:, None, :] * delta[:, :, None]).astype(F)
    out = (S * q[:, None, :]).sum(-1, dtype=F)
    return S, out


def prework(conv, qkv):
    """the conv over [conv state; rows] -> q, k, v per row, and the new conv state (the last 3 rows)"""
    full = np.concatenate([conv, qkv]).astype(F)
    S = qkv.shape[0]
    win = np.stack([full[s:s + 4].sum(0, dtype=F) for s in range(S)])               # [S, C]
    q = np.tanh(win[:, None, :DK] + np.arange(H, dtype=F)[None, :, None]).astype(F)  # [S, H, DK]
    k = np.cos(win[:, None, :DK] * F(0.5) + np.arange(H, dtype=F)[None, :, None]).astype(F)
    v = np.sin(win[:, None, :DV] + np.arange(H, dtype=F)[None, :, None]).astype(F)   # [S, H, DV]
    return q, k, v, full[-3:].copy()


def replay(state, rows):
    """the recurrence over rows = (q, k, v, a, b) from state -> (outputs, end state): today's kernel and today's commit"""
    q, k, v, a, b = rows
    S, outs = state, []
    for s in range(q.shape[0]):
        g, beta = gates(a[s], b[s])
        S, o = step(S, q[s], k[s], v[s], g, beta)
        outs.append(o)
    return (np.stack(outs) if outs else np.zeros((0, H, DV), F)), S


PW = 2 * H * DK + H * DV + 2 * H                     # the pack's row: q | k | v | a | b


def pack_rows(q, k, v, a, b):
    n = q.shape[0]
    return np.concatenate([q.reshape(n, -1), k.reshape(n, -1), v.reshape(n, -1), a, b], axis=1).astype(F)


def unpack(pack):
    n, o = pack.shape[0], 0
    out = []
    for shape in ((H, DK), (H, DK), (H, DV), (H,), (H,)):
        w = int(np.prod(shape))
        out.append(pack[:, o:o + w].reshape((n,) + shape))
        o += w
    return tuple(out)


def kernel_lazy(base, pack, rows, counters=None):
    """[kernel gdn_body_lazy] phase B's loop body over the pack's rows (state only), the state written once (the committed
    one; a copy of the input when nothing is pending), then the new rows; emits the new rows' pack"""
    S = base
    if pack is not None:
        _, S = replay(S, unpack(pack))
    committed = S.copy()
    outs, _ = replay(S, rows)                         # the end state stays in registers and is dropped
    if counters is not None:
        counters["reads"] += 1; counters["writes"] += 1
        counters["steps"] += (pack.shape[0] if pack is not None else 0) + rows[0].shape[0]
    return outs, committed, pack_rows(*rows)


class Entry:
    """a GDN layer's cache entry (mlx_lm ArraysCache): .state is the list of leaves, [i] indexes it"""
    def __init__(self, conv, state):
        self.state = [conv, state]
        self._rows = None                             # MTPLX's capture stash (qkv, q, k, v, a, b)
        self._lz_c = None                             # lazy: (committed base, S) of the LAST forward if it was lazy
    def __getitem__(self, i): return self.state[i]
    def __setitem__(self, i, v): self.state[i] = v


def snapshot(entry):
    """cache_state.snapshot_untrimmable_cache_lazy: a view of every leaf (NEW objects: identity is not preserved)"""
    return tuple(x.copy() for x in entry.state)


def rollback(entry, snap):
    """cache_state.restore_cache: clones of the snapshot's leaves; the entry takes the snapshot's arity"""
    cloned = [x.copy() for x in snap]
    if len(entry.state) == len(cloned):
        entry.state[:] = cloned
    else:
        entry.state = cloned
    entry._rows = None


# ---------------------------------------------------------------- today's protocol
def stock_forward(entry, qkv, a, b, counters=None):
    q, k, v, conv = prework(entry[0], qkv)
    outs, end = replay(entry[1], (q, k, v, a, b))
    entry._rows = (qkv, q, k, v, a, b)
    entry[0], entry[1] = conv, end
    if counters is not None:
        counters["reads"] += 1; counters["writes"] += 1; counters["steps"] += qkv.shape[0]
    return outs


def stock_commit(entry, pre, keep, verified, counters=None):
    """MTPLX commit_verified_window for one pure-GDN layer (it only ever touches leaves 0 and 1)"""
    if keep == verified:
        entry._rows = None
        return True
    qkv, q, k, v, a, b = entry._rows
    _, new_state = replay(pre[1], (q[:keep], k[:keep], v[:keep], a[:keep], b[:keep]))
    entry[0] = np.concatenate([pre[0], qkv[:keep]])[-3:].copy()
    entry[1] = new_state
    entry._rows = None
    if counters is not None:
        counters["reads"] += 1; counters["writes"] += 1; counters["steps"] += keep; counters["replays"] += 1
    return True


# ---------------------------------------------------------------- the lazy protocol  [ref2]
def lz_true(base, pack, counters=None):
    """[ref2 _lz_true] the true state of (base, pack): today's replay of the pack's rows on the base"""
    _, st = replay(base, unpack(pack))
    if counters is not None:
        counters["reads"] += 1; counters["writes"] += 1; counters["steps"] += pack.shape[0]; counters["flushes"] += 1
    return st


def lz_flush(entry, counters=None):
    """[ref2 gdn_lazy_flush] the 3-leaf form -> the 2-leaf form"""
    if len(entry.state) != 3:
        return False
    entry.state = [entry.state[0], lz_true(entry.state[1], entry.state[2], counters)]
    return True


def lazy_forward(entry, qkv, a, b, counters=None, body=True):
    """[ref2 _gdn_call / _gdn_body_call with CFG gdn_lazy_commit] body=False: a call the body declines (stock path)"""
    entry._lz_c = None                                # every forward of the entry clears it; only the lazy body sets it
    if not body:
        lz_flush(entry, counters)
        return stock_forward(entry, qkv, a, b, counters)
    pend = entry.state[2] if len(entry.state) == 3 else None
    q, k, v, conv = prework(entry[0], qkv)
    outs, committed, pack = kernel_lazy(entry[1], pend, (q, k, v, a, b), counters)
    entry._rows = (qkv, q, k, v, a, b)
    entry.state = [conv, committed, pack]             # the post-window state = committed + the pack's S rows
    entry._lz_c = (committed, qkv.shape[0])
    return outs


def lazy_commit(entry, pre, keep, verified, counters=None):
    """[ref2 _commit with _lz_commit_pre / _lz_commit_post] MTPLX's commit, wrapped"""
    win = entry._lz_c if entry._lz_c is not None and entry._lz_c[1] == verified else None
    if win is None and len(pre) == 3:                 # pre: a stock-path window over a 3-leaf snapshot
        pre = (pre[0], lz_true(pre[1], pre[2], counters))
    ok = stock_commit(entry, pre, keep, verified, counters=None if win is not None else counters)
    if ok and win is not None and keep < verified:    # post: the committed base back, the kept rows pending
        entry.state = [entry.state[0], win[0], entry.state[2][:keep]]
    entry._lz_c = None
    return ok


def peek_state(entry):
    """[ref2 gdn_lazy_state] the true recurrent state for a reader, the entry unchanged"""
    return entry.state[1] if len(entry.state) != 3 else replay(entry.state[1], unpack(entry.state[2]))[1]
