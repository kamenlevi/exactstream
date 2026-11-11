"""D7b-NAX (Leap, Wed 2026-11-04 16:2x; TABLE_350 'BP-4'): the D7b drafter's 4-bit projections on Leap's 16-row NAX kernel
(qmm16.qmm16c, REF2's) instead of MLX's quantized_matmul. r57_screen calls ref2.remove() before every propose, so today
the D7b drafter (d7b_live.apply: the 5 layers' Linears 4-bit RTN g64) runs MLX's 16-row 4-bit kernel, which BP-2a timed
no faster than bf16 (44-61 GB/s); R-85b's 6.3 ms D7b propose had REF2's qmm16c inside. This puts qmm16c back for the
drafter's own layers only, whatever REF2's state.
install(draft): every nn.QuantizedLinear under draft.layers with bits 4, group 64, bf16 scales, N % 64 == 0, K % 64 == 0 and
N >= 256 takes qmm16c for bf16 inputs of <= 16 rows (zero-padded to 16; qmm16c is row-invariant); anything else (the
first long context prefill, odd shapes) keeps MLX's call. uninstall(draft) restores the class. Drafts change only by
fp32 accumulation order (same 4-bit weights); the drafter's outputs are verified, so the output stays exact."""
import os, sys
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_f24k/Leap_scripts"))
from qmm16 import qmm16c

COUNTS = {"nax": 0, "mlx": 0, "nax32": 0, "naxks": 0}
# D17-NAX (Leap, Fri 2026-11-06 08:2x; TABLE_350 'DRAFTER PROFILE'): with R57_D17 the draft block is 17 rows, so every block
# projection fell off the <= 16-row path onto MLX's quantized_matmul (52-57 GB/s: gate|up / down 219 us, q / o 107 us).
# D7BNAX_32=1 (off by default): 17-32 rows run as two 16-row qmm16c halves (weights >= 4 MB), and every <= 16-row NaxQL
# (D7BNAX_32=2: 17-32 rows on qmm32.qmm32c instead, one weight read, every eligible shape; Fri 09:0x)
# call takes a per-shape K split
# (_ks: the smallest ks dividing K / 64, <= 10, that gives >= 320 simdgroups). Drafts change only by fp32 accumulation
# order; the drafter's outputs are verified, so the output stays exact.
D32 = [int(os.environ.get("D7BNAX_32", "0") or "0")]   # 1: two qmm16c halves (>= 4 MB); 2: qmm32.qmm32c, one weight read, all shapes


def _ks(N, K):
    if not D32[0]:
        return 1
    g, tiles, ks = K // 64, N // 32, 1
    for k in (1, 2, 4, 5, 8, 10):
        if g % k == 0:
            ks = k
            if tiles * k >= 320:
                break
    if ks > 1:
        COUNTS["naxks"] += 1
    return ks


class NaxQL(nn.QuantizedLinear):
    """nn.QuantizedLinear whose <= 16-row bf16 calls run qmm16c"""

    def __call__(self, x):
        rows = x.size // x.shape[-1] if x.ndim >= 2 else 1
        if x.dtype == mx.bfloat16 and rows <= 16 and "bias" not in self:
            K = x.shape[-1]
            x2 = x.reshape(rows, K)
            if rows < 16:
                x2 = mx.concatenate([x2, mx.zeros((16 - rows, K), x2.dtype)])
            y = qmm16c(x2, self["weight"], self["scales"], self["biases"], _ks(self["weight"].shape[0], K))
            COUNTS["nax"] += 1
            return y[:rows].reshape(*x.shape[:-1], y.shape[-1])
        if D32[0] >= 2 and x.dtype == mx.bfloat16 and rows <= 32 and "bias" not in self:   # D17-NAX v2: qmm32c (09:0x
            K = x.shape[-1]                                                                   # bench: == the halves bitwise,
            from qmm32 import qmm32c                                                          # -0.44 ms per propose vs them)
            N = self["weight"].shape[0]
            g, tiles, ks = K // 64, N // 32, 1
            for k in (1, 2, 4, 5):                       # qmm32c's partials are 2x qmm16c's: ks <= 5 keeps them <= 16 KB
                if g % k == 0:
                    ks = k
                    if tiles * k >= 320:
                        break
            if ks > 1:
                COUNTS["naxks"] += 1
            y = qmm32c(x.reshape(rows, K), self["weight"], self["scales"], self["biases"], ks)
            COUNTS["nax32"] += 1
            return y.reshape(*x.shape[:-1], N)
        if (D32[0] and x.dtype == mx.bfloat16 and rows <= 32 and "bias" not in self      # D17-NAX: two 16-row halves,
                and self["weight"].shape[0] * self["weight"].shape[1] * 8 >= 8_000_000):   # for weights >= 4 MB only (the
            # 08:19 bench: k|v 1024 x 2560 and 640 x 2560 ran faster on MLX's 17-row kernel than as two halves)
            K = x.shape[-1]
            x2 = x.reshape(rows, K)
            ks = _ks(self["weight"].shape[0], K)
            ya = qmm16c(x2[:16], self["weight"], self["scales"], self["biases"], ks)
            yb = qmm16c(x2[16:], self["weight"], self["scales"], self["biases"], ks)
            COUNTS["nax32"] += 1
            return mx.concatenate([ya, yb], axis=0).reshape(*x.shape[:-1], ya.shape[-1])
        COUNTS["mlx"] += 1
        return super().__call__(x)


class NaxNQL(NaxQL):
    """NaxQL on DENSE-N's fragment-native copy (qmm26.qmm16n, bitwise == qmm16c at ks 1; built at install, kept beside
    the MLX layout for the > 16-row calls)"""

    def __call__(self, x):
        rows = x.size // x.shape[-1] if x.ndim >= 2 else 1
        if x.dtype == mx.bfloat16 and rows <= 16 and "bias" not in self:
            K = x.shape[-1]
            x2 = x.reshape(rows, K)
            wn, sn, bn, N = self._nax_native
            y = _qmm16n(x2, wn, sn, bn, N)
            COUNTS["nax"] += 1
            return y[:rows].reshape(*x.shape[:-1], N)
        COUNTS["mlx"] += 1
        return nn.QuantizedLinear.__call__(self, x)


def _qmm16n(x, wn, sn, bn, N):
    import qmm26
    return qmm26.qmm16n(x, wn, sn, bn, N)


def _eligible(m):
    if not isinstance(m, nn.QuantizedLinear) or m.bits != 4 or m.group_size != 64 or "bias" in m:
        return False
    N, K = m["weight"].shape[0], m["weight"].shape[1] * 8
    return m["scales"].dtype == mx.bfloat16 and N % 64 == 0 and K % 64 == 0 and N >= 256


def install(draft, native=False):
    """native=True: DENSE-N's layout (qmm16n; a native copy per weight, +~0.26 GB for the D7b drafter)"""
    n = 0
    for layer in draft.layers:
        for _, m in layer.named_modules():
            if type(m) is nn.QuantizedLinear and _eligible(m):
                if native:
                    if getattr(m, "_nax_native", None) is None:
                        import qmm26
                        wn, sn, bn = (mx.array(a) if not isinstance(a, mx.array) else a
                                      for a in qmm26.repack(m["weight"], m["scales"], m["biases"]))
                        mx.eval(wn, sn, bn)
                        object.__setattr__(m, "_nax_native", (wn, sn, bn, int(m["weight"].shape[0])))
                    m.__class__ = NaxNQL
                else:
                    m.__class__ = NaxQL
                n += 1
    return n


def uninstall(draft):
    n = 0
    for layer in draft.layers:
        for _, m in layer.named_modules():
            if type(m) in (NaxQL, NaxNQL):
                m.__class__ = nn.QuantizedLinear
                n += 1
    return n


# ---- BP-6 (Wed 16:2x): the 8-bit drafter for the exact default (qmm16c8.py, BP-5) ----
_KS8 = {}                                               # (N, K) -> split-K from BP-5's best per drafter shape


def _ks8(N, K):
    if (N, K) in _KS8:
        return _KS8[(N, K)]
    if N <= 1024 or N == 7680:
        ks = 4
    elif N == 2560 and K in (4096, 7680):
        ks = 2
    else:
        ks = 1
    while K % (64 * ks):
        ks //= 2
    return ks


class NaxQL8(nn.QuantizedLinear):
    """an 8-bit g64 nn.QuantizedLinear whose <= 16-row bf16 calls run qmm16c8 (one fixed split-K per weight shape)"""

    def __call__(self, x):
        rows = x.size // x.shape[-1] if x.ndim >= 2 else 1
        if x.dtype == mx.bfloat16 and rows <= 16 and "bias" not in self:
            from qmm16c8 import qmm16c8
            K = x.shape[-1]
            N = self["weight"].shape[0]
            y = qmm16c8(x.reshape(rows, K), self["weight"], self["scales"], self["biases"], ks=_ks8(N, K))
            COUNTS["nax"] += 1
            return y.reshape(*x.shape[:-1], N)
        COUNTS["mlx"] += 1
        return nn.QuantizedLinear.__call__(self, x)


def apply8(draft, fc=True):
    """BP-6: quantize the drafter's layer Linears (and fc) to 8-bit RTN g64 in place and route their <= 16-row calls to
    qmm16c8 (the exact default's drafter: near lossless vs bf16, unlike D7b's 4-bit). Returns the number of NaxQL8."""
    nn.quantize(draft, group_size=64, bits=8,
                class_predicate=lambda p, m: isinstance(m, nn.Linear) and (p.startswith("layers") or (fc and p == "fc")))
    n = 0
    for _, m in draft.named_modules():
        if type(m) is nn.QuantizedLinear and m.bits == 8 and m.group_size == 64 and "bias" not in m:
            N, K = m["weight"].shape[0], m["weight"].shape[1] * 4
            if m["scales"].dtype == mx.bfloat16 and N % 32 == 0 and K % 64 == 0:
                m.__class__ = NaxQL8
                n += 1
    mx.eval(draft.parameters())
    return n


# ---- BP-7 (Wed 16:3x): the draft head on qmm16c ----
def head_logits(draft, hid):
    """the drafter's compute_logits with the (target's 4-bit g64) lm_head on qmm16c at <= 16 rows: the same output
    multiplier and softcap after it. Falls back to draft.compute_logits for anything else (a converted / non-4-bit head,
    more than 16 rows)."""
    lm = getattr(draft, "lm_head", None)
    rows = hid.size // hid.shape[-1]
    if (not isinstance(lm, nn.QuantizedLinear) or lm.bits != 4 or lm.group_size != 64 or "bias" in lm
            or lm["weight"].ndim != 2 or rows > 16 or hid.dtype != mx.bfloat16 or lm["weight"].shape[0] % 32):
        return draft.compute_logits(hid)
    lg = qmm16c(hid.reshape(rows, hid.shape[-1]), lm["weight"], lm["scales"], lm["biases"]).reshape(*hid.shape[:-1], -1)
    COUNTS["nax"] += 1
    lg = lg * draft.config.output_multiplier
    cap = draft.config.final_logit_softcapping
    if cap is not None and cap > 0:
        lg = mx.tanh(lg / cap) * cap
    return lg


# ---- BP-1c / BP-9 (Wed 17:0x): the screened draft head on qmm16c (FR-Spec top-98,304 + session ids) ----
class DraftSubset:
    """the drafter's head over a token subset (exact-safe: drafts are verified). BP-1c: at 98,304 + session ids the
    walk keeps code x1.0000 / math x0.9995. Base rows gathered once from the (target's 4-bit g64) lm_head; up to cap
    session ids (add(ids): the prompt, then each check's committed ids; reset() per prompt). The subset length stays a
    multiple of 128 (base 98,304 + cap 4,096), unused extra rows masked to -inf, so head2s's two-stage top-K runs as is."""

    def __init__(self, lm_head, base_n=98304, cap=4096, rank_path=None):
        import numpy as np
        rank = np.load(rank_path or os.path.expanduser("~/Leap/data/frspec_rank.npy"))
        ids = np.unique(np.asarray(rank[:base_n], np.int64)).astype(np.int32)
        assert (ids.size + cap) % 128 == 0, (ids.size, cap)
        self.np = np
        self.w, self.s, self.b = lm_head["weight"], lm_head["scales"], lm_head["biases"]
        self.ids = mx.array(ids)
        self.set = set(ids.tolist())
        self.wb = mx.contiguous(mx.take(self.w, self.ids, axis=0))
        self.sb = mx.contiguous(mx.take(self.s, self.ids, axis=0))
        self.bb = mx.contiguous(mx.take(self.b, self.ids, axis=0))
        self.cap, self.extra = int(cap), []
        self.xids = mx.zeros((cap,), mx.int32)
        self.xw = mx.zeros((cap, self.w.shape[1]), self.w.dtype)
        self.xs = mx.zeros((cap, self.s.shape[1]), self.s.dtype)
        self.xb = mx.zeros((cap, self.b.shape[1]), self.b.dtype)
        self.xok = mx.zeros((cap,), mx.bool_)
        mx.eval(self.ids, self.wb, self.sb, self.bb, self.xids, self.xw, self.xs, self.xb, self.xok)

    def add(self, token_ids):
        new = [t for t in dict.fromkeys(int(t) for t in self.np.asarray(token_ids).reshape(-1)) if t not in self.set]
        new = new[:max(0, self.cap - len(self.extra))]
        if not new:
            return 0
        a, n = len(self.extra), len(new)
        idx = mx.array(self.np.asarray(new, self.np.int32))
        self.xids[a:a + n] = idx
        self.xw[a:a + n] = mx.take(self.w, idx, axis=0)
        self.xs[a:a + n] = mx.take(self.s, idx, axis=0)
        self.xb[a:a + n] = mx.take(self.b, idx, axis=0)
        self.xok[a:a + n] = True
        mx.eval(self.xids, self.xw, self.xs, self.xb, self.xok)
        self.extra += new
        self.set.update(new)
        return n

    def reset(self):
        self.set -= set(self.extra)
        self.extra = []
        self.xok = mx.zeros((self.cap,), mx.bool_)

    def logits(self, draft, hid):
        rows = hid.size // hid.shape[-1]
        x = hid.reshape(rows, hid.shape[-1])
        q = lambda w, s_, b: (qmm16c(x, w, s_, b) if rows <= 16 else      # > 16 rows (generic; D17's head sees 16): chunks, the
                              mx.concatenate([qmm16c(x[i:i + 16], w, s_, b) for i in range(0, rows, 16)], axis=0))   # same bits
        yb = q(self.wb, self.sb, self.bb)
        yx = mx.where(self.xok[None], q(self.xw, self.xs, self.xb), mx.array(-float("inf"), yb.dtype))
        lg = mx.concatenate([yb, yx], axis=-1).reshape(*hid.shape[:-1], -1)
        lg = lg * draft.config.output_multiplier
        cap = draft.config.final_logit_softcapping
        if cap is not None and cap > 0:
            lg = mx.tanh(lg / cap) * cap
        return lg

    def ids_all(self):
        return mx.concatenate([self.ids, self.xids])


def screened_head2s(draft, hid, sub, K=16):
    """r57_screen's head2s (two-stage top-K over 128-wide blocks) over the subset's logits; candidates mapped back to
    token ids"""
    lg = sub.logits(draft, hid)
    B, s = 128, lg.shape
    blk = lg.reshape(*s[:-1], s[-1] // B, B)
    tb = mx.argpartition(blk.max(axis=-1), -K, axis=-1)[..., -K:]
    sel = mx.take_along_axis(blk, mx.broadcast_to(tb[..., None], (*s[:-1], K, B)), axis=-2).reshape(*s[:-1], K * B)
    j = mx.argpartition(sel, -K, axis=-1)[..., -K:]
    c = mx.take_along_axis(tb, j // B, axis=-1) * B + j % B
    return mx.take(sub.ids_all(), c), mx.take_along_axis(lg, c, axis=-1)
