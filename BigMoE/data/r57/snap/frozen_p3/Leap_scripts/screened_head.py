"""MTP-STEP-1 (Leap, Wed 2026-11-04 15:3x; TABLE_350 'MTP-STEP'): a screened 4-bit lm_head for the MTP chain's pick (FAST:
the top-1 and its softmax p are taken over a token subset). The pick needs the MTP's top-1 token and p per depth
(MTP Head, RERANK-0 / E1); the full head is ~1.7 of a ~2.2 ms step (MTP-STEP-0).
ScreenedHead(lm_head, base_ids, extra_cap=4096): the base set's rows (FR-Spec top-N: ~/Leap/data/frspec_rank.npy[:N])
gathered once into one contiguous 4-bit matrix; a second matrix of up to extra_cap rows for the session's own tokens
outside the base set (add(ids) appends them; rows past the count are padding, masked to -inf).
  __call__(h [B, 1, D] or [B, D]) -> (token ids [B], p [B] = softmax over base + extra rows, fp32)
No host sync inside __call__ (the pick can stay on the GPU: where(p > cf, tok, draft))."""
import numpy as np
import mlx.core as mx


class ScreenedHead:
    def __init__(self, lm_head, base_ids, extra_cap=4096):
        self.gs, self.bits = lm_head.group_size, lm_head.bits
        self.W, self.S, self.B = lm_head["weight"], lm_head["scales"], lm_head["biases"]
        base = np.unique(np.asarray(base_ids, np.int64)).astype(np.int32)
        self.base_set = set(base.tolist())
        self.ids = mx.array(base)
        self.w = mx.contiguous(mx.take(self.W, self.ids, axis=0))
        self.s = mx.contiguous(mx.take(self.S, self.ids, axis=0))
        self.b = mx.contiguous(mx.take(self.B, self.ids, axis=0))
        self.cap = int(extra_cap)
        self.n_extra = 0
        self.extra_list = []
        self.xids = mx.zeros((self.cap,), mx.int32)
        self.xw = mx.zeros((self.cap, self.W.shape[1]), self.W.dtype)
        self.xs = mx.zeros((self.cap, self.S.shape[1]), self.S.dtype)
        self.xb = mx.zeros((self.cap, self.B.shape[1]), self.B.dtype)
        self.xmask = mx.full((self.cap,), -float("inf"), mx.float32)
        mx.eval(self.ids, self.w, self.s, self.b, self.xids, self.xw, self.xs, self.xb, self.xmask)

    def add(self, token_ids):
        """append the session's tokens that are outside the base set (once each; ignored past extra_cap)"""
        new = [t for t in dict.fromkeys(int(t) for t in token_ids) if t not in self.base_set and t not in self.extra_list]
        new = new[:max(0, self.cap - self.n_extra)]
        if not new:
            return 0
        a, n = self.n_extra, len(new)
        idx = mx.array(np.asarray(new, np.int32))
        self.xids[a:a + n] = idx
        self.xw[a:a + n] = mx.take(self.W, idx, axis=0)
        self.xs[a:a + n] = mx.take(self.S, idx, axis=0)
        self.xb[a:a + n] = mx.take(self.B, idx, axis=0)
        self.xmask[a:a + n] = 0.0
        self.extra_list += new
        self.n_extra += n
        mx.eval(self.xids, self.xw, self.xs, self.xb, self.xmask)
        return n

    def logits(self, h):
        h2 = h.reshape(-1, h.shape[-1])
        q = lambda w, s, b: mx.quantized_matmul(h2, w, scales=s, biases=b, transpose=True, group_size=self.gs, bits=self.bits)
        lg = q(self.w, self.s, self.b).astype(mx.float32)
        if self.n_extra:
            lg = mx.concatenate([lg, q(self.xw, self.xs, self.xb).astype(mx.float32) + self.xmask], axis=-1)
        return lg

    def __call__(self, h):
        lg = self.logits(h)
        j = mx.argmax(lg, axis=-1)
        p = mx.exp(mx.take_along_axis(lg, j[:, None], axis=-1)[:, 0] - mx.logsumexp(lg, axis=-1))
        allids = mx.concatenate([self.ids, self.xids]) if self.n_extra else self.ids
        return mx.take(allids, j), p
