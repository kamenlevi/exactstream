"""R57_CHATSH drop-in (Leap, Wed 2026-11-04; BigMoE's ask 15:5x; TABLE_350 'MTP-STEP'): the chat switch's MTP chain
(r57_screen.py MTPHead: D-20's native chain + D-23c's early exit) picking each draft over ScreenedHead's token subset
(FR-Spec top-32K + the session's own ids) instead of the full 4-bit lm_head. MTPSCREEN (TrainLab b36b990): a step
2.571 -> 1.353 ms at 1 row; drafts change only where the full top-1 is outside the subset (0.3 % of code depths).
The output stays exact: drafts are verified by the target model, whose lm_head is untouched.
Mechanism: the model's MTP logits come from one hook attribute: mtplx qwen4_exp (the 125B) calls the bound
`_mtp_draft_head_logits` (its __init__ binds it to _head_logits = lm_head); mtplx's generic MTP patch takes
`_mtplx_draft_lm_head` when set. The hook is swapped for the duration of this class's own mtp_forward calls and
restored after (nothing else sees it; the verify pass's lm_head is never touched). A pick over the subset is an
index into the subset, mapped back to a token id before it feeds the next step.
  MTPHead = r57_screen's class (the base, as constructed today); model = the model whose mtp_forward the base calls.
  cls = screened_mtphead(MTPHead, model, lm_head)   # lm_head: model.language_model.lm_head (4-bit QuantizedLinear)
  head = cls(); head.add(prompt_ids)               # then as the base: head.append(hidden, next_ids); head.draft_cut(k, t)
append() adds its committed ids to the subset (ids outside FR-Spec 32K, up to extra_cap=4096 per head object)."""
import os
from contextlib import contextmanager
import numpy as np
import mlx.core as mx
from screened_head import ScreenedHead

RANK = os.path.expanduser("~/Leap/data/frspec_rank.npy")


def _hook_name(tm):
    """the attribute the model's mtp_forward takes its draft logits from: mtplx qwen4_exp (the 125B) calls the bound
    `_mtp_draft_head_logits` (set in __init__ to _head_logits); mtplx's generic MTP patch takes `_mtplx_draft_lm_head`"""
    return "_mtp_draft_head_logits" if hasattr(tm, "_mtp_draft_head_logits") else "_mtplx_draft_lm_head"


@contextmanager
def _draft_head(tm, fn):
    name = _hook_name(tm)
    prev = getattr(tm, name, None)
    object.__setattr__(tm, name, fn)
    try:
        yield
    finally:
        object.__setattr__(tm, name, prev)


def screened_mtphead(MTPHead, model, lm_head, base_n=32768, extra_cap=4096, rank_path=RANK):
    """a subclass of the screen's MTPHead whose chain picks over ScreenedHead(lm_head, FR-Spec top-base_n + session ids)"""
    tm = model.language_model if hasattr(model, "language_model") else model
    sh_proto = (lm_head, np.load(rank_path)[:base_n], extra_cap)

    class ScreenedMTPHead(MTPHead):
        def __init__(self):
            super().__init__()
            self.sh = ScreenedHead(*sh_proto)
            self._lg = lambda z: self.sh.logits(z).reshape(*z.shape[:-1], -1)   # [B, T, subset] like the full head's

        def add(self, ids):
            """the session's ids (prompt, committed tokens) outside the base set join the subset"""
            return self.sh.add(np.asarray(ids).reshape(-1).tolist())

        def _ids(self):
            return mx.concatenate([self.sh.ids, self.sh.xids]) if self.sh.n_extra else self.sh.ids

        def append(self, hidden, next_ids):
            self.add(next_ids)
            with _draft_head(tm, self._lg):
                super().append(hidden, next_ids)

        def draft_cut(self, k, t):
            """the base's draft_cut (D-23c's running-product cut at t), each pick mapped from the subset to its token id"""
            ids = self._ids()
            l0 = self.logits[0, -1].astype(mx.float32)
            tk, p = mx.take(ids, mx.argmax(l0)), mx.softmax(l0).max()
            mx.eval(tk, p)
            toks, ps, h, run_, n_fw = [int(tk.item())], [float(p.item())], self.h, float(p.item()), 0
            with _draft_head(tm, self._lg):
                while run_ >= t and len(toks) < k:
                    lgx, h = model.mtp_forward(h, mx.array([[toks[-1]]]), mtp_cache=self.cache, return_hidden=True)
                    n_fw += 1
                    lx = lgx[0, -1].astype(mx.float32)
                    tk, p = mx.take(ids, mx.argmax(lx)), mx.softmax(lx).max()
                    mx.eval(tk, p)
                    toks.append(int(tk.item())); ps.append(float(p.item())); run_ *= ps[-1]
            if n_fw:
                self.cache[0].trim(n_fw)                     # the chain's own entries are not committed history
            return toks[:int((np.cumprod(np.array(ps)) >= t).sum())], ps

    return ScreenedMTPHead
