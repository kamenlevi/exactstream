"""MTP Head lever 'scr' (WEEK98 S18 SCREEN; NewIdeas 19:5x): train the head under the drafter's vocab screen, the sandbox
analog of M3's H8 on the 125B (FR-Spec's top-N ids + the session's own tokens; logits outside are masked).
The screen of a row anchored at s = the SCREEN_N (8192) most frequent ids of Sandbox98's data/shortlist_counts.npy
(Magicoder python + the training captures' answers; no MBPP / HumanEval text) + every id the window has committed by
the anchor (ids[0 .. s + 1]: the head at anchor s reads token s + 1). The screen is the same at every depth of a check.
Training: core._KLG['vg'] is wrapped once (as levers/accset_base.py). For the stepping arm (marker on its standin) the
head's logits outside the screen are masked (softmax over the screen); the target's out-of-screen top-32 mass moves to
kl_top's 'rest' bucket; a row's hit is the screened argmax, so accept-until-fail follows the screened drafter.
The chain is unchanged: standin is the identity and records the anchors of each depth >= 2. Depth 1 (no standin call)
uses a0 + arange(n). A strong reference to the step's window object tells a new step from the same one.
Read-outs apply the same screen through row_mask (sb98/tree8.py, spec suffix '!s')."""
import os
import numpy as np
import mlx.core as mx

N = int(os.environ.get("SCREEN_N", "8192"))
_G, _ST = {}, {"x": None, "anc": None, "fresh": False}


def glob_mask(V):
    if V not in _G:
        cnt = np.load(os.path.expanduser("~/Sandbox98/data/shortlist_counts.npy"))
        m = np.zeros(V, bool)
        m[np.argsort(-cnt, kind="stable")[:N]] = True
        _G[V] = m
    return _G[V]


def row_mask(ids, anc, V):
    """[n, V] bool numpy: the screen of rows anchored at anc"""
    fp = np.full(V, 1 << 40, np.int64)
    u, i = np.unique(np.asarray(ids, np.int64), return_index=True)
    fp[u] = i
    return glob_mask(V)[None, :] | (fp[None, :] <= np.asarray(anc, np.int64)[:, None] + 1)


def standin(j, h, anc):
    from sb import core as C
    _ST.update(x=C.CUR["x"], anc=np.array(anc), fresh=True)
    return h


standin._mh_screen = True


def setup(D, args):
    from sb import core as C
    if C._KLG.get("mh_screen"):
        return
    base = C._kl_hit

    def kl_hit(hn, tk, lp, mask, tgt):
        if not getattr(C.REPL["fn"], "_mh_screen", False):
            return base(hn, tk, lp, mask, tgt)
        x, n = C.CUR["x"], tk.shape[0]
        if _ST["x"] is x and _ST["fresh"] and len(_ST["anc"]) == n:
            anc = _ST["anc"]
        else:
            anc = x["a0"] + np.arange(n)                            # depth 1: every anchor from a0
        _ST["fresh"] = False
        lg = C.lm_logits_train(hn)[0]
        Mn = row_mask(x["ids"], anc, lg.shape[-1])
        lgm = mx.where(mx.array(Mn), lg, -1e30)
        lq = mx.take_along_axis(lgm, tk, -1) - mx.logsumexp(lgm, -1, keepdims=True)
        intk = mx.array(np.take_along_axis(Mn, np.array(tk).astype(np.int64), -1))
        lpm = mx.where(intk, lp, -1e30)                             # out-of-screen target mass -> the 'rest' bucket
        kl = (C.kl_top(mx.where(intk, lq, -1e30), lpm) * mask).sum() / mx.maximum(mask.sum(), 1.0)
        hit = mx.stop_gradient((mx.argmax(lgm, -1) == tgt).astype(mx.float32))
        return kl, hit
    C._KLG["vg"] = mx.value_and_grad(kl_hit)
    C._KLG["mh_screen"] = True
