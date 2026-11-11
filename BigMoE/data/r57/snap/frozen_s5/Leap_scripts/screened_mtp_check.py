"""R57_CHATSH drop-in check (Leap, Wed 2026-11-04): screened_mtp.screened_mtphead on a stub MTP model (mtplx's logits path:
`_mtplx_draft_lm_head` if set, else lm_head, on the post-norm hidden) with a real 4-bit lm_head (V 40,000, D 256), against
the base MTPHead (r57_screen.py's class, verbatim, on the same stub). Two stub styles: 'generic' (mtplx's MTP patch:
_mtplx_draft_lm_head if set) and 'qwen4' (mtplx qwen4_exp, the 125B: mtp_forward calls the bound
_mtp_draft_head_logits, which __init__ binds to _head_logits = lm_head):
 (a) subset = every id: the same drafts and p as the base, bitwise;
 (b) subset = 20,000 of 40,000 ids: each chain matches the base up to its first depth whose full top-1 is outside the
     subset (prefix rule);
 (c) after any call the stub's draft head is unset again (the verify path never sees the subset);
 (d) add(): ids outside the base set join the subset once.
usage: [BigMoE venv] screened_mtp_check.py -> ~/Leap/data/screened_mtp_check.txt (GPU, MEM < 0.5 GB)"""
import os, sys, time, tempfile
import numpy as np
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5/Leap_scripts"))
import mlx.core as mx
import mlx.nn as nn
from screened_mtp import screened_mtphead

OUT = os.path.expanduser("~/Leap/data/screened_mtp_check.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nR57_CHATSH drop-in check {time.strftime('%a %H:%M:%S')}")
V, D = 40000, 256
mx.random.seed(5)

class Cache:
    def __init__(self): self.n = 0
    def trim(self, n): self.n -= n

class Stub(nn.Module):
    def __init__(self):
        super().__init__()
        lin = nn.Linear(D, V, bias=False)
        lin.weight = (lin.weight * 3.0).astype(mx.bfloat16)
        self.lm_head = nn.QuantizedLinear.from_linear(lin, 64, 4)
        self.Wm = (mx.random.normal((D, D)) / D ** 0.5).astype(mx.bfloat16)
        self.E = (mx.random.normal((V, D)) * 0.5).astype(mx.bfloat16)
        if STYLE == "qwen4":
            object.__setattr__(self, "_mtp_draft_head_logits", self._head_logits)
    def _core(self, h, ids, mtp_cache):
        x = mx.tanh(h @ self.Wm + self.E[ids])
        mtp_cache[0].n += ids.shape[-1]
        post = x * mx.rsqrt(mx.mean(x.astype(mx.float32) ** 2, -1, keepdims=True) + 1e-6).astype(x.dtype)
        if STYLE == "qwen4":
            return self._mtp_draft_head_logits(post), x
        dh = getattr(self, "_mtplx_draft_lm_head", None)
        return (dh(post) if dh is not None else self.lm_head(post)), x
    def _head_logits(self, h):
        return self.lm_head(h)
    def mtp_forward(self, h, ids, mtp_cache=None, return_hidden=False):
        lg, x = self._core(h, ids, mtp_cache)
        return (lg, x) if return_hidden else lg
    def mtp_update_cache(self, h, ids, mtp_cache=None):
        return self._core(h, ids, mtp_cache)[1]
    def make_mtp_cache(self):
        return [Cache()]


class MTPHead:                                           # r57_screen.py's class, verbatim (uses the module's `model`)
    def __init__(self):
        self.cache = model.make_mtp_cache()
        self.logits = self.h = None

    def append(self, hidden, next_ids):
        if hidden.shape[1] > 1:
            model.mtp_update_cache(hidden[:, :-1], mx.array([next_ids[:-1]]), mtp_cache=self.cache)
        self.logits, self.h = model.mtp_forward(hidden[:, -1:], mx.array([next_ids[-1:]]), mtp_cache=self.cache,
                                                return_hidden=True)

    def draft_cut(self, k, t):
        l0 = self.logits[0, -1].astype(mx.float32)
        tk, p = mx.argmax(l0), mx.softmax(l0).max()
        mx.eval(tk, p)
        toks, ps, h, run_, n_fw = [int(tk.item())], [float(p.item())], self.h, float(p.item()), 0
        while run_ >= t and len(toks) < k:
            lgx, h = model.mtp_forward(h, mx.array([[toks[-1]]]), mtp_cache=self.cache, return_hidden=True)
            n_fw += 1
            lx = lgx[0, -1].astype(mx.float32)
            tk, p = mx.argmax(lx), mx.softmax(lx).max()
            mx.eval(tk, p)
            toks.append(int(tk.item())); ps.append(float(p.item())); run_ *= ps[-1]
        if n_fw:
            self.cache[0].trim(n_fw)
        return toks[:int((np.cumprod(np.array(ps)) >= t).sum())], ps

def full_chain(h, tok, k):                               # the full head's greedy chain (no cut), for the prefix rule
    out = []
    for _ in range(k):
        lg, h = model.mtp_forward(h, mx.array([[tok]]), mtp_cache=[Cache()], return_hidden=True)
        tok = int(mx.argmax(lg[0, -1]).item()); out.append(tok)
    return out

def hook_ok():
    if STYLE == "qwen4":                                  # restored to the bound _head_logits of this model
        f = model.__dict__.get("_mtp_draft_head_logits")
        return f is not None and getattr(f, "__func__", None) is Stub._head_logits and f.__self__ is model
    return getattr(model, "_mtplx_draft_lm_head", None) is None

d = tempfile.mkdtemp()
rng = np.random.default_rng(1)
perm = rng.permutation(V).astype(np.int64)
np.save(os.path.join(d, "all.npy"), perm)
ok_a = ok_b = ok_c = True
n_match = n_cut = 0
n_in = n_ts = n_base_out = 0
for STYLE, name, base_n in (("generic", "all", V), ("generic", "half", 20000), ("qwen4", "all", V), ("qwen4", "half", 20000)):
    mx.random.seed(5)
    model = Stub()
    mx.eval(model.parameters())
    cls = screened_mtphead(MTPHead, model, model.lm_head, base_n=base_n, extra_cap=64, rank_path=os.path.join(d, "all.npy"))
    for trial in range(12):
        hid = (mx.random.normal((1, 3, D)) * 0.7).astype(mx.bfloat16)
        nxt = rng.integers(0, V, 3).tolist()
        b, s = MTPHead(), cls()
        b.append(hid, nxt); s.append(hid, nxt)
        tb, pb = b.draft_cut(8, 0.0)
        ts, ps_ = s.draft_cut(8, 0.0)
        ok_c &= hook_ok()
        if name == "all":
            ok_a &= tb == ts and pb == ps_
        else:
            inset = set(perm[:base_n].tolist()) | set(s.sh.extra_list)
            j = next((i for i, tkn in enumerate(tb) if tkn not in inset), len(tb))   # first depth the subset cannot pick
            ok_b &= ts[:j] == tb[:j]
            n_match += j; n_cut += len(tb)
            n_in += sum(tkn in inset for tkn in ts); n_ts += len(ts)
            n_base_out += sum(tkn not in inset for tkn in tb)
say(f"(a) subset = every id (generic + qwen4 stubs): drafts and p == the base's, bitwise: {ok_a}")
say(f"(b) subset = half the ids: chains == the base's up to the first depth outside the subset: {ok_b} "
    f"({n_match} of {n_cut} depths before it)")
ok_e = n_in == n_ts and n_base_out > 0
say(f"(e) the subset is really in use: every screened draft inside it {n_in} of {n_ts}; base drafts outside it "
    f"{n_base_out} -> {ok_e}")
say(f"(c) the hook restored after every call (generic: unset; qwen4: the bound _head_logits): {ok_c}")
cls = screened_mtphead(MTPHead, model, model.lm_head, base_n=20000, extra_cap=64, rank_path=os.path.join(d, "all.npy"))
s = cls()
outside = perm[20000:20010].tolist()
n1 = s.add(outside + outside[:3]); n2 = s.add(outside)
ok_d = n1 == 10 and n2 == 0 and s.sh.extra_list == outside
say(f"(d) add(): 10 new ids once, repeats ignored: {ok_d}")
say(f"R57_CHATSH drop-in: {'PASS' if ok_a and ok_b and ok_c and ok_d and ok_e else 'FAIL'}")
