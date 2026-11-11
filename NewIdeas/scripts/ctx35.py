"""NewIdeas I5 kill test: is frugal's damage in the CONTEXT or in the current token? (35B proxy, decode mode, 2026-10-21)

Idea (I5, "lazy exactness"): decode with frugal routing (no stalls for ranks >= 2), and repair the past exactly in the
background: an exact batched pass over committed tokens (reads amortised across tokens, never on the critical path)
replaces their K/V and GDN state. That only pays if most of frugal's KLD comes from the corrupted context.
Four decode streams over the same teacher-forced tokens, each one token per call:
  exact      every top-8 computed (reference logits, and the exact context)
  current    the token computed frugally ON TOP OF THE EXACT CONTEXT (snapshot / restore of the exact cache): repair lag 0
  lagR       frugal stream whose cache is replaced by a copy of the exact cache every R tokens (repair lag 0..R-1)
  frugal     frugal everywhere (its own context): what frugal does today
Frugal = per-layer LRU of C slots per stream, ranks < W loaded and inserted, other non-resident ranks skipped (not
inserted), no renormalisation. Experts at 4-bit g64 as defer35.py. KLD(exact || stream) over the full vocabulary and
top-1 agreement, positions >= WARM after the 32-token exact prefill.
usage: ctx35.py [C W R T]    defaults 26 2 16 700
"""
import os, sys, time
from collections import OrderedDict
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import qwen3_next
from mlx_lm.models.cache import KVCache, ArraysCache

sys.path.insert(0, os.path.dirname(__file__))
from defer35 import MODEL, segments, to4bit

C, W, R, T = (int(v) for v in (sys.argv[1:5] if len(sys.argv) >= 5 else ("26", "2", "16", "700")))
T0, WARM = 32, int(os.environ.get("NI_WARM", "128"))
OUT = os.path.expanduser("~/NewIdeas/data/ctx35"); os.makedirs(OUT, exist_ok=True)


class S:
    frugal = False
    lru = None            # list of per-layer OrderedDicts for the active stream
    miss = 0; slots = 0


def moe_call(self, x):
    gates = mx.softmax(self.gate(x), axis=-1, precise=True)
    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if self.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    y = self.switch_mlp(x, inds)
    shared = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
    if not S.frugal or x.shape[1] != 1:
        return (y * scores[..., None]).sum(axis=-2) + shared
    sc = np.array(scores[0, 0].astype(mx.float32)); ix = np.array(inds[0, 0])
    cache = S.lru[self._ni_layer]
    keep = np.ones(k, np.float32)
    for r in np.argsort(-sc):
        e = int(ix[r]); rank = int((sc > sc[r]).sum())
        if e in cache:
            cache.move_to_end(e)
        elif rank < W:
            if len(cache) >= C:
                cache.popitem(last=False)
            cache[e] = True
        else:
            keep[r] = 0.0
    S.miss += int((keep == 0).sum()); S.slots += k
    w = scores * mx.array(keep)[None, None, :].astype(scores.dtype)
    return (y * w[..., None]).sum(axis=-2) + shared


def snap(cache):
    return [(c.offset,) if isinstance(c, KVCache) else (list(c.cache),) for c in cache]


def restore(cache, s):
    for c, v in zip(cache, s):
        if isinstance(c, KVCache):
            c.trim(c.offset - v[0])
        else:
            c.cache = list(v[0])


def copy_cache(src, model):
    dst = model.make_cache()
    for d, c in zip(dst, src):
        if isinstance(c, KVCache):
            d.keys, d.values, d.offset = c.keys + 0, c.values + 0, c.offset
        else:
            d.cache = [None if a is None else a + 0 for a in c.cache]
    return dst


def step(model, tok, cache, frugal, lru=None):
    S.frugal, S.lru = frugal, lru
    lg = model(mx.array([[tok]]), cache=cache)[0, -1].astype(mx.float32)
    lp = lg - mx.logsumexp(lg)
    mx.eval(lp)
    return lp


def kld(p_lp, q_lp):
    return float((mx.exp(p_lp) * (p_lp - q_lp)).sum())


def main():
    t0 = time.time()
    model, tok = load(MODEL, lazy=True)
    to4bit(model)
    lm = model.language_model if hasattr(model, "language_model") else model
    for i, l in enumerate(lm.model.layers):
        l.mlp._ni_layer = i
    nl = len(lm.model.layers)
    qwen3_next.Qwen3NextSparseMoeBlock.__call__ = moe_call
    segs = segments(tok)
    res = {k: [] for k in ("current", f"lag{R}", "frugal")}
    agree = {k: [] for k in res}
    miss = {k: [0, 0] for k in res}
    for si, s in enumerate(segs):
        ids = [int(v) for v in np.array(s[0])][:T + 1]
        ce = model.make_cache()
        S.frugal = False
        model(mx.array([ids[:T0]]), cache=ce)
        cf, cl = copy_cache(ce, model), copy_cache(ce, model)
        lru = {k: [OrderedDict() for _ in range(nl)] for k in res}
        for t in range(T0, T):
            if (t - T0) % R == 0:
                cl = copy_cache(ce, model)                     # repair: the lag stream gets the exact context
            sn = snap(ce)
            S.miss = S.slots = 0
            lpc = step(model, ids[t], ce, True, lru["current"]); miss["current"][0] += S.miss; miss["current"][1] += S.slots
            restore(ce, sn)
            lpe = step(model, ids[t], ce, False)
            S.miss = S.slots = 0
            lpl = step(model, ids[t], cl, True, lru[f"lag{R}"]); miss[f"lag{R}"][0] += S.miss; miss[f"lag{R}"][1] += S.slots
            S.miss = S.slots = 0
            lpf = step(model, ids[t], cf, True, lru["frugal"]); miss["frugal"][0] += S.miss; miss["frugal"][1] += S.slots
            if t - T0 >= WARM:
                a = int(mx.argmax(lpe))
                for kname, lq in (("current", lpc), (f"lag{R}", lpl), ("frugal", lpf)):
                    res[kname].append(kld(lpe, lq)); agree[kname].append(int(mx.argmax(lq)) == a)
        print(f"segment {si}: {T - T0} steps x 4 streams, {time.time() - t0:.0f}s", flush=True)
    line = f"C={C} W={W} R={R} T={T}: " + "  ".join(
        f"{k} KLD {np.mean(v):.5f} (P99 {np.percentile(v, 99):.3f}) top-1 {np.mean(agree[k]):.4f} missing slots "
        f"{miss[k][0] / max(1, miss[k][1]):.3f}" for k, v in res.items()) + f"  ({len(res['frugal'])} positions, " \
        f"{time.time() - t0:.0f}s, peak {mx.get_peak_memory() / 1e9:.1f} GB)"
    print(line, flush=True)
    open(os.path.join(OUT, "results.txt"), "a").write(line + "\n")


if __name__ == "__main__":
    main()
    extra = os.path.expanduser("~/NewIdeas/data/after_ctx_arms.txt")   # more 35B arms in the same lock (exec frees RAM)
    if os.path.exists(extra) and T >= 700:
        arms = open(extra).read().split()
        os.rename(extra, extra + ".started")
        os.execv(sys.executable, [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "defer35.py")]
                 + arms)
