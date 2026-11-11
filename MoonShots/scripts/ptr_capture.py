"""D-06 PTR capture (MoonShots Wed 16:0x, for NewIdeas' drafter track D98): where do the 125B's QSA heads point?
Teacher-forced on Chat1000's agent125 turns (prompt + the 125B's own 160-token greedy continuation): chunked prefill of the
prompt up to X[L-2], then ONE 161-row call on X[L-2 : L+159] (row s has input X[L-2+s] and predicts X[L-1+s]) with a hook on
every QSA Attention (12 layers): q of those rows recomputed from the layer input (q_proj, q_norm, standard rope; the cache
holds the roped keys), DENSE causal softmax over all keys (the real layer intersects it with the indexer's 2,048-token
selection), per row: top-16 positions by the max over the 24 heads, and each head's top-4. Plus the model's own softmax
max / argmax per row. Output data/ptr/<turn>.npz (analysis: ptr_analyze.py, CPU).
usage: [BigMoE venv] ptr_capture.py FRAC turn ...   (inside a quiet lock; GPU)"""
import os, re, sys, time
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import numpy as np
import mlx.core as mx
from bigmoe.loader import load_streamed
from bigmoe.fuse import reset_generation_state
from mtplx.models import qwen4_exp as Q

MD = os.path.expanduser("~/BigMoE/models/bare")
OUT = os.path.expanduser("~/MoonShots/data/ptr"); os.makedirs(OUT, exist_ok=True)
frac, TURNS = float(sys.argv[1]), sys.argv[2:]
KTOP, HTOP, SINK = 16, 4, 4


class CAP:
    on = False
    lay = {}
    store = {}


_orig = Q.Attention.__call__


def _hook(self, x, cache):
    if not CAP.on:
        return _orig(self, x, cache)
    pos_start = int(cache.offset)
    fixed = bool(getattr(cache, "fixed_capacity", False))
    rope_start = cache.rope_offset if fixed else pos_start
    out = _orig(self, x, cache)
    assert not fixed, "fixed-capacity cache: capture not supported"
    B, S, _ = x.shape
    fused = getattr(self, "qkv_fused", None)
    q = fused(x)[0] if fused is not None else self.q_proj(x)
    q, _ = mx.split(q.reshape(B, S, self.n_heads, -1), 2, axis=-1)
    q = self.q_norm(q)
    cos, sin = Q._rope_cos_sin(rope_start + mx.arange(S, dtype=mx.int32), self._inv_freq, self._rope_attention_scaling)
    q = Q._apply_partial_rope(q, cos, sin).transpose(0, 2, 1, 3).astype(mx.float32)     # B, H, S, d
    T = pos_start + S
    k = cache.kv.keys[:, :, :T, :].astype(mx.float32)                                    # B, KV, T, d
    g = self.n_heads // self.n_kv_heads
    sc = (q.reshape(B, self.n_kv_heads, g, S, -1) @ k[:, :, None].transpose(0, 1, 2, 4, 3)) * self.scale
    sc = sc.reshape(self.n_heads, S, T)
    causal = mx.arange(T)[None, :] <= (pos_start + mx.arange(S))[:, None]
    p = mx.softmax(mx.where(causal[None], sc, -mx.inf), axis=-1)                          # H, S, T
    pm = p.max(0)
    ti = mx.argsort(-pm, axis=-1)[:, :KTOP]
    hi = mx.argsort(-p, axis=-1)[:, :, :HTOP]
    mx.eval(ti, hi)
    L = CAP.lay[id(self)]
    CAP.store[L] = (np.array(ti, np.int32), np.array(mx.take_along_axis(pm, ti, -1), np.float32),
                    np.array(hi, np.int32), np.array(mx.take_along_axis(p, hi, -1), np.float32))
    return out


Q.Attention.__call__ = _hook

t0 = time.time()
C = max(16, round(frac * 512))
rt, st = load_streamed(MD, C, mtp=False)
for name, m in rt.model.named_modules():
    if isinstance(m, Q.Attention):
        CAP.lay[id(m)] = int(re.search(r"layers\.(\d+)\.", name).group(1))
print(f"loaded {time.time() - t0:.0f}s (C={C}), QSA layers {sorted(CAP.lay.values())}", flush=True)
for name in TURNS:
    t1 = time.time()
    z = np.load(os.path.expanduser(f"~/Chat1000/data/agent125/{name}.npz"))
    prompt, out = [int(t) for t in z["prompt"]], [int(t) for t in z["out"]]
    X = prompt + out
    L = len(prompt)
    reset_generation_state(rt.model)
    cache = rt.model.make_cache()
    ids = mx.array([X[:L - 2]])
    for i in range(0, ids.shape[1], 2048):
        rt.model(ids[:, i:i + 2048], cache=cache, emit_logits=False)
        mx.eval([c.state for c in cache if hasattr(c, "state")])
    t2 = time.time()
    CAP.on, CAP.store = True, {}
    lg = rt.model(mx.array([X[L - 2:L + 159]]), cache=cache)
    lg = (lg[0] if isinstance(lg, tuple) else lg)[0].astype(mx.float32)
    pr = mx.softmax(lg, axis=-1)
    pmax, amax = pr.max(-1), pr.argmax(-1)
    mx.eval(pmax, amax)
    CAP.on = False
    st.gate.check()
    lays = sorted(CAP.store)
    am = np.array(amax)
    np.savez_compressed(os.path.join(OUT, f"{name}.npz"), X=np.array(X), L=L, layers=np.array(lays),
                        top=np.stack([CAP.store[l][0] for l in lays]), topv=np.stack([CAP.store[l][1] for l in lays]),
                        htop=np.stack([CAP.store[l][2] for l in lays]), htopv=np.stack([CAP.store[l][3] for l in lays]),
                        pmax=np.array(pmax), amax=am, mtp=z["mtp"])
    agree = float(np.mean(am[1:161] == np.array(out)))
    print(f"{name}: prompt {L}, prefill {t2 - t1:.0f}s, capture {time.time() - t2:.1f}s, layers {len(lays)}, "
          f"argmax == recorded greedy on {100 * agree:.1f} % of the 160 tokens", flush=True)
print(f"done in {time.time() - t0:.0f}s, peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
