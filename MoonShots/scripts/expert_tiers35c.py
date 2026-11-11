"""Round 19 PoC C (35B quality, NOT bit-exact): gate-mass-adaptive bit plans vs fixed rank plans. Per token, the gate
mass before a rank decides its tier: exact 4-bit while mass-before < t4, MSB-3 while < t3, else MSB-2 (peaked gates spend
fewer bits, flat ones more). Compared with fixed rank plans at the same average expert bytes. Tiers are masked on the fly
from the 4-bit codes (as expert_tiers35b.py). Relative bytes: 4.5 / 3.5 / 2.5 per rank-use.
usage: expert_tiers35c.py   (GPU, ~25 GB: exclusive window)"""
import math, os, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

MODEL = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit")
model, tok = load(MODEL)
tm = model.language_model.model
text = ""
for p in ("~/SpecDecode/FINDINGS.md", "~/BigMoE/FINDINGS.md", "~/MoEPrefetch/FINDINGS.md"):
    text += open(os.path.expanduser(p)).read() + "\n\n"
ids = mx.array(tok.encode(text)[:2049])[None]
inp, tgt = ids[:, :-1], ids[:, 1:]
P3 = ("gate_proj", "up_proj", "down_proj")


def score(ref_top=None):
    out = model(inp)
    lg = (out[0] if isinstance(out, tuple) else out)[0].astype(mx.float32)
    nll = -mx.take_along_axis(lg - mx.logsumexp(lg, axis=-1, keepdims=True), tgt[0][:, None], axis=-1)[:, 0]
    top = mx.argmax(lg, axis=-1)
    mx.eval(nll, top)
    agree = float((top == ref_top).astype(mx.float32).mean().item()) if ref_top is not None else 1.0
    return float(nll.mean().item()), agree, top


base_nll, _, base_top = score()
print(f"4-bit reference: ppl {math.exp(base_nll):.4f} over {inp.shape[1]} tokens", flush=True)
sws = [l.mlp.switch_mlp for l in tm.layers]
orig = [{p: (getattr(sw, p).weight, getattr(sw, p).scales, getattr(sw, p).biases) for p in P3} for sw in sws]
assert all(getattr(sw, p).bits == 4 and getattr(sw, p).group_size == 64 for sw in sws for p in P3)
KEEP = {3: mx.array(0xEEEEEEEE, dtype=mx.uint32), 2: mx.array(0xCCCCCCCC, dtype=mx.uint32)}


def tier(li, p, n):
    w, s, b = orig[li][p]
    if n == 4:
        return w, s, b
    low = (1 << (4 - n)) - 1
    lsum = None
    for j in range(8):
        part = (w >> (4 * j)) & low
        lsum = part if lsum is None else lsum + part
    off = lsum.reshape(*s.shape, 8).sum(-1).astype(mx.float32) / 64
    return w & KEEP[n], s, (b.astype(mx.float32) + s.astype(mx.float32) * off).astype(s.dtype)


def glu(sw, x, idx, li, n):
    x, sidx, inv = _gather_sort(mx.expand_dims(x, (-2, -3)), idx)

    def proj(p, h):
        w, s, b = tier(li, p, n)
        return mx.gather_qmm(h, w, s, b, rhs_indices=sidx, transpose=True, group_size=64, bits=4, sorted_indices=True)

    h = sw.activation(proj("up_proj", x), proj("gate_proj", x))
    return _scatter_unsort(proj("down_proj", h), inv, idx.shape).squeeze(-2)


CUR = {"bytes": [], "plan": None, "t": None}
BYTES = {4: 4.5, 3: 3.5, 2: 2.5}


def moe_call(self, x):
    gates = mx.softmax(self.gate(x), axis=-1, precise=True)
    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if self.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    order = mx.argsort(-scores, axis=-1)
    inds, scores = mx.take_along_axis(inds, order, axis=-1), mx.take_along_axis(scores, order, axis=-1)
    if CUR["plan"] is not None:                                  # fixed bits per rank
        bits = mx.broadcast_to(mx.array(CUR["plan"]), scores.shape)
    else:                                                         # gate-mass-adaptive
        t4, t3 = CUR["t"]
        frac = scores / scores.sum(axis=-1, keepdims=True)
        before = mx.cumsum(frac, axis=-1) - frac
        bits = mx.where(before < t4, 4, mx.where(before < t3, 3, 2))
    CUR["bytes"].append(mx.where(bits == 4, 4.5, mx.where(bits == 3, 3.5, 2.5)).mean())
    y = None
    for n in (4, 3, 2):
        Y = mx.where((bits == n)[..., None], glu(self.switch_mlp, x, inds, self._li, n), 0)
        y = Y if y is None else y + Y
    y = (y * scores[..., None]).sum(axis=-2)
    return y + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)


for li, l in enumerate(tm.layers):
    object.__setattr__(l.mlp, "_li", li)
type(tm.layers[0].mlp).__call__ = moe_call


def run(name, plan=None, t=None):
    CUR.update(plan=plan, t=t, bytes=[])
    t0 = time.perf_counter()
    nll, ag, _ = score(base_top)
    rel = float(mx.stack(CUR["bytes"]).mean().item()) / 4.5
    print(f"{name:24s}: ppl {math.exp(nll):.4f} ({100 * (math.exp(nll - base_nll) - 1):+.2f} %, dNLL {nll - base_nll:+.4f}), "
          f"agreement {ag:.4f}, expert bytes {100 * (rel - 1):+.1f} % ({time.perf_counter() - t0:.0f}s)", flush=True)


run("fixed 44444444 (check)", plan=(4,) * 8)
for plan in ((4, 4, 3, 3, 2, 2, 2, 2), (4, 3, 3, 3, 2, 2, 2, 2), (4, 4, 4, 3, 3, 2, 2, 2), (4, 4, 4, 4, 2, 2, 2, 2)):
    run("fixed " + "".join(map(str, plan)), plan=plan)
for t in ((0.25, 0.6), (0.3, 0.7), (0.35, 0.75), (0.4, 0.8), (0.5, 0.85), (0.3, 0.85), (0.5, 0.7), (0.2, 0.5)):
    run(f"adaptive t4={t[0]} t3={t[1]}", t=t)
