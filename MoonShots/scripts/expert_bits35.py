"""How few bits do fine-grained MoE experts need? (memory-halving hunt; NOT bit-exact: quality is the measure)
Qwen3.6-35B-A3B 4-bit (256 experts, top-8), fully resident; ~2K tokens of real text, teacher-forced. Routed experts only are
requantized (dense layers, routers and shared experts stay 4-bit). Per variant: perplexity, mean NLL delta, and greedy top-1
agreement with the 4-bit model.
  b3        : all routed experts RTN 3-bit g64 (from the dequantized 4-bit weights)
  b2 / b2g32: RTN 2-bit g64 / g32
  mixR      : gate-rank-aware precision: each token's top-R experts (by gate) at 4-bit, the rest at 2-bit g32 (errors of
              low-gate experts are scaled by their small gate weight). Memory model: hot experts kept at 4-bit, the rest 2-bit.
usage: expert_bits35.py   (GPU, ~30 GB: exclusive window)"""
import math, os, sys, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

MODEL = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit")
model, tok = load(MODEL)
tm = model.language_model.model
text = ""
for p in ("~/SpecDecode/FINDINGS.md", "~/BigMoE/FINDINGS.md", "~/MoEPrefetch/FINDINGS.md"):
    text += open(os.path.expanduser(p)).read() + "\n\n"
ids = mx.array(tok.encode(text)[:2049])[None]
inp, tgt = ids[:, :-1], ids[:, 1:]


def logits():
    out = model(inp)
    return out[0] if isinstance(out, tuple) else out


def score(lg, ref_top=None):
    lp = lg[0].astype(mx.float32) - mx.logsumexp(lg[0].astype(mx.float32), axis=-1, keepdims=True)
    nll = -mx.take_along_axis(lp, tgt[0][:, None], axis=-1)[:, 0]
    top = mx.argmax(lg[0], axis=-1)
    mx.eval(nll, top)
    agree = float((top == ref_top).astype(mx.float32).mean().item()) if ref_top is not None else 1.0
    return float(nll.mean().item()), agree, top


base_nll, _, base_top = score(logits())
print(f"4-bit reference: ppl {math.exp(base_nll):.4f} over {inp.shape[1]} tokens", flush=True)

sws = [l.mlp.switch_mlp for l in tm.layers]
orig = [{p: (getattr(sw, p).weight, getattr(sw, p).scales, getattr(sw, p).biases, getattr(sw, p).bits, getattr(sw, p).group_size)
         for p in ("gate_proj", "up_proj", "down_proj")} for sw in sws]


def requant(bits, gs):
    out = []
    for sw, o in zip(sws, orig):
        d = {}
        for p, (w, s, b, ob, og) in o.items():
            deq = mx.dequantize(w, s, b, group_size=og, bits=ob)
            d[p] = tuple(mx.quantize(deq, group_size=gs, bits=bits)) + (bits, gs)   # quantize returns a list
        mx.eval([v[:3] for v in d.values()])
        out.append(d)
    return out


def install(qs):
    for sw, d in zip(sws, qs):
        for p, (w, s, b, bits, gs) in d.items():
            m = getattr(sw, p); m.weight, m.scales, m.biases, m.bits, m.group_size = w, s, b, bits, gs


def restore():
    install([{p: v for p, v in o.items()} for o in orig])


for name, bits, gs in (("b3", 3, 64), ("b2", 2, 64), ("b2g32", 2, 32)):
    t0 = time.perf_counter()
    install(requant(bits, gs))
    nll, ag, _ = score(logits(), base_top)
    restore(); mx.clear_cache()
    print(f"{name:6s}: ppl {math.exp(nll):.4f} (dNLL {nll - base_nll:+.4f}), greedy top-1 agreement {ag:.4f}  ({time.perf_counter()-t0:.0f}s)", flush=True)

# gate-rank-aware mix: run each MoE block with two SwitchGLU weight sets, top-R ranks on 4-bit, the rest on 2-bit g32
q2 = requant(2, 32)
MoE = type(tm.layers[0].mlp)
orig_call = MoE.__call__
R = [None]


def mixed_call(self, x):
    gates = self.gate(x)
    gates = mx.softmax(gates, axis=-1, precise=True)
    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if getattr(self, "norm_topk_prob", False):
        scores = scores / scores.sum(axis=-1, keepdims=True)
    order = mx.argsort(-scores, axis=-1)                        # rank by gate
    inds = mx.take_along_axis(inds, order, axis=-1); scores = mx.take_along_axis(scores, order, axis=-1)
    sw = self.switch_mlp
    y = sw(x, inds[..., :R[0]])                                 # MLX is lazy: each call captures the arrays installed now
    if R[0] < k:
        keep = {p: (getattr(sw, p).weight, getattr(sw, p).scales, getattr(sw, p).biases, getattr(sw, p).bits, getattr(sw, p).group_size)
                for p in ("gate_proj", "up_proj", "down_proj")}
        for p, (w, s, b, bits, gs) in q2[self._li].items():
            m = getattr(sw, p); m.weight, m.scales, m.biases, m.bits, m.group_size = w, s, b, bits, gs
        lo = sw(x, inds[..., R[0]:])
        for p, (w, s, b, bits, gs) in keep.items():
            m = getattr(sw, p); m.weight, m.scales, m.biases, m.bits, m.group_size = w, s, b, bits, gs
        y = mx.concatenate([y, lo], axis=-2)
    y = (y * scores[..., None]).sum(axis=-2)
    sh = self.shared_expert(x)
    if hasattr(self, "shared_expert_gate"):
        sh = mx.sigmoid(self.shared_expert_gate(x)) * sh
    return y + sh


for li, l in enumerate(tm.layers):
    object.__setattr__(l.mlp, "_li", li)
MoE.__call__ = mixed_call
for r in (8, 1, 2, 4, 6):                                       # mix8 = all ranks 4-bit: must match the reference
    R[0] = r
    nll, ag, _ = score(logits(), base_top)
    print(f"mix{r}  : top-{r} ranks 4-bit, rest 2-bit g32: ppl {math.exp(nll):.4f} (dNLL {nll - base_nll:+.4f}), agreement {ag:.4f}", flush=True)
MoE.__call__ = orig_call
