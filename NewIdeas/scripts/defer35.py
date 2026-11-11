"""NewIdeas I1 kill test: deferred (late-added) expert contributions, on the Qwen3.6-35B-A3B proxy (2026-10-21).

Idea: a routed expert that is not resident is not waited for and not dropped: its weights are read in the background,
it is computed later on the SAME saved input x_L, and its gate-weighted output is added to the residual stream d layers
later (at the end of layer L + d) instead of at layer L. Layers L+1 .. L+d-1 run without it; everything after has it.
If that costs far less than dropping (frugal), SSD reads stop being latency stalls: a d-layer window hides each read.

Model: unsloth Qwen3.6-35B-A3B-MLX-8bit with the routed experts requantized to affine 4-bit g64 on load (~20 GB peak,
everything else 8-bit); the reference is the SAME model with every expert computed in place, so the numbers measure
only the policy. Missing experts come from a per-layer LRU cache simulation over each teacher-forced segment:
capacity C experts per layer (of 256); ranks < W (by gate score) are always waited (loaded now, inserted); a rank >= W
that is not resident is MISSING. drop: missing ones are skipped and never enter the cache (frugal). deferN: missing
ones are loaded in the background (they enter the cache for later tokens) and added N layers later; contributions
whose target passes the last layer are added at the last layer (= waited at the end of the token). end: added at the
last layer. Arms with "W0" wait for nothing (every miss deferred).
floor: every MoE output x (1 + 2^-7, exact in bf16) (a tiny exact-ish perturbation: the routing-flip floor of this model).

Scoring as Breakthrough's noise35.py: 3 teacher-forced segments (prose, GSM8K, code; SEG tokens each), KLD(ref || arm)
over the reference's top-64 + tail mass, top-1 agreement, also over positions >= 128 (warm cache).
usage: defer35.py ARM [ARM ...]    ARM = exact | floor | drop_C26_W2 | defer2_C26_W2 | end_C26_W2 | ...
"""
import glob, json, os, sys, time
from collections import OrderedDict
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import qwen3_5, qwen3_next

MODEL = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MLX-8bit/snapshots/*/"))[0]
OUT = os.path.expanduser("~/NewIdeas/data/defer35"); os.makedirs(OUT, exist_ok=True)
SEG = int(os.environ.get("NI_SEG", "1365"))
TOPK = 64
WARM = 128


def segments(tok):
    """The three teacher-forced segments, FROZEN: token ids are saved on first use (data/defer35/seg_ids_<SEG>.npz) and
    reused, because the source files (BigMoE FINDINGS, MoonShots scripts) keep changing (batch 3 broke on that)."""
    fz = os.path.join(OUT, f"seg_ids_{SEG}.npz")
    if os.path.exists(fz):
        z = np.load(fz)
        return [mx.array(z[f"s{i}"])[None] for i in range(3)]
    out = _segments_from_files(tok)
    np.savez(fz, **{f"s{i}": np.array(o[0]) for i, o in enumerate(out)})
    return out


def _segments_from_files(tok):                       # same recipe as Breakthrough/scripts/noise35.py
    srcs = [[os.path.expanduser("~/BigMoE/FINDINGS.md")], None,
            sorted(glob.glob(os.path.expanduser("~/MoonShots/scripts/*.py")))]
    gsm = [json.loads(l) for l in open(os.path.expanduser("~/MoEPrefetch/data/gsm8k/test.jsonl"))][200:400]
    out = []
    for k, files in enumerate(srcs):
        text = ("\n\n".join("Question: " + p["question"] + "\nAnswer: " + p["answer"] for p in gsm) if files is None
                else "\n\n".join(open(f).read() for f in files[:40]))
        ids = tok.encode(text)[:SEG + 1]
        assert len(ids) == SEG + 1, (k, len(ids))
        out.append(mx.array(ids)[None])
    return out


def to4bit(model):
    n = 0
    for name, m in model.named_modules():
        if ".switch_mlp." in name and hasattr(m, "scales") and m.bits != 4:
            w = mx.dequantize(m.weight, m.scales, m.biases, group_size=m.group_size, bits=m.bits)
            m.weight, m.scales, m.biases = mx.quantize(w, group_size=64, bits=4)
            m.group_size, m.bits = 64, 4
            mx.eval(m.weight, m.scales, m.biases)
            del w
            n += 1
    mx.eval(model.parameters())
    mx.clear_cache()
    return n


# ---------------------------------------------------------------- policy state (set per arm / per segment)
class P:
    mode = "exact"; C = 26; W = 2; d = 0; delta = 0.0; half = False
    pending_half = {}
    pending = {}           # target layer -> [1, T, D] contribution to add at the end of that layer
    stats = None           # per layer: [slot uses, missing uses, gate mass total, missing mass]
    nlayers = 40


def parse(arm):
    P.mode, P.d, P.half = arm, 0, False
    if arm in ("exact", "floor"):
        return
    parts = arm.split("_")
    head = parts[0]
    P.C = int(parts[1][1:]); P.W = int(parts[2][1:])
    if head.startswith("bias"):                      # cache-aware near-tie routing: +delta on resident experts' logits
        P.mode, P.delta = "bias", int(head[4:]) / 100
    elif head == "drop":
        P.mode = "drop"
    elif head == "end":
        P.mode, P.d = "defer", 10 ** 6
    elif head.startswith("defer"):
        P.mode, P.d = "defer", int(head[5:])
    elif head == "half":                             # late by half a layer: added after layer L+1's attention
        P.mode, P.d, P.half = "defer", 1, True
    else:
        raise ValueError(arm)


def lru_missing(order, C, W, insert_missing):
    """order: [T, k] expert ids sorted by score (desc). Returns bool [T, k] missing mask. C > k, so the LRU victim is
    never one of the current token's experts (they were all touched last)."""
    T, k = order.shape
    miss = np.zeros((T, k), bool)
    if C == 0:                                       # nothing resident: every rank >= W is missing (W ranks waited)
        miss[:, W:] = True
        return miss
    cache = OrderedDict()
    for t in range(T):
        row = order[t]
        for r in range(k):
            e = int(row[r])
            if e in cache:
                cache.move_to_end(e)
                continue
            if r >= W:
                miss[t, r] = True
                if not insert_missing:
                    continue
            if len(cache) >= C:
                cache.popitem(last=False)
            cache[e] = True
    return miss


def bias_call(self, x):
    """Router logits + delta for experts resident in this layer's LRU (sequential over tokens); select the top-k of the
    biased logits, weight them by their ORIGINAL softmax probabilities (renormalised over the selection as the model
    does); ranks < W (by original logit) are loaded if cold, other cold ones are dropped (W >= k: exact streaming)."""
    L = self._ni_layer
    lgx = self.gate(x)
    probs = mx.softmax(lgx, axis=-1, precise=True)
    lg = np.array(lgx[0].astype(mx.float32)); pr = np.array(probs[0].astype(mx.float32))
    T, E = lg.shape; k = self.top_k
    cache = OrderedDict()
    sel = np.zeros((T, k), np.int64); miss = np.zeros((T, k), bool); loads = 0; changed = 0
    for t in range(T):
        b = lg[t] + 0.0
        if cache and P.delta:
            b[np.fromiter(cache.keys(), np.int64)] += P.delta
        top = np.argpartition(-b, k)[:k]
        top = top[np.argsort(-lg[t][top])]
        changed += k - len(np.intersect1d(top, np.argpartition(-lg[t], k)[:k]))
        for r, e in enumerate(top):
            e = int(e)
            if e in cache:
                cache.move_to_end(e)
                continue
            if r < P.W:
                loads += 1
                if len(cache) >= P.C:
                    cache.popitem(last=False)
                cache[e] = True
            else:
                miss[t, r] = True
        sel[t] = top
    inds = mx.array(sel[None].astype(np.uint32))
    scores = mx.take_along_axis(probs, inds, axis=-1)
    if self.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    y = self.switch_mlp(x, inds)
    shared = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
    sc = np.array(scores[0].astype(mx.float32))
    st = P.stats.setdefault(L, np.zeros(6))
    st += [miss.size, miss.sum(), sc.sum(), (sc * miss).sum(), loads, changed]
    w = scores * mx.array((~miss)[None].astype(np.float32)).astype(scores.dtype)
    return (y * w[..., None]).sum(axis=-2) + shared


def moe_call(self, x):
    if P.mode == "bias":
        return bias_call(self, x)
    gates = mx.softmax(self.gate(x), axis=-1, precise=True)
    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if self.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    y = self.switch_mlp(x, inds)                                     # [B, T, k, D]
    shared = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
    L = self._ni_layer
    if P.mode == "exact":
        return (y * scores[..., None]).sum(axis=-2) + shared
    if P.mode == "floor":
        return ((y * scores[..., None]).sum(axis=-2) + shared) * (1.0 + 2.0 ** -7)
    sc = np.array(scores[0].astype(mx.float32)); ix = np.array(inds[0])
    srt = np.argsort(-sc, axis=-1)
    order = np.take_along_axis(ix, srt, axis=-1)
    miss_sorted = lru_missing(order, P.C, P.W, insert_missing=(P.mode == "defer"))
    miss = np.zeros_like(miss_sorted)
    np.put_along_axis(miss, srt, miss_sorted, axis=-1)               # back to argpartition order
    st = P.stats.setdefault(L, np.zeros(4))
    st += [miss.size, miss.sum(), sc.sum(), (sc * miss).sum()]
    m = mx.array(miss[None].astype(np.float32)).astype(scores.dtype)
    wres, wmis = scores * (1 - m), scores * m
    res = (y * wres[..., None]).sum(axis=-2) + shared
    if P.mode == "defer":
        late = (y * wmis[..., None]).sum(axis=-2)
        tgt = min(L + P.d, P.nlayers - 1)
        box = P.pending_half if (P.half and L + 1 < P.nlayers) else P.pending
        box[tgt] = late if tgt not in box else box[tgt] + late
    return res


def layer_call(self, x, mask=None, cache=None):
    if self.is_linear:
        r = self.linear_attn(self.input_layernorm(x), mask, cache)
    else:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
    h = x + r
    addh = P.pending_half.pop(self.mlp._ni_layer, None)
    if addh is not None:
        h = h + addh
    out = h + self.mlp(self.post_attention_layernorm(h))
    add = P.pending.pop(self.mlp._ni_layer, None)
    if add is not None:
        out = out + add
    return out


def logprobs(model, seg):
    P.pending, P.pending_half = {}, {}
    out = model(seg[:, :-1])
    lg = (out[0] if isinstance(out, tuple) else out)[0].astype(mx.float32)
    assert not P.pending and not P.pending_half, "undelivered deferred contributions"
    return lg - mx.logsumexp(lg, axis=-1, keepdims=True)


def main():
    arms = sys.argv[1:]
    extra = os.path.join(OUT, "extra_arms.txt")         # arms added after the runner was queued (main run only)
    if SEG == 1365 and os.path.exists(extra):
        arms += [a for a in open(extra).read().split() if a not in arms]
    t0 = time.time()
    model, tok = load(MODEL, lazy=True)
    n4 = to4bit(model)
    layers = model.language_model.model.layers
    P.nlayers = len(layers)
    for i, l in enumerate(layers):
        l.mlp._ni_layer = i
    qwen3_next.Qwen3NextSparseMoeBlock.__call__ = moe_call
    qwen3_5.DecoderLayer.__call__ = layer_call
    segs = segments(tok)
    print(f"load + {n4} expert tensors -> 4-bit: {time.time() - t0:.0f}s, peak {mx.get_peak_memory() / 1e9:.1f} GB",
          flush=True)
    refp = os.path.join(OUT, f"ref_{SEG}.npz")
    if not os.path.exists(refp):
        parse("exact")
        d = {}
        for i, s in enumerate(segs):
            lp = logprobs(model, s)
            top = mx.argpartition(-lp, TOPK - 1, axis=-1)[:, :TOPK]
            tlp = mx.take_along_axis(lp, top, axis=-1)
            a1 = mx.argmax(lp, axis=-1)
            mx.eval(top, tlp, a1)
            d[f"top{i}"], d[f"tlp{i}"], d[f"a1{i}"] = np.array(top), np.array(tlp), np.array(a1)
        np.savez(refp, **d)
        print(f"ref written ({time.time() - t0:.0f}s)", flush=True)
    ref = np.load(refp)
    for arm in arms:
        if arm == "exact":
            continue
        t1 = time.time()
        parse(arm)
        P.stats = {}
        kl, agree, seg_kl = [], [], []
        for i, s in enumerate(segs):
            lp = logprobs(model, s)
            top, p_lp = mx.array(ref[f"top{i}"]), mx.array(ref[f"tlp{i}"])
            q_lp = mx.take_along_axis(lp, top, axis=-1)
            p, q = mx.exp(p_lp), mx.exp(q_lp)
            pt, qt = mx.maximum(1 - p.sum(-1), 1e-12), mx.maximum(1 - q.sum(-1), 1e-12)
            k = (p * (p_lp - q_lp)).sum(-1) + pt * (mx.log(pt) - mx.log(qt))
            a1 = mx.argmax(lp, axis=-1)
            mx.eval(k, a1)
            kl.append(np.array(k)); agree.append(np.array(a1) == ref[f"a1{i}"])
            seg_kl.append(float(kl[-1].mean()))
        klw = np.concatenate([x[WARM:] for x in kl]); agw = np.concatenate([x[WARM:] for x in agree])
        kla, aga = np.concatenate(kl), np.concatenate(agree)
        if P.stats:
            tot = sum(P.stats.values())
            ms = f"missing slots {tot[1] / tot[0]:.3f} mass {tot[3] / tot[2]:.3f}"
            if len(tot) == 6:                          # bias arms: cold loads and re-routed slots per token (40 layers)
                ntok = tot[0] / 8 / P.nlayers
                ms += f" loads/token {tot[4] / ntok:.1f} rerouted slots/token {tot[5] / ntok:.1f}"
        else:
            ms = "missing slots 0 mass 0"
        line = (f"seg{SEG} {arm:16s} KLD {kla.mean():.5f} (warm {klw.mean():.5f}, P99 {np.percentile(kla, 99):.4f}; "
                f"prose/gsm/code {seg_kl[0]:.4f}/{seg_kl[1]:.4f}/{seg_kl[2]:.4f})  top-1 {aga.mean():.4f} "
                f"(warm {agw.mean():.4f})  {ms}  {time.time() - t1:.0f}s peak {mx.get_peak_memory() / 1e9:.1f} GB")
        print(line, flush=True)
        open(os.path.join(OUT, "results.txt"), "a").write(line + "\n")


if __name__ == "__main__":
    main()
