"""NewIdeas: one policy language for the 35B frontier (reads/token vs KLD), 2026-10-21. Builds on defer35.py (same model,
segments, reference, late-add plumbing). An arm is key-value parts joined by "_":
  C26   LRU cache slots per layer (of 256)          b25   +0.25 on resident experts' router logits (near-tie re-route)
  w2    ranks always on time (loaded if cold)        m85   also on time: ranks covering 85 % of the top-8 mass
  lhalf / l1 / l2 / ldrop   what happens to the other cold ranks: added before layer L+1's router (half a layer late),
        1 or 2 layers late (all loaded in the background and inserted), or dropped (not read, not inserted)
  x70   late only for cold ranks inside the first 70 % of the mass; colder ones beyond it are dropped (default x100)
Ranks are by the ORIGINAL router score among the selected experts; weights are the original softmax probabilities,
renormalised over the selection (the model's norm_topk_prob); no renormalisation for skipped ones.
Reports: KLD vs the exact reference, top-1, cold reads per token (on-time + late), late and dropped slots and mass.
usage: pol35.py ARM [ARM ...]    -> ~/NewIdeas/data/defer35/results.txt (seg-tagged lines starting with 'pol')
"""
import os, sys, time
from collections import OrderedDict
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import qwen3_5, qwen3_next

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import defer35 as D

Q = {}


def parse(arm):
    q = {"C": 26, "b": 0.0, "w": 2, "m": 0.0, "l": "drop", "x": 1.0, "B": (0, 10 ** 6), "k": 99, "n": 0, "r": 99}
    for p in arm.split("_"):
        if p[0] == "C": q["C"] = int(p[1:])
        elif p[0] == "b": q["b"] = int(p[1:]) / 100
        elif p[0] == "w": q["w"] = int(p[1:])
        elif p[0] == "m": q["m"] = int(p[1:]) / 100
        elif p[0] == "x": q["x"] = int(p[1:]) / 100
        elif p[0] == "l": q["l"] = p[1:]
        elif p[0] == "B": q["B"] = tuple(int(v) for v in p[1:].split("-"))   # policy only in layers a..b-1
        elif p[0] == "k": q["k"] = int(p[1:])          # static: only the top-k by score are ever computed
        elif p[0] == "n": q["n"] = int(p[1:])          # weights = p / sum of the top-n probabilities (n >= 8)
        elif p[0] == "r": q["r"] = int(p[1:])          # late only for ranks < r (colder ranks beyond are dropped)
        else: raise ValueError(arm)
    return q


def moe_call(self, x):
    q, L, k = Q["q"], self._ni_layer, self.top_k
    if not (q["B"][0] <= L < q["B"][1]):                  # outside the band: exact
        g = mx.softmax(self.gate(x), axis=-1, precise=True)
        ii = mx.argpartition(g, kth=-k, axis=-1)[..., -k:]
        ss = mx.take_along_axis(g, ii, axis=-1)
        ss = ss / ss.sum(axis=-1, keepdims=True) if self.norm_topk_prob else ss
        return (self.switch_mlp(x, ii) * ss[..., None]).sum(axis=-2) + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
    lgx = self.gate(x)
    probs = mx.softmax(lgx, axis=-1, precise=True)
    lg = np.array(lgx[0].astype(mx.float32))
    T = lg.shape[0]
    cache = OrderedDict()
    sel = np.zeros((T, k), np.int64); cat = np.zeros((T, k), np.int8)     # 0 on time, 1 late, 2 dropped
    loads = 0
    for t in range(T):
        b = lg[t] + 0.0
        if q["b"] and cache:
            b[np.fromiter(cache.keys(), np.int64)] += q["b"]
            b[np.argpartition(-lg[t], 2)[:2]] = 1e9                  # never displace the router's ranks 0-1
        top = np.argpartition(-b, k)[:k]
        top = top[np.argsort(-lg[t][top])]
        p = np.exp(lg[t][top] - lg[t][top].max()); p /= p.sum(); cp = np.cumsum(p)
        rw = max(q["w"], int(np.searchsorted(cp, q["m"] - 1e-9)) + 1 if q["m"] else 0)
        for r, e in enumerate(top):
            e = int(e)
            if r >= q["k"]:
                cat[t, r] = 2; continue
            if e in cache:
                cache.move_to_end(e); continue
            if r < rw:
                c = 0
            elif q["l"] != "drop" and cp[r] - p[r] < q["x"] - 1e-9 and r < q["r"]:
                c = 1
            else:
                c = 2
            cat[t, r] = c
            if c < 2:
                loads += 1
                if q["C"] > 0:
                    if len(cache) >= q["C"]:
                        cache.popitem(last=False)
                    cache[e] = True
        sel[t] = top
    inds = mx.array(sel[None].astype(np.uint32))
    scores = mx.take_along_axis(probs, inds, axis=-1)
    if q["n"]:
        topn = mx.sort(probs, axis=-1)[..., -q["n"]:].sum(axis=-1, keepdims=True)
        scores = scores / topn
    elif self.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    y = self.switch_mlp(x, inds)
    shared = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
    sc = np.array(scores[0].astype(mx.float32))
    st = Q["stats"].setdefault(L, np.zeros(11))
    nl_ = (cat == 1).sum(1); nw_ = ((cat == 0) & (sel >= 0)).sum(1)
    st += [cat.size, loads, (cat == 1).sum(), (cat == 2).sum(), (sc * (cat == 1)).sum(), (sc * (cat == 2)).sum(),
           (nl_ == 0).sum(), (nl_ == 1).sum(), (nl_ == 2).sum(), (nl_ >= 3).sum(), max(0, loads - (cat == 1).sum())]
    on = mx.array((cat[None] == 0).astype(np.float32)).astype(scores.dtype)
    out = (y * (scores * on)[..., None]).sum(axis=-2) + shared
    if (cat == 1).any():
        late = mx.array((cat[None] == 1).astype(np.float32)).astype(scores.dtype)
        lo = (y * (scores * late)[..., None]).sum(axis=-2)
        d = {"half": 1}.get(q["l"]) or int(q["l"])
        tgt = min(L + d, D.P.nlayers - 1)
        box = D.P.pending_half if (q["l"] == "half" and L + 1 < D.P.nlayers) else D.P.pending
        box[tgt] = lo if tgt not in box else box[tgt] + lo
    return out


def main():
    arms = sys.argv[1:]
    t0 = time.time()
    model, tok = load(D.MODEL, lazy=True)
    D.to4bit(model)
    layers = model.language_model.model.layers
    D.P.nlayers = len(layers)
    for i, l in enumerate(layers):
        l.mlp._ni_layer = i
    qwen3_next.Qwen3NextSparseMoeBlock.__call__ = moe_call
    qwen3_5.DecoderLayer.__call__ = D.layer_call
    segs = D.segments(tok)
    refp = os.path.join(D.OUT, f"ref_{D.SEG}.npz")
    if not os.path.exists(refp):                       # exact reference with defer35's exact path
        qwen3_next.Qwen3NextSparseMoeBlock.__call__ = D.moe_call
        D.parse("exact")
        d = {}
        for i, s in enumerate(segs):
            lp = D.logprobs(model, s)
            top = mx.argpartition(-lp, D.TOPK - 1, axis=-1)[:, :D.TOPK]
            tlp = mx.take_along_axis(lp, top, axis=-1); a1 = mx.argmax(lp, axis=-1)
            mx.eval(top, tlp, a1)
            d[f"top{i}"], d[f"tlp{i}"], d[f"a1{i}"] = np.array(top), np.array(tlp), np.array(a1)
        np.savez(refp, **d)
        qwen3_next.Qwen3NextSparseMoeBlock.__call__ = moe_call
    ref = np.load(refp)
    print(f"load {time.time() - t0:.0f}s", flush=True)
    for arm in arms:
        t1 = time.time()
        Q["q"], Q["stats"] = parse(arm), {}
        kl, agree = [], []
        for i, s in enumerate(segs):
            lp = D.logprobs(model, s)
            top, p_lp = mx.array(ref[f"top{i}"]), mx.array(ref[f"tlp{i}"])
            q_lp = mx.take_along_axis(lp, top, axis=-1)
            p, qq = mx.exp(p_lp), mx.exp(q_lp)
            pt, qt = mx.maximum(1 - p.sum(-1), 1e-12), mx.maximum(1 - qq.sum(-1), 1e-12)
            kk = (p * (p_lp - q_lp)).sum(-1) + pt * (mx.log(pt) - mx.log(qt))
            a1 = mx.argmax(lp, axis=-1)
            mx.eval(kk, a1)
            kl.append(np.array(kk)); agree.append(np.array(a1) == ref[f"a1{i}"])
        kla, aga = np.concatenate(kl), np.concatenate(agree)
        tot = sum(Q["stats"].values()); ntok = tot[0] / 8 / D.P.nlayers
        ps = "/".join(f"{x.mean():.4f}" for x in kl)
        line = (f"seg{D.SEG} pol {arm:22s} KLD {kla.mean():.5f} (prose/gsm/code {ps}; P99 {np.percentile(kla, 99):.4f})  top-1 {aga.mean():.4f}"
                f"  reads/token {tot[1] / ntok:6.1f}  late {tot[2] / tot[0]:.3f} (mass {tot[4] / ntok / D.P.nlayers:.3f})"
                f"  dropped {tot[3] / tot[0]:.3f} (mass {tot[5] / ntok / D.P.nlayers:.3f})  waited cold/token {tot[10] / ntok:.1f}"
                f"  late per token-layer 0/1/2/3+: {tot[6] / (tot[0] / 8):.3f}/{tot[7] / (tot[0] / 8):.3f}/{tot[8] / (tot[0] / 8):.3f}/{tot[9] / (tot[0] / 8):.3f}"
                f"  {time.time() - t1:.0f}s")
        print(line, flush=True)
        open(os.path.join(D.OUT, "results.txt"), "a").write(line + "\n")


if __name__ == "__main__":
    main()
    extra = os.path.expanduser("~/NewIdeas/data/after_pol_arms.txt")   # more arms in the same lock (exec frees RAM)
    if os.path.exists(extra) and D.SEG == 1365:
        arms = open(extra).read().split()
        os.rename(extra, extra + ".started")
        os.execv(sys.executable, [sys.executable, os.path.abspath(__file__)] + arms)
    nxt = os.path.expanduser("~/NewIdeas/data/after_pol_exec.txt")   # then one more script in the same lock
    if os.path.exists(nxt) and D.SEG == 1365:
        cmd = open(nxt).read().split()
        os.rename(nxt, nxt + ".started")
        os.execv(sys.executable, [sys.executable] + cmd)
