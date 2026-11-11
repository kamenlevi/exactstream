"""Model-level test of the activation-aware refit of truncated expert tiers (IDEAS 33c) on the 35B proxy (Qwen3.6-35B-A3B,
lmstudio 4-bit g64, 40 layers, 256 experts, top-8). NOT bit-exact; quality study.
1. Calibration: the 4-bit reference runs on texts disjoint from the eval (gsm8k test 0-399 Q+A, scripts/*.py, MoEPrefetch +
   SpecDecode findings); every layer's MoE input x and score-sorted top-8 ids are kept.
2. Refit, per layer / expert / projection / tier n in {3, 2}: codes stay m = q >> (4 - n) (nested planes); the 2 G group
   scales and biases of each output row are a ridge least-squares fit of the row's outputs on the expert's routed calibration
   inputs (gate/up: x; down: h = silu(gate x) * up x from the 4-bit weights), ridge toward the stored truncation; experts with
   too few routed rows keep the truncation. Applied as gather_qmm(bits=4) on the masked codes with scale alpha / 2^(4-n).
3. Eval: prefill KL(4-bit ref || plan) per token, per corpus part (gsm solutions 1000-1149, MoonShots patches code, NewIdeas
   docs), plus dNLL and top-1 agreement, for 44333333 / 44222222 / 42222222 / 43222222, truncated vs refit.
usage: refit35.py   (GPU, ~25 GB, quiet lock; ~10 min)"""
import glob, json, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refit_lib import fit_proj, codes, solve

T0 = time.time()
MODEL = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit")
model, tok = load(MODEL)
tm = model.language_model.model
NL = len(tm.layers)
P3 = ("gate_proj", "up_proj", "down_proj")
SEG = 2048
rd = lambda p: open(os.path.expanduser(p)).read()
gsm = [json.loads(l) for l in open(os.path.expanduser("~/MoEPrefetch/data/gsm8k/test.jsonl"))]
qa = lambda a, b: "\n\n".join("Question: " + p["question"] + "\nAnswer: " + p["answer"] for p in gsm[a:b])
cat = lambda fs: "\n\n".join(rd(f) for f in fs)
CAL = [qa(0, 400), cat(sorted(glob.glob(os.path.expanduser("~/MoonShots/scripts/*.py")))),
       rd("~/MoEPrefetch/FINDINGS.md") + "\n\n" + rd("~/SpecDecode/FINDINGS.md")]
EVAL = {"gsm": qa(1000, 1150), "code": cat(sorted(glob.glob(os.path.expanduser("~/MoonShots/patches/*/*.py")))),
        "general": rd("~/NewIdeas/GAPS.md") + "\n\n" + rd("~/NewIdeas/IDEAS.md") + "\n\n" + rd("~/NewIdeas/CHARTER.md")}
segs = lambda text, n: [mx.array(tok.encode(text)[i * SEG:(i + 1) * SEG + 1])[None] for i in range(n)]
cal_ids = [s for t in CAL for s in segs(t, 2)]                       # 6 segments x 2048 = 12,288 calibration tokens
eval_ids = {k: segs(t, 1)[0] for k, t in EVAL.items()}
sws = [l.mlp.switch_mlp for l in tm.layers]
orig = [{p: (getattr(sw, p).weight, getattr(sw, p).scales, getattr(sw, p).biases) for p in P3} for sw in sws]
assert all(getattr(sw, p).bits == 4 and getattr(sw, p).group_size == 64 for sw in sws for p in P3)
KEEP = {3: mx.array(0xEEEEEEEE, dtype=mx.uint32), 2: mx.array(0xCCCCCCCC, dtype=mx.uint32)}
TIERS = {}                                                           # (li, p, n) -> refit (scale, bias) [E, out, G]


def trunc_sb(li, p, n):
    w, s, b = orig[li][p]
    low = (1 << (4 - n)) - 1
    lsum = None
    for j in range(8):
        part = (w >> (4 * j)) & low
        lsum = part if lsum is None else lsum + part
    off = lsum.reshape(*s.shape, 8).sum(-1).astype(mx.float32) / 64
    return s, (b.astype(mx.float32) + s.astype(mx.float32) * off).astype(s.dtype)


def tier(li, p, n, mode):
    """(codes, scale, bias) for gather_qmm(bits=4): tier n keeps the top n bits of every code"""
    w, s, b = orig[li][p]
    if n == 4:
        return w, s, b
    s2, b2 = TIERS[(li, p, n)] if mode == "refit" else trunc_sb(li, p, n)
    return w & KEEP[n], s2, b2


CUR = {"plan": (4,) * 8, "mode": "trunc", "cap": None}


def glu(sw, x, idx, li, n, mode):
    x, sidx, inv = _gather_sort(mx.expand_dims(x, (-2, -3)), idx)

    def proj(p, h):
        w, s, b = tier(li, p, n, mode)
        return mx.gather_qmm(h, w, s, b, rhs_indices=sidx, transpose=True, group_size=64, bits=4, sorted_indices=True)

    h = sw.activation(proj("up_proj", x), proj("gate_proj", x))
    return _scatter_unsort(proj("down_proj", h), inv, idx.shape).squeeze(-2)


def moe_call(self, x):
    gates = mx.softmax(self.gate(x), axis=-1, precise=True)
    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if self.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    order = mx.argsort(-scores, axis=-1)
    inds, scores = mx.take_along_axis(inds, order, axis=-1), mx.take_along_axis(scores, order, axis=-1)
    if CUR["cap"] is not None:
        CUR["cap"][self._li].append((x[0].astype(mx.float16), inds[0].astype(mx.int32), scores[0].astype(mx.float32)))
    plan, y = CUR["plan"], None
    for n in sorted(set(plan), reverse=True):
        sel = mx.array([b == n for b in plan])[:, None]
        Y = mx.where(sel, glu(self.switch_mlp, x, inds, self._li, n, CUR["mode"]), 0)
        y = Y if y is None else y + Y
    y = (y * scores[..., None]).sum(axis=-2)
    return y + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)


for li, l in enumerate(tm.layers):
    object.__setattr__(l.mlp, "_li", li)
type(tm.layers[0].mlp).__call__ = moe_call

# ---- 1. calibration capture (4-bit reference)
CUR["cap"] = [[] for _ in range(NL)]
for s in cal_ids:
    out = model(s[:, :-1])
    mx.eval(out, [c[-1] for c in CUR["cap"]])
cap = [tuple(mx.concatenate([t[j] for t in c]) for j in range(3)) for c in CUR["cap"]]
CUR["cap"] = None
mx.eval(cap)
print(f"calibration: {cap[0][0].shape[0]} tokens x {NL} layers captured ({time.time() - T0:.0f} s)", flush=True)


# ---- 2. refit
POOL = ThreadPoolExecutor(8)
LAM0 = float(os.environ.get("REFIT_LAMBDA", "1e-3"))
WEIGHT = os.environ.get("REFIT_WEIGHT", "none")                      # none | score2 (rows weighted by gate score^2)
MINRANK = int(os.environ.get("REFIT_MINRANK", "0"))                  # fit only on rows where the expert is at rank >= this (0-based)
print(f"refit options: lambda {LAM0}, weight {WEIGHT}, min rank {MINRANK}", flush=True)
MINROWS = 3                                                          # x 2G params (gate/up 192 rows, down 48)
nfit = {3: [0, 0], 2: [0, 0]}
for li in range(NL):
    X, I, SC = cap[li][0].astype(mx.float32), np.array(cap[li][1]), np.array(cap[li][2])
    S = {p: orig[li][p][1].astype(mx.float32) for p in P3}
    B = {p: orig[li][p][2].astype(mx.float32) for p in P3}
    new = {(p, n): ([], []) for p in P3 for n in (3, 2)}
    pending = []
    for e in range(256):
        hit = I == e
        hit[:, :MINRANK] = False
        rows = np.nonzero(hit.any(1))[0]
        Xe = X[mx.array(rows)] if len(rows) else None
        we = mx.array((SC[rows][hit[rows]] ** 2).astype(np.float32)) if (WEIGHT == "score2" and len(rows)) else None
        Qe = {p: codes(orig[li][p][0][e][None])[0] for p in P3}            # [out, K] codes 0..15 of expert e
        wdq = {p: (S[p][e][..., None] * Qe[p].reshape(*S[p][e].shape, 64).astype(mx.float32) + B[p][e][..., None]) for p in P3}
        if Xe is not None:
            g_, u_ = Xe @ wdq["gate_proj"].reshape(wdq["gate_proj"].shape[0], -1).T, Xe @ wdq["up_proj"].reshape(wdq["up_proj"].shape[0], -1).T
            He = g_ * mx.sigmoid(g_) * u_
        jobs = []
        for p in P3:
            out, G = S[p][e].shape
            inp = None if Xe is None else (He if p == "down_proj" else Xe)
            for n in (3, 2):
                r = fit_proj(inp, Qe[p], S[p][e], B[p][e], n, MINROWS, we, LAM0)
                if r[2] is None:
                    jobs.append((p, n, None, (r[0], r[1])))
                    nfit[n][1] += 1
                    continue
                mx.eval(r[0], r[1])
                jobs.append((p, n, POOL.submit(solve, np.array(r[0]), np.array(r[1])), r[2]))
                nfit[n][0] += 1
        pending.append(jobs)
        while len(pending) > 4 or (e == 255 and pending):              # collect with a lag: GPU and CPU solves overlap
            for p, n, fut, extra in pending.pop(0):
                if fut is None:
                    a, b = extra
                else:
                    th = mx.array(fut.result())
                    a, b = th[:, :extra], th[:, extra:]
                new[(p, n)][0].append(a)
                new[(p, n)][1].append(b)
    for p in P3:
        s0 = orig[li][p][1]
        for n in (3, 2):
            a = mx.stack(new[(p, n)][0]).astype(s0.dtype)                 # alpha in bf16 (the stored value)
            b = mx.stack(new[(p, n)][1]).astype(s0.dtype)
            TIERS[(li, p, n)] = ((a.astype(mx.float32) / (1 << (4 - n))).astype(s0.dtype), b)
            mx.eval(TIERS[(li, p, n)])
    if li % 8 == 7:
        print(f"  refit layers 0-{li} ({time.time() - T0:.0f} s; fits tier3 {nfit[3]}, tier2 {nfit[2]} [fitted, kept])", flush=True)
del cap
print(f"refit done ({time.time() - T0:.0f} s): tier 3 {nfit[3][0]} fitted / {nfit[3][1]} kept truncated, tier 2 {nfit[2][0]} / "
      f"{nfit[2][1]}", flush=True)

# ---- 3. eval: prefill KL vs the 4-bit reference per corpus part
PLANS = [("44333333", "trunc"), ("44333333", "refit"), ("44222222", "trunc"), ("44222222", "refit"),
         ("42222222", "trunc"), ("42222222", "refit"), ("43222222", "trunc"), ("43222222", "refit")]
if os.environ.get("REFIT_PLANS"):
    keep = os.environ["REFIT_PLANS"].split(",")
    PLANS = [(p, m) for p, m in PLANS if p in keep and (m == "refit" or os.environ.get("REFIT_TRUNC", "1") == "1")]
res = {}
logits = lambda inp: (lambda o: (o[0] if isinstance(o, tuple) else o)[0])(model(inp))
CH = 256
for part, ids in eval_ids.items():
    inp, tgt = ids[:, :-1], ids[0, 1:]
    T = inp.shape[1]
    CUR["plan"], CUR["mode"] = (4,) * 8, "trunc"
    lo = logits(inp)
    lr, nll_r, top_r = [], 0.0, []
    for c in range(0, T, CH):
        l = lo[c:c + CH].astype(mx.float32)
        l = l - mx.logsumexp(l, axis=-1, keepdims=True)
        nll_r += -mx.take_along_axis(l, tgt[c:c + CH, None], axis=-1).sum().item()
        top_r.append(mx.argmax(l, axis=-1))
        lr.append(l.astype(mx.bfloat16))
        mx.eval(lr[-1], top_r[-1])
    del lo
    for plan, mode in PLANS:
        CUR["plan"], CUR["mode"] = tuple(int(c) for c in plan), mode
        lo = logits(inp)
        kls, nll, ag = [], 0.0, 0.0
        for i, c in enumerate(range(0, T, CH)):
            l = lo[c:c + CH].astype(mx.float32)
            l = l - mx.logsumexp(l, axis=-1, keepdims=True)
            r = lr[i].astype(mx.float32)
            kl = (mx.exp(r) * (r - l)).sum(-1)
            nll += -mx.take_along_axis(l, tgt[c:c + CH, None], axis=-1).sum().item()
            ag += (mx.argmax(l, axis=-1) == top_r[i]).astype(mx.float32).sum().item()
            mx.eval(kl)
            kls.append(np.array(kl))
        del lo
        kl = np.concatenate(kls)
        res[(part, plan, mode)] = (kl.mean(), (nll - nll_r) / T, ag / T)
        print(f"{part:8s} {plan} {mode:5s}: KL {kl.mean():.5f} (p99 {np.percentile(kl, 99):.4f}), dNLL {(nll - nll_r) / T:+.4f}, "
              f"top-1 agreement {ag / T:.4f}", flush=True)
    del lr
print("--- refit / trunc KL ratio per plan and part:", flush=True)
for plan in sorted({p for p, _ in PLANS}):
    print(f"{plan}: " + "  ".join(f"{part} {res[(part, plan, 'refit')][0] / res[(part, plan, 'trunc')][0]:.3f}" for part in eval_ids),
          flush=True)
print(f"done ({time.time() - T0:.0f} s)", flush=True)
