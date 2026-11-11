"""One scenario of the T1 self-test, in its own process (CPU device only): the decision map's harness code (dm_common.py,
dm_ablations.py, from <src>) driven with a small FAKE model in place of the 125B, so the real wrapper code runs.
Usage: harness_case.py <src dir> <out prefix> <variant: streamed | fused | mixed>     (DM_XS from the environment)
The fake model mirrors the two real paths: >= 32 rows -> SparseMoeBlock._prefill_call -> switch.sorted_experts(x, inds)
and the fused combine imported at call time; narrower -> switch(x, inds). "streamed" gives every block a subclass that
overrides __call__ and calls the base class explicitly (BigMoE's ...Streamed); "fused" uses the base class itself;
"mixed" uses both (the nested-wrapper case)."""
import json, os, sys, types
import numpy as np
import mlx.core as mx
mx.set_default_device(mx.cpu)
src, out, variant = sys.argv[1], sys.argv[2], sys.argv[3]
D, E, K, INNER = 128, 12, 4, 8
NL = 48 if variant.startswith("measure") else 3
LOG = {"switch": [], "router": [], "shared": [], "sgate": []}
NPMASK = {"c": None}

def mask_np(x, c):
    """independent reference: keep a member iff fewer than c members of its 64-group are strictly larger in |x|"""
    if c is None or c >= 64: return x
    a = np.abs(x.astype(np.float32)).reshape(*x.shape[:-1], -1, 64)
    bigger = (a[..., None, :] > a[..., :, None]).sum(-1)
    return np.where((bigger < c).reshape(x.shape), x, np.zeros_like(x))

class Rec:
    def __init__(self, W, key): self.W, self.key = W, key
    def __call__(self, x):
        LOG[self.key].append(np.array(x)); return x @ self.W.T

class Sw:
    def __init__(self, rng):
        f = lambda *s: mx.array((rng.standard_normal(s) / np.sqrt(s[-1])).astype(np.float32))
        self.Wg, self.Wu, self.Wd = f(E, INNER, D), f(E, INNER, D), f(E, D, INNER)
    def _sorted(self, x, indices):
        xn = np.array(x)
        if NPMASK["c"] is not None: xn = mask_np(xn, NPMASK["c"]); x = mx.array(xn)
        LOG["switch"].append(xn)
        rows = x.reshape(-1, D); idx = indices.reshape(-1).astype(mx.int32)
        order = mx.argsort(idx); inv = mx.argsort(order)
        xs_ = mx.take(mx.repeat(rows, K, axis=0), order, axis=0); es = mx.take(idx, order, axis=0)
        g = (mx.take(self.Wg, es, axis=0) @ xs_[..., None]).squeeze(-1)
        u = (mx.take(self.Wu, es, axis=0) @ xs_[..., None]).squeeze(-1)
        y = (mx.take(self.Wd, es, axis=0) @ (g * mx.sigmoid(g) * u)[..., None]).squeeze(-1)
        return y[:, None, :], inv
    def __call__(self, x, indices):
        y, inv = self._sorted(x, indices)
        return mx.take(y, inv, axis=0).reshape(*indices.shape, D)
    def sorted_experts(self, x, indices):
        y, inv = self._sorted(x, indices)
        return y.reshape(y.shape[0], -1), inv
SwStreamed = type("SwStreamed", (Sw,), {"__call__": lambda self, x, indices, *a, **kw: Sw.__call__(self, x, indices, *a, **kw)})

def _applies(block, x):
    return x.ndim >= 2 and x.size // x.shape[-1] >= 32

class SparseMoeBlock:
    top_k = K; norm_topk_prob = True
    def __init__(self, rng, cls):
        f = lambda *s: mx.array((rng.standard_normal(s) / np.sqrt(s[-1])).astype(np.float32))
        self.gate = Rec(f(E, D) * 4, "router"); self.switch_mlp = cls(rng)
        self.shared_expert = Rec(f(D, D), "shared"); self.shared_expert_gate = Rec(f(1, D), "sgate")
    def _route(self, x):
        gates = mx.softmax(self.gate(x), axis=-1, precise=True); k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob: scores = scores / scores.sum(axis=-1, keepdims=True)
        return inds, scores, k
    def __call__(self, x):
        if _applies(self, x): return self._prefill_call(x)
        inds, scores, k = self._route(x)
        y = self.switch_mlp(x, inds); y = (y * scores[..., None]).sum(axis=-2)
        return y + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
    def _prefill_call(self, x):
        from mtplx.kernels.qwen4_moe_prefill_combine import moe_prefill_combine
        inds, scores, k = self._route(x)
        y_sorted, inv_order = self.switch_mlp.sorted_experts(x, inds)
        shared = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        rows = inds.size // k
        return moe_prefill_combine(y_sorted, inv_order, scores.reshape(rows, k), shared.reshape(rows, -1)).reshape(x.shape)

def moe_prefill_combine(y_sorted, inv_order, scores, shared):
    rows, k = scores.shape
    return (mx.take(y_sorted, inv_order, axis=0).reshape(rows, k, -1) * scores[..., None]).sum(axis=1) + shared

def mod(name, **kw):
    m = types.ModuleType(name); m.__dict__.update(kw); sys.modules[name] = m; return m
dummy = lambda nm: type(nm, (), {"__call__": lambda self, *a, **k: None})
Q = mod("mtplx.models.qwen4_exp", SparseMoeBlock=SparseMoeBlock, _moe_prefill_combine_applies=_applies,
        GatedDeltaNet=dummy("GatedDeltaNet"), Attention=dummy("Attention"), GatedResidual=dummy("GatedResidual"),
        Qwen4ExpTextModel=dummy("Qwen4ExpTextModel"))
PCm = mod("mtplx.kernels.qwen4_moe_prefill_combine", moe_prefill_combine=moe_prefill_combine)
Gm = mod("mtplx.generation")
mod("mtplx.models", qwen4_exp=Q); mod("mtplx.kernels", qwen4_moe_prefill_combine=PCm)
mod("mtplx", models=sys.modules["mtplx.models"], kernels=sys.modules["mtplx.kernels"], generation=Gm)

os.environ["DM_OUT"] = out + "_out"
sys.path.insert(0, src)
import dm_common as C
import dm_ablations as A
C.install()
rng = np.random.default_rng(7)
classes = {"streamed": [SwStreamed] * NL, "fused": [Sw] * NL, "mixed": [SwStreamed, Sw, SwStreamed],
           "measure": [SwStreamed] * NL, "measure_broken": [SwStreamed] * NL}[variant]
blocks = [SparseMoeBlock(rng, c) for c in classes]
for i, b in enumerate(blocks): C.LAYERS[id(b)] = (i, "moe")
xw = mx.array(rng.standard_normal((1, 40, D)).astype(np.float32)); xn = mx.array(rng.standard_normal((1, 8, D)).astype(np.float32))
XS = getattr(C, "_XS", None)

def run(arm, npmask=None):
    C.clear()
    if arm: A.ablation(arm)()
    NPMASK["c"] = npmask
    for k in LOG: LOG[k].clear()
    r0 = (XS.XS["rows"], XS.XS["calls"]) if XS is not None else (0, 0)
    outs = []
    for x in (xw, xn):
        h = x
        for b in blocks: h = h + b(h)
        outs.append(np.array(h))
    n = len(LOG["router"])
    flags = dict(calls=n, n_switch=len(LOG["switch"]), n_shared=len(LOG["shared"]), n_sgate=len(LOG["sgate"]))
    if n == len(LOG["switch"]) == len(LOG["shared"]) == len(LOG["sgate"]):
        flags["shared_is_x"] = all(np.array_equal(LOG["shared"][i], LOG["router"][i]) and np.array_equal(LOG["sgate"][i], LOG["router"][i]) for i in range(n))
        for c in (64, 48, 40, 32):
            flags[f"switch_is_mask{c}"] = all(np.array_equal(LOG["switch"][i], mask_np(LOG["router"][i], c)) for i in range(n))
    if XS is not None:
        flags.update(xs_rows=XS.XS["rows"] - r0[0], xs_calls=XS.XS["calls"] - r0[1], xs_c_during=XS.XS["c"], wrapped=sorted(c.__name__ for c in XS._WRAPPED))
    NPMASK["c"] = None; C.clear()
    if XS is not None: flags["xs_c_after_clear"] = XS.XS["c"]
    return outs, flags

if variant.startswith("measure"):
    # the patched measure(): its per-window record of masked switch rows and its WARNING when a path escapes the mask
    def fake_run_window(model, ids):
        n = len(ids) - 1
        use = blocks[:-1] if variant == "measure_broken" else blocks       # broken: one layer's experts are never called
        for lo, hi in ((0, min(n, 40)), (40, n)):                           # a wide chunk, then a narrow tail
            if hi <= lo: continue
            h = mx.array(np.random.default_rng(lo).standard_normal((1, hi - lo, D)).astype(np.float32))
            for b in use: h = h + b(h)
            mx.eval(h)
        return np.zeros((n, 8), np.int64), np.zeros((n, 8), np.float32)
    C.run_window = fake_run_window
    wins = [("code", "w_a.npz", np.arange(47, dtype=np.int32), 0, 5), ("multi", "w_b.npz", np.arange(60, dtype=np.int32), 0, 9)]
    meta = dict(src=src, variant=variant, DM_XS=os.environ.get("DM_XS"))
    for arm in ("xs64", "xs48", "topk3+xs32"):
        A.ABLATIONS.setdefault("topk3", A.topk_experts(3))
        r = C.measure(None, wins, arm, A.ablation(arm))
        meta[arm] = {w: dict(xs=v.get("xs"), warn=v.get("warn"), rows=len(v["tk"])) for w, v in r.items()}
        meta[arm + "_file"] = os.path.exists(os.path.join(C.OUT, arm + ".json"))
    json.dump(meta, open(out + ".json", "w"), indent=1); print("measure case done"); sys.exit(0)

res, arrays = {}, {}
cases = [("baseline", None, None), ("topk8", "topk3", None), ("ref_mask32", None, 32), ("ref_topk_mask32", "topk3", 32), ("ref_mask48", None, 48)]
A.ABLATIONS["topk3"] = A.topk_experts(3)                 # the fake has top_k 4: a rank cut that bites (stands in for topk8)
if XS is not None:
    cases += [("xs64", "xs64", None), ("xs32", "xs32", None), ("xs48", "xs48", None), ("topk+xs32", "topk3+xs32", None), ("baseline_again", None, None)]
for tag, arm, npm in cases:
    outs, flags = run(arm, npm); res[tag] = flags; arrays[tag + "_wide"], arrays[tag + "_narrow"] = outs
res["_meta"] = dict(src=src, variant=variant, DM_XS=os.environ.get("DM_XS"), xs_imported=XS is not None,
                    xs_arms=sorted(k for k in A.ABLATIONS if k.startswith("xs")), device=str(mx.default_device()))
np.savez(out + ".npz", **arrays); json.dump(res, open(out + ".json", "w"), indent=1)
print("case done:", res["_meta"])
