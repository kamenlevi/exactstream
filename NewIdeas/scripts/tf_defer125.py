"""NewIdeas I1 on the 125B: deferred (late-added) experts, teacher-forced step KLD with the Eval suite's scorer
(2026-10-21). Built on Breakthrough's tf_frugal_var.py (read-only reuse of fork125bt + the Eval suite's reference).

Per 1-token decode call and layer L: the top-10 are read (sync, CPU residency as tf_frugal_var's SYNCGATE path) except
ranks beyond the mass rule; ranks < W and resident ranks are combined now; every OTHER computed rank (one that was not
resident before the call) is computed on the same input but its gate-weighted output is written into the hyper-
connection streams at the end of layer L + D, with layer L's own inject weights (the streams are only ever added to,
so this is exactly where it would have landed, minus its effect on layers L+1 .. L+D-1). Weights: p / sum(top-10 p)
(no renormalisation, as noren / m85_nr). The loads are those of exact streaming (or of the mass rule).
VAR: dD_wW[_mNN][_bNN] or dh_wW[_mNN][_bNN] (half a layer late; bNN: +NN/100 on resident experts' router
logits before the top-10, weights from the original probabilities)   e.g. dh_w2, d1_w2, dh_w2_m85, d1_w10_b25 (=
exact streaming with near-tie re-routing: w10 waits every rank)
usage: tf_defer125.py VAR FRAC SETS EVERY      -> ~/NewIdeas/data/tfdefer/results.txt
"""
import json, os, sys, time
os.environ["EVALSUITE_FRUGAL"] = "1"
os.environ["BIGMOE_SPARE"] = "0"
os.environ["BIGMOE_PF_NOPUBLISH"] = "1"
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
sys.path.insert(0, os.path.expanduser("~/Breakthrough/fork125bt"))
sys.path.insert(1, os.path.expanduser("~/MoEPrefetch/scripts"))
import numpy as np
import mlx.core as mx
from bigmoe.loader import load_streamed
from fused_route import topk_probs
from bigmoe.gated import ext
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
from evalsuite import stages as SG, engine as EN

VAR, frac, SETS, EVERY = sys.argv[1], float(sys.argv[2]), sys.argv[3].split(","), int(sys.argv[4])
D, W, THETA, HALF, BIAS = 2, 2, None, False, 0.0
for p in VAR.split("_"):
    if p.startswith("b"): BIAS = int(p[1:]) / 100; continue      # near-tie re-route: +BIAS on resident experts' logits
    if p == "dh": D, HALF = 1, True               # half a layer late: before layer L+1's MLP read (router)
    elif p.startswith("d"): D = int(p[1:])
    elif p.startswith("w"): W = int(p[1:])
    elif p.startswith("m"): THETA = int(p[1:]) / 100
OUT = os.path.expanduser("~/NewIdeas/data/tfdefer"); os.makedirs(OUT, exist_ok=True)
MD = os.path.expanduser("~/BigMoE/models/bare")
cfg = json.load(open(os.path.join(MD, "config.json"))); cfg = cfg.get("text_config", cfg)
E, K, NL = cfg["num_experts"], cfg["num_experts_per_tok"], cfg["num_hidden_layers"]
KC = K
C = max(16, round(frac * E))
rt, st = load_streamed(MD, C, mtp=False)
from mtplx.models import qwen4_exp
from mtplx.models.qwen4_exp import _hyper_residual_write
from mtplx.attention_math import attention_gate
orig_call = qwen4_exp.SparseMoeBlock.__call__
STASH, PEND, PENDH = {}, {}, {}
STAT = {"calls": 0, "late": 0, "now": 0, "drop": 0, "ld": 0, "late_mass": 0.0}


def block(self, x):
    sw = self.switch_mlp
    bm = getattr(sw, "_bm", None)
    if bm is None or x.shape[1] != 1 or bm["full"]:
        return orig_call(self, x)
    if bm["L"] == bm["first"]:
        bm["gate"].new_pass()
    L = bm["L"]
    if BIAS:
        lg = self.gate(x)[0].astype(mx.float32)                    # [1, E] router logits
        res_all = np.array(ext.lookup(L, list(range(E)))) >= 0
        b = lg + BIAS * mx.array(res_all.astype(np.float32))[None]
        keep2 = mx.argpartition(-lg, 1, axis=-1)[:, :2]               # never displace the router's ranks 0-1
        b = mx.put_along_axis(b, keep2, mx.array(1e9, b.dtype), axis=-1)
        top = mx.argpartition(-b, K - 1, axis=-1)[:, :K]
        pr = mx.softmax(lg, axis=-1, precise=True)
        ps = mx.take_along_axis(pr, top, axis=-1)
        o = mx.argsort(-ps, axis=-1)
        ids, pk = mx.take_along_axis(top, o, axis=-1).astype(mx.int32), mx.take_along_axis(ps, o, axis=-1)
    else:
        ids, pk = topk_probs(self.gate(x)[0], KC)                  # [1, K] by score
    p = np.array(pk[0, :K].astype(mx.float32)); p10 = p.sum()
    nload = K
    if THETA is not None:
        c = np.cumsum(p) / p10
        nload = max(1, int(np.searchsorted(c, THETA - 1e-9)) + 1)
    idl = [int(v) for v in np.array(ids[0])]
    pre = np.array(ext.lookup(L, idl))                             # residency before this call
    need = [e for r_, e in enumerate(idl) if r_ < nload or pre[r_] >= 0]
    ext.ensure(L, mx.array(np.array(need, np.int32)))              # read the computed cold ranks, touch resident ones
    slots = np.array(ext.lookup(L, idl))
    STAT["ld"] += int((pre[:nload] < 0).sum())
    r = np.arange(K)
    late = (r < nload) & (r >= W) & (pre < 0)
    now = ((r < nload) | (pre >= 0)) & ~late         # resident ranks are always used on time (the mass cap drops COLD ones only)
    STAT["calls"] += 1; STAT["late"] += int(late.sum()); STAT["now"] += int(now.sum()); STAT["drop"] += int(K - (now | late).sum())
    STAT["late_mass"] += float(p[late].sum() / p10)
    safe = np.where(slots >= 0, slots, slots[0]).astype(np.uint32)
    base = type(sw).__mro__[1]
    yk = base.__call__(sw, x, mx.array(safe).reshape(1, 1, K))     # [1, 1, K, D] per-rank outputs
    w = p / p10
    wn = mx.array((w * now).astype(np.float32)).astype(x.dtype)
    y = (yk * wn[None, None, :, None]).sum(axis=-2)
    if late.any():
        wl = mx.array((w * late).astype(np.float32)).astype(x.dtype)
        STASH[L] = (yk * wl[None, None, :, None]).sum(axis=-2)
    return y + attention_gate(self.shared_expert(x), self.shared_expert_gate(x))


def layer_call(self, hidden, *, input_ids, ssm_mask, cache):
    if "ple" in self:
        hidden = hidden + self.ple(hidden, input_ids, cache)
    mixed, hyper, inject = self.attn_hyper_connection(hidden)
    if self.is_linear:
        block_out = self.linear_attn(mixed, ssm_mask, cache)
    else:
        block_out = self.self_attn(mixed, cache)
    L = self._ni_L
    for lo, inj in PENDH.pop(L, []):                  # half-layer-late experts of layer L-1: seen by this layer's router
        hyper = _hyper_residual_write(hyper, lo, inj)
    mixed, hyper, inject = self.mlp_hyper_connection(hyper, pending=(block_out, inject))
    block_out = self.mlp(mixed)
    hidden = _hyper_residual_write(hyper, block_out, inject)
    lo = STASH.pop(L, None)
    if lo is not None:
        if HALF and L + 1 < NL:
            PENDH.setdefault(L + 1, []).append((lo, inject))
        else:
            PEND.setdefault(min(L + D, NL - 1), []).append((lo, inject))
    for lo, inj in PEND.pop(L, []):
        hidden = _hyper_residual_write(hidden, lo, inj)
    return hidden


for i, l in enumerate(rt.model.language_model.model.layers if hasattr(rt.model, "language_model") else rt.model.model.layers):
    l._ni_L = i
qwen4_exp.SparseMoeBlock.__call__ = block
qwen4_exp.DecoderLayer.__call__ = layer_call
items = SG.tf_items()
head = EN.var_head(SG.load_ref_tf(items), SG._targets(items))
seqs = [i for i, it in enumerate(items) if it["set"] in SETS][::EVERY]
out = {s: [] for s in SETS}
t0 = time.time()
for i in seqs:
    it = items[i]
    r = SG.step_forward(rt, it["ids"], it["pos"], lambda pos, lg, i=i: head(i, pos, lg))
    assert not PEND and not PENDH and not STASH, "undelivered late contributions"
    st.gate.check()
    out[it["set"]].append(r)
cat = {s: np.concatenate([r["kld"] for r in v]) for s, v in out.items() if v}
agr = {s: np.concatenate([r["agree"] for r in v]) for s, v in out.items() if v}
allk = np.concatenate(list(cat.values()))
n = max(1, STAT["calls"])
line = (f"{VAR:10s} C={C}: KLD {allk.mean():.4f} over {len(allk)} positions ({len(seqs)} seqs, every {EVERY}) | "
        + " ".join(f"{s} {cat[s].mean():.4f} (top-1 {agr[s].mean():.3f})" for s in cat)
        + f" | per token: {NL * STAT['now'] / n:.1f} now, {NL * STAT['late'] / n:.1f} late (mass {STAT['late_mass'] / n:.3f}"
        f" per layer), {NL * STAT['drop'] / n:.1f} dropped, {NL * STAT['ld'] / n:.1f} reads | {time.time() - t0:.0f}s")
print(line, flush=True)
TAG = "" if set(SETS) == {"tf_ifeval", "tf_multi"} else "_S" + "".join(x[3] for x in SETS)   # Breakthrough's naming
np.savez(os.path.join(OUT, f"{VAR}_r{round(frac * 100):02d}{TAG}_e{EVERY}.npz"), **{f"kld_{s}": cat[s] for s in cat}, **{f"agree_{s}": agr[s] for s in agr})
open(os.path.join(OUT, "results.txt"), "a").write(line + "\n")
