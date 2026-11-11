"""TrainLab: chained-drafting fine-tune of the 125B's native MTP head (NewIdeas 23:05; exact-safe: drafts are verified).
Chat1000 (data/agent125): the stock head's chain on real Claude Code turns gives tau ~3.9 (bash 3.4, text 3.8, edit
5.6, write 2.8) vs the 35B head's 8-10. Kill if tau(16) does not rise >= ~30 %.

Head (MTPLX's reconstruction, qwen4_exp.Qwen4ExpMTP): the trunk's pre-mixer WIDENED stream W [S, 4 x 2560] -> full-
width RMSNorm -> shared fc_hidden per substream, + fc_embedding(norm(embed(t+1))) -> one QSA + HC + 512-expert MoE
DecoderLayer -> pre-mixer output (the recursion state) -> hyper_connection_mixer -> the trunk's lm_head.
Training forward (here, plain MLX ops so autodiff works; MoonShots 23:25 + code study): dense causal attention (QSA's
indexer selects everything below ~2051 tokens), MTPLX's eager hyper-connection chain (MTPLX_QWEN4_HC_PREFILL_READ=0),
the MoE with mx.gather_qmm on the frozen 4-bit experts. Trained (fp32 copies): fc_hidden, fc_embedding, attention
q/k/v/o, router, shared expert. Frozen: experts, HC mixers, norms, lm_head, embeddings.
Depths (DeepSeek-V3's MTP recipe, no custom masks): depth j is a causal pass over the previous depth's outputs with
token t+j; KL to the trunk's distribution at position t+j (top-64 + tail) at every depth.
Data: Kamen's Claude Code turns (SpecDecode extract/render, read-only), disjoint from SpecDecode's 29 samples (which
contain Chat1000's 12 test turns); the trunk prefills each turn once (streamed BigMoE engine) and every position of it
becomes training data (no disk: features live in RAM for one turn).
  mtp125.py probe | check | train --run NAME [--depth 4] [--lr 1e-5] [--time-limit S] | export --run NAME
"""
import argparse, glob, json, math, os, random, sys, time
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
if os.environ.get("TL_INFER") != "1":                       # scoring only (sim125, agent125_tl) keeps the fused read:
    os.environ["MTPLX_QWEN4_HC_PREFILL_READ"] = "0"       # bit-identical, faster; training needs the eager one (VJP)
if os.environ.get("TL_PLAIN_QSA") == "1":                    # everything custom off (also slows the trunk's prefill)
    for _k in ("MTPLX_QSA_DENSE_BAND_SDPA", "MTPLX_QSA_FLASH", "MTPLX_QSA_GATHER", "MTPLX_QSA_PREFILL", "MTPLX_FUSED_QSA_QKV",
               "MTPLX_FUSED_QSA_INDEXER", "MTPLX_FUSED_MOE_VERIFY", "MTPLX_FUSED_MOE_DECODE", "MTPLX_FUSED_HC", "MTPLX_FUSED_HC_V3",
               "MTPLX_MOE_SORTED_GATHER_KERNEL"):
        os.environ[_k] = "0"
elif os.environ.get("TL_PLAIN_MIN") == "1":                  # the minimal set: turn 7 showed MTPLX's QSA attention is
    os.environ["MTPLX_MOE_SORTED_GATHER_KERNEL"] = "0"       # differentiable at prefill width with default flags; only the
                                                             # NAX sorted gather (no VJP) must go for the head's MoE
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("~/SpecDecode"))
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

TL = os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p4s/TrainLab_root")
MD = os.path.expanduser("~/BigMoE/models/bare")
RES = os.path.join(TL, "data/mtp125_results.txt")
TOPK = 64
CHUNK = 2048


def limit_memory():
    mx.set_cache_limit(int(float(os.environ.get("TL_CACHE_GB", "2")) * 2 ** 30))
    mx.set_memory_limit(int(float(os.environ.get("TL_MEM_GB", "36")) * 2 ** 30))


def load(frac=0.25):
    from bigmoe.loader import load_streamed
    cfg = json.load(open(os.path.join(MD, "config.json"))); tc = cfg.get("text_config", cfg)
    rt, st = load_streamed(MD, max(16, round(frac * tc["num_experts"])), mtp=True)
    lm = rt.model.language_model
    return rt, lm, lm.mtp


def dequant_linear(q):
    """A trainable fp32 Linear from a (Quantized)Linear."""
    if isinstance(q, nn.QuantizedLinear):
        w = mx.dequantize(q.weight, q.scales, q.biases, group_size=q.group_size, bits=q.bits)
    else:
        w = q.weight
    lin = nn.Linear(w.shape[1], w.shape[0], bias=False)
    lin.weight = w.astype(mx.float32)
    return lin


class Trainable(nn.Module):
    """fp32 copies of the head's trained parts; the frozen rest is read from the MTPLX module."""

    def __init__(self, head):
        super().__init__()
        layer = head.layers[0]
        att, moe = layer.self_attn, layer.mlp
        self.fc_hidden = dequant_linear(head.fc_hidden)
        self.fc_embedding = dequant_linear(head.fc_embedding)
        self.q_proj, self.k_proj = dequant_linear(att.q_proj), dequant_linear(att.k_proj)
        self.v_proj, self.o_proj = dequant_linear(att.v_proj), dequant_linear(att.o_proj)
        self.router = dequant_linear(moe.gate)
        se = moe.shared_expert
        if hasattr(se, "gu_weight"):                                   # MTPLX's fused gate+up shared expert
            gu = mx.dequantize(se.gu_weight, se.gu_scales, se.gu_biases, group_size=se.group_size, bits=se.bits)
            g, u = mx.split(gu, 2, axis=0)
            self.se_gate, self.se_up = nn.Linear(g.shape[1], g.shape[0], bias=False), nn.Linear(u.shape[1], u.shape[0], bias=False)
            self.se_gate.weight, self.se_up.weight = g.astype(mx.float32), u.astype(mx.float32)
        else:
            self.se_gate, self.se_up = dequant_linear(se.gate_proj), dequant_linear(se.up_proj)
        self.se_down = dequant_linear(se.down_proj)
        self.se_gatew = dequant_linear(moe.shared_expert_gate) if hasattr(moe, "shared_expert_gate") else None


def experts_call(moe, x, inds):
    """Frozen routed experts with plain gather_qmm (sorted by expert as mlx_lm's SwitchGLU), [B, S, k, D]."""
    sw = moe.switch_mlp
    B, S, D = x.shape
    k = inds.shape[-1]
    xf = mx.expand_dims(x, (-2, -3))                                   # [B, S, 1, 1, D]
    idx = inds.flatten()
    order = mx.argsort(idx)
    inv = mx.argsort(order)
    xs = xf.flatten(0, -3)[order // k]                                  # [B*S*k, 1, D]
    ids = idx[order]
    if hasattr(sw, "gu_weight"):                                       # MTPLX's fused gate+up layout
        gu = mx.gather_qmm(xs, sw.gu_weight, sw.gu_scales, sw.gu_biases, rhs_indices=ids, transpose=True,
                           group_size=sw.group_size, bits=sw.bits, sorted_indices=True)
        g, u = mx.split(gu, 2, axis=-1)
        down = sw.down_proj
    else:
        gp, up = sw.gate_proj, sw.up_proj
        g = mx.gather_qmm(xs, gp.weight, gp.scales, gp.biases, rhs_indices=ids, transpose=True,
                          group_size=gp.group_size, bits=gp.bits, sorted_indices=True)
        u = mx.gather_qmm(xs, up.weight, up.scales, up.biases, rhs_indices=ids, transpose=True,
                          group_size=up.group_size, bits=up.bits, sorted_indices=True)
        down = sw.down_proj
    h = nn.silu(g) * u
    y = mx.gather_qmm(h, down.weight, down.scales, down.biases, rhs_indices=ids, transpose=True,
                      group_size=down.group_size, bits=down.bits, sorted_indices=True)
    y = y[inv].reshape(B, S, k, -1)
    return y


def moe_train(moe, P, x):
    gates = mx.softmax(P.router(x.astype(mx.float32)), axis=-1)
    k = moe.top_k
    inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if getattr(moe, "norm_topk_prob", True):
        scores = scores / scores.sum(axis=-1, keepdims=True)
    y = experts_call(moe, x, inds)
    out = (y * scores[..., None].astype(y.dtype)).sum(axis=-2)
    xf = x.astype(mx.float32)
    sh = P.se_down(nn.silu(P.se_gate(xf)) * P.se_up(xf))
    if P.se_gatew is not None:
        sh = mx.sigmoid(P.se_gatew(xf)) * sh
    return out.astype(mx.float32) + sh


def attn_train(att, P, x, offset):
    import mtplx.models.qwen4_exp as Q
    B, S, _ = x.shape
    xf = x.astype(mx.float32)
    q, gate = mx.split(P.q_proj(xf).reshape(B, S, att.n_heads, -1), 2, axis=-1)
    gate = gate.reshape(B, S, -1)
    k = P.k_proj(xf).reshape(B, S, att.n_kv_heads, -1)
    v = P.v_proj(xf).reshape(B, S, att.n_kv_heads, -1)
    q, k = att.q_norm(q), att.k_norm(k)
    pos = offset + mx.arange(S, dtype=mx.int32)
    cos, sin = Q._rope_cos_sin(pos, att._inv_freq, att._rope_attention_scaling)
    q, k = Q._apply_partial_rope(q, cos, sin), Q._apply_partial_rope(k, cos, sin)
    q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=att.scale, mask="causal")
    o = o.transpose(0, 2, 1, 3).reshape(B, S, -1)
    return P.o_proj(o * mx.sigmoid(gate))


def head_train(head, P, W, emb, offset):
    """One MTP step for every position: (W[t], emb(t+1)) -> pre-mixer widened output [B, S, 4*2560]."""
    B, S, Wd = W.shape
    hc = head._hc
    hn = head.pre_fc_norm_hidden(W).reshape(B, S, hc, -1).astype(mx.float32)
    en = P.fc_embedding(head.pre_fc_norm_embedding(emb).astype(mx.float32))
    h = (P.fc_hidden(hn) + en[:, :, None, :]).reshape(B, S, Wd)
    layer = head.layers[0]
    hb = h.astype(W.dtype)
    mixed, hyper, inject = layer.attn_hyper_connection(hb)
    blk = attn_train(layer.self_attn, P, mixed, offset).astype(W.dtype)
    mixed, hyper, inject = layer.mlp_hyper_connection(hyper, pending=(blk, inject))
    blk = moe_train(layer.mlp, P, mixed).astype(W.dtype)
    return hyper + (blk[..., None, :] * inject[..., :, None]).reshape(*hyper.shape)


def draft_logits(lm, head, hout):
    return lm.lm_head(head.hyper_connection_mixer(hout))


def kl_top(lq, tlp):
    p, q = mx.exp(tlp), mx.exp(lq)
    pt = mx.maximum(1 - p.sum(-1), 1e-12); qt = mx.maximum(1 - q.sum(-1), 1e-12)
    return (p * (tlp - lq)).sum(-1) + pt * (mx.log(pt) - mx.log(qt))


def depth_losses(lm, head, P, W, ids, top, tlp, K, start=0):
    """W [1, S, Wd] widened stream, ids [1, S] tokens, top/tlp [1, S, 64] trunk targets. Depth j (1..K): chain t uses
    the previous depth's output at t and token ids[t + j]; target = trunk distribution at t + j (predicts t + j + 1)."""
    embed = lm.model.embed_tokens
    S = W.shape[1]
    feat, per, tot = W, [], 0.0
    for j in range(1, K + 1):
        n = S - j - 1
        if n <= start + 8:
            break
        tok = ids[:, j:j + n]
        hout = head_train(head, P, feat[:, :n], embed(tok), j - 1)
        lg = draft_logits(lm, head, hout).astype(mx.float32)
        lq = mx.take_along_axis(lg, top[:, j:j + n], -1) - mx.logsumexp(lg, -1, keepdims=True)
        kl = kl_top(lq, tlp[:, j:j + n])[:, start:]
        per.append(kl.mean()); tot = tot + per[-1]
        feat = hout
    return tot / max(len(per), 1), per


# ---------------------------------------------------------------- trunk features (teacher-forced prefill)
def trunk_features(lm, ids):
    """Prefill the streamed trunk over ids (chunks of CHUNK) -> widened [1, S, Wd] (bf16) and top-64 [1, S, 64]."""
    cache = lm.make_cache()
    Ws, tops, tlps = [], [], []
    for a in range(0, len(ids), CHUNK):
        x = mx.array([ids[a:a + CHUNK]])
        logits, W = lm(x, cache=cache, return_hidden=True)
        lg = logits.astype(mx.float32)
        lp = lg - mx.logsumexp(lg, -1, keepdims=True)
        tp = mx.argpartition(-lp, TOPK - 1, axis=-1)[..., :TOPK]
        tops.append(tp); tlps.append(mx.take_along_axis(lp, tp, -1)); Ws.append(W)
        mx.eval(tops[-1], tlps[-1], Ws[-1], [c.state for c in cache if hasattr(c, "state")])
    return mx.concatenate(Ws, 1), mx.concatenate(tops, 1), mx.concatenate(tlps, 1)


def samples(n, seed=2):
    import extract as X
    from render import render
    held = [json.loads(l) for l in open(os.path.expanduser("~/SpecDecode/data/samples.jsonl"))]
    held_targets = {json.dumps(s["target"], sort_keys=True)[:400] for s in held}
    random.seed(seed)
    out = []
    for path in sorted(glob.glob(f"{X.PROJ}/*/*.jsonl")):
        if "SpecDecode" in path:
            continue
        msgs = X.load_session(path)
        for i, m in enumerate(msgs):
            if m["role"] != "assistant" or i < 4:
                continue
            ctx = X.context_before(msgs, i)
            kind = X.classify(m, ctx)
            if kind and json.dumps(m, sort_keys=True)[:400] not in held_targets:
                out.append({"kind": kind, "project": path.split("/")[-2], "messages": ctx, "target": m})
    random.shuffle(out)
    return out[:n], render


def general_items():
    """The 125B's own greedy text outside every held-out set: EvalSuite's reference decodes (results/ref/decode.jsonl:
    IFEval 100, MATH-500 50, HumanEval 60; prompts in data/items/dec_*.jsonl). Disjoint from tf_gen (sim125's held-out
    general text: other IFEval ids, other HumanEval tasks, GSM8K, multilingual), checked 10-22 03:10. Shuffled, fixed."""
    E = os.path.expanduser("~/EvalSuite")
    comp = {}
    for l in open(f"{E}/results/ref/decode.jsonl"):
        r = json.loads(l); comp[r["key"]] = r["completion_ids"]
    out = []
    for f in ("dec_ifeval", "dec_math", "dec_humaneval"):
        for l in open(f"{E}/data/items/{f}.jsonl"):
            it = json.loads(l)
            if it["key"] in comp and len(comp[it["key"]]) >= 16:
                out.append(list(it["prompt_ids"]) + list(comp[it["key"]]))
    random.Random(7).shuffle(out)
    return out


def rendered_full(s, render, tok):
    """prompt ids (forced prefix) + Claude's real continuation (teacher-forced text; the trunk scores it)."""
    ids, rest = render(s, tok)
    cont = tok.encode(rest, add_special_tokens=False)[:512]
    full = ids + cont
    if len(full) > 9000:
        full = full[:600] + full[-8400:]
    return full


# ---------------------------------------------------------------- commands
def cmd_probe(a):
    rt, lm, head = load(a.frac)
    layer = head.layers[0]
    att, moe = layer.self_attn, layer.mlp
    print("head", type(head).__name__, "layer", type(layer).__name__, "attn", type(att).__name__, "moe", type(moe).__name__)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj", "qkv_fused", "indexer", "q_norm", "k_norm"):
        m = getattr(att, name, None)
        print(f"  attn.{name}: {type(m).__name__}", getattr(m, "bits", ""), getattr(getattr(m, "weight", None), "shape", ""))
    print("  n_heads", att.n_heads, "n_kv", att.n_kv_heads, "head_dim", getattr(att, "head_dim", None), "scale", att.scale)
    for name in ("gate", "switch_mlp", "shared_expert", "shared_expert_gate"):
        m = getattr(moe, name, None)
        print(f"  moe.{name}: {type(m).__name__}", getattr(m, "bits", ""))
    print("  top_k", moe.top_k, "norm_topk_prob", getattr(moe, "norm_topk_prob", None))
    print("  fc_hidden", type(head.fc_hidden).__name__, head.fc_hidden.weight.dtype, "lm_head", type(lm.lm_head).__name__)
    print(f"peak {mx.get_peak_memory() / 1e9:.1f} GB")


def cmd_check(a):
    """my head_train (depth 1) vs MTPLX's own mtp_forward on a real turn: logits must agree."""
    rt, lm, head = load(a.frac)
    tok = rt.tokenizer
    sm, render = samples(4)
    ids = rendered_full(sm[0], render, tok)[:1500]
    W, top, tlp = trunk_features(lm, ids)
    P = Trainable(head)
    x = mx.array([ids])
    S = len(ids) - 1
    hout = head_train(head, P, W[:, :S], lm.model.embed_tokens(x[:, 1:S + 1]), 0)
    mine = draft_logits(lm, head, hout).astype(mx.float32)
    cache = lm.make_mtp_cache()
    ref = lm.mtp_forward(W[:, :S], x[:, 1:S + 1], mtp_cache=cache).astype(mx.float32)
    d = mx.abs(mine - ref)
    am = (mx.argmax(mine, -1) == mx.argmax(ref, -1)).astype(mx.float32).mean()
    print(f"check: S {S}: max |dlogit| {d.max().item():.4f}, mean {d.mean().item():.5f}, argmax agreement {am.item():.4f}")
    _, per = depth_losses(lm, head, P, W, x, top, tlp, 4)
    print("check: stock KL per depth " + " ".join(f"{float(v):.3f}" for v in per), f"peak {mx.get_peak_memory() / 1e9:.1f} GB")


def cmd_check2(a):
    """Localize the mismatch: MTPLX's own layer step by step vs mine, and whether autodiff passes MTPLX's modules."""
    rt, lm, head = load(a.frac)
    tok = rt.tokenizer
    sm, render = samples(4)
    ids = rendered_full(sm[0], render, tok)[:600]
    W, top, tlp = trunk_features(lm, ids)
    P = Trainable(head)
    x = mx.array([ids]); S = len(ids) - 1
    Wd, emb = W[:, :S], lm.model.embed_tokens(x[:, 1:S + 1])
    layer = head.layers[0]
    def rel(a_, b_):
        a_, b_ = a_.astype(mx.float32), b_.astype(mx.float32)
        return f"max {mx.abs(a_ - b_).max().item():.4g} mean {mx.abs(a_ - b_).mean().item():.4g} (|ref| mean {mx.abs(b_).mean().item():.4g})"
    h0_ref = head._prepare_inputs_eager(Wd, emb)
    hn = head.pre_fc_norm_hidden(Wd).reshape(1, S, head._hc, -1).astype(mx.float32)
    h0_mine = (P.fc_hidden(hn) + P.fc_embedding(head.pre_fc_norm_embedding(emb).astype(mx.float32))[:, :, None, :]).reshape(1, S, -1)
    print("check2 prepare:", rel(h0_mine, h0_ref), flush=True)
    mixed, hyper, inject = layer.attn_hyper_connection(h0_ref)
    import mtplx.models.qwen4_exp as Q
    c = lm.make_mtp_cache()
    blk_ref = layer.self_attn(mixed, c[0])
    blk_mine = attn_train(layer.self_attn, P, mixed, 0)
    print("check2 attention:", rel(blk_mine, blk_ref), flush=True)
    mixed2, hyper2, inject2 = layer.mlp_hyper_connection(hyper, pending=(blk_ref, inject))
    moe_ref = layer.mlp(mixed2)
    moe_mine = moe_train(layer.mlp, P, mixed2)
    print("check2 moe:", rel(moe_mine, moe_ref), flush=True)
    hout_ref = hyper2 + (moe_ref[..., None, :] * inject2[..., :, None]).reshape(*hyper2.shape)
    lg_head = lm.lm_head(head.hyper_connection_mixer(hout_ref)).astype(mx.float32)
    lg_draft = lm._mtp_draft_head_logits(head.hyper_connection_mixer(hout_ref)).astype(mx.float32)
    print("check2 draft head vs lm_head:", rel(lg_head, lg_draft), "same object:", lm._mtp_draft_head_logits == lm.lm_head.__call__, flush=True)
    ref_full = lm.mtp_forward(Wd, x[:, 1:S + 1], mtp_cache=lm.make_mtp_cache()).astype(mx.float32)
    print("check2 stepwise ref vs mtp_forward:", rel(lg_draft, ref_full), flush=True)
    # target alignment: depth-1 KL of MTPLX's own logits vs the trunk's distribution at t + 1
    lp = ref_full - mx.logsumexp(ref_full, -1, keepdims=True)
    lq = mx.take_along_axis(lp, top[:, 1:S + 1], -1)
    print("check2 depth-1 KL of MTPLX's head vs trunk(t+1):", kl_top(lq, tlp[:, 1:S + 1]).mean().item(),
          " vs trunk(t+2):", kl_top(mx.take_along_axis(lp[:, :S - 1], top[:, 2:S + 1], -1), tlp[:, 2:S + 1]).mean().item(), flush=True)
    # does autodiff pass MTPLX's own attention / MoE modules?
    wrapped = NativeMoE(layer.mlp)
    d = mx.abs(wrapped(mixed2).astype(mx.float32) - moe_ref.astype(mx.float32))
    print(f"check2 NativeMoE vs MTPLX mlp: max {d.max().item():.4g} mean {d.mean().item():.4g}", flush=True)
    for name, f in (("self_attn", lambda m: layer.self_attn(m, lm.make_mtp_cache()[0]).astype(mx.float32).sum()),
                    ("mlp", lambda m: layer.mlp(m).astype(mx.float32).sum()),
                    ("NativeMoE", lambda m: wrapped(m).astype(mx.float32).sum())):
        try:
            g = mx.grad(f)(mixed if name == "self_attn" else mixed2); mx.eval(g)
            print(f"check2 grad through MTPLX {name}: OK |g| {mx.abs(g).max().item():.3g}", flush=True)
        except Exception as e:
            print(f"check2 grad through MTPLX {name}: FAIL {type(e).__name__}: {str(e)[:160]}", flush=True)


class NativeMoE(nn.Module):
    """MTPLX's plain SparseMoeBlock path (qwen4_exp ~L1432-1440) with stop_gradient on the routing indices (its own
    call feeds argpartition indices to take_along_axis, which autodiff cannot cross). Same ops, same order."""

    def __init__(self, orig):
        super().__init__()
        self.gate, self.switch_mlp = orig.gate, orig.switch_mlp
        self.shared_expert, self.shared_expert_gate = orig.shared_expert, orig.shared_expert_gate
        self.top_k, self.norm_topk_prob = orig.top_k, orig.norm_topk_prob

    def __call__(self, x):
        from mtplx.attention_math import attention_gate
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        k = self.top_k
        inds = mx.stop_gradient(mx.argpartition(gates, kth=-k, axis=-1)[..., -k:])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        return y + attention_gate(self.shared_expert(x), self.shared_expert_gate(x))


TRAIN_MODULES = ("fc_hidden", "fc_embedding", "layers.0.self_attn.q_proj", "layers.0.self_attn.k_proj",
                 "layers.0.self_attn.v_proj", "layers.0.self_attn.o_proj", "layers.0.mlp.gate", "layers.0.mlp.shared_expert",
                 "layers.0.mlp.shared_expert_gate")


def native_prepare(head):
    """Make the head's trained parts fp32 Linear modules in place; freeze everything else."""
    install(head, Trainable(head), dtype=mx.float32)
    if not isinstance(head.layers[0].mlp, NativeMoE):
        head.layers[0].mlp = NativeMoE(head.layers[0].mlp)
    head.freeze()
    for name, m in head.named_modules():
        if any(name == t or name.startswith(t + ".") for t in TRAIN_MODULES) and isinstance(m, nn.Linear):
            m.unfreeze()
    return sum(v.size for _, v in tree_flatten(head.trainable_parameters()))


_LOSS_CK = {}


def _depth_kl(lm, head):
    """Per-depth top-64 KL vs the trunk from the head's pre-mixer output, under mx.checkpoint: the [n, vocab] logits are
    recomputed in the backward pass instead of being kept for every depth (1 GB per depth at n = 1024)."""
    if id(head) not in _LOSS_CK:
        def f(hout, tp, tl):
            tp, tl = mx.stop_gradient(tp), mx.stop_gradient(tl)     # checkpoint differentiates every input otherwise
            lg = lm.lm_head(head.hyper_connection_mixer(hout)).astype(mx.float32)
            lq = mx.take_along_axis(lg, tp, -1) - mx.logsumexp(lg, -1, keepdims=True)
            return kl_top(lq, tl).mean()
        _LOSS_CK[id(head)] = mx.checkpoint(f)
    return _LOSS_CK[id(head)]


def native_depth_losses(lm, head, W, ids, top, tlp, K, wdecay=1.0, ck=False):
    """DeepSeek-style depths through MTPLX's own head (fresh MTP cache per depth pass: a constant rope offset leaves
    relative attention unchanged). Depth j weighs wdecay^(j-1) (~ the chance a chain reaches it; 1 = uniform)."""
    embed = lm.model.embed_tokens
    S = W.shape[1]
    feat, per, tot, wsum = W, [], 0.0, 0.0
    for j in range(1, K + 1):
        n = S - j - 1
        if n <= 8:
            break
        hout = head.fuse_and_run(feat[:, :n], embed(ids[:, j:j + n]), lm.make_mtp_cache())
        if ck:
            l = _depth_kl(lm, head)(hout, top[:, j:j + n], tlp[:, j:j + n])
        else:
            lg = lm.lm_head(head.hyper_connection_mixer(hout)).astype(mx.float32)
            lq = mx.take_along_axis(lg, top[:, j:j + n], -1) - mx.logsumexp(lg, -1, keepdims=True)
            l = kl_top(lq, tlp[:, j:j + n]).mean()
        w = wdecay ** (j - 1)
        per.append(l); tot = tot + w * l; wsum += w
        feat = hout
    return tot / max(wsum, 1e-9), per


def cmd_train(a):
    t0 = time.time()
    limit = a.time_limit or float(os.environ.get("TL_LIMIT", "1e9"))
    rt, lm, head = load(a.frac)
    tok = rt.tokenizer
    ck = os.path.join(TL, "ckpt", a.run + ".mtp125")
    state = {"turns": 0, "tokens": 0, "step": 0, "seed": a.seed, "native": a.native}
    if a.native:
        ntr = native_prepare(head)
        print(f"native: {ntr / 1e6:.1f}M trainable parameters inside MTPLX's head", flush=True)
        P = head
        src = ck if os.path.exists(ck + ".json") else (os.path.join(TL, "ckpt", a.init + ".mtp125") if a.init else None)
        if src:                                                # --init: warm start from another run, on the transcripts
            state = json.load(open(src + ".json")); head.load_weights(src + ".safetensors", strict=False)   # it has not seen
            if src != ck:
                state["init"] = a.init
                print(f"warm start from {a.init} ({state['turns']} turns, {state['tokens']} tokens)", flush=True)
        lossf = nn.value_and_grad(head, lambda hd, W, x, top, tlp: native_depth_losses(lm, hd, W, x, top, tlp, a.depth,
                                                                                         a.wdecay, a.ck_loss))
    else:
        P = Trainable(head)
        if os.path.exists(ck + ".json"):
            state = json.load(open(ck + ".json")); P.load_weights(ck + ".safetensors")
        P.freeze(); P.unfreeze()
        lossf = nn.value_and_grad(P, lambda Pm, W, x, top, tlp: depth_losses(lm, head, Pm, W, x, top, tlp, a.depth))
    opt = optim.Adam(learning_rate=lambda s: a.lr * mx.minimum(1.0, (s + 1) / 20.0))
    sm, render = samples(a.turns, state["seed"])
    gen = general_items() if a.mix_general > 0 else []
    print(f"train: {len(sm)} turns, resuming at turn {state['turns']}" + (f"; + {a.mix_general:g} x its tokens of the 125B's "
          f"own general text per turn ({len(gen)} items)" if gen else ""), flush=True)

    def train_seq(ids, hist):
        W, top, tlp = trunk_features(lm, ids)
        x = mx.array([ids])
        for s in range(0, len(ids) - a.depth - 2, a.window):
            e = min(len(ids), s + a.window + a.depth + 2)
            (loss, per), g = lossf(P, W[:, s:e], x[:, s:e], top[:, s:e], tlp[:, s:e])
            g, gn = optim.clip_grad_norm(g, 1.0)
            mx.eval(loss, gn)
            if not np.isfinite(float(loss)):
                continue
            opt.update(P, g); mx.eval(P.trainable_parameters(), opt.state)
            state["step"] += 1; state["tokens"] += e - s
            hist.append([float(v) for v in per])

    show = lambda m: " ".join(f"{v:.3f}" for v in m) if len(m) <= 4 else " ".join(f"d{j + 1} {m[j]:.3f}" for j in (0, 1, 3, 7, len(m) - 1) if j < len(m))
    while state["turns"] < len(sm) and time.time() - t0 < limit - 300:   # one turn takes up to ~190 s + the save
        ids = rendered_full(sm[state["turns"]], render, tok)
        t1 = time.time()
        hist = []
        train_seq(ids, hist)
        state["turns"] += 1
        m = np.mean([h for h in hist if len(h) == a.depth], 0) if hist else []
        line = f"turn {state['turns']} ({len(ids)} tokens, {time.time() - t1:.0f}s) KL {show(m)}"
        if gen:
            t2, used, gh = time.time(), 0, []
            while used < a.mix_general * len(ids):
                g_ids = gen[state.get("gen", 0) % len(gen)]; state["gen"] = state.get("gen", 0) + 1
                train_seq(g_ids, gh); used += len(g_ids)
            mg = np.mean([h for h in gh if len(h) == a.depth], 0) if gh else []
            line += f" | general {used} tokens ({time.time() - t2:.0f}s) KL {show(mg)}"
        print(line + f" | steps {state['step']} tokens {state['tokens']} {time.time() - t0:.0f}s", flush=True)
        if state["turns"] % 2 == 0:
            mx.save_safetensors(ck + ".safetensors", dict(tree_flatten(P.trainable_parameters()))); json.dump(state, open(ck + ".json", "w"))
    mx.save_safetensors(ck + ".safetensors", dict(tree_flatten(P.trainable_parameters()))); json.dump(state, open(ck + ".json", "w"))
    print(f"saved {a.run}: {state['turns']} turns, {state['tokens']} tokens", flush=True)


def install(head, P, dtype=None):
    """Write the trained fp32 copies back into the MTPLX head as Linear modules (bf16 for MTPLX's inference path; fp32
    for native training, where these modules are the trained parameters)."""
    dtype = dtype or mx.bfloat16
    layer = head.layers[0]
    att, moe = layer.self_attn, layer.mlp

    def lin(src):
        l = nn.Linear(src.weight.shape[1], src.weight.shape[0], bias=False)
        l.weight = src.weight.astype(dtype)
        return l
    head.fc_hidden, head.fc_embedding = lin(P.fc_hidden), lin(P.fc_embedding)
    att.q_proj, att.k_proj, att.v_proj, att.o_proj = lin(P.q_proj), lin(P.k_proj), lin(P.v_proj), lin(P.o_proj)
    if getattr(att, "qkv_fused", None) is not None:
        att.qkv_fused = None                                  # the fused q/k/v would keep the stock weights
    moe.gate = lin(P.router)
    moe.shared_expert = SharedMLP(lin(P.se_gate), lin(P.se_up), lin(P.se_down))   # replaces a fused gate+up module too
    if P.se_gatew is not None:
        moe.shared_expert_gate = lin(P.se_gatew)
    object.__setattr__(head, "_mtp_prepare_inputs_impl", head._prepare_inputs_eager)   # no compiled graph on old weights
    if not isinstance(layer.mlp, NativeMoE):
        layer.mlp = NativeMoE(layer.mlp)
    mx.eval(head.parameters())


class SharedMLP(nn.Module):
    def __init__(self, g, u, d):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = g, u, d

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


def main():
    limit_memory()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("--run", default="mtp125_d4")
    ap.add_argument("--frac", type=float, default=0.25); ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--turns", type=int, default=400); ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--time-limit", type=float, default=0)
    ap.add_argument("--native", action="store_true")
    ap.add_argument("--wdecay", type=float, default=1.0); ap.add_argument("--ck-loss", action="store_true")
    ap.add_argument("--mix-general", type=float, default=0.0); ap.add_argument("--init", default="")
    a = ap.parse_args()
    {"probe": cmd_probe, "check": cmd_check, "check2": cmd_check2, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
