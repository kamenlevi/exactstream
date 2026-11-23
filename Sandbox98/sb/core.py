"""Sandbox98 core (Sat 14 Nov 2026): a minute-scale copy of the 125B deep-drafter setup on a small local model.

- Target: Qwen3.5-2B (mlx-community 4-bit, on disk). Same family as the 125B (Qwen3.8-Flash-Next): hybrid Gated-DeltaNet +
  gated full attention, native one-layer MTP head. Frozen; only the drafter learns.
- Drafter: the target's NATIVE MTP head (fetched alone, dl/qwen35_2b_mtp), fp32 trainable copy: fc over
  [norm(emb(token)), norm(hidden)] -> one gated-attention decoder layer -> norm -> the target's tied lm_head (frozen, 4-bit).
- Chain = D-22's training-time test (TrainLab sticky/d22_ttt.py, as BigMoE dd_d22_ttt.py), ported without the 125B's
  hyper-connections. Anchor s (last committed pair): depth 1 runs (W[s], ids[s+1]) at position s and predicts ids[s+2];
  depth j >= 2 runs (the head's own depth-(j-1) output = the carried STAND-IN for W[s+j-1], token ids[s+j]) at position
  s+j-1, attends to the true pairs u <= s plus its own chain entries, and predicts ids[s+j+1].
- Loss = b5 / t30b's: per-depth KL to the target's top-32 (kl_top), accept-until-fail (Spec-AUF) support, depth weight
  wdecay^(j-1) x the depth's kept share. Read-out = dd_ttt.py's eval: per-depth bands over all valid cells of the
  teacher-forced chain, plus kept per 32-row check (walk from the answer's first anchor).
- REPLACE hooks (the day-1 harness): at depth j >= 2 the stand-in can be replaced by rows of any [S, d] array R (R = the
  TRUE W -> the oracle; R = a map's prediction -> a candidate), optionally only every K depths (BLOCK).
W = the hidden the head reads = the target's final hidden AFTER its final norm (vLLM's Qwen3-Next / Qwen3.5 MTP input;
checked against the pre-norm stream on depth-1 accuracy, see results/headcheck.json)."""
import glob, json, os, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn

H = os.path.expanduser
ROOT = H("~/Sandbox98")
MODEL = sorted(glob.glob(H("~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-2B-MLX-4bit/snapshots/*")))[0]
MTPW = os.path.join(ROOT, "dl/qwen35_2b_mtp/mtp.safetensors")
DATA = os.path.join(ROOT, "data")
CKPT = os.path.join(ROOT, "ckpt")
RES = os.path.join(ROOT, "results")
PAUSE = H("~/.local/share/quiet/PAUSE")
TAPS = (2, 7, 12, 17, 22)            # residual after these layers (0-based) of 24 ~ the 125B's taps {5,15,25,35,45} of 48
BANDS = [(0, 0), (1, 2), (3, 6), (7, 10), (11, 14), (15, 22), (23, 30), (7, 14)]   # 7-14: the fast (depth-15) screens' band
DEEP_BANDS = [(31, 38), (39, 46), (47, 54), (55, 62)]   # read-outs past 31 deep (Leap's 48-64-row question)
NEG = -1e9
CFG = {"wsrc": "post", "order": "ew"}  # W = post-norm final hidden; fc input order [emb, hidden] (vLLM). headcheck decides


def limit_memory(gb=None):
    mx.set_cache_limit(int(float(os.environ.get("SB_CACHE_GB", "1")) * 2 ** 30))
    mx.set_memory_limit(int(float(gb or os.environ.get("SB_MEM_GB", "8")) * 2 ** 30))


def wait_pause():
    while os.path.exists(PAUSE):
        time.sleep(1)


# ---------------------------------------------------------------- the target
_T = {}
def load_target():
    if "m" not in _T:
        from mlx_lm import load
        m, tok = load(MODEL)
        m.eval()
        m.freeze()
        _T["m"], _T["tok"] = m, tok
    return _T["m"], _T["tok"]


def tm():
    return load_target()[0].language_model.model


def drop_trunk():
    """training and read-outs need only the embedding, the final norm and the lm_head: free the 24 trunk layers (~1.3 GB)
    so a run stays under the by-day ~5 GB line. (Capture needs the trunk: never call it there.)"""
    import gc
    tm().layers = []
    gc.collect(); mx.clear_cache()


def target_states(ids, taps=TAPS, logits=True):
    """One full-sequence forward of the frozen target (no cache). ids [S] ->
    res [S, d] (final residual, pre-norm), post [S, d] (after the final norm), {L: [S, d]} taps, logits [S, V] (fp32)."""
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    t = tm()
    x = t.embed_tokens(mx.array(np.asarray(ids, np.int32))[None])
    fa, ssm = create_attention_mask(x, None), create_ssm_mask(x, None)
    tp = {}
    for i, layer in enumerate(t.layers):
        x = layer(x, mask=ssm if layer.is_linear else fa, cache=None)
        if i in taps:
            tp[i] = x[0]
    post = t.norm(x)
    return x[0], post[0], tp, (t.embed_tokens.as_linear(post)[0].astype(mx.float32) if logits else None)


def final_norm(res):
    """post-norm W from a stored pre-norm residual [.., d] (what the head reads when CFG wsrc = post)"""
    return tm().norm(res.astype(mx.bfloat16))


def lm_logits(hn):
    """the target's tied lm_head (frozen, 4-bit), exactly as the target applies it, on normed rows [.., d] -> fp32 logits
    (read-outs; its forward runs at ~13 TFLOPS here)"""
    return tm().embed_tokens.as_linear(hn.astype(mx.bfloat16)).astype(mx.float32)


_LMW = {}
def lm_logits_train(hn):
    """the same lm_head as a bf16 dequantized copy (1 GB), for training: the 4-bit path's backward to its input is ~5x
    slower (bench 18:4x: 2048 rows fwd+bwd 1247 ms 4-bit vs 260 ms bf16)"""
    if "w" not in _LMW:
        E = tm().embed_tokens
        _LMW["w"] = mx.dequantize(E.weight, E.scales, E.biases, group_size=E.group_size, bits=E.bits).astype(mx.bfloat16)
        mx.eval(_LMW["w"])
    return (hn.astype(mx.bfloat16) @ _LMW["w"].T).astype(mx.float32)


def embed(ids):
    return tm().embed_tokens(ids).astype(mx.float32)


# ---------------------------------------------------------------- the drafter (native MTP head, fp32)
class Drafter(nn.Module):
    def __init__(self, d=2048, nh=8, nkv=2, hd=256, ffn=6144, eps=1e-6, rope_dims=64, rope_base=1e7):
        super().__init__()
        self.pre_fc_norm_hidden = nn.RMSNorm(d, eps=eps)
        self.pre_fc_norm_embedding = nn.RMSNorm(d, eps=eps)
        self.fc = nn.Linear(2 * d, d, bias=False)
        self.input_layernorm = nn.RMSNorm(d, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(d, eps=eps)
        self.q_proj = nn.Linear(d, nh * hd * 2, bias=False)
        self.k_proj = nn.Linear(d, nkv * hd, bias=False)
        self.v_proj = nn.Linear(d, nkv * hd, bias=False)
        self.o_proj = nn.Linear(nh * hd, d, bias=False)
        self.q_norm = nn.RMSNorm(hd, eps=eps)
        self.k_norm = nn.RMSNorm(hd, eps=eps)
        self.gate_proj = nn.Linear(d, ffn, bias=False)
        self.up_proj = nn.Linear(d, ffn, bias=False)
        self.down_proj = nn.Linear(ffn, d, bias=False)
        self.norm = nn.RMSNorm(d, eps=eps)
        self.nh, self.nkv, self.hd, self.d = nh, nkv, hd, d
        self.scale = hd ** -0.5
        self.rope = nn.RoPE(rope_dims, traditional=False, base=rope_base)

    def mlp(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


NATIVE = {"mtp.fc.weight": "fc.weight", "mtp.norm.weight": "norm.weight",
          "mtp.pre_fc_norm_hidden.weight": "pre_fc_norm_hidden.weight",
          "mtp.pre_fc_norm_embedding.weight": "pre_fc_norm_embedding.weight"}
for _k in ("input_layernorm", "post_attention_layernorm"):
    NATIVE[f"mtp.layers.0.{_k}.weight"] = f"{_k}.weight"
for _k in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
    NATIVE[f"mtp.layers.0.self_attn.{_k}.weight"] = f"{_k}.weight"
for _k in ("gate_proj", "up_proj", "down_proj"):
    NATIVE[f"mtp.layers.0.mlp.{_k}.weight"] = f"{_k}.weight"


def load_drafter(ckpt=None, strict=True):
    """ckpt None / 'native': the native head (zero-centred norms shifted +1, as mlx_lm's sanitize does for the trunk);
    else a saved checkpoint name in ckpt/ (or a path)"""
    D = Drafter()
    if ckpt in (None, "native"):
        w = mx.load(MTPW)
        out = {}
        for k, v in w.items():
            v = v.astype(mx.float32)
            if v.ndim == 1:                  # every 1-D mtp.* tensor is a zero-centred RMSNorm gain (incl. pre_fc_norm_*)
                v = v + 1.0
            out[NATIVE[k]] = v
        D.load_weights(list(out.items()))
    else:
        p = ckpt if os.path.exists(ckpt) else os.path.join(CKPT, ckpt + ".safetensors")
        D.load_weights(p, strict=strict)   # strict=False: a lever's extra modules load after its setup()
    mx.eval(D.parameters())
    return D


# ---------------------------------------------------------------- the chain (D-22's TTT, no hyper-connections)
def fuse(D, Win, emb):
    """the head's input for pairs (hidden Win [1,n,d], token embedding emb [1,n,d])"""
    hn = D.pre_fc_norm_hidden(Win.astype(mx.float32))
    en = D.pre_fc_norm_embedding(emb)
    return D.fc(mx.concatenate([en, hn] if CFG["order"] == "ew" else [hn, en], axis=-1))


def rope_rows(D, x, pos):
    """nn.RoPE(64, traditional=False, base 1e7) with a position per row: x [1,H,n,hd], pos [n] (mx int)"""
    half = D.rope.dims // 2
    inv = D.rope.base ** (-mx.arange(0, D.rope.dims, 2, dtype=mx.float32) / D.rope.dims)
    ang = pos.astype(mx.float32)[:, None] * inv[None, :]
    c, s = mx.cos(ang).astype(x.dtype), mx.sin(ang).astype(x.dtype)
    x1, x2 = x[..., :half], x[..., half:2 * half]
    return mx.concatenate([x1 * c - x2 * s, x1 * s + x2 * c, x[..., 2 * half:]], -1)


def qkv(D, x, offset=None, pos=None):
    """rows x [1,n,d] (already input-normed) at positions offset + arange(n), or per-row positions pos [n] ->
    q [1,H,n,hd], gate [1,n,H*hd], k, v [1,KV,n,hd]"""
    B, n, _ = x.shape
    q, gate = mx.split(D.q_proj(x).reshape(B, n, D.nh, 2 * D.hd), 2, axis=-1)
    k = D.k_proj(x).reshape(B, n, D.nkv, D.hd)
    v = D.v_proj(x).reshape(B, n, D.nkv, D.hd)
    q, k = D.q_norm(q).transpose(0, 2, 1, 3), D.k_norm(k).transpose(0, 2, 1, 3)
    if pos is None:
        q, k = D.rope(q, offset=offset), D.rope(k, offset=offset)
    else:
        q, k = rope_rows(D, q, pos), rope_rows(D, k, pos)
    return q, gate.reshape(B, n, -1), k, v.transpose(0, 2, 1, 3)


def ttt_attn(scale, q, K1, V1, ck, cv, anc):
    """d22's: query row i (anchor anc[i]) attends to the true pairs u <= anc[i] (K1 / V1 [1,KV,S1,hd]) and to its own
    chain entries (ck / cv: one [1,KV,n,hd] per depth 2..j, same rows). q [1,H,n,hd] -> [1,H,n,hd]"""
    _, Hh, n, hd = q.shape
    KV, S1 = K1.shape[1], K1.shape[2]
    r = Hh // KV
    qg = q.reshape(1, KV, r, n, hd)
    st = (qg @ K1[:, :, None].swapaxes(-1, -2)) * scale                         # [1,KV,r,n,S1]
    ok = mx.arange(S1)[None, :] <= anc[:, None]
    if WINDOW["n"] > 0:
        ok = ok & (mx.arange(S1)[None, :] > anc[:, None] - WINDOW["n"])
    st = mx.where(ok, st, NEG)
    sc = mx.stack([(qg * k[:, :, None]).sum(-1) for k in ck], -1) * scale     # [1,KV,r,n,J]
    m = mx.stop_gradient(mx.maximum(st.max(-1, keepdims=True), sc.max(-1, keepdims=True)))
    et, ec = mx.exp(st - m), mx.exp(sc - m)
    o = et @ V1[:, :, None]
    for i, v in enumerate(cv):
        o = o + ec[..., i:i + 1] * v[:, :, None]
    o = o / (et.sum(-1, keepdims=True) + ec.sum(-1, keepdims=True))
    return o.reshape(1, Hh, n, hd)


def block(D, x, blk, xa=None):
    """the decoder layer around an attention output blk = o_proj(attn * sigmoid(gate)): x + blk, then (PREFIX EYES, when a
    hook gives xa: + the cross-attention xa(h) into the trunk's prefix K / V) + MLP. xa None = exactly as before."""
    h = x + blk
    if xa is not None:
        h = h + xa(h)
    return h + D.mlp(D.post_attention_layernorm(h))


class Layer(nn.Module):
    """one more gated-attention decoder layer for a deeper drafter (the native layer's parts, same names)"""
    def __init__(self, d=2048, nh=8, nkv=2, hd=256, ffn=6144, eps=1e-6, rope_dims=64, rope_base=1e7):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(d, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(d, eps=eps)
        self.q_proj = nn.Linear(d, nh * hd * 2, bias=False)
        self.k_proj = nn.Linear(d, nkv * hd, bias=False)
        self.v_proj = nn.Linear(d, nkv * hd, bias=False)
        self.o_proj = nn.Linear(nh * hd, d, bias=False)
        self.q_norm = nn.RMSNorm(hd, eps=eps)
        self.k_norm = nn.RMSNorm(hd, eps=eps)
        self.gate_proj = nn.Linear(d, ffn, bias=False)
        self.up_proj = nn.Linear(d, ffn, bias=False)
        self.down_proj = nn.Linear(ffn, d, bias=False)
        self.nh, self.nkv, self.hd, self.d = nh, nkv, hd, d
        self.scale = hd ** -0.5
        self.rope = nn.RoPE(rope_dims, traditional=False, base=rope_base)

    def mlp(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


def layers_of(D):
    """the drafter's decoder layers in order: the native layer (D itself), then any a lever stacked (D._stack)"""
    return getattr(D, "_stack", None) or [D]


def _win_mask(n1, a0):
    """bool [n1 - a0, n1]: query row i (pair a0 + i) sees pairs u with a0 + i - WINDOW < u <= a0 + i (WINDOW["n"] > 0)"""
    s = a0 + mx.arange(n1 - a0)[:, None]; u = mx.arange(n1)[None, :]
    return (u <= s) & (u > s - WINDOW["n"])


def fut_in(D, j, anc):
    """BREAK98 Screen 1 (FAR's FUTURE-STATE ROWS; sb/futstate.py, levers/fut*.py mark their standin hook with _fut): the
    added head input at depth j for rows with anchors anc, i.e. P(norm(the target's state at placeholder offset j under the
    anchor)), [1,n,d], or None (no hook, or j past the chain). None leaves every path exactly as before."""
    fu = getattr(REPL["fn"], "_fut", None)
    return None if fu is None else fu(D, j, anc)


def kvx_in(D, j, anc):
    """BREAK98c PREFIX EYES (FAR's IDEAS_1117c; sb/kvx.py, levers/kvx*.py mark their standin hook with _kvx): a function
    h -> the cross-attention output for rows h [1,n,d] (the residual after the head's self-attention; anchors anc, depth
    j) into the frozen trunk's cached K / V of the verified prefix (positions <= the anchor), or None (no hook: every
    path exactly as before). Native one-layer head, full attention only."""
    fu = getattr(REPL["fn"], "_kvx", None)
    if fu is None:
        return None
    assert len(layers_of(D)) == 1 and WINDOW["n"] == 0, "PREFIX EYES: native one-layer head, full attention only"
    return lambda h: fu(D, j, h, anc)


def depth1(D, W, ida, a0):
    """depth 1 over every true pair u = 0..S-2 (keys / values) with queries from anchors a0..S-2 -> h [1,n,d], K1, V1.
    K1 / V1 stack every layer's keys / values on a leading axis: [L, 1, KV, S1, hd] (L = 1 for the native head). A
    layer below the last runs over every pair (the next layer's keys need it); the last only over the anchors.
    With a future-state hook (fut_in), the anchors' QUERY rows (and their residual) add the depth-1 future input, while
    every pair's keys / values stay future-free (the head's cache of true pairs does not depend on where chains hung)."""
    S = W.shape[1]
    n1 = S - 1
    x = fuse(D, W[:, :n1], embed(ida[None, 1:n1 + 1]))
    Ls = layers_of(D)
    fx = fut_in(D, 1, a0 + mx.arange(n1 - a0))
    if fx is not None:
        assert len(Ls) == 1 and WINDOW["n"] == 0, "future-state input: native one-layer head, full attention only"
    Ks, Vs = [], []
    for li, L in enumerate(Ls):
        q, gate, K, V = qkv(L, L.input_layernorm(x), offset=0)
        Ks.append(K); Vs.append(V)
        if li < len(Ls) - 1:
            o = mx.fast.scaled_dot_product_attention(q, K, V, scale=L.scale, mask=_win_mask(n1, 0) if WINDOW["n"] else "causal")
            blk = L.o_proj(o.transpose(0, 2, 1, 3).reshape(1, n1, -1) * mx.sigmoid(gate))
            x = block(L, x, blk)
        else:
            xr = x[:, a0:]
            if fx is not None:                            # same shapes as above, so a zero input is bitwise today's
                xq = mx.concatenate([x[:, :a0], xr + fx], 1)
                q, gate, _, _ = qkv(L, L.input_layernorm(xq), offset=0)
                xr = xq[:, a0:]
            o = mx.fast.scaled_dot_product_attention(q[:, :, a0:], K, V, scale=L.scale,
                                                     mask=_win_mask(n1, a0) if WINDOW["n"] else "causal")   # bottom-right aligned
            n = n1 - a0
            blk = L.o_proj(o.transpose(0, 2, 1, 3).reshape(1, n, -1) * mx.sigmoid(gate[:, a0:]))
            x = block(L, xr, blk, kvx_in(D, 1, a0 + mx.arange(n)))        # PREFIX EYES: the anchors' rows only
    return x, mx.stack(Ks), mx.stack(Vs)


def deeper(D, h, ida, anc, j, K1, V1, ck, cv):
    """depth j >= 2 for rows with anchors anc [n] (mx int) and carried stand-ins h [1,n,d]: input (h, ids[anc + j]) at
    positions anc + j - 1, through every layer; appends the rows' chain entries (all layers stacked, [L, 1, KV, n, hd])
    to ck / cv (aligned to the same rows). A future-state hook adds its depth-j input to the fused row (fut_in)."""
    x = fuse(D, h, embed(ida[anc + j])[None])
    fx = fut_in(D, j, anc)
    if fx is not None:
        x = x + fx
    xa = kvx_in(D, j, anc)                                # PREFIX EYES (None without the hook)
    ks, vs = [], []
    for li, L in enumerate(layers_of(D)):
        q, gate, k, v = qkv(L, L.input_layernorm(x), pos=anc + j - 1)
        ks.append(k); vs.append(v)
        o = ttt_attn(L.scale, q, K1[li], V1[li], [c[li] for c in ck] + [k], [c[li] for c in cv] + [v], anc)
        blk = L.o_proj(o.transpose(0, 2, 1, 3).reshape(1, h.shape[1], -1) * mx.sigmoid(gate))
        x = block(L, x, blk, xa)
    ck.append(mx.stack(ks)); cv.append(mx.stack(vs))
    return x


REPL = {"R": None, "K": 1, "fn": None}   # REPLACE hook: R [1,S,d] rows replace the stand-in at depth j >= 2
CUR = {"f": None, "train": False, "x": None}   # the current capture file (levers may load its taps), training or read-out, window
WINDOW = {"n": 0}                                # eval-only: cap the head's attention to the last n true pairs (0 = all)


def standin(j, h, anc):
    """the stand-in depth j reads: the carried one, or REPL's replacement rows / transform (anc = the rows' anchors)"""
    R, K = REPL["R"], REPL["K"]
    if R is not None and (j - 1) % K == 0:
        h = R[:, anc + j - 1]
    if REPL["fn"] is not None:
        h = REPL["fn"](j, h, anc)
    return h


def chain(D, W, ida, Dp, on_depth, a0=0):
    """the read-out chain, every anchor at every depth (no compaction). W [1,S,d] (the hidden the head reads at every
    committed pair), ida [S] (mx int32). Anchors s = a0..S-j-1 run depth j; on_depth(j, h, anc) gets h [1,n,d] (pre-norm),
    row i = anchor anc[i] = a0 + i, which predicts ids[anc + j + 1]."""
    S = W.shape[1]
    h, K1, V1 = depth1(D, W, ida, a0)
    on_depth(1, h, a0 + mx.arange(h.shape[1]))
    ck, cv = [], []
    for j in range(2, Dp + 1):
        n = S - j - a0
        if n <= 4:                                        # dd_ttt / d22's chain stops here too
            break
        anc = a0 + mx.arange(n)
        ck, cv = [c[..., :n, :] for c in ck], [c[..., :n, :] for c in cv]
        h = deeper(D, standin(j, h[:, :n], anc), ida, anc, j, K1, V1, ck, cv)
        on_depth(j, h, anc)


# ---------------------------------------------------------------- loss (b5 / t30b's AUF KL), row-compacted
def kl_top(lq, tlp):
    """mtp125.kl_top: KL(target || draft) on the target's top-k plus one 'rest' bucket"""
    p, q = mx.exp(tlp), mx.exp(lq)
    pt = mx.maximum(1 - p.sum(-1), 1e-12); qt = mx.maximum(1 - q.sum(-1), 1e-12)
    return (p * (tlp - lq)).sum(-1) + pt * (mx.log(pt) - mx.log(qt))


_CK = {}
def depth_kl_auf():
    """checkpointed: the frozen lm_head (bf16 copy) on normed rows -> (masked mean KL, argmax hits) (d22's depth_kl_auf)"""
    if "fa" not in _CK:
        def f(hn, tk, lp, mask, tgt):
            tk, lp, mask = mx.stop_gradient(tk), mx.stop_gradient(lp), mx.stop_gradient(mask)
            lg = lm_logits_train(hn)[0]
            lq = mx.take_along_axis(lg, tk, -1) - mx.logsumexp(lg, -1, keepdims=True)
            hit = mx.stop_gradient((mx.argmax(lg, -1) == tgt).astype(mx.float32))
            return (kl_top(lq, lp) * mask).sum() / mx.maximum(mask.sum(), 1.0), hit
        _CK["fa"] = mx.checkpoint(f)
    return _CK["fa"]


def ttt_loss(D, x, Dp=30, wdecay=0.97, extra=None):
    """t30b's loss: per-depth KL to the target's top-32 over depths 1..Dp, accept-until-fail support (a row counts at depth
    j only while its argmax was right at every depth < j; the failing depth itself counts), depth weight wdecay^(j-1) x
    the depth's kept share of depth 1's rows. Rows that failed are DROPPED from the chain (identical loss: each anchor's
    chain only attends to its own entries and the true pairs; ~3-6x less lm_head work). x = window_at(...).
    extra(j, h, anc, m, x) -> an added per-depth term (weighted like the KL). -> (loss, kl, extra)"""
    W, ids = x["W"], x["ids"]
    S, a0, r = len(ids), x["a0"], x["r"]
    T = x["tk"].shape[0]                                  # rows with top-32 (= S - 1)
    tk, lp, asst = mx.array(x["tk"]), mx.array(x["lp"]), np.asarray(x["asst"])
    ida = mx.array(ids)
    tot, wsum, ex, m1 = mx.array(0.0), 0.0, mx.array(0.0), None
    h, K1, V1 = depth1(D, W, ida, a0)
    anc_np = np.arange(a0, a0 + r)                        # the slice's anchors (the window holds their whole chains)
    h = h[:, :r]
    ck, cv = [], []
    for j in range(1, Dp + 1):
        if j >= 2:
            anc = mx.array(anc_np)
            h = deeper(D, standin(j, h, anc), ida, anc, j, K1, V1, ck, cv)
        ok = (anc_np + j + 1 <= S - 1) & (anc_np + j <= T - 1)
        tgt_i = np.minimum(anc_np + j + 1, S - 1); row_i = np.minimum(anc_np + j, T - 1)
        m_np = ok.astype(np.float32) * asst[tgt_i].astype(np.float32)
        m = mx.array(m_np)
        l, hit = depth_kl_auf()(D.norm(h), tk[mx.array(row_i)], lp[mx.array(row_i)], m, ida[mx.array(tgt_i)])
        cnt = float(m_np.sum())
        if j == 1:
            m1 = max(cnt, 1.0)
        w = wdecay ** (j - 1) * cnt / m1
        tot = tot + w * l; wsum += w
        if extra is not None:
            ex = ex + w * extra(j, h, mx.array(anc_np), m, x)
        keep = np.flatnonzero((np.array(hit) > 0) & (anc_np + j + 1 <= S - 1))   # d22: alive = hit (any target)
        if len(keep) == 0 or j == Dp:
            break
        if len(keep) < len(anc_np):                       # accept-until-fail: failed rows leave the chain
            kk = mx.array(keep)
            h = h[:, kk]; ck = [c[..., kk, :] for c in ck]; cv = [c[..., kk, :] for c in cv]
            anc_np = anc_np[keep]
    wsum = max(wsum, 1e-6)
    return (tot + ex) / wsum, tot / wsum, ex / wsum


def _kl_hit(hn, tk, lp, mask, tgt):
    lg = lm_logits_train(hn)[0]
    lq = mx.take_along_axis(lg, tk, -1) - mx.logsumexp(lg, -1, keepdims=True)
    hit = mx.stop_gradient((mx.argmax(lg, -1) == tgt).astype(mx.float32))
    return (kl_top(lq, lp) * mask).sum() / mx.maximum(mask.sum(), 1.0), hit


def _kl_hit_rank(hn, tk, lp, mask, tgt):
    """_kl_hit plus the true token's rank under the head: the number of tokens with a strictly higher logit (0 = top,
    ties counted for the true token). Used only by the tree-aware mask (levers/treemask*.py)."""
    lg = lm_logits_train(hn)[0]
    lq = mx.take_along_axis(lg, tk, -1) - mx.logsumexp(lg, -1, keepdims=True)
    hit = mx.stop_gradient((mx.argmax(lg, -1) == tgt).astype(mx.float32))
    rank = mx.stop_gradient((lg > mx.take_along_axis(lg, tgt[:, None], -1)).sum(-1))
    return (kl_top(lq, lp) * mask).sum() / mx.maximum(mask.sum(), 1.0), hit, rank


_KLG = {}
def ttt_step(D, x, Dp=30, wdecay=0.97, extra=None, aux=None):
    """ttt_loss's value AND its exact gradient for D's trainable parameters with memory flat in depth (the one-graph
    backward held a full weight-sized gradient per depth for every head matrix, ~242 MB x 30 depths, plus several depths'
    248K-wide logits: 5.8-7.8 GB peaks):
      pass 1, no autograd: the chain (row-compacted). At each depth: KL_j, the hits and g_j = dKL_j / d(norm(h_j)),
              evaluated at once; the rows alive at each depth and every depth's h / k / v (absolute row slots) are kept.
      pass 2, a manual backward through the chain, deepest depth first: each depth is recomputed inside mx.grad of
              c_j <norm(h_j), g_j> + <outputs, their cotangents from deeper depths>, giving that depth's parameter
              gradient (added to a running sum) and the cotangents of its inputs (h_{j-1}, the earlier depths' k / v,
              and depth 1's K1 / V1); everything is evaluated per depth. Checked equal to the one-graph gradient
              (results/twopass_check.txt).
    extra(j, h, anc, m, x): a lever's added per-depth term (weighted like the KL), autograd in pass 2.
    aux(j, anc_np, ctx) -> (tk2, lp2, lam) or None: a second top-32 target for the same rows (e.g. a teacher), added as
    lam x its kl_top in pass 1 (its gradient joins g_j; no vocab-wide tensor reaches autograd). ctx = {D, h1 (depth-1
    output of every anchor a0..S-2), a0, x}. -> (loss, kl, extra, grads)"""
    from mlx.utils import tree_map
    if "vg" not in _KLG:
        _KLG["vg"] = mx.value_and_grad(_kl_hit)
    W, ids = x["W"], x["ids"]
    S, a0, r = len(ids), x["a0"], x["r"]
    T = x["tk"].shape[0]
    tk, lp, asst = mx.array(x["tk"]), mx.array(x["lp"]), np.asarray(x["asst"])
    ida = mx.array(ids)
    d, KV, hd = D.d, D.nkv, D.hd
    CUR["f"], CUR["train"], CUR["x"] = x.get("f"), True, x
    # ---- pass 1
    rec, m1 = [], None
    h, K1, V1 = depth1(D, W, ida, a0)
    ctx = {"D": D, "h1": h, "a0": a0, "x": x}
    h = h[:, :r]
    idx = np.arange(r)
    auxv = 0.0
    ck, cv = [], []
    # FAR's WALK-MIX (THINK_1116c; levers/walkmix*.py mark their standin hook with _walkmix_w = w): a row that missed
    # stays in the chain, its later depths at KL weight w (survivors 1, the failing depth itself 1). The depth weight
    # uses the weighted row count, so w = 0 is today's accept-until-fail loss and w = 1 the 'allrows' loss.
    wmix = getattr(REPL["fn"], "_walkmix_w", None)
    # TREE-AWARE MASK (NewIdeas 20:3x, the follow-up to WALK-MIX; levers/treemask*.py mark their hook with _treemask_k
    # = k and _treemask_w = w): a row that missed stays in the chain only when its true token was in the head's top-k at
    # that depth (where a tree still walks on), at KL weight w from the next depth on; a miss outside the top-k drops the
    # row as today. k = 1 is today's accept-until-fail loss exactly (the hit decides, never the rank).
    tmk = getattr(REPL["fn"], "_treemask_k", None)
    tmw = float(getattr(REPL["fn"], "_treemask_w", 1.0))
    if tmk is not None and "vgr" not in _KLG:
        _KLG["vgr"] = mx.value_and_grad(_kl_hit_rank)
    dead = np.zeros(r, bool)
    for j in range(1, Dp + 1):
        anc_np = a0 + idx
        if j >= 2:
            anc = mx.array(anc_np)
            h = deeper(D, standin(j, h, anc), ida, anc, j, K1, V1, ck, cv)
        ok = (anc_np + j + 1 <= S - 1) & (anc_np + j <= T - 1)
        tgt_i = np.minimum(anc_np + j + 1, S - 1); row_i = np.minimum(anc_np + j, T - 1)
        m_np = ok.astype(np.float32) * asst[tgt_i].astype(np.float32)
        if wmix is not None:
            m_np = m_np * np.where(dead[idx], np.float32(wmix), np.float32(1.0))
        elif tmk is not None:
            m_np = m_np * np.where(dead[idx], np.float32(tmw), np.float32(1.0))
        m = mx.array(m_np)
        hn_ = mx.stop_gradient(D.norm(h))
        if tmk is not None:
            (kl_j, hit, rk), g = _KLG["vgr"](hn_, tk[mx.array(row_i)], lp[mx.array(row_i)], m, ida[mx.array(tgt_i)])
        else:
            (kl_j, hit), g = _KLG["vg"](hn_, tk[mx.array(row_i)], lp[mx.array(row_i)], m, ida[mx.array(tgt_i)])
        a2 = aux(j, anc_np, ctx) if aux is not None else None
        if a2 is not None:
            tk2, lp2, lam = a2
            (kl2, _), g2 = _KLG["vg"](hn_, tk2, lp2, m, ida[mx.array(tgt_i)])
            g = g + lam * g2
            mx.eval(kl2, g)
            kl2f = lam * float(kl2)
        else:
            kl2f = 0.0
        ia = mx.array(idx)
        h_abs = mx.zeros((1, r, d), mx.float32); h_abs[:, ia] = h.astype(mx.float32)
        e = {"idx": ia, "g": g, "m": m, "h_abs": h_abs}
        if j >= 2:
            k_abs = mx.zeros(ck[-1].shape[:-2] + (r, hd), ck[-1].dtype); k_abs[..., ia, :] = ck[-1]
            v_abs = mx.zeros(cv[-1].shape[:-2] + (r, hd), cv[-1].dtype); v_abs[..., ia, :] = cv[-1]
            e["k_abs"], e["v_abs"] = k_abs, v_abs
        mx.eval(kl_j, hit, g, h, h_abs, *([e["k_abs"], e["v_abs"]] if j >= 2 else []), *([rk] if tmk is not None else []))
        cnt = float(m_np.sum())
        if j == 1:
            m1 = max(cnt, 1.0)
        e["w"], e["kl"], e["aux"] = wdecay ** (j - 1) * cnt / m1, float(kl_j), kl2f
        rec.append(e)
        if wmix is not None:                              # WALK-MIX: misses stay (at weight w from the next depth on)
            dead[idx[np.array(hit) == 0]] = True
            keep = np.flatnonzero(anc_np + j + 1 <= S - 1)
        elif tmk is not None:                             # TREE-AWARE MASK: a miss stays only inside the top-k
            hit_np = np.array(hit) > 0
            near = (np.array(rk) < tmk) if tmk > 1 else np.zeros(len(idx), bool)
            dead[idx[~hit_np]] = True
            keep = np.flatnonzero((hit_np | near) & (anc_np + j + 1 <= S - 1))
        else:
            keep = np.flatnonzero((np.array(hit) > 0) & (anc_np + j + 1 <= S - 1))   # d22: alive = hit (any target)
        if len(keep) == 0 or j == Dp:
            break
        if len(keep) < len(idx):
            kk = mx.array(keep)
            h = h[:, kk]; ck = [c[..., kk, :] for c in ck]; cv = [c[..., kk, :] for c in cv]
            idx = idx[keep]
    mx.eval(K1, V1)
    wsum = max(sum(e["w"] for e in rec), 1e-6)
    kl = sum(e["w"] * e["kl"] for e in rec) / wsum
    auxv = sum(e["w"] * e["aux"] for e in rec) / wsum
    # ---- pass 2: the manual backward, deepest depth first
    P0 = D.trainable_parameters()
    J = len(rec)
    exv = [0.0]

    def Lj(params, h_prev, K1_, V1_, ks, vs, j, ct_h, ct_k, ct_v):
        D.update(params)
        e = rec[j - 1]
        ia = e["idx"]; anc = a0 + ia
        h_in = standin(j, h_prev[:, ia], anc)
        ck_ = [k[..., ia, :] for k in ks]; cv_ = [v[..., ia, :] for v in vs]
        hj = deeper(D, h_in, ida, anc, j, K1_, V1_, ck_, cv_)
        c = e["w"] / wsum
        out = c * (D.norm(hj) * e["g"]).sum()
        if extra is not None:
            ex = c * extra(j, hj, anc, e["m"], x)
            exv[0] = exv[0] + ex
            out = out + ex
        out = out + (hj * ct_h[:, ia]).sum()
        if ct_k is not None:
            out = out + (ck_[-1] * ct_k[..., ia, :]).sum() + (cv_[-1] * ct_v[..., ia, :]).sum()
        return out

    def L1(params, ct_h, ct_K1, ct_V1):
        D.update(params)
        h1, K1_, V1_ = depth1(D, W, ida, a0)
        h1 = h1[:, :r]
        e = rec[0]
        c = e["w"] / wsum
        out = c * (D.norm(h1) * e["g"]).sum()
        if extra is not None:
            ex = c * extra(1, h1, a0 + e["idx"], e["m"], x)
            exv[0] = exv[0] + ex
            out = out + ex
        return out + (h1 * ct_h).sum() + (K1_ * ct_K1).sum() + (V1_ * ct_V1).sum()

    gLj = mx.grad(Lj, argnums=(0, 1, 2, 3, 4, 5))
    gL1 = mx.grad(L1, argnums=0)
    ct_h = mx.zeros((1, r, d), mx.float32)
    ct_K1, ct_V1 = mx.zeros(K1.shape, K1.dtype), mx.zeros(V1.shape, V1.dtype)
    ct_k = {j: mx.zeros(rec[j - 1]["k_abs"].shape, rec[j - 1]["k_abs"].dtype) for j in range(2, J + 1)}
    ct_v = {j: mx.zeros(rec[j - 1]["v_abs"].shape, rec[j - 1]["v_abs"].dtype) for j in range(2, J + 1)}
    grads = None
    for j in range(J, 1, -1):
        ks = [rec[i - 1]["k_abs"] for i in range(2, j)]; vs = [rec[i - 1]["v_abs"] for i in range(2, j)]
        dp, dh, dK1, dV1, dks, dvs = gLj(P0, rec[j - 2]["h_abs"], K1, V1, ks, vs, j, ct_h, ct_k[j], ct_v[j])
        grads = dp if grads is None else tree_map(lambda a, b: a + b, grads, dp)
        ct_h = dh
        ct_K1, ct_V1 = ct_K1 + dK1, ct_V1 + dV1
        for i, (a_, b_) in enumerate(zip(dks, dvs)):
            ct_k[i + 2] = ct_k[i + 2] + a_; ct_v[i + 2] = ct_v[i + 2] + b_
        mx.eval(grads, ct_h, ct_K1, ct_V1, *[ct_k[i] for i in range(2, j)], *[ct_v[i] for i in range(2, j)], exv[0])
    dp = gL1(P0, ct_h, ct_K1, ct_V1)
    grads = dp if grads is None else tree_map(lambda a, b: a + b, grads, dp)
    mx.eval(grads)
    D.update(P0)
    exf = (float(exv[0]) if isinstance(exv[0], mx.array) else float(exv[0])) + auxv
    return kl + exf, kl, exf, grads


# ---------------------------------------------------------------- data
def files(st):
    return sorted(f for f in glob.glob(os.path.join(DATA, st, "*.npz")) if not f.endswith(".taps.npz"))


def load_W(z):
    """the hidden the head reads, [1,S,d] bf16, from a capture file's pre-norm residual"""
    res = mx.array(z["res"])
    return (final_norm(res) if CFG["wsrc"] == "post" else res.astype(mx.bfloat16))[None]


def window_at(f, b, r, Dp):
    """a training window: anchors b .. b + r - 1 (the whole prefix as context), cropped at b + r + Dp + 2"""
    z = np.load(f)
    ids = np.asarray(z["ids"], np.int32)
    end = min(len(ids), b + r + Dp + 2)
    res = mx.array(z["res"][:end - 1])
    W = (final_norm(res) if CFG["wsrc"] == "post" else res.astype(mx.bfloat16))[None]
    return {"W": W, "ids": ids[:end], "tk": np.asarray(z["tk"][:end - 1], np.int32),
            "lp": np.asarray(z["lp"][:end - 1], np.float32), "asst": np.asarray(z["asst"], np.float32)[:end], "a0": b,
            "r": r, "f": f}


def first_anchor(asst):
    a = np.flatnonzero(np.asarray(asst) > 0)
    return max(0, int(a[0]) - 2) if len(a) else 0


def pass_slices(fs, R, seed):
    """t30b's one-pass order: every file's anchors (2 before the first answer token .. len - 3) cut into ceil(n / R)
    near-equal consecutive slices, all slices shuffled once with the seed"""
    out = []
    for f in fs:
        z = np.load(f)
        n_ids = len(z["ids"]); first = first_anchor(z["asst"])
        n = (n_ids - 3) - first + 1
        if n <= 0:
            continue
        k = -(-n // R)
        bs = [first + (i * n) // k for i in range(k + 1)]
        out += [(f, bs[i], bs[i + 1] - bs[i]) for i in range(k)]
    order = np.random.default_rng(seed).permutation(len(out))
    return [out[i] for i in order]


# ---------------------------------------------------------------- read-out (dd_ttt.py's eval)
ARGMAX_ROWS = 2048    # rows per lm_head + argmax chunk in chain_table: ~3 GB of logits in flight at 2048 (SIB uses 256)


def chain_table(D, f, Dp=31, R=None, K=1, fn=None):
    """per answer file: rt [n_anchors, Dp] int8 (1 right, 0 wrong, -1 no target) of the teacher-forced chain from every
    anchor s = gen - 2 .., plus gen and a0. R: replace the stand-in by rows of R (REPLACE, every K depths); fn: transform
    it (a lever's standin). The argmax uses the target's own 4-bit lm_head, in row chunks after the chain."""
    z = np.load(f)
    ids = np.asarray(z["ids"], np.int32)
    gen = int(z["gen"])
    W = load_W(z)
    a0 = max(0, gen - 2)
    S = len(ids)
    n = S - 1 - a0
    rt = np.full((n, Dp), -1, np.int8)
    REPL.update(R=R, K=K, fn=fn)
    CUR["f"], CUR["train"], CUR["x"] = f, False, None
    hs = []

    def on_depth(j, h, anc):
        m = min(h.shape[1], S - 1 - j - a0)
        if m > 0:
            hs.append((j, m, D.norm(h[0, :m]).astype(mx.bfloat16)))
    try:
        chain(D, W, mx.array(ids), Dp, on_depth, a0)
    finally:
        REPL.update(R=None, K=1, fn=None)
    Hs = mx.concatenate([x for _, _, x in hs], 0)
    am = []
    for c0 in range(0, Hs.shape[0], ARGMAX_ROWS):
        am.append(mx.argmax(lm_logits(Hs[c0:c0 + ARGMAX_ROWS]), -1))
        mx.eval(am[-1])
    am = np.array(mx.concatenate(am))
    o = 0
    for j, m, _ in hs:
        a = am[o:o + m]; o += m
        rt[:m, j - 1] = (a == ids[a0 + j + 1:a0 + j + 1 + m]).astype(np.int8)
    return rt, gen, a0


def kept_walk(rt, r0, Dp=31):
    sub = rt[r0:]; out = []; p = 0
    Dp = min(Dp, rt.shape[1])
    while p < sub.shape[0]:
        k = 0
        while k < Dp and sub[p, k] == 1:
            k += 1
        out.append(k + 1); p += k + 1
    return out


def summarize(tables):
    """tables: list of (key, rt, gen, a0) -> bands (all valid cells, dd_ttt's definition), strict conditional per depth
    (right at j given right at 1..j-1), kept per 32, plus per-window band counts for paired comparisons"""
    Dmax = max((t[1].shape[1] for t in tables), default=31)
    bands = BANDS + [b for b in DEEP_BANDS if b[0] < Dmax]
    band = {b: [0, 0] for b in bands}
    cond = np.zeros((2, max(Dmax, 31)))
    kept48, kept64 = [], []
    kept, per = [], {}
    for key, rt, gen, a0 in tables:
        pw = {}
        for lo, hi in bands:
            c = rt[:, lo:hi + 1]
            r, v = int((c == 1).sum()), int((c >= 0).sum())
            band[(lo, hi)][0] += r; band[(lo, hi)][1] += v
            pw[f"{lo}-{hi}"] = [r, v]
        alive = np.ones(rt.shape[0], bool)
        for j in range(rt.shape[1]):
            v = alive & (rt[:, j] >= 0)
            cond[0, j] += int((v & (rt[:, j] == 1)).sum()); cond[1, j] += int(v.sum())
            alive &= rt[:, j] == 1
        kk = kept_walk(rt, max(0, gen - 1 - a0))
        kept += kk
        if rt.shape[1] >= 47:
            kept48 += kept_walk(rt, max(0, gen - 1 - a0), 47)
        if rt.shape[1] >= 63:
            kept64 += kept_walk(rt, max(0, gen - 1 - a0), 63)
        pw["kept"] = [int(sum(kk)), len(kk)]
        per[key] = pw
    out = {f"{lo}-{hi}": (band[(lo, hi)][0] / band[(lo, hi)][1] if band[(lo, hi)][1] else None) for lo, hi in bands}
    out["kept_per_32"] = float(np.mean(kept)) if kept else None   # kept per check (max = kept_max: 32 at depth 31)
    out["kept_max"] = min(int(max(t[1].shape[1] for t in tables)), 31) + 1 if tables else None   # kept_per_32's cap
    out["cond"] = [float(cond[0, j] / cond[1, j]) if cond[1, j] else None for j in range(cond.shape[1])]
    if kept48:
        out["kept_per_48"] = float(np.mean(kept48))
    if kept64:
        out["kept_per_64"] = float(np.mean(kept64))
    out["cond_n"] = [int(c) for c in cond[1]]
    out["windows"] = len(tables)
    return out, per
