"""TrainLab: can cheap LoRA self-distillation make the 35B proxy tolerate expert dataflows that give SSD reads slack?
(2026-10-21; charter: ~/.local/share/pacer/TASKS.md, "Trainable dataflow").

Model, segments and reference are NewIdeas' defer35.py harness (unsloth Qwen3.6-35B-A3B-MLX-8bit, routed experts
requantized to 4-bit g64 on load; 3 frozen held-out segments of 1365 tokens; exact reference ref_1365.npz), so a flow's
zero-shot KLD here must reproduce defer35's number for the same dataflow.

Teacher = the same weights with the exact dataflow and the LoRA switched off (computed online, Metal GDN kernel).
Student = the same weights + LoRA (attention, GDN projections, shared expert; optionally the router) + optional
pre-routers, run under the modified dataflow. Loss = forward KL(teacher || student) over the teacher's top-64 + tail.
Routed experts are frozen. Training uses a chunked, differentiable GDN (gdn_chunk.py) and per-layer checkpointing.

Dataflow spec, parts joined by "_" (anything outside the band B is exact):
  exact
  late      routed ranks >= W are added at the END of layer L+1 (W default 0: the whole routed output; ScMoE-like)
  half      routed ranks >= W are added after layer L+1's attention, before its router (W default 1)
  wN        ranks < N (by router score) stay on time
  pre/prew  layer L's experts are chosen at layer L-1 by a trainable pre-router (init: layer L's router + norm) that
            reads layer L-1's post-attention state; weights from the pre-router (pre) or from layer L's router (prew).
            With CN (+drop): only the W waited ranks are pre-routed (loaded a layer ahead); the other k - W slots are
            the true router's best outside them under the cache rule (prew_drop_w2_C26 = pre-routed frugal)
  kN        static top-N instead of 8;  nM  weights = p / (sum of the top-M probabilities) instead of renormalising
  drop      ranks >= W (default 2) are not computed (with CN: only cold ones, never inserted = frugal / pol35 ldrop)
  xNN       with CN and late/half: a cold rank >= W is late only if the selected mass before it is < NN %, colder
            ones are dropped (pol35's x; half_w2_C13_x50 = C13_w2_lhalf_x50)
  CN        cache-dependent (with late/half/drop): a per-layer LRU of N experts per sequence (pol35's semantics): resident
            experts and cold ranks < W are on time (cold ones loaded and inserted), cold ranks >= W are late (loaded in
            the background, inserted). half_w2_C26 = pol35's C26_w2_lhalf. Selection by f32 router logits as pol35.
  dN        pre-routing distance: layer L's experts are chosen at layer L - N (default 1; N > 1: eval only)
  Ba-b      the flow applies in layers a .. b-1
  capM      sticky routing by a hard cap (Temporal MoE's constraint): at most M of a token's experts per layer are new
            vs the previous token's; the M best new ones (current router) + the best k - M of the previous set; true
            weights renormalised. Eval prints switch / union / LRU metrics and the error energy (missed true top-k
            w^2 + used non-top-k w^2, w = p / sum of the true top-k p)
  sticky    exact dataflow; training adds ReMoE's soft reuse loss (2605.27081) per layer as the aux term:
            -log(mean_t sum_{e in top-k(t-1)} p_t(e)); use with --targets routerfull (or router) and --aux lambda
usage:
  train35.py eval FLOW [FLOW ...] [--run NAME]            held-out KLD per flow, zero-shot or with a run's LoRA
  train35.py train --flow FLOW --run NAME [--tokens N] [--seq 1024] [--batch 1] [--lr 1e-4] [--rank 16]
             [--targets gdn,attn,shared,router] [--eval-every 262144] [--aux 1.0] [--time-limit S]   (resumes NAME)
"""
import argparse, glob, json, math, os, sys, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten, tree_map
from mlx_lm import load

sys.path.insert(0, os.path.expanduser("~/NewIdeas/scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import defer35 as D
from gdn_chunk import gated_delta_chunked

TL = os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel1/TrainLab_root")
RES = os.path.join(TL, "data/results.txt")
TOPK = 64
WARM = 128


# ---------------------------------------------------------------- dataflow spec
class Flow:
    def __init__(self, spec):
        self.spec = spec
        self.late = self.half = self.drop = self.sticky = self.floor = False
        self.W = None; self.pre = None; self.k = None; self.n = None; self.band = (0, 10 ** 6); self.C = None; self.x = None; self.d = 1; self.cap = None
        if spec in ("exact", "floor"):
            self.floor = spec == "floor"              # floor: every MoE output x (1 + 2^-7), defer35's routing-flip floor
            return
        for p in spec.split("_"):
            if p == "late": self.late = True
            elif p == "drop": self.drop = True
            elif p == "sticky": self.sticky = True
            elif p == "half": self.half = True
            elif p in ("pre", "prew"): self.pre = p
            elif p[0] == "w": self.W = int(p[1:])
            elif p[0] == "C": self.C = int(p[1:])
            elif p[0] == "x": self.x = int(p[1:]) / 100
            elif p[0] == "d" and p[1:].isdigit(): self.d = int(p[1:])
            elif p.startswith("cap"): self.cap = int(p[3:])
            elif p[0] == "k": self.k = int(p[1:])
            elif p[0] == "n": self.n = int(p[1:])
            elif p[0] == "B": self.band = tuple(int(v) for v in p[1:].split("-"))
            else: raise ValueError(spec)
        if self.W is None:
            self.W = 1 if self.half else (2 if self.drop else 0)

    def on(self, L):
        return self.band[0] <= L < self.band[1]

    def pre_at(self, L):                              # layer L's experts are pre-routed (chosen at L - d)
        return self.pre is not None and L >= self.d and self.on(L)


# ---------------------------------------------------------------- LoRA and pre-routers
class LoRA(nn.Module):
    on = True

    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        in_d = base.weight.shape[1] * 32 // base.bits if isinstance(base, nn.QuantizedLinear) else base.weight.shape[1]
        s = 1 / math.sqrt(in_d)
        self.lora_a = mx.random.uniform(-s, s, (in_d, r))
        self.lora_b = mx.zeros((r, base.weight.shape[0]))
        self.scale = alpha / r

    def __call__(self, x):
        y = self.base(x)
        if LoRA.on:
            y = y + (self.scale * ((x.astype(mx.float32) @ self.lora_a) @ self.lora_b)).astype(y.dtype)
        return y


class PreRouter(nn.Module):
    """Chooses layer L's experts from layer L-1's post-attention state: rms_norm (own weight) + linear (E x D), both
    initialised from layer L's post-attention norm and router."""

    def __init__(self, norm, gate):
        super().__init__()
        self.norm_w = norm.weight.astype(mx.float32)
        if isinstance(gate, nn.QuantizedLinear):
            w = mx.dequantize(gate.weight, gate.scales, gate.biases, group_size=gate.group_size, bits=gate.bits)
        else:
            w = gate.weight
        self.w = w.astype(mx.float32)
        self.eps = norm.eps

    def __call__(self, h):
        x = mx.fast.rms_norm(h.astype(mx.float32), self.norm_w, self.eps)
        return x @ self.w.T


class RouterFull(nn.Module):
    """A trainable fp32 router (dequantized from the 8-bit one); output in bf16 like the original's."""

    def __init__(self, w):
        super().__init__()
        self.w = w

    def __call__(self, x):
        return (x.astype(mx.float32) @ self.w.T).astype(x.dtype)


def gate_lin(m):                                     # the router, under a LoRA wrapper or not
    return m.gate


def add_lora(model, targets, r, alpha):
    names = {"gdn": ("linear_attn", ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")),
             "attn": ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
             "shared": ("mlp.shared_expert", ("gate_proj", "up_proj", "down_proj")),
             "router": ("mlp", ("gate",))}
    n = 0
    for layer in model.language_model.model.layers:
        for t in targets:
            path, leaves = names[t]
            mod = layer
            ok = True
            for part in path.split("."):
                if not hasattr(mod, part):
                    ok = False; break
                mod = getattr(mod, part)
            if not ok:
                continue
            for leaf in leaves:
                base = getattr(mod, leaf)
                setattr(mod, leaf, LoRA(base, r, alpha))
                n += 1
    return n


# ---------------------------------------------------------------- per-layer LRU (cache-dependent flows)
class Trace:
    """Eval: record the selected experts per layer ([B, T, k] numpy) for routing-locality metrics."""
    rec = None


class Cache:
    """Eval: the LRU runs inside the forward (numpy, per sequence, per layer; record=True keeps each layer's residency
    map [B, T, E] before every token). Train: the maps recorded in a no-grad pass of the student are used as constants,
    so a rare routing flip between the two passes still gets a well-defined cold/resident status."""
    maps = None          # train: {layer: bool [B, T, E] resident-before-token}
    persist = None       # decode: {(sequence id, layer): OrderedDict} kept across forward calls
    seq_ids = None       # decode: the sequence id of each batch row
    valid = None         # decode: bool [B, T] (False = left padding: no LRU update, output unused)
    record = None        # eval: dict to fill with maps, or None
    stats = None         # eval: [slots, late slots, cold loads, dropped slots]


def lru_late(order, C, W, E, record, insert_late=True, cpb=None, x=None, state=None, valid=None):
    """order [T, k] expert ids by rank; cpb [T, k] selected mass before each rank (for the x cap). Returns the late
    mask (cold, rank >= W, within the cap), the residency map, loads, and the dropped mask (cold beyond the cap)."""
    T, k = order.shape
    late = np.zeros((T, k), bool)
    drop = np.zeros((T, k), bool)
    res = np.zeros((T, E), bool) if record else None
    from collections import OrderedDict
    cache, loads = (OrderedDict() if state is None else state), 0
    for t in range(T):
        if valid is not None and not valid[t]:
            continue
        if record and cache:
            res[t, np.fromiter(cache.keys(), np.int64)] = True
        for r in range(k):
            e = int(order[t, r])
            if e in cache:
                cache.move_to_end(e); continue
            if r >= W:
                if not insert_late or (x is not None and cpb[t, r] >= x - 1e-9):
                    drop[t, r] = True                    # dropped (frugal / beyond the cap): not read, not inserted
                    continue
                late[t, r] = True
            loads += 1
            if len(cache) >= C:
                cache.popitem(last=False)
            cache[e] = True
    return late, res, loads, drop


def cap_select(lg, k, m):
    """lg [T, E] f32 router logits of one sequence -> [T, k] expert ids (rank order) with |E_t - E_(t-1)| <= m."""
    T = lg.shape[0]
    out = np.zeros((T, k), np.int64)
    prev = None
    for t in range(T):
        top = np.argpartition(-lg[t], k)[:k]
        top = top[np.argsort(-lg[t][top])]
        if prev is not None:
            pset = set(prev.tolist())
            new = [e for e in top if e not in pset]
            if len(new) > m:
                keep_new = new[:m]
                old = prev[np.argsort(-lg[t][prev])][:k - m]
                sel = np.array(list(old) + keep_new)
                top = sel[np.argsort(-lg[t][sel])]
        out[t] = top
        prev = top
    return out


def cache_late(L, inds, rank, flow, E, cpb):
    """bool masks [B, T, k] for the selected experts (rank order): late (cold, rank >= W, inside the x cap) and dropped
    (cold, rank >= W, beyond the cap, or every such one for drop flows). cpb: selected mass before each rank."""
    if Cache.maps is not None:
        res = Cache.maps[L]
        cold = mx.logical_and(mx.logical_not(mx.take_along_axis(res, inds.astype(mx.int32), axis=-1)), rank >= flow.W)
        if flow.drop:
            return mx.zeros(cold.shape, mx.bool_), cold
        if flow.x is None:
            return cold, mx.zeros(cold.shape, mx.bool_)
        inside = cpb < flow.x - 1e-9
        return mx.logical_and(cold, inside), mx.logical_and(cold, mx.logical_not(inside))
    ii, cp = np.array(inds), np.array(cpb)
    B, T, k = ii.shape
    late, drop, maps = np.zeros((B, T, k), bool), np.zeros((B, T, k), bool), []
    for b in range(B):
        st = None
        if Cache.persist is not None:
            from collections import OrderedDict
            st = Cache.persist.setdefault((Cache.seq_ids[b], L), OrderedDict())
        late[b], res, loads, drop[b] = lru_late(ii[b], flow.C, flow.W, E, Cache.record is not None, not flow.drop,
                                                cp[b], flow.x, st, None if Cache.valid is None else Cache.valid[b])
        maps.append(res)
        if Cache.stats is not None:
            Cache.stats += [T * k, late[b].sum(), loads, drop[b].sum()]
    if Cache.record is not None:
        Cache.record[L] = mx.array(np.stack(maps))
    return mx.array(late), mx.array(drop)


# ---------------------------------------------------------------- forward pieces
def gdn_train(m, inputs):
    """GatedDeltaNet.__call__ without a cache, with the chunked differentiable recurrence."""
    B, S, _ = inputs.shape
    qkv = m.in_proj_qkv(inputs)
    z = m.in_proj_z(inputs).reshape(B, S, m.num_v_heads, m.head_v_dim)
    b = m.in_proj_b(inputs)
    a = m.in_proj_a(inputs)
    conv_state = mx.zeros((B, m.conv_kernel_size - 1, m.conv_dim), dtype=inputs.dtype)
    conv_out = nn.silu(m.conv1d(mx.concatenate([conv_state, qkv], axis=1)))
    q, k, v = [t.reshape(B, S, h, d) for t, h, d in zip(
        mx.split(conv_out, [m.key_dim, 2 * m.key_dim], -1),
        [m.num_k_heads, m.num_k_heads, m.num_v_heads], [m.head_k_dim, m.head_k_dim, m.head_v_dim])]
    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale ** 2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
    log_g = -mx.exp(m.A_log.astype(mx.float32)) * nn.softplus((a + m.dt_bias).astype(mx.float32))
    out, _ = gated_delta_chunked(q, k, v, log_g, mx.sigmoid(b), None)
    out = m.norm(out, z)
    return m.out_proj(out.reshape(B, S, -1))


def moe(m, L, x, flow, pre_logits):
    """Returns (on-time output incl. shared expert, late routed output or None, aux KL of the pre-router or None)."""
    k = flow.k if (flow.k and flow.on(L)) else m.top_k
    gates = mx.softmax(gate_lin(m)(x), axis=-1, precise=True)
    aux = None
    if flow.pre_at(L) and pre_logits is not None and flow.C is None:
        pl = pre_logits.astype(mx.float32)
        inds = mx.stop_gradient(mx.argpartition(-pl, kth=k - 1, axis=-1)[..., :k])
        pp = mx.softmax(pl, axis=-1)
        src = gates if flow.pre == "prew" else pp.astype(gates.dtype)
        scores = mx.take_along_axis(src, inds, axis=-1)
        scores = scores / scores.sum(axis=-1, keepdims=True)
        tg = mx.stop_gradient(gates.astype(mx.float32))
        aux = (tg * (mx.log(tg + 1e-20) - mx.log(pp + 1e-20))).sum(-1).mean()
    else:
        inds = mx.stop_gradient(mx.argpartition(gates, kth=-k, axis=-1)[..., -k:])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if flow.n and flow.on(L):
            top_n = mx.topk(gates, flow.n, axis=-1) if flow.n != k else scores
            scores = scores / top_n.sum(axis=-1, keepdims=True)
        elif m.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
    if flow.C is not None and flow.on(L):                                   # pol35 selection: f32 router logits
        lgf = gate_lin(m)(x).astype(mx.float32)
        if flow.pre and flow.pre_at(L) and pre_logits is not None:
            # pre-routed cache flow: the W waited ranks are the pre-router's top-W (chosen at layer L-1, loaded a layer
            # ahead); the other k - W slots are the true router's best experts outside them, under the cache rule
            pl = pre_logits.astype(mx.float32)
            ptop = mx.argpartition(-pl, kth=flow.W - 1, axis=-1)[..., :flow.W]
            ptop = mx.take_along_axis(ptop, mx.argsort(-mx.take_along_axis(pl, ptop, axis=-1), axis=-1), axis=-1)
            E = lgf.shape[-1]
            hit = (mx.arange(E)[None, None, :, None] == ptop[..., None, :]).any(-1)
            rest_l = mx.where(hit, -mx.inf, lgf)
            rest = mx.argpartition(-rest_l, kth=k - flow.W - 1, axis=-1)[..., :k - flow.W]
            rest = mx.take_along_axis(rest, mx.argsort(-mx.take_along_axis(lgf, rest, axis=-1), axis=-1), axis=-1)
            inds = mx.stop_gradient(mx.concatenate([ptop, rest], axis=-1))
            pp = mx.softmax(pl, axis=-1)                                   # pre-router aux: KL to the true router
            tg = mx.stop_gradient(gates.astype(mx.float32))
            aux = (tg * (mx.log(tg + 1e-20) - mx.log(pp + 1e-20))).sum(-1).mean()
        else:
            inds = mx.argpartition(-lgf, kth=k - 1, axis=-1)[..., :k]
            inds = mx.stop_gradient(mx.take_along_axis(inds, mx.argsort(-mx.take_along_axis(lgf, inds, axis=-1),
                                                                         axis=-1), axis=-1))   # rank order, as pol35
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = scores / scores.sum(axis=-1, keepdims=True)
        rank = mx.broadcast_to(mx.arange(k), inds.shape)
        psel = mx.softmax(mx.take_along_axis(lgf, inds, axis=-1), axis=-1)        # pol35's p over the selection
        cpb = mx.cumsum(psel, axis=-1) - psel
        late_m, drop_m = cache_late(L, inds, rank, flow, gates.shape[-1], cpb)
    if flow.cap is not None and flow.on(L):
        lgn = np.array(gate_lin(m)(x).astype(mx.float32))                 # [B, T, E] (mid-trace evaluation)
        sel = np.stack([cap_select(lgn[b], k, flow.cap) for b in range(lgn.shape[0])])
        inds = mx.stop_gradient(mx.array(sel.astype(np.uint32)))
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = scores / scores.sum(axis=-1, keepdims=True)
        if Cache.stats is not None:                                       # Chat1000's error energy per token-layer
            gp = np.array(gates.astype(mx.float32))
            tk = np.argpartition(-lgn, k, axis=-1)[..., :k]
            den = np.take_along_axis(gp, tk, -1).sum(-1, keepdims=True)
            w = gp / den
            in_true = np.zeros(gp.shape, bool); np.put_along_axis(in_true, tk, True, -1)
            used = np.zeros(gp.shape, bool); np.put_along_axis(used, sel, True, -1)
            en = ((w ** 2) * (in_true & ~used)).sum(-1) + ((w ** 2) * (used & ~in_true)).sum(-1)
            Cache.stats += [en.size, 0, 0, 0] if len(Cache.stats) == 4 else 0
            Cache.energy = getattr(Cache, "energy", []) + [float(en.mean())]
    if Trace.rec is not None:
        Trace.rec[L] = np.array(inds)
    if flow.sticky:                                   # soft reuse mass of the previous token's experts (ReMoE)
        prev = mx.stop_gradient(inds[:, :-1])
        gf = gates.astype(mx.float32)
        mass = mx.take_along_axis(gf[:, 1:], prev, axis=-1).sum(-1)       # [B, T-1]
        aux = -mx.log(mass.mean() + 1e-8)
    y = m.switch_mlp(x, inds)                                               # [B, T, k, D]
    shared = mx.sigmoid(m.shared_expert_gate(x)) * m.shared_expert(x)
    if (flow.late or flow.half or flow.drop) and flow.on(L):
        if flow.C is not None:
            msk = mx.stop_gradient(late_m.astype(scores.dtype))
            if flow.drop:
                msk = mx.stop_gradient(drop_m.astype(scores.dtype))
            else:                                   # the x cap: beyond it, cold ranks are dropped
                scores = scores * mx.stop_gradient(1 - drop_m.astype(scores.dtype))
        else:
            order = mx.argsort(-scores.astype(mx.float32), axis=-1)
            rank = mx.argsort(order, axis=-1)
            msk = mx.stop_gradient((rank >= flow.W).astype(scores.dtype))   # 1 = late
        wres, wmis = scores * (1 - msk), scores * msk
        if flow.drop:
            return (y * wres[..., None]).sum(axis=-2) + shared, None, aux
        return (y * wres[..., None]).sum(axis=-2) + shared, (y * wmis[..., None]).sum(axis=-2), aux
    if flow.floor:
        return ((y * scores[..., None]).sum(axis=-2) + shared) * (1.0 + 2.0 ** -7), None, aux
    return (y * scores[..., None]).sum(axis=-2) + shared, None, aux


def layer_body(layer, L, nl, flow, train, h, add_full, add_half, pre_logits, cache=None, masks=None):
    """One decoder layer as a pure function. add_full: added at the end of this layer; add_half: added after this
    layer's attention. Returns (h, late for L+1 (full), late for L+1 (half), pre-router logits for L+1, aux)."""
    x = layer.input_layernorm(h)
    if layer.is_linear:
        r = gdn_train(layer.linear_attn, x) if train else layer.linear_attn(x, masks[1] if masks else None, cache)
    else:
        r = layer.self_attn(x, masks[0] if masks else ("causal" if x.shape[1] > 1 else None), cache)
    h = x_res = h + r
    if add_half is not None:
        h = h + add_half
    nxt_pre = None
    if flow.pre_at(L + flow.d) and L + flow.d < nl:                         # issued before this layer's MoE
        nxt_pre = (layer.next_router if flow.d == 1 else getattr(layer, f"next_router{flow.d}"))(h)
    on, late, aux = moe(layer.mlp, L, layer.post_attention_layernorm(h), flow, pre_logits)
    out = h + on
    lf = lh = None
    if late is not None:
        if L + 1 >= nl:                                                     # last layer: late = end of this layer
            add_full = late if add_full is None else add_full + late
        elif flow.half:
            lh = late
        else:
            lf = late
    if add_full is not None:
        out = out + add_full
    return out, lf, lh, nxt_pre, aux


def forward(model, ids, flow, train, ckpt=True, caches=None):
    """caches (eval only): mlx_lm per-layer caches (batched, left padding) for prefill + decode."""
    lm = model.language_model
    layers = lm.model.layers
    nl = len(layers)
    h = lm.model.embed_tokens(ids)
    lf = lh = pre = None
    pre_for = {}
    auxs = []
    masks = None
    if caches is not None:
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask
        masks = (create_attention_mask(h, caches[lm.model.fa_idx]), create_ssm_mask(h, caches[lm.model.ssm_idx]))
    for L, layer in enumerate(layers):
        if train and ckpt:
            assert flow.d == 1, "training supports pre-routing distance 1 only"
            h, lf, lh, pre, aux = ckpt_layer(layer, L, nl, flow, h, lf, lh, pre)
        elif flow.d == 1:
            h, lf, lh, pre, aux = layer_body(layer, L, nl, flow, train, h, lf, lh, pre,
                                             None if caches is None else caches[L], masks)
        else:                                                    # distance d: pre-logits wait d layers
            h, lf, lh, nxt, aux = layer_body(layer, L, nl, flow, train, h, lf, lh, pre_for.pop(L, None),
                                             None if caches is None else caches[L], masks)
            if nxt is not None:
                pre_for[L + flow.d] = nxt
        if aux is not None:
            auxs.append(aux)
    return lm.model.norm(h), auxs


_CK = {}


def ckpt_layer(layer, L, nl, flow, h, lf, lh, pre):
    """mx.checkpoint around layer_body; None carries are passed as a flag pattern so the traced function is pure."""
    key = (flow.spec, L, lf is None, lh is None, pre is None)
    if key not in _CK:
        flags = key[2:]

        def inner(params, h, *carry):
            layer.update(params)
            it = iter(carry)
            a_f = None if flags[0] else next(it)
            a_h = None if flags[1] else next(it)
            p_l = None if flags[2] else next(it)
            out, nf, nh, npre, aux = layer_body(layer, L, nl, flow, True, h, a_f, a_h, p_l)
            res = [out]
            res += [nf] if nf is not None else []
            res += [nh] if nh is not None else []
            res += [npre] if npre is not None else []
            res += [aux] if aux is not None else []
            return tuple(res)

        _CK[key] = mx.checkpoint(inner)
    carry = [c for c in (lf, lh, pre) if c is not None]
    outs = list(_CK[key](layer.trainable_parameters(), h, *carry))
    out = outs.pop(0)
    # which outputs exist is static per (layer, flow): recompute the pattern from the flow
    nf = nh = npre = aux = None
    late_here = (flow.late or flow.half) and flow.on(L)
    if late_here and L + 1 < nl:
        if flow.half: nh = outs.pop(0)
        else: nf = outs.pop(0)
    if flow.pre_at(L + 1) and L + 1 < nl:
        npre = outs.pop(0)
    if (flow.pre_at(L) and pre is not None) or flow.sticky:
        aux = outs.pop(0)
    assert not outs
    return out, nf, nh, npre, aux


# ---------------------------------------------------------------- losses and evaluation
def teacher_top(model, ids):
    """Teacher = exact dataflow, LoRA off, kernel GDN: top-64 ids and log-probs per position [B, T, K]."""
    LoRA.on = False
    h, _ = forward(model, ids, Flow("exact"), False)
    LoRA.on = True
    tops, tlps = [], []
    for s in range(0, h.shape[1], 512):
        lg = model.language_model.lm_head(h[:, s:s + 512]).astype(mx.float32)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        top = mx.argpartition(-lp, TOPK - 1, axis=-1)[..., :TOPK]
        tops.append(top); tlps.append(mx.take_along_axis(lp, top, axis=-1))
        mx.eval(tops[-1], tlps[-1])
    return mx.concatenate(tops, axis=1), mx.concatenate(tlps, axis=1)


def kl_top(lp_q_top, p_lp, lse_q=None):
    """KL(p || q) over p's top-K + one tail bucket; p_lp, lp_q_top [.., K] log-probs."""
    p, q = mx.exp(p_lp), mx.exp(lp_q_top)
    pt = mx.maximum(1 - p.sum(-1), 1e-12)
    qt = mx.maximum(1 - q.sum(-1), 1e-12)
    return (p * (p_lp - lp_q_top)).sum(-1) + pt * (mx.log(pt) - mx.log(qt))


def _chunk_kl(head, top, tlp):
    def f(h):                                        # the teacher arrays are closed over (constants, int ids)
        lg = head(h).astype(mx.float32)
        lq = mx.take_along_axis(lg, top, axis=-1) - mx.logsumexp(lg, axis=-1, keepdims=True)
        return kl_top(lq, tlp).sum()
    return mx.checkpoint(f)


def student_loss(model, ids, top, tlp, flow, aux_w, ckpt=True):
    h, auxs = forward(model, ids, flow, True, ckpt)
    head = model.language_model.lm_head
    tot = 0.0
    for s in range(0, h.shape[1], 256):
        tot = tot + _chunk_kl(head, top[:, s:s + 256], tlp[:, s:s + 256])(h[:, s:s + 256])
    kd = tot / (h.shape[0] * h.shape[1])
    aux = sum(auxs) / len(auxs) if auxs else mx.array(0.0)
    return kd + aux_w * aux, (kd, aux)


def heldout(tok):
    """defer35's 3 frozen segments (prose, gsm, code) + TrainLab's extra held-out (multilingual, chat) if built."""
    segs = [("prose", None), ("gsm", None), ("code", None)]
    out = [(n, s) for (n, _), s in zip(segs, D.segments(tok))]
    fz = os.path.join(TL, "data/heldout_extra_ids.npz")
    if os.path.exists(fz):
        z = np.load(fz)
        out += [(k, mx.array(z[k])[None]) for k in z.files]
    return out


def reference(model, segs):
    """Exact reference (exact flow, no LoRA, kernel GDN) for every held-out segment, computed here and stored with the
    segment ids it was computed from (data/ref_heldout.npz); recomputed for any segment whose ids changed. Independent
    of NewIdeas' ref_1365.npz (rebuilt after its 16:16 frozen-segment fix); same model and ids -> the same numbers."""
    rp = os.path.join(TL, "data/ref_heldout.npz")
    have = dict(np.load(rp)) if os.path.exists(rp) else {}
    todo = [(n, s) for n, s in segs if not (f"ids_{n}" in have and np.array_equal(have[f"ids_{n}"], np.array(s[0])))]
    if todo:
        on = LoRA.on; LoRA.on = False
        for n, s in todo:
            lp = logprobs(model, s, Flow("exact"))
            top = mx.argpartition(-lp, TOPK - 1, axis=-1)[:, :TOPK]
            tlp = mx.take_along_axis(lp, top, axis=-1)
            a1 = mx.argmax(lp, axis=-1)
            mx.eval(top, tlp, a1)
            have[f"top_{n}"], have[f"tlp_{n}"], have[f"a1_{n}"] = np.array(top), np.array(tlp), np.array(a1)
            have[f"ids_{n}"] = np.array(s[0])
        LoRA.on = on
        np.savez(rp, **have)
        print(f"reference written for {[n for n, _ in todo]}", flush=True)
    return {n: (have[f"top_{n}"], have[f"tlp_{n}"], have[f"a1_{n}"]) for n, _ in segs}


def logprobs(model, seg, flow):
    h, _ = forward(model, seg[:, :-1], flow, False)
    lg = model.language_model.lm_head(h)[0].astype(mx.float32)
    return lg - mx.logsumexp(lg, axis=-1, keepdims=True)


def locality(traces):
    """Routing locality of the held-out segments (Chat1000's union_metrics): switch rate, expert union per layer at
    K = 16/32/64 rows, LRU hit rate at C = 26/13, averaged over segments."""
    sys.path.insert(0, os.path.expanduser("~/Chat1000/scripts"))
    import union_metrics as U
    vals = {}
    for ids in traces:
        for name, v in [("switch", U.switch_rate(ids))] + [(f"U{K}", U.union(ids, K)) for K in (16, 32, 64)] + \
                       [(f"LRU{C}", U.lru_hits(ids, C)) for C in (26, 13)]:
            vals.setdefault(name, []).append(v)
    return " ".join(f"{k} {np.mean(v):.3f}" if k[0] != "U" else f"{k} {np.mean(v):.1f}" for k, v in vals.items())


def evaluate(model, segs, R, flow, trace=False):
    per, allk, allw, agree = {}, [], [], []
    Cache.stats = np.zeros(4) if (flow.C is not None or flow.cap is not None) else None
    Cache.energy = []
    traces = []
    for n, s in segs:
        Trace.rec = {} if (trace or flow.sticky or flow.cap is not None or flow.spec == "exact") and \
            n in ("prose", "gsm", "code") else None
        lp = logprobs(model, s, flow)
        top, p_lp, a1r = R[n]
        q = mx.take_along_axis(lp, mx.array(top), axis=-1)
        k = kl_top(q, mx.array(p_lp))
        a1 = mx.argmax(lp, axis=-1)
        mx.eval(k, a1)
        k = np.array(k); per[n] = float(k.mean())
        if Trace.rec is not None:                                      # [T, L, k] routing of this segment
            traces.append(np.stack([Trace.rec[L][0] for L in sorted(Trace.rec)], axis=1).astype(np.int64))
            Trace.rec = None
        if n in ("prose", "gsm", "code"):
            allk.append(k); allw.append(k[WARM:]); agree.append(np.array(a1) == a1r)
    ka = np.concatenate(allk)
    cs = None
    if flow.cap is not None:
        cs = f"error energy/token-layer {np.mean(Cache.energy):.4f}"
        Cache.stats = None
    if Cache.stats is not None:
        ntok = Cache.stats[0] / 8 / len(model.language_model.model.layers)
        cs = (f"late slots {Cache.stats[1] / Cache.stats[0]:.3f} dropped {Cache.stats[3] / Cache.stats[0]:.3f} "
              f"cold reads/token {Cache.stats[2] / ntok:.1f}")
        Cache.stats = None
    if traces:
        cs = (cs + " | " if cs else "") + locality(traces)
    return {"cache": cs, "kld": float(ka.mean()), "warm": float(np.concatenate(allw).mean()), "p99": float(np.percentile(ka, 99)),
            "top1": float(np.concatenate(agree).mean()), "per": per}


def fmt(r):
    per = " ".join(f"{k} {v:.4f}" for k, v in r["per"].items())
    c = f" | {r['cache']}" if r.get("cache") else ""
    return f"KLD {r['kld']:.5f} (warm {r['warm']:.5f}, P99 {r['p99']:.4f}) top-1 {r['top1']:.4f} | {per}{c}"


# ---------------------------------------------------------------- setup
def setup(targets=(), r=16, alpha=32, pre_flows=()):
    t0 = time.time()
    model, tok = load(D.MODEL, lazy=True)
    n4 = D.to4bit(model)
    nlora, ntr = prepare(model, targets, r, alpha, pre_flows)
    print(f"load {time.time() - t0:.0f}s, {n4} expert tensors -> 4-bit, {nlora} LoRA modules, {ntr / 1e6:.1f}M "
          f"trainable, peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
    return model, tok


def prepare(model, targets, r, alpha, pre_flows):
    layers = model.language_model.model.layers
    # pre-routers for every flow that needs them (layer L holds the pre-router of layer L+1)
    dists = {Flow(f).d for f in pre_flows if Flow(f).pre}
    for d in sorted(dists):
        for L in range(len(layers) - d):
            nxt = layers[L + d]
            setattr(layers[L], "next_router" if d == 1 else f"next_router{d}",
                    PreRouter(nxt.post_attention_layernorm, nxt.mlp.gate))
    if "routerfull" in targets:
        for layer in layers:
            g = layer.mlp.gate
            w = mx.dequantize(g.weight, g.scales, g.biases, group_size=g.group_size, bits=g.bits) \
                if isinstance(g, nn.QuantizedLinear) else g.weight
            lin = RouterFull(w.astype(mx.float32))
            layer.mlp.gate = lin
        targets = [t for t in targets if t != "routerfull"]
    nlora = add_lora(model, targets, r, alpha) if targets else 0
    model.freeze()
    for _, m in model.named_modules():
        if isinstance(m, LoRA):
            m.unfreeze(keys=["lora_a", "lora_b"], recurse=False)
        if isinstance(m, PreRouter):
            m.unfreeze(keys=["norm_w", "w"], recurse=False)
        if isinstance(m, RouterFull):
            m.unfreeze(keys=["w"], recurse=False)
    mx.eval(model.parameters())
    return nlora, sum(v.size for _, v in tree_flatten(model.trainable_parameters()))


class TwoAdam:
    """Adam over flat parameter names; the pre-routers (full E x D matrices initialised from the routers) get their
    own, smaller step. State saves to one safetensors file ('group|param|m')."""

    def __init__(self, lr, lr_pre):
        warm = lambda base: (lambda step: base * mx.minimum(1.0, (step + 1) / 20.0))      # 20-step linear warm-up
        self.opts = {"a": optim.Adam(learning_rate=warm(lr)), "b": optim.Adam(learning_rate=warm(lr_pre))}

    @staticmethod
    def _grp(k):                                    # pre-routers and full routers: the smaller step
        return "b" if ("next_router" in k or k.endswith("mlp.gate.w")) else "a"

    def update(self, model, grads):
        params = dict(tree_flatten(model.trainable_parameters()))
        new = {}
        for g, o in self.opts.items():
            gg = {k: v for k, v in tree_flatten(grads) if self._grp(k) == g}
            if gg:
                new.update(o.apply_gradients(gg, {k: params[k] for k in gg}))
        model.update(tree_unflatten(list(new.items())))

    def arrays(self):
        out = []
        for g, o in self.opts.items():
            for k, st in o.state.items():
                if isinstance(st, dict):
                    out += [v for v in st.values()]
                else:
                    out.append(st)
        return out

    def save(self, path):
        d = {}
        for g, o in self.opts.items():
            for k, st in o.state.items():
                if isinstance(st, dict):
                    d.update({f"{g}|{k}|{kk}": vv for kk, vv in st.items()})
                else:
                    d[f"{g}|{k}"] = st
        mx.save_safetensors(path, d)

    def load(self, path):
        for key, v in mx.load(path).items():
            parts = key.split("|")
            st = self.opts[parts[0]].state
            if len(parts) == 2:
                st[parts[1]] = v
            else:
                st.setdefault(parts[1], {})[parts[2]] = v


def make_opt(lr, lr_pre):
    return TwoAdam(lr, lr_pre)


def log_result(line):
    print(line, flush=True)
    with open(RES, "a") as f:
        f.write(line + "\n")


def cmd_eval(flows, run=None):
    """Zero-shot (no LoRA), or with a trained run's LoRA / pre-routers (--run NAME: its targets and rank)."""
    if run:
        st = json.load(open(os.path.join(TL, "ckpt", run + ".json")))
        model, tok = setup(targets=st["targets"].split(","), r=st["rank"], alpha=2 * st["rank"],
                           pre_flows=flows + [st["flow"]])
        model.load_weights(os.path.join(TL, "ckpt", run + ".safetensors"), strict=False)
        tag = f"ckpt {run}@{st['tokens']}"
    else:
        model, tok = setup(pre_flows=flows)
        tag = "zeroshot"
    segs = heldout(tok)
    R = reference(model, segs)
    for f in flows:
        t1 = time.time()
        r = evaluate(model, segs, R, Flow(f))
        log_result(f"{time.strftime('%m-%d %H:%M')} {tag} {f:18s} {fmt(r)}  ({time.time() - t1:.0f}s, peak "
                   f"{mx.get_peak_memory() / 1e9:.1f} GB)")


def cmd_train(a):
    t_start = time.time()
    limit = a.time_limit or float(os.environ.get("TL_LIMIT", "1e9"))
    flow = Flow(a.flow)
    model, tok = setup(targets=a.targets.split(","), r=a.rank, alpha=2 * a.rank, pre_flows=[a.flow])
    segs = heldout(tok)
    R = reference(model, segs)
    stream = np.load(os.path.join(TL, "data/train_tokens.npy"), mmap_mode="r")
    ck = os.path.join(TL, "ckpt", a.run)
    state = {"tokens": 0, "step": 0, "seed": 0, "flow": a.flow, "lr": a.lr, "targets": a.targets, "rank": a.rank}
    opt = make_opt(a.lr, a.lr_pre)
    if os.path.exists(ck + ".json"):
        state = json.load(open(ck + ".json"))
        assert state["flow"] == a.flow, (state["flow"], a.flow)
        model.load_weights(ck + ".safetensors", strict=False)
        print(f"resumed {a.run} at {state['tokens']} tokens", flush=True)
    if os.path.exists(ck + ".opt.safetensors"):
        opt.load(ck + ".opt.safetensors")
    os.makedirs(os.path.join(TL, "data/teacher"), exist_ok=True)

    def save():
        mx.save_safetensors(ck + ".safetensors", dict(tree_flatten(model.trainable_parameters())))
        opt.save(ck + ".opt.safetensors")
        json.dump(state, open(ck + ".json", "w"))

    def do_eval(tag):
        t1 = time.time()
        r = evaluate(model, segs, R, flow)
        log_result(f"{time.strftime('%m-%d %H:%M')} train {a.run} {a.flow} tokens {state['tokens']} {tag} {fmt(r)} "
                   f"({time.time() - t1:.0f}s)")
        return r

    if state["tokens"] == 0:
        do_eval("step0")
    lossf = nn.value_and_grad(model, lambda m, ids, top, tlp: student_loss(m, ids, top, tlp, flow, a.aux))
    T, Bt = a.seq, a.batch
    next_eval = (state["tokens"] // a.eval_every + 1) * a.eval_every
    t_last, tok_last, hist, gn_hist, bad = time.time(), state["tokens"], [], [], 0
    acc_g, acc_n, T_ = None, 0, a.seq
    eval_cost = 60
    while state["tokens"] < a.tokens:
        if time.time() - t_start > limit - eval_cost - 90:
            print("time limit: stopping", flush=True)
            break
        # windows are a pure function of the step, so every run sees the same text in the same order and the
        # teacher's top-64 (exact flow, LoRA off) is computed once per step and cached on disk for all runs
        t_step = time.time()
        offs = np.random.default_rng(10 ** 6 + state["step"]).integers(0, stream.shape[0] - T - 1, size=Bt)
        ids = mx.array(np.stack([np.asarray(stream[o:o + T]) for o in offs]).astype(np.int32))
        tf = os.path.join(TL, f"data/teacher/s{T}x{Bt}_{state['step']:06d}.npz")
        if os.path.exists(tf):
            z = np.load(tf)
            top, tlp = mx.array(z["top"]), mx.array(z["tlp"].astype(np.float32))
        else:
            top, tlp = teacher_top(model, ids)
            tlp = tlp.astype(mx.float16).astype(mx.float32)          # the same precision as the cached steps
            np.savez(tf, top=np.array(top), tlp=np.array(tlp).astype(np.float16))
        if flow.C is not None:                       # pass 1: the student's own routing -> per-layer residency maps
            Cache.record = {}
            hh, _ = forward(model, ids, flow, False)
            mx.eval(hh, list(Cache.record.values()))
            Cache.maps, Cache.record = Cache.record, None
        (loss, (kd, aux)), grads = lossf(model, ids, top, tlp)
        Cache.maps = None
        if a.accum > 1:                                   # gradient accumulation: one update per a.accum windows
            acc_g = grads if acc_g is None else tree_map(lambda x, y: x + y, acc_g, grads)
            acc_n += 1
            mx.eval(acc_g)
            if acc_n < a.accum:
                state["step"] += 1; state["tokens"] += T_ * Bt
                hist.append((float(kd), float(aux)))
                continue
            grads = tree_map(lambda x: x / acc_n, acc_g)
            acc_g, acc_n = None, 0
        grads, gnorm = optim.clip_grad_norm(grads, a.clip)
        mx.eval(loss, gnorm)
        if not (np.isfinite(float(loss)) and np.isfinite(float(gnorm))):    # never let a bad step into the weights
            bad += 1
            print(f"step {state['step']}: non-finite loss {float(loss)} / grad norm {float(gnorm)}: skipped", flush=True)
            if bad >= 5:
                print("5 non-finite steps in a row: stopping", flush=True)
                break
            state["step"] += 1
            continue
        bad = 0
        opt.update(model, grads)
        mx.eval(model.trainable_parameters(), opt.arrays(), loss)
        gn_hist.append(float(gnorm))
        if a.duty < 1:                               # duty cycle (quiet nights): idle (1/duty - 1) x the step time
            time.sleep((time.time() - t_step) * (1 / a.duty - 1))
        state["step"] += 1; state["tokens"] += T * Bt
        hist.append((float(kd), float(aux)))
        if state["step"] % 10 == 0:
            dt = time.time() - t_last
            kd_m = np.mean([h[0] for h in hist]); ax_m = np.mean([h[1] for h in hist]); hist = []
            gn_m = np.mean(gn_hist) if gn_hist else float("nan"); gn_hist = []
            print(f"step {state['step']} tokens {state['tokens']} kd {kd_m:.4f} aux {ax_m:.4f} |g| {gn_m:.3f} "
                  f"{(state['tokens'] - tok_last) / dt:.0f} tok/s peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
            t_last, tok_last = time.time(), state["tokens"]
        if state["tokens"] >= next_eval:
            t1 = time.time(); do_eval("eval"); eval_cost = time.time() - t1 + 30
            save()
            next_eval += a.eval_every
    do_eval("final")
    save()
    for f in [x for x in a.final_evals.split(",") if x]:            # the trained LoRA under other flows
        t1 = time.time()
        r = evaluate(model, segs, R, Flow(f))
        log_result(f"{time.strftime('%m-%d %H:%M')} ckpt {a.run}@{state['tokens']} {f:18s} {fmt(r)}  ({time.time() - t1:.0f}s)")


def limit_memory():
    """Keep the process footprint near MLX's active memory: a 2 GB cap on the freed-buffer cache (turn 2's run reached
    a 45 GB footprint at ~31 GB active and pushed the Mac into swap) and a soft limit for the active set."""
    mx.set_cache_limit(int(float(os.environ.get("TL_CACHE_GB", "2")) * 2 ** 30))
    mx.set_memory_limit(int(float(os.environ.get("TL_MEM_GB", "36")) * 2 ** 30))


def main():
    limit_memory()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("flows", nargs="*")
    ap.add_argument("--flow"); ap.add_argument("--run")
    ap.add_argument("--tokens", type=int, default=10 ** 9)
    ap.add_argument("--seq", type=int, default=1024); ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--targets", default="gdn,attn,shared")
    ap.add_argument("--eval-every", type=int, default=262144)
    ap.add_argument("--aux", type=float, default=1.0)
    ap.add_argument("--lr-pre", type=float, default=2e-5)
    ap.add_argument("--time-limit", type=float, default=0)
    ap.add_argument("--duty", type=float, default=1.0)
    ap.add_argument("--final-evals", default="")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "eval":
        cmd_eval(a.flows, a.run)
    elif a.cmd == "train":
        cmd_train(a)


if __name__ == "__main__":
    main()
