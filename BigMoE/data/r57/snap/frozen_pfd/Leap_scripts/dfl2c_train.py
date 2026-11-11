"""Leap L3 step 2 (2026-10-23): fine-tune DFlash2 on agentic windows with regeneration-free labels (DFL2C_PREREG.md).
Inputs from dfl2c_extract.py: h = hidden_norm(fc(taps)) (bf16 bits), the 125B's top-32 ids / log-probs, ids, asst.
Frozen: fc + hidden_norm (precomputed), the 125B's 4-bit embedding and LM head. Trained (fp32 master weights): the 5
layers, the final norm, the candidate selector.
Blocks exactly as the official loop: [anchor, MASK x 7] at absolute positions a..a+7; context rows r <= a-1 with
(a + i) - r < 2048 for block position i (the official sliding mask, temporal order); full attention inside the block.
A step holds S groups of up to A consecutive assistant anchors (within 2A positions), each group sharing one union of
context rows. Loss at depths j = 1..7, weight w = [text tokens a+1..a+j-1 equal the 125B's greedy tokens] (on-path) x
[token a+j is in an assistant turn]: soft cross-entropy to the 125B's top-32 + the selector's cross-entropy over its 16
candidates with the true predecessor.
Modes: --check (rope + training forward vs the official path on the same blocks), --time (step time and memory),
default = train for --epochs within --minutes, then save the trained weights (bf16) to --out.
Usage: [BigMoE venv] dfl2c_train.py [--check|--time] [--epochs 1] [--minutes 75] [--lr 2e-5]"""
import argparse, glob, json, math, os, signal, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfd/Leap_third_party"))
import dflash_model_mlx as D

ap = argparse.ArgumentParser()
ap.add_argument("--check", action="store_true"); ap.add_argument("--time", action="store_true")
ap.add_argument("--epochs", type=float, default=1.0); ap.add_argument("--minutes", type=float, default=75)
ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--S", type=int, default=4)
ap.add_argument("--A", type=int, default=64); ap.add_argument("--val-every", type=int, default=100)
ap.add_argument("--out", default="~/Leap/data/dfl2c_ft.safetensors"); ap.add_argument("--resume", action="store_true")
ap.add_argument("--micro", type=int, default=2); ap.add_argument("--cache-gb", type=float, default=2.0)
ap.add_argument("--data", default="~/Leap/data/dfl2c")         # L4: ~/Leap/data/dfl2d (the 125B's own outputs)
ap.add_argument("--bs", type=int, default=8)                   # L5: 16 = depths 1-15 (block-16 extension)
ap.add_argument("--pmask", type=float, default=0.0)            # L1-20: share of training blocks with revealed positions
A = ap.parse_args()
W, BS, KD, H, KV, HD = 2048, A.bs, A.bs - 1, 32, 8, 128
# Memory (batch 2's first start, 20:47, was stopped by the swap guard after 10 steps: swap +4.4 GB). MLX keeps freed
# buffers of every context length it has seen, up to ~1.5x the GPU working set by default, so the footprint grows far
# past the 32 GB active peak. Cap that cache, round the context length up to a multiple of 128 so buffers are reused,
# and run each step's S groups as micro-batches of --micro groups with summed gradients (same loss and gradient).
(getattr(mx, "set_cache_limit", None) or mx.metal.set_cache_limit)(int(A.cache_gb * 2 ** 30))

class Target(nn.Module):
    """only the 125B's input embedding and LM head (4-bit affine, group 64), which draft.bind() needs"""
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.QuantizedEmbedding(248320, 2560, group_size=64, bits=4)
        self.lm_head = nn.QuantizedLinear(2560, 248320, bias=False, group_size=64, bits=4)

def load_target():
    w = mx.load(os.path.expanduser("~/BigMoE/models/bare/model-00016-of-00017.safetensors"))
    t = Target()
    t.load_weights([(f"embed_tokens.{s}", w[f"language_model.model.embed_tokens.{s}"]) for s in ("weight", "scales", "biases")]
                   + [(f"lm_head.{s}", w[f"language_model.lm_head.{s}"]) for s in ("weight", "scales", "biases")])
    mx.eval(t.parameters())
    return t

class Id(nn.Module):
    def __call__(self, x):
        return x

draft = D.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
draft.bind(load_target())
MASK, OUTM, BASE = int(draft.config.mask_token_id), float(draft.config.output_multiplier), float(draft.config.rope_theta)
draft.freeze()
TRAIN = list(draft.layers) + [draft.norm, draft.candidate_selector]
for m in TRAIN:
    m.unfreeze()
    m.update(tree_map(lambda p: p.astype(mx.float32), m.parameters()))
mx.eval(draft.parameters())
INV = mx.array((BASE ** (-np.arange(0, HD, 2) / HD)).astype(np.float32))

def bf16(hb):                       # uint16 bf16 bits -> mx bf16, exact
    return mx.array((hb.astype(np.uint32) << 16).view(np.float32)).astype(mx.bfloat16)

def rope(x, pos):                   # non-traditional RoPE (rotate halves), as nn.RoPE(traditional=False)
    ang = pos[..., None].astype(mx.float32) * INV
    c, s = mx.cos(ang), mx.sin(ang)
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    return mx.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1).astype(x.dtype)

def attn(sa, x, hc, kpos, qpos, neg, S, Aa, U):
    B = S * Aa
    q = rope(sa.q_norm(sa.q_proj(x).reshape(B, BS, H, HD)).transpose(0, 2, 1, 3), qpos) * sa.scale  # (B, H, 8, HD)
    kb = rope(sa.k_norm(sa.k_proj(x).reshape(B, BS, KV, HD)).transpose(0, 2, 1, 3), qpos)          # (B, KV, 8, HD)
    vb = sa.v_proj(x).reshape(B, BS, KV, HD).transpose(0, 2, 1, 3)
    kc = rope(sa.k_norm(sa.k_proj(hc).reshape(S, U, KV, HD)).transpose(0, 2, 1, 3), kpos)          # (S, KV, U, HD)
    vc = sa.v_proj(hc).reshape(S, U, KV, HD).transpose(0, 2, 1, 3)
    qg = q.reshape(S, Aa, KV, 4, BS, HD).transpose(0, 2, 1, 3, 4, 5).reshape(S, KV, Aa * 4 * BS, HD)
    sc = (qg @ kc.transpose(0, 1, 3, 2)).reshape(S, KV, Aa, 4, BS, U) + neg[:, None, :, None]
    sb = (q.reshape(B, KV, 4 * BS, HD) @ kb.transpose(0, 1, 3, 2)).reshape(S, Aa, KV, 4, BS, BS)
    p = mx.softmax(mx.concatenate([sc, sb.transpose(0, 2, 1, 3, 4, 5)], axis=-1), axis=-1)
    oc = (p[..., :U].reshape(S, KV, Aa * 4 * BS, U) @ vc).reshape(S, KV, Aa, 4 * BS, HD)
    oc = oc.transpose(0, 2, 1, 3, 4).reshape(B, KV, 4 * BS, HD)
    pb = p[..., U:].transpose(0, 2, 1, 3, 4, 5).reshape(B, KV, 4 * BS, BS)
    o = (oc + pb @ vb).reshape(B, H, BS, HD)
    return sa.o_proj(o.transpose(0, 2, 1, 3).reshape(B, BS, H * HD))

def fwd(hctx, upos, anc, tok):
    """hctx (S, U, D) context rows, upos (S, U) their absolute positions (-1 = pad), anc (S, A) anchors,
    tok (S, A, 8) block tokens -> normed hidden (S*A, 7, D) at block positions 1..7"""
    S, Aa = anc.shape; U = hctx.shape[1]; B = S * Aa
    h = (draft.embed_tokens(tok.reshape(B, BS)) * draft.embed_scale).astype(mx.float32)
    qpos = (anc.reshape(B, 1) + mx.arange(BS)[None]).reshape(B, 1, BS)
    r, a, i = upos[:, None, None, :], anc[:, :, None, None], mx.arange(BS)[None, None, :, None]
    neg = mx.where((r >= 0) & (r <= a - 1) & ((a + i) - r < W), 0.0, -1e9).astype(mx.float32)
    hc = hctx.astype(mx.float32)
    for L in draft.layers:
        x, k = L.attention_conv.prepare(L.input_layernorm(h))
        h = h + L.attention_conv.finish(attn(L.self_attn, x, hc, upos[:, None, :], qpos, neg, S, Aa, U), k)
        x, k = L.mlp_conv.prepare(L.post_attention_layernorm(h))
        h = h + L.mlp_conv.finish(L.mlp(x), k)
    return draft.norm(h[:, 1:])

def heads(hid, pred, cand=None):
    logits = draft.lm_head(hid.astype(mx.bfloat16)).astype(mx.float32) * OUTM
    if cand is None:
        cand = mx.stop_gradient(mx.argpartition(logits, -16, axis=-1)[..., -16:])
    sel = draft.candidate_selector
    sc = mx.take_along_axis(logits, cand, axis=-1) + (sel.predecessor_codebook(pred)[:, :, None]
         * sel.hidden_projection(hid)[:, :, None] * sel.successor_codebook(cand)).sum(-1)
    return logits, cand, sc

def loss_fn(batch, norm):           # norm = the whole step's weight, so micro-batch losses and gradients add up
    hctx, upos, anc, tok, tkl, lpl, y, pred, w = batch
    logits, cand, sc = heads(fwd(hctx, upos, anc, tok), pred)
    ce = mx.logsumexp(logits, axis=-1) - (mx.softmax(lpl, axis=-1) * mx.take_along_axis(logits, tkl, axis=-1)).sum(-1)
    inc = cand == y[..., None]
    sce = (mx.logsumexp(sc, axis=-1) - (sc * inc).sum(-1)) * inc.any(-1)
    return (w * (ce + sce)).sum() / norm

# ---------------------------------------------------------------- data
def load_windows():
    wins = []
    for p in sorted(glob.glob(os.path.join(os.path.expanduser(A.data), "*.npz"))):
        z = np.load(p)
        ids, asst, tk = z["ids"], z["asst"].astype(bool), z["tk"]
        n = tk.shape[0]
        match = np.zeros(n + 1, bool); match[1:] = tk[:, 0] == ids[1:n + 1]
        anchors = [a for a in range(n) if asst[a] and asst[a + 1]]
        wins.append(dict(key=os.path.basename(p)[:-4], split=str(z["split"]), h=z["h"], ids=ids, asst=asst, tk=tk,
                         lp=z["lp"], match=match, anchors=anchors, n=n))
    return wins

def groups_of(win):
    out, cur = [], []
    for a in win["anchors"]:
        if cur and (len(cur) == A.A or a - cur[0] >= 2 * A.A):
            out.append((win, cur)); cur = []
        cur.append(a)
    if cur:
        out.append((win, cur))
    return out

PM_RNG = np.random.default_rng(20)

def make_batch(items, pmask=0.0):
    """L1-20: with pmask > 0, that share of blocks reveals a random 10-60 % of its valid positions as the true (125B
    greedy) token in the block input, with no loss there, so the drafter learns to refine partly filled blocks"""
    S = len(items)
    U = -(-max(1, max(g[-1] - max(0, g[0] - (W - 1)) for _, g in items)) // 128) * 128   # padded rows are masked
    hc = np.zeros((S, U, 2560), np.uint16); upos = np.full((S, U), -1, np.int32)
    anc = np.zeros((S, A.A), np.int32); tok = np.full((S, A.A, BS), MASK, np.int32)
    B = S * A.A
    tkl = np.zeros((B, KD, 32), np.int32); lpl = np.zeros((B, KD, 32), np.float32)
    y = np.zeros((B, KD), np.int32); pred = np.zeros((B, KD), np.int32); w = np.zeros((B, KD), np.float32)
    for s, (win, g) in enumerate(items):
        lo, hi = max(0, g[0] - (W - 1)), g[-1]
        hc[s, :hi - lo] = win["h"][lo:hi]; upos[s, :hi - lo] = np.arange(lo, hi)
        for k in range(A.A):
            a = g[k] if k < len(g) else g[0]
            anc[s, k] = a; tok[s, k, 0] = win["ids"][a]
            if k >= len(g):
                continue
            b = s * A.A + k
            for j in range(1, KD + 1):
                r = a + j - 1
                if r > win["n"] - 1 or not win["asst"][a + j] or (j > 1 and not win["match"][a + j - 1]):
                    break
                tkl[b, j - 1] = win["tk"][r]; lpl[b, j - 1] = win["lp"][r]; y[b, j - 1] = win["tk"][r, 0]
                pred[b, j - 1] = win["ids"][a] if j == 1 else win["tk"][r - 1, 0]
                w[b, j - 1] = 1.0
            if pmask > 0 and PM_RNG.random() < pmask:
                valid = np.nonzero(w[b] > 0)[0]
                if valid.size > 1:
                    show = valid[PM_RNG.random(valid.size) < PM_RNG.uniform(0.1, 0.6)]
                    tok[s, k, show + 1] = y[b, show]
                    w[b, show] = 0.0
    return (bf16(hc), mx.array(upos), mx.array(anc), mx.array(tok), mx.array(tkl), mx.array(lpl), mx.array(y),
            mx.array(pred), mx.array(w))

def evaluate(groups):
    """weighted on-path accuracy per depth: drafter argmax and the selector's teacher-forced pick vs the 125B greedy"""
    hit_l, hit_s, den = np.zeros(KD), np.zeros(KD), np.zeros(KD)
    for k in range(0, len(groups), A.micro):
        hctx, upos, anc, tok, tkl, lpl, y, pred, w = make_batch(groups[k:k + A.micro])
        logits, cand, sc = heads(fwd(hctx, upos, anc, tok), pred)
        pick = mx.take_along_axis(cand, mx.argmax(sc, axis=-1)[..., None], axis=-1)[..., 0]
        mx.eval(logits, pick)
        wn = np.array(w)
        hit_l += (np.array(mx.argmax(logits, axis=-1) == y) * wn).sum(0); hit_s += (np.array(pick == y) * wn).sum(0)
        den += wn.sum(0)
    return hit_l / np.maximum(den, 1), hit_s / np.maximum(den, 1), den

# ---------------------------------------------------------------- modes
if A.check:
    x = mx.random.normal((1, H, BS, HD)); off = 3001
    d_rope = float(mx.abs(draft.rope(x, offset=off) - rope(x, mx.arange(off, off + BS)[None, None])).max())
    wins = load_windows()
    win = max(wins, key=lambda w_: len([a for a in w_["anchors"] if a > W]))     # full windows where possible
    anchors = ([a for a in win["anchors"] if a > W] or win["anchors"])[:A.A]
    batch = make_batch([(win, anchors)])
    logits_m, _, _ = heads(fwd(*batch[:4]), batch[7])
    fc, hn = draft.fc, draft.hidden_norm
    draft.fc, draft.hidden_norm = Id(), Id()
    ref = []
    for a in anchors:
        cache = draft.make_cache(); start = max(0, a - (W - 1))
        for c in cache:
            c.offset = start
        hid = draft.hidden_states(mx.array([[int(win["ids"][a])] + [MASK] * (BS - 1)]), bf16(win["h"][start:a])[None],
                                  cache, logits_start=1)
        ref.append(draft.compute_logits(hid).astype(mx.float32)[0])
    draft.fc, draft.hidden_norm = fc, hn
    ref = mx.stack(ref); mine = logits_m[:len(anchors)]
    agree = float((mx.argmax(ref, -1) == mx.argmax(mine, -1)).astype(mx.float32).mean())
    print(f"[dfl2c-t] CHECK rope max|diff| {d_rope:.2e}; forward vs official on {len(anchors)} blocks x 7: "
          f"max|logit diff| {float(mx.abs(ref - mine).max()):.3e}, argmax agreement {agree:.4f} "
          f"({'PASS' if agree >= 0.99 else 'FAIL'})", flush=True)
    sys.exit(0 if agree >= 0.99 else 1)

wins = load_windows()
tr = [g for w_ in wins if w_["split"] == "train" for g in groups_of(w_)]
va = [g for w_ in wins if w_["split"] == "val" for g in groups_of(w_)]
rng = np.random.default_rng(0)
va = [va[i] for i in sorted(rng.choice(len(va), min(len(va), 24), replace=False))] if va else []
n_anchor = sum(len(g) for _, g in tr)
print(f"[dfl2c-t] {len(wins)} windows; train {len(tr)} groups / {n_anchor} anchors; val {len(va)} groups", flush=True)
loss_and_grad = nn.value_and_grad(draft, lambda m, b, norm: loss_fn(b, norm))
steps = max(1, int(math.ceil(A.epochs * len(tr) / A.S))) if not A.time else 5
warm = min(20, steps // 10 + 1)
opt = optim.AdamW(learning_rate=A.lr, weight_decay=0.0)

def lr_at(s):                       # linear warm-up, then cosine to 10 %
    if s < warm:
        return A.lr * (0.01 + 0.99 * s / warm)
    return A.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, steps - warm)))))

def step(items):
    parts = [make_batch(items[k:k + A.micro], A.pmask) for k in range(0, len(items), A.micro)]
    norm = mx.array(max(1.0, sum(float(b[-1].sum()) for b in parts)))
    loss, acc = 0.0, None
    for b in parts:
        l, g = loss_and_grad(draft, b, norm)
        acc = g if acc is None else tree_map(mx.add, acc, g)
        mx.eval(l, acc)
        loss += float(l)
    acc, _ = optim.clip_grad_norm(acc, 1.0)
    opt.update(draft, acc)
    mx.eval(draft.trainable_parameters(), opt.state)
    return loss

def mem():
    return (f"mem {mx.get_active_memory() / 1e9:.1f} + cache {mx.get_cache_memory() / 1e9:.1f} GB "
            f"(peak {mx.get_peak_memory() / 1e9:.1f})")

if A.time:
    ts = []
    for k in range(steps):
        t0 = time.time(); l = step([tr[int(i)] for i in rng.choice(len(tr), A.S)]); ts.append(time.time() - t0)
    st = float(np.median(ts[1:]))
    print(f"[dfl2c-t] TIME S={A.S} (micro {A.micro}) A={A.A}: step {st:.2f}s (first {ts[0]:.1f}s), loss {l:.3f}, "
          f"{mem()}; one epoch = {len(tr) / A.S:.0f} steps = {len(tr) / A.S * st / 60:.0f} min", flush=True)
    sys.exit(0)

# Pause / stop safety (orchestrator 10-23): SIGTERM or SIGINT -> finish the step, checkpoint, exit 3; a checkpoint every
# 10 min. Checkpoint = the trained weights in bf16 at --out (same file as the final) + .state.json (step, data order,
# rng, training time, log). --resume reloads them; AdamW moments restart, with a 10-step learning-rate ramp.
# Training time counts steps only (a step over 60 s counts as 60 s, so a paused job does not eat the budget).
STOP = {"now": False}
signal.signal(signal.SIGTERM, lambda *_: STOP.update(now=True))
signal.signal(signal.SIGINT, lambda *_: STOP.update(now=True))
out = os.path.expanduser(A.out); stf = out.replace(".safetensors", ".state.json")

def save(state, status):
    tmp = out.replace(".safetensors", ".tmp.safetensors")
    mx.save_safetensors(tmp, {k_: v.astype(mx.bfloat16) for k_, v in tree_flatten(draft.trainable_parameters())})
    os.replace(tmp, out)
    state["status"] = status
    json.dump(state, open(stf + ".tmp", "w")); os.replace(stf + ".tmp", stf)

def val(tag):
    if not va:
        return
    al, as_, den = evaluate(va)
    state["log"]["val"].append({"tag": tag, "argmax": al.tolist(), "selector": as_.tolist(), "n": den.tolist()})
    print(f"[dfl2c-t] VAL {tag}: on-path top-1 by depth " + " ".join(f"{x:.3f}" for x in al) + " | selector "
          + " ".join(f"{x:.3f}" for x in as_) + f" | n {int(den[0])}..{int(den[-1])}", flush=True)

if A.resume:
    state = json.load(open(stf))
    draft.load_weights(out, strict=False)
    for m in TRAIN:
        m.update(tree_map(lambda p: p.astype(mx.float32), m.parameters()))
    mx.eval(draft.parameters())
    rng.bit_generator.state = state["rng"]
    state["resumes"] += 1
    print(f"[dfl2c-t] RESUME at step {state['s']}/{steps} ({state['train_time'] / 60:.1f} min trained)", flush=True)
else:
    state = {"s": 0, "order": [], "train_time": 0.0, "ema": None, "resumes": 0, "rng": None, "status": "running",
             "log": {"args": vars(A), "steps": steps, "train_groups": len(tr), "train_anchors": n_anchor, "val": [],
                     "loss": []}}
    val("before")
s0, last_ck = state["s"], time.time()
while state["s"] < steps:
    s = state["s"]
    if not state["order"]:
        state["order"] = [int(x) for x in rng.permutation(len(tr))]
    items = [tr[state["order"].pop()] for _ in range(min(A.S, len(state["order"])))]
    opt.learning_rate = lr_at(s) * (min(1.0, (s - s0 + 1) / 10) if state["resumes"] else 1.0)
    t1 = time.time(); l = step(items); state["train_time"] += min(time.time() - t1, 60.0)
    state["s"] = s + 1; state["ema"] = l if state["ema"] is None else 0.98 * state["ema"] + 0.02 * l
    state["log"]["loss"].append(l); state["rng"] = rng.bit_generator.state
    if (s + 1) % 20 == 0 or s == s0:
        print(f"[dfl2c-t] step {s + 1}/{steps} loss {l:.3f} (ema {state['ema']:.3f}) lr {float(opt.learning_rate):.2e} "
              f"{state['train_time'] / 60:.1f} min trained, {mem()}", flush=True)
    if STOP["now"]:
        save(state, "stopped")
        print(f"[dfl2c-t] STOPPED by signal at step {s + 1}: checkpoint saved; continue with --resume", flush=True)
        sys.exit(3)
    if (s + 1) % A.val_every == 0:
        val(f"step {s + 1}")
    if time.time() - last_ck > 600:
        save(state, "running"); last_ck = time.time()
    if state["train_time"] > A.minutes * 60:
        print(f"[dfl2c-t] time budget reached at step {s + 1}/{steps}", flush=True)
        break
val("after")
save(state, "done")
print(f"[dfl2c-t] DONE: saved {out} after {state['train_time'] / 60:.1f} min of training, {state['s']} steps", flush=True)
