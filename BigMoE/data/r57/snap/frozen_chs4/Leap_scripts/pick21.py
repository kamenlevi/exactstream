"""L1-21 (Leap, Sat 2026-10-24, pre-registered in ~/Leap/TABLE_350.md): an autoregressive picker over DFlash2's top-64
lattice. L1-20's oracle: the 125B's token is in the drafter's top-64 at >= 91 % of depths (14.11 tokens per 16-row pass if
picked perfectly) but the shipped first-order selector reaches 9.54.
Steps (one process, < 10 GB):
 1. Lattices on L1-20's 109 windows (data/l120: the 125B's greedy outputs, precomputed context features h), every
    P21_STRIDE-th assistant anchor, official block layout [anchor, MASK x (bs-1)] with the sliding context cache:
    the drafter's hidden state per depth, its top-64 candidates (sorted) and logits, the 125B's greedy token per depth
    (labels and validity exactly as dfl2c_train.py's make_batch).
 2. Picker (~5 M parameters): 3-layer causal transformer over depths, d 512. Input at depth j: the drafter's hidden
    state at j + the 125B's embedding of the token at j-1 (anchor at j = 1) + a depth embedding. Candidate i at depth j:
    the selector's successor codebook row (frozen) through a linear map + rank embedding + its drafter log-softmax
    (within the lattice) features; score = <q_j, r_ji> / 16 + a * log-softmax. Teacher forcing on the 125B's tokens;
    cross-entropy over the 64 candidates where the true token is in the lattice. Train split only; val split reported.
 3. Held-out replay (data/dfl2d_code, 24 HumanEval samples, the official loop as cand21.py): selector vs picker
    (greedy left to right, computed as Jacobi iterations: same fixed point) vs oracle, tokens per pass, and tok/s at
    the stack with running-confidence truncation (t 0 / 0.1 / 0.3 / 0.5) for both expert settings (p 0.8: 5.06 GB,
    top-8: 5.52 GB per 16-row pass).
Usage: [BigMoE venv] [P21_WEIGHTS=data/l120_ft.safetensors] [P21_TAG=_ft] [P21_BS=16,32] [P21_STRIDE=2] pick21.py
 -> ~/Leap/data/pick21{tag}.txt"""
import os, sys, glob, time, math
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx.utils import tree_flatten
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs4/Leap_third_party"))
import dflash_model_mlx as D

(getattr(mx, "set_cache_limit", None) or mx.metal.set_cache_limit)(int(3 * 2 ** 30))
WEIGHTS = os.environ.get("P21_WEIGHTS", "")
TAG = os.environ.get("P21_TAG", "")
BSS = [int(x) for x in os.environ.get("P21_BS", "16,32").split(",")]
STRIDE = int(os.environ.get("P21_STRIDE", "2"))
K, W = 64, 2048
SMOKE = os.environ.get("P21_SMOKE") == "1"          # 3 windows, 20 steps, 2 replay samples: catches bugs in ~1 min

class Target(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.QuantizedEmbedding(248320, 2560, group_size=64, bits=4)
        self.lm_head = nn.QuantizedLinear(2560, 248320, bias=False, group_size=64, bits=4)
w = mx.load(os.path.expanduser("~/BigMoE/models/bare/model-00016-of-00017.safetensors"))
tgt = Target()
tgt.load_weights([(f"embed_tokens.{s}", w[f"language_model.model.embed_tokens.{s}"]) for s in ("weight", "scales", "biases")]
                 + [(f"lm_head.{s}", w[f"language_model.lm_head.{s}"]) for s in ("weight", "scales", "biases")])
del w
draft = D.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
draft.bind(tgt)
if WEIGHTS:
    draft.load_weights(os.path.expanduser(WEIGHTS), strict=False)
mx.eval(draft.parameters(), tgt.parameters())
MASK = int(draft.config.mask_token_id)
SEL = draft.candidate_selector
FC, HN = draft.fc, draft.hidden_norm
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

class Id(nn.Module):
    def __call__(self, x):
        return x

def bf16(hb):
    return mx.array((hb.astype(np.uint32) << 16).view(np.float32)).astype(mx.bfloat16)

def lattice(hid):
    """top-K candidates per depth, sorted by the drafter's logit, their logits (float32), and the raw logits"""
    logits = draft.compute_logits(hid)[0]                                    # [D, V], as the shipped selector sees them
    cand = mx.argpartition(logits, -K, axis=-1)[:, -K:]
    u = mx.take_along_axis(logits, cand, axis=-1).astype(mx.float32)
    order = mx.argsort(-u, axis=-1)
    return mx.take_along_axis(cand, order, axis=-1), mx.take_along_axis(u, order, axis=-1), logits

def select_conf(anchor, hid, logits):
    """the shipped selector at temperature 0 (as propose) and the selected candidate's softmax per depth"""
    cand = mx.argpartition(logits[None], -SEL.top_k, axis=-1)[..., -SEL.top_k:]
    unary = mx.take_along_axis(logits[None], cand, axis=-1)
    hp = SEL.hidden_projection(hid)
    pred, path, conf = mx.array([anchor]), [], []
    for p in range(hp.shape[1]):
        sc = unary[:, p] + mx.sum(SEL.predecessor_codebook(pred)[:, None] * hp[:, p, None] * SEL.successor_codebook(cand[:, p]), axis=-1)
        s = mx.argmax(sc, axis=-1)
        conf.append(mx.take_along_axis(mx.softmax(sc.astype(mx.float32), axis=-1), s[:, None], axis=-1)[:, 0])
        pred = mx.take_along_axis(cand[:, p], s[:, None], axis=-1)[:, 0]
        path.append(pred)
    return mx.stack(path, axis=1)[0], mx.stack(conf, axis=1)[0]

# ---------------------------------------------------------------- 1. lattices on the training windows
def load_windows():
    wins = []
    for p in sorted(glob.glob(os.path.expanduser("~/Leap/data/l120/*.npz"))):
        z = np.load(p)
        ids, asst, tk = z["ids"], z["asst"].astype(bool), z["tk"]
        n = tk.shape[0]
        match = np.zeros(n + 1, bool); match[1:] = tk[:, 0] == ids[1:n + 1]
        anchors = [a for a in range(n) if asst[a] and asst[a + 1]]
        wins.append(dict(split=str(z["split"]), h=z["h"], ids=ids, asst=asst, tk=tk, match=match, anchors=anchors, n=n))
    if SMOKE:
        wins = [w_ for w_ in wins if w_["split"] == "train"][:2] + [w_ for w_ in wins if w_["split"] == "val"][:1]
    return wins

def dump(wins, bs):
    Dp = bs - 1
    draft.fc, draft.hidden_norm = Id(), Id()
    out = {k: [] for k in ("H", "C", "U", "Y", "P", "V", "S")}
    t0 = time.time()
    for win in wins:
        anchors = win["anchors"][::STRIDE]
        if not anchors:
            continue
        cache = draft.make_cache(); start = max(0, anchors[0] - (W - 1))
        for c in cache:
            c.offset = start
        fed = start
        for a in anchors:
            if a <= fed:
                continue
            ctx = bf16(win["h"][fed:a])[None]
            hid = draft.hidden_states(mx.array([[int(win["ids"][a])] + [MASK] * Dp]), ctx, cache, 1)
            cand, u, _ = lattice(hid)
            fed = a
            if (trim := cache[0].offset - a) > 0:
                D._trim_recent_cache(cache, trim)
            y = np.zeros(Dp, np.int32); pv = np.zeros(Dp, np.int32); v = np.zeros(Dp, bool)
            for j in range(1, Dp + 1):
                r = a + j - 1
                if r > win["n"] - 1 or not win["asst"][a + j] or (j > 1 and not win["match"][a + j - 1]):
                    break
                y[j - 1] = win["tk"][r, 0]; pv[j - 1] = win["ids"][a] if j == 1 else win["tk"][r - 1, 0]; v[j - 1] = True
            if not v[0]:
                continue
            mx.eval(hid, cand, u)
            out["H"].append(hid[0].astype(mx.bfloat16)); out["C"].append(cand); out["U"].append(u)
            out["Y"].append(y); out["P"].append(pv); out["V"].append(v); out["S"].append(win["split"] == "train")
    draft.fc, draft.hidden_norm = FC, HN
    H = mx.stack(out["H"]); C = mx.stack(out["C"]); U = mx.stack(out["U"])
    Y, P, V = (np.stack(out[k]) for k in ("Y", "P", "V")); S = np.array(out["S"])
    mx.eval(H, C, U)
    inl = (np.array(C) == Y[..., None]).any(-1) & V
    say(f"  lattices block {bs}: {len(S)} blocks ({S.sum()} train / {(~S).sum()} val), {time.time() - t0:.0f} s; "
        f"true token in top-{K} at valid depths: {inl.sum() / V.sum():.3f} (train {inl[S].sum() / V[S].sum():.3f}, "
        f"val {inl[~S].sum() / V[~S].sum():.3f})")
    return H, C, U, Y, P, V, S

# ---------------------------------------------------------------- 2. the picker
class Picker(nn.Module):
    def __init__(self, Dmax, d=512, L=3, nh=8, r=256):
        super().__init__()
        self.ph = nn.Linear(2560, d, bias=False)
        self.pe = nn.Linear(2560, d, bias=False)
        self.pos = nn.Embedding(Dmax, d)
        self.layers = [nn.TransformerEncoderLayer(d, nh, 4 * d, norm_first=True) for _ in range(L)]
        self.ln = nn.LayerNorm(d)
        self.q = nn.Linear(d, r, bias=False)
        self.pc = nn.Linear(256, r, bias=False)
        self.rank = nn.Embedding(K, r)
        self.uf = nn.Linear(4, r, bias=False)
        self.ua = mx.array([1.0])
        self.ue = mx.array([1.0])

    def __call__(self, Hd, Ep, Cc, U, Pc):
        """Hd drafter hidden [B, D, 2560]; Ep 125B embedding of the previous token; Cc successor codes of the candidates
        [B, D, K, 256]; U their drafter logits; Pc predecessor codes of the previous token [B, D, 256]. Starts as the
        shipped selector (unary + first-order edge) plus a learned term."""
        B, Dn, _ = Hd.shape
        x = self.ph(Hd) + self.pe(Ep) + self.pos(mx.arange(Dn))
        mask = nn.MultiHeadAttention.create_additive_causal_mask(Dn).astype(x.dtype)
        for l in self.layers:
            x = l(x, mask)
        q = self.q(self.ln(x))
        lsm = U - mx.logsumexp(U, axis=-1, keepdims=True)
        edge = (Pc[:, :, None, :] * SEL.hidden_projection(Hd.astype(mx.bfloat16)).astype(mx.float32)[:, :, None, :]
                * Cc).sum(-1)
        edge = mx.stop_gradient(edge)
        f = mx.stack([lsm, U - U[..., :1], mx.exp(lsm), edge / 10.0], axis=-1)
        rr = self.pc(Cc) + self.rank(mx.arange(K)) + self.uf(f)
        return (q[:, :, None, :] * rr).sum(-1) / 16.0 + self.ua * lsm + self.ue * edge

def emb(tokens):
    return draft.embed_tokens(tokens).astype(mx.float32)

def codes(cand):
    return SEL.successor_codebook(cand).astype(mx.float32)

def batch_inputs(H, C, U, P, idx):
    i = mx.array(idx); pv = mx.array(P[idx])
    return (H[i].astype(mx.float32), emb(pv), codes(C[i]), U[i],
            SEL.predecessor_codebook(pv).astype(mx.float32))

def labels(C, Y, V, idx):
    hit = (np.array(C[mx.array(idx)]) == Y[idx][..., None])
    lab = np.where(hit.any(-1) & V[idx], hit.argmax(-1), -1)
    return mx.array(lab)

def loss_fn(model, Hd, Ep, Cc, U, Pc, lab):
    s = model(Hd, Ep, Cc, U, Pc).astype(mx.float32)
    m = (lab >= 0).astype(mx.float32)
    tgt = mx.take_along_axis(s, mx.maximum(lab, 0)[..., None], axis=-1)[..., 0]
    return ((mx.logsumexp(s, axis=-1) - tgt) * m).sum() / mx.maximum(m.sum(), 1.0)

def teacher_forced(model, H, C, U, Y, P, V, idx):
    """per-depth accuracy (true prefix) of: picker, the drafter's top-1, and accepted length from each anchor"""
    hit_p, hit_1, den, acc_p, acc_1 = 0, 0, 0, [], []
    for k in range(0, len(idx), 128):
        b = idx[k:k + 128]
        s = model(*batch_inputs(H, C, U, P, b))
        pick = mx.take_along_axis(C[mx.array(b)], mx.argmax(s, axis=-1)[..., None], axis=-1)[..., 0]
        pk, t1 = np.array(pick), np.array(C[mx.array(b)][..., 0])
        v = V[b]; y = Y[b]
        hit_p += ((pk == y) & v).sum(); hit_1 += ((t1 == y) & v).sum(); den += v.sum()
        for row in range(len(b)):
            for arr, store in ((pk, acc_p), (t1, acc_1)):
                ok = (arr[row] == y[row]) & v[row]
                store.append(int(np.argmin(ok)) if not ok.all() else int(ok.sum()))
    return hit_p / den, hit_1 / den, np.mean(acc_p), np.mean(acc_1)

def train_picker(data, bs):
    H, C, U, Y, P, V, S = data
    tr, va = np.nonzero(S)[0], np.nonzero(~S)[0]
    model = Picker(bs - 1)
    mx.eval(model.parameters())
    nparam = sum(v.size for _, v in tree_flatten(model.parameters()))
    epochs, B = 6, 64
    steps = 20 if SMOKE else epochs * (len(tr) // B)
    sched = optim.cosine_decay(3e-4, steps, 3e-5)
    opt = optim.AdamW(learning_rate=sched, weight_decay=0.01)
    lg = nn.value_and_grad(model, loss_fn)
    rng = np.random.default_rng(21); t0 = time.time(); ema = None
    for s in range(steps):
        idx = rng.choice(tr, B, replace=False)
        loss, g = lg(model, *batch_inputs(H, C, U, P, idx), labels(C, Y, V, idx))
        opt.update(model, g); mx.eval(model.parameters(), opt.state, loss)
        ema = float(loss) if ema is None else 0.98 * ema + 0.02 * float(loss)
        if (s + 1) % 200 == 0 or s + 1 == steps:
            print(f"    step {s + 1}/{steps} loss {ema:.3f} ({time.time() - t0:.0f} s)", flush=True)
    a_p, a_1, l_p, l_1 = teacher_forced(model, H, C, U, Y, P, V, va)
    t_p, t_1, tl_p, tl_1 = teacher_forced(model, H, C, U, Y, P, V, tr[:len(va)])
    say(f"  picker block {bs}: {nparam / 1e6:.1f} M parameters, {steps} steps, {time.time() - t0:.0f} s. Teacher-forced "
        f"accuracy (val / train): picker {a_p:.3f} / {t_p:.3f}, drafter top-1 {a_1:.3f} / {t_1:.3f}; accepted run from "
        f"every anchor (val): picker {l_p:.2f}, drafter top-1 {l_1:.2f}")
    return model

def pick_path(model, anchor, hid, cand, u):
    """greedy left-to-right pick, computed as Jacobi iterations (same fixed point); per-depth confidence"""
    Dn = cand.shape[0]
    Cc, Hd, U1 = codes(cand)[None], hid.astype(mx.float32)[None], u[None]
    y = np.array(cand[:, 0]); conf = None
    for _ in range(Dn):
        prev = mx.array(np.concatenate([[anchor], y[:-1]]))
        s = model(Hd, emb(prev)[None], Cc, U1, SEL.predecessor_codebook(prev).astype(mx.float32)[None])[0]
        pi = mx.argmax(s, axis=-1)
        new = np.array(mx.take_along_axis(cand, pi[:, None], axis=-1)[:, 0])
        conf = np.array(mx.take_along_axis(mx.softmax(s, axis=-1), pi[:, None], axis=-1)[:, 0])
        if (new == y).all():
            break
        y = new
    return y, conf

# ---------------------------------------------------------------- 3. held-out replay, priced
def pass_ms(rows, bs, experts):
    return (2.1 + experts * (rows / 16) ** 0.72) / 224 * 1000 + (3.5 if bs <= 16 else 4.0)

samples = []
for path in sorted(glob.glob(os.path.expanduser("~/Leap/data/dfl2d_code/*.npz"))):
    z = np.load(path)
    samples.append((z["f"].reshape(z["f"].shape[0], -1), z["ids"], int(z["off"]), int(z["gen"])))
if SMOKE:
    samples = samples[:2]

def replay(bs, policy, t, model=None):
    tok = blocks = rows = 0; ms = ms8 = 0.0
    for f, ids, off, gen in samples:
        n = len(ids); cache = draft.make_cache()
        start = max(off, gen - (draft.config.sliding_window - 1))
        for c in cache:
            c.offset = start
        fed, a = start, gen
        while a < off + n - 1:
            ctx = mx.array(f[fed - off:a - off][None]).astype(mx.bfloat16)
            anchor = int(ids[a - off])
            hid = draft.hidden_states(mx.array([[anchor] + [MASK] * (bs - 1)]), ctx, cache, 1)
            fed = a
            true = np.asarray(ids[a - off + 1:a - off + bs])
            cand, u, logits = lattice(hid)
            if policy == "selector":
                p, c = select_conf(anchor, hid, logits); mx.eval(p, c); d, conf = np.array(p), np.array(c)
                hit = d[:len(true)] == true
            elif policy == "picker":
                d, conf = pick_path(model, anchor, hid[0], cand, u)
                hit = d[:len(true)] == true
            else:
                mx.eval(cand); hit = (np.array(cand)[:len(true)] == true[:, None]).any(-1); conf = np.ones(bs - 1)
            keep = int(np.sum(np.cumprod(conf.astype(np.float64)) >= t))
            k = int(np.argmin(hit)) if not hit.all() else len(true)
            acc = min(min(k, keep) + 1, off + n - 1 - a)
            tok += acc; blocks += 1; rows += keep + 1
            ms += pass_ms(keep + 1, bs, 5.06); ms8 += pass_ms(keep + 1, bs, 5.52)
            if (trim := cache[0].offset - a) > 0:
                D._trim_recent_cache(cache, trim)
            a = a + acc
    return tok / blocks, rows / blocks, tok / ms * 1000, tok / ms8 * 1000

say(f"L1-21: autoregressive picker over DFlash2's top-{K} lattice; drafter {WEIGHTS or 'incoai (original)'}; "
    f"training lattices every {STRIDE}nd anchor of L1-20's windows; held-out replay on 24 HumanEval samples")
wins = load_windows()
for bs in BSS:
    data = dump(wins, bs)
    model = train_picker(data, bs)
    del data
    for policy in ("selector", "picker", "oracle"):
        for t in ((0.0,) if policy == "oracle" else (0.0, 0.1, 0.3, 0.5)):
            t0 = time.time()
            tpp, rw, ts, ts8 = replay(bs, policy, t, model)
            say(f"  block {bs:2d} {policy:8s} truncate t {t:.1f}: {tpp:5.2f} tokens per pass at {rw:4.1f} rows; at the "
                f"stack {ts:4.0f} tok/s (top-8: {ts8:4.0f}) ({time.time() - t0:.0f} s)")
say("Lines (L1-21): kill < 10.5 tokens per 16-row pass (FT selector 9.54) and < +10 % priced at block 32; "
    "pass >= 13.1 per 16-row pass or >= 350 tok/s at the top-8 stack")
open(os.path.expanduser(f"~/Leap/data/pick21{TAG}.txt"), "w").write("\n".join(lines) + "\n")
