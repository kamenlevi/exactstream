"""TTS1 (MTP Head, Tue 11-03 night; pre-registered in REPORT 'TTS1'; MoonShots' WALL2_KEPT item 1, Test-Time Speculation
arXiv 2605.09329): a teacher-forced replay of DFlash2 L1-20 along the 125B's own windows, with one Adam step on a LoRA
adapter after every check, against the frozen drafter in the same harness.
Loop: the live official loop with the t 0.3 cut (drafts kept while the product of their cf >= 0.3; NewIdeas'
live_segments rule): at anchor A the drafter proposes 15 (own selector chain + cf); k = accepted vs the 125B's tokens;
L = drafts inside the cut; kept = min(k, L) + 1; next anchor = A + kept.
Update (leak-free): after the check, the labels the live engine has are the verified rows 0..min(k, L) (the accepted
drafts and the bonus row), all conditioned on the true prefix. Loss = the L1-20 trainer's (Leap dfl2c_train.py): soft
cross-entropy to the 125B's top-32 (D-02a sets; hard CE on the 125B's token where no top-32 is stored: Leap's code
set) + the selector's cross-entropy over its 16 candidates with the true predecessor; mean over the rows. One Adam
step (K = 1; with one step from q_t, TTS's proximal term has zero gradient, so it is omitted). LoRA r 16, alpha 32, on
every Linear of the 5 layers' attention and MLP; the base, fc, norms, convs, selector and the 125B's head stay frozen.
The context K / V the drafter attends to are the ones its drafting pass cached (recorded per layer); the update does
not recompute old context rows (live semantics). Reset (fresh LoRA, fresh Adam) per window.
Arms: TTS_LRS (default 0, 1e-4, 1e-3; lr 0 = the frozen drafter through the same code path, adapter B = 0).
Read-out per set: kept per check (cut), raw accepted + 1 per check, rows verified per check, by answer position
(< 150 / 150-300 / >= 300), ms per update. Validation: the update's forward reproduces the drafting pass (first checks).
Usage: [BigMoE venv] tts_replay.py   (env TTS_SETS 'tag=glob,...', TTS_NMAX windows per set, TTS_LRS)
Output: data/tts_replay.txt (appended) + data/tts_replay_<tag>.npz (per check records per arm)."""
import glob, os, sys, time, zlib
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfx/Leap_third_party"))
import dflash_model_mlx as DF
from mlx_lm.tuner.lora import LoRALinear

V, K, BS = 248320, 16, 16
LD, TD = os.path.expanduser("~/Leap/data"), os.path.expanduser("~/TrainLab/data")
LIVE_T = 0.3
LRS = [float(x) for x in os.environ.get("TTS_LRS", "0,1e-4,1e-3").split(",")]
R, ALPHA = 16, 32.0
SETS = os.environ.get("TTS_SETS", "code=~/Leap/data/dfl2d_code/*.npz,ifeval=~/TrainLab/data/d02a/tf_ifeval_*.npz,"
                      "multi=~/TrainLab/data/d02a/tf_multi_*.npz,agent=~/TrainLab/data/d02a/agent_*.npz")
NMAX = int(os.environ.get("TTS_NMAX", "0"))
NCHECK = int(os.environ.get("TTS_NCHECK", "3"))      # checks per arm whose update forward is validated
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)


class Target(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.QuantizedEmbedding(V, 2560, group_size=64, bits=4)
        self.lm_head = nn.QuantizedLinear(2560, V, bias=False, group_size=64, bits=4)


w = mx.load(os.path.expanduser("~/BigMoE/models/bare/model-00016-of-00017.safetensors"))
tgt = Target()
tgt.load_weights([(f"embed_tokens.{s}", w[f"language_model.model.embed_tokens.{s}"]) for s in ("weight", "scales", "biases")]
                 + [(f"lm_head.{s}", w[f"language_model.lm_head.{s}"]) for s in ("weight", "scales", "biases")])
del w
mx.eval(tgt.parameters())
d = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
d.load_weights(os.environ.get("TL_DRAFT", os.path.join(LD, "l120_ft.safetensors")), strict=False)
d.bind(tgt)
mx.eval(d.parameters())
MASK = int(d.config.mask_token_id)
d.freeze()
LORAS = []
for li, layer in enumerate(d.layers):
    for part in (layer.self_attn, layer.mlp):
        for name, m in list(part.children().items()):
            if isinstance(m, nn.Linear) and not isinstance(m, nn.QuantizedLinear):
                lo = LoRALinear.from_base(m, r=R, scale=ALPHA / R)
                setattr(part, name, lo)
                LORAS.append((f"{li}.{name}", lo))
NTRAIN = sum(v.size for _, v in tree_flatten(d.trainable_parameters()))
say(f"TTS1 {time.strftime('%m-%d %H:%M')}: DFlash2 L1-20 + LoRA r {R} alpha {ALPHA:.0f} on {len(LORAS)} Linears "
    f"({NTRAIN / 1e6:.2f} M trainable); lrs {LRS}; cut t {LIVE_T}")


def reset_lora(seed):
    key = mx.random.key(seed)
    for _, lo in LORAS:
        key, sub = mx.random.split(key)
        ind, r = lo.lora_a.shape
        lo.lora_a = mx.random.uniform(-1 / ind ** 0.5, 1 / ind ** 0.5, (ind, r), key=sub)
        lo.lora_b = mx.zeros((r, lo.lora_b.shape[1]))
    mx.eval(d.trainable_parameters())


def head2s(hid):
    """the exact two-stage top-16 (R-06), as d02b_walk.py"""
    lg = d.compute_logits(hid)
    B, s = 128, lg.shape
    blk = lg.reshape(*s[:-1], s[-1] // B, B)
    tb = mx.argpartition(blk.max(axis=-1), -K, axis=-1)[..., -K:]
    sel = mx.take_along_axis(blk, mx.broadcast_to(tb[..., None], (*s[:-1], K, B)), axis=-2).reshape(*s[:-1], K * B)
    j = mx.argpartition(sel, -K, axis=-1)[..., -K:]
    c = mx.take_along_axis(tb, j // B, axis=-1) * B + j % B
    return c, mx.take_along_axis(lg, c, axis=-1)


def chain_cf(blk, unary, cand, hp):
    """the own selector chain and each pick's cf (selector softmax at the pick), as d02b_walk.py chain_cf"""
    sel = d.candidate_selector
    pred, path, conf = blk[:, 0], [], []
    for p in range(hp.shape[1]):
        sc = unary[:, p] + mx.sum(sel.predecessor_codebook(pred)[:, None] * hp[:, p, None]
                                  * sel.successor_codebook(cand[:, p]), axis=-1)
        s_ = mx.argmax(sc, axis=-1)
        pick = mx.take_along_axis(cand[:, p], s_[:, None], axis=-1)[:, 0]
        path.append(pick)
        conf.append(mx.take_along_axis(mx.softmax(sc.astype(mx.float32), axis=-1), s_[:, None], axis=-1)[:, 0])
        pred = pick
    return mx.stack(path, axis=1), mx.stack(conf, axis=1)


def train_hidden(model, blk, rec, a):
    """DFlash2DecoderLayer / DFlashAttention's forward for the block alone, attending to the context K / V the drafting
    pass returned from its caches (rec, constants); RoPE offset = the anchor's position a"""
    h = model.embed_tokens(blk) * model.embed_scale
    for layer, (Kc, Vc) in zip(model.layers, rec):
        att = layer.self_attn
        res = h
        x, k1 = layer.attention_conv.prepare(layer.input_layernorm(h))
        B, L, _ = x.shape
        q = att.q_norm(att.q_proj(x).reshape(B, L, att.n_heads, -1)).transpose(0, 2, 1, 3)
        pk = att.k_norm(att.k_proj(x).reshape(B, L, att.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        pv = att.v_proj(x).reshape(B, L, att.n_kv_heads, -1).transpose(0, 2, 1, 3)
        q = model.rope(q, offset=a)
        pk = model.rope(pk, offset=a)
        Kc, Vc = mx.stop_gradient(Kc), mx.stop_gradient(Vc)
        ctx_len = Kc.shape[2]
        keys = mx.concatenate([Kc, pk], axis=2)
        values = mx.concatenate([Vc, pv], axis=2)
        mask = DF.create_causal_mask(L, offset=ctx_len) if att.is_causal else None
        if att.is_sliding:
            query = ctx_len + mx.arange(L)[:, None]
            key = mx.arange(ctx_len + L)[None]
            context = (key < ctx_len) & (query - key < att.sliding_window)
            block = key >= ctx_len
            if att.is_causal:
                block = block & (key <= query)
            mask = context | block
        o = mx.fast.scaled_dot_product_attention(q, keys, values, scale=att.scale, mask=mask)
        h = res + layer.attention_conv.finish(att.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1)), k1)
        res = h
        x, k2 = layer.mlp_conv.prepare(layer.post_attention_layernorm(h))
        h = res + layer.mlp_conv.finish(layer.mlp(x), k2)
    return model.norm(h[:, 1:])


def loss_fn(model, blk, rec, a, n, lab, pred, tkl, lpl):
    hid = train_hidden(model, blk, rec, a)[:, :n]
    lg = model.compute_logits(hid).astype(mx.float32)                     # [1, n, V]
    lse = mx.logsumexp(lg, axis=-1)
    if tkl is not None:                                                    # soft CE to the 125B's top-32 (L1-20's loss)
        ce = lse - (mx.softmax(lpl, axis=-1) * mx.take_along_axis(lg, tkl, axis=-1)).sum(-1)
    else:                                                                  # hard CE on the 125B's token
        ce = lse - mx.take_along_axis(lg, lab[..., None], axis=-1)[..., 0]
    cand = mx.stop_gradient(mx.argpartition(lg, -K, axis=-1)[..., -K:])
    unary = mx.take_along_axis(lg, cand, axis=-1)
    sel = model.candidate_selector
    hp = sel.hidden_projection(hid)
    sc = unary + mx.sum(sel.predecessor_codebook(pred)[:, :, None] * hp[:, :, None]
                        * sel.successor_codebook(cand), axis=-1).astype(mx.float32)
    inc = cand == lab[..., None]
    sce = (mx.logsumexp(sc, axis=-1) - (sc * inc).sum(-1)) * inc.any(-1)
    return (ce + sce).mean()


LVG = nn.value_and_grad(d, loss_fn)


def run_window(smp, lr, seed, tag, val):
    f, ids, off, gen, tk, lp = smp
    reset_lora(seed)
    opt = optim.Adam(learning_rate=lr) if lr > 0 else None
    cache = d.make_cache()
    rec = [None] * len(cache)
    for i, c in enumerate(cache):
        orig = c.update_and_fetch
        def uf(k_, v_, i=i, orig=orig):
            kv = orig(k_, v_); rec[i] = kv
            return kv
        c.update_and_fetch = uf
    start = max(off, gen - (d.config.sliding_window - 1))
    for c in cache:
        c.offset = start
    fed, a, n, out, tms = start, gen, len(ids), [], []
    while a < off + n - 1:
        ctx = mx.array(f[fed - off:a - off][None]).astype(mx.bfloat16)
        blk = mx.array([[int(ids[a - off])] + [MASK] * (BS - 1)])
        true = ids[a - off + 1:a - off + BS]
        m = len(true)
        hid = d.hidden_states(blk, ctx, cache, 1)
        cand, unary = head2s(hid)
        picks, cf = chain_cf(blk, unary, cand, d.candidate_selector.hidden_projection(hid))
        mx.eval(picks, cf, hid)
        pk_, cf_ = np.array(picks[0]).tolist(), np.array(cf[0]).astype(np.float64)
        k = 0
        while k < m and pk_[k] == true[k]:
            k += 1
        L = int((np.cumprod(cf_[:m]) >= LIVE_T).sum())
        kv = min(k, L)
        out.append((a - gen, k, L, kv, m))
        if val[0] > 0:                                     # the update's forward must reproduce the drafting pass
            h2 = train_hidden(d, blk, rec, a)
            mx.eval(h2)
            dmax = float(mx.abs(h2.astype(mx.float32) - hid.astype(mx.float32)).max())
            val[1] = max(val[1], dmax); val[0] -= 1
        if opt is not None and kv < m:                     # labels: rows 0..kv (accepted drafts + the bonus row)
            nr = kv + 1
            lab = mx.array(np.array(true[:nr], np.int32)[None])
            pred = mx.array(np.array([int(ids[a - off])] + [int(x) for x in true[:nr - 1]], np.int32)[None])
            if tk is not None:
                tkl = mx.array(tk[a - gen:a - gen + nr][None].astype(np.int32))
                lpl = mx.array(lp[a - gen:a - gen + nr][None].astype(np.float32))
            else:
                tkl = lpl = None
            t0 = time.perf_counter()
            loss, grads = LVG(d, blk, rec, a, nr, lab, pred, tkl, lpl)
            opt.update(d, grads)
            mx.eval(d.trainable_parameters(), opt.state, loss)
            tms.append(time.perf_counter() - t0)
        fed = a
        if (trim := cache[0].offset - a) > 0:
            DF._trim_recent_cache(cache, trim)
        a = a + kv + 1
    return out, tms


AGR = []
def load_set(pattern):
    smp, names = [], []; AGR.clear()
    for p in sorted(glob.glob(os.path.expanduser(pattern))):
        z = np.load(p)
        f = z["f"].reshape(z["f"].shape[0], -1)
        ids, off, gen = z["ids"], int(z["off"]), int(z["gen"])
        tk = lp = None
        if "tk" in z.files:                                # the 125B's top-32 per answer position: tk[i] at gen + i
            tk, lp = z["tk"], z["lp"]
            if tk.shape[0] == len(ids):                    # d02a's tf_* windows hold one row per position from off
                tk, lp = tk[gen - off:], lp[gen - off:]
            m = min(tk.shape[0], len(ids) - (gen - off) - 1)
            ok = float(np.mean(tk[:m, 0] == ids[gen - off + 1:gen - off + 1 + m]))
            AGR.append(ok)                                 # aligned: the target's top-1 vs the stored answer, multi /
            assert ok > 0.8, f"{p}: top-32 not aligned at gen ({ok:.3f})"    # ifeval 0.97-0.99, agent 0.899-1.0;
                                                           # misaligned <= 0.09 (smoke 11-04 09:0x)
        smp.append((f, ids, off, gen, tk, lp)); names.append(os.path.basename(p))
        if NMAX and len(smp) >= NMAX:
            break
    return smp, names


def summary(recs, label, base=None):
    r = np.array([x for w_ in recs for x in w_], np.int64)            # pos, k, L, kv, m
    kept = (r[:, 3] + 1).mean(); raw = (r[:, 1] + 1).mean(); rows = (np.minimum(r[:, 2], r[:, 4]) + 1).mean()
    bins = [(r[:, 0] < 150), (r[:, 0] >= 150) & (r[:, 0] < 300), (r[:, 0] >= 300)]
    kb = [(r[b, 3] + 1).mean() if b.any() else float("nan") for b in bins]
    s = (f"    {label}: {len(r)} checks; kept per check (cut) {kept:.3f}; raw accepted + 1 {raw:.3f}; rows {rows:.2f}; "
         f"by position <150 / 150-300 / >=300: " + " / ".join(f"{x:.2f}" for x in kb))
    if base is not None:
        s += f"; vs frozen {100 * (kept / base - 1):+.1f} %"
    return s, kept


for item in SETS.split(","):
    tag, pat = item.split("=", 1)
    smp, names = load_set(pat)
    say(f"  {tag}: {len(smp)} windows ({'top-32 soft CE' if smp[0][4] is not None else 'hard CE'}"
        + (f"; target top-1 vs answer, min / mean {min(AGR):.3f} / {np.mean(AGR):.3f}" if AGR else "") + ")")
    base, saved = None, {}
    for lr in LRS:
        t0 = time.time(); val = [NCHECK, 0.0]; recs, tms = [], []
        for wi, s_ in enumerate(smp):
            o_, t_ = run_window(s_, lr, zlib.crc32(names[wi].encode()), tag, val)
            recs.append(o_); tms += t_
        s, kept = summary(recs, f"lr {lr:g}", base)
        if base is None and lr == 0:
            base = kept
        say(s + (f"; update {1e3 * np.median(tms):.1f} ms median" if tms else "") + f"; validation max |d hid| "
            f"{val[1]:.2e}; {time.time() - t0:.0f} s")
        saved[f"lr{lr:g}"] = np.array([x + (wi,) for wi, w_ in enumerate(recs) for x in w_], np.int64)
    np.savez(os.path.join(TD, f"tts_replay_{tag}.npz"), names=np.array(names), **saved)
    open(os.path.join(TD, "tts_replay.txt"), "a").write("\n".join(lines) + "\n"); lines.clear()
