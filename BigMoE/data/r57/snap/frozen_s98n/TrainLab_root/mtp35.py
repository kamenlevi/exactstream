"""TrainLab: chained-drafting fine-tune of the 35B's native MTP head (Chat1000's DRAFTER.md option A: the kill test for
a trained target-conditioned drafter). 2026-10-21.

Head: SpecDecode's MLX rebuild (~/SpecDecode/mtp.py, weights data/mtp/mtp.safetensors): fc([norm_e(embed(t_{i+1})),
norm_h(h_i)]) -> one full-attention + 256-expert MoE layer -> norm -> the main lm_head; chained on its own post-norm
output; main model's post-norm hidden states (SpecDecode: 91.8 % depth-1). Draft vocabulary: ids < 100K (99.3 % of
generated tokens), as SpecDecode deploys it.
Training (EAGLE-3's training-time test): unroll K depths; depth 1 is the causal pass over the true pairs (h_i, t_{i+1});
at depth j the input is (own feature from depth j-1, true token t_{i+j}) at position i+j-1, attending to the true
pairs <= i and to the chain's own depth 2..j entries (diagonal), exactly what the inference cache holds for an
accepted chain. Loss: forward KL to the target's top-64 + tail (restricted to the draft vocabulary) at every depth.
Trained: fc, norms, attention, router, shared expert (fp32); routed experts frozen at 4 bits.
Eval (Chat1000's protocol): tau(K) by simulated passes on ~/Chat1000/data/sot35/texts.json (30 greedy answers) and
SpecDecode's 9 route4 agentic turns; depth-1 alpha exact and under the ratio-0.1 rule.
usage: mtp35.py eval [--run NAME] [--K 8,16,32]   |   mtp35.py train --run NAME [--depth 6] [--lr 3e-5] ...
"""
import argparse, json, math, os, sys, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/SpecDecode"))
import train35 as T
import mtp as SD

SD.MTP_PATH = os.path.expanduser("~/SpecDecode/data/mtp/mtp.safetensors")
V = 100_000
RES = os.path.join(T.TL, "data/mtp_results.txt")
EOS = {248044, 248046}


# ---------------------------------------------------------------- the head
def load_head(model, run=None):
    head = SD.load_mtp(model, bits=None)
    nn.quantize(head, group_size=64, bits=4, class_predicate=lambda p, m: hasattr(m, "to_quantized") and "switch_mlp" in p)
    # trainable parts in fp32
    head.update(nn.utils.tree_map(lambda a: a.astype(mx.float32) if a.dtype == mx.bfloat16 else a,
                                  head.trainable_parameters()))
    head.freeze()
    head.unfreeze()
    for name, m in head.named_modules():
        if "switch_mlp" in name and hasattr(m, "scales"):
            m.freeze()
    if run:
        head.load_weights(os.path.join(T.TL, "ckpt", run + ".mtp.safetensors"), strict=False)
    mx.eval(head.parameters())
    return head


def draft_logits(lm_head, y):
    """logits over ids < V with the main model's (quantized) lm_head rows."""
    return mx.quantized_matmul(y, lm_head.weight[:V], lm_head.scales[:V], lm_head.biases[:V], transpose=True,
                               group_size=lm_head.group_size, bits=lm_head.bits)


def _qkv(attn, x, offset):
    B, L, _ = x.shape
    q, gate = mx.split(attn.q_proj(x).reshape(B, L, attn.num_attention_heads, -1), 2, axis=-1)
    gate = gate.reshape(B, L, -1)
    k = attn.k_norm(attn.k_proj(x).reshape(B, L, attn.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
    v = attn.v_proj(x).reshape(B, L, attn.num_key_value_heads, -1).transpose(0, 2, 1, 3)
    q = attn.q_norm(q).transpose(0, 2, 1, 3)
    return attn.rope(q, offset=offset), attn.rope(k, offset=offset), v, gate


def unrolled(head, embed, h, tok, K):
    """h [B, T, D] target post-norm hidden states, tok [B, T]. Returns the post-norm features [B, T, D] of depths
    1..K; depth j at chain i predicts tok[i + j + 1] (the target's distribution at position i + j)."""
    layer = head.layers[0]
    attn = layer.self_attn
    B, Tn, D = h.shape
    rep = attn.num_attention_heads // attn.num_key_value_heads
    causal = mx.tril(mx.ones((Tn, Tn), dtype=mx.bool_))
    feats, feat, ks, vs = [], h, [], []
    for j in range(1, K + 1):
        tin = mx.concatenate([tok[:, j:], mx.zeros((B, j), tok.dtype)], axis=1)       # t_{i+j}
        x = head.fc(mx.concatenate([head.pre_fc_norm_embedding(embed(tin).astype(mx.float32)),
                                    head.pre_fc_norm_hidden(feat.astype(mx.float32))], -1))
        q, k, v, gate = _qkv(attn, layer.input_layernorm(x), j - 1)
        if j == 1:
            k1, v1 = mx.repeat(k, rep, axis=1), mx.repeat(v, rep, axis=1)
            o = mx.fast.scaled_dot_product_attention(q, k1, v1, scale=attn.scale, mask="causal")
        else:
            ks.append(mx.repeat(k, rep, axis=1)); vs.append(mx.repeat(v, rep, axis=1))
            s1 = mx.where(causal, (q @ k1.swapaxes(-1, -2)) * attn.scale, -mx.inf)       # true pairs <= i
            sd = mx.stack([(q * kd).sum(-1) * attn.scale for kd in ks], axis=-1)         # own chain (diagonal)
            p = mx.softmax(mx.concatenate([s1, sd], axis=-1).astype(mx.float32), axis=-1).astype(q.dtype)
            o = p[..., :Tn] @ v1 + (p[..., Tn:, None] * mx.stack(vs, axis=-2)).sum(-2)
        o = o.transpose(0, 2, 1, 3).reshape(B, Tn, -1)
        hm = x + attn.o_proj(o * mx.sigmoid(gate))
        y = hm + T.moe(layer.mlp, -1, layer.post_attention_layernorm(hm), T.Flow("exact"), None)[0]
        feat = head.norm(y)
        feats.append(feat)
    return feats


def kd_depths(head, lm_head, feats, top, tlp):
    """Mean KL over depths and valid positions; top/tlp: the target's top-64 at every position [B, T, 64]."""
    B, Tn = top.shape[:2]
    tot, per = 0.0, []
    for j, f in enumerate(feats, start=1):
        n = Tn - j - 1                                      # chains i < n have a target at position i + j
        ttop, ttlp = top[:, j:j + n], tlp[:, j:j + n]
        inv = ttop < V
        tl = mx.where(inv, ttlp, -mx.inf)                   # teacher mass on ids >= V goes to the tail
        def chunk_kl(fc, tt, tlc, inc):
            lg = draft_logits(lm_head, fc).astype(mx.float32)
            lq = mx.take_along_axis(lg, mx.where(inc, tt, 0), axis=-1) - mx.logsumexp(lg, axis=-1, keepdims=True)
            lq = mx.where(inc, lq, -mx.inf)
            p, q = mx.exp(tlc), mx.exp(lq)
            pt = mx.maximum(1 - p.sum(-1), 1e-12); qt = mx.maximum(1 - q.sum(-1), 1e-12)
            return (mx.where(inc, p * (tlc - lq), 0.0)).sum(-1).sum() + (pt * (mx.log(pt) - mx.log(qt))).sum()
        s = 0.0
        for a in range(0, n, 256):
            b = min(a + 256, n)
            s = s + mx.checkpoint(lambda fc, tt=ttop[:, a:b], tlc=tl[:, a:b], inc=inv[:, a:b]: chunk_kl(fc, tt, tlc, inc))(
                f[:, a:b])
        per.append(s / (B * n))
        tot = tot + per[-1]
    return tot / len(feats), per


# ---------------------------------------------------------------- target features
def target_features(model, ids):
    """Exact 35B (kernel GDN, LoRA-free): post-norm hidden states [B, T, D] and top-64 log-probs [B, T, 64]."""
    if ids.shape[0] == 1 and ids.shape[1] > 2048:
        h = chunked_forward(model, ids)
    else:
        assert ids.shape[0] * ids.shape[1] * 8 <= 32767, "gather rows above MLX 0.32.2's unaligned bound"
        h, _ = T.forward(model, ids, T.Flow("exact"), False)
    tops, tlps = [], []
    for s in range(0, h.shape[1], 512):
        lg = model.language_model.lm_head(h[:, s:s + 512]).astype(mx.float32)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        tp = mx.argpartition(-lp, T.TOPK - 1, axis=-1)[..., :T.TOPK]
        tops.append(tp); tlps.append(mx.take_along_axis(lp, tp, axis=-1))
    top, tlp = mx.concatenate(tops, 1), mx.concatenate(tlps, 1)
    mx.eval(h, top, tlp)
    return h, top, tlp


def chunked_forward(model, ids, chunk=2048):
    """Exact-flow post-norm hidden states in chunks of <= 2048 tokens with mlx_lm caches: MLX 0.32.2's sorted tensor-unit
    gather_qmm leaves rows unwritten above 32,767 unaligned routed rows (mtplx/moe_sorted_gather.py), i.e. for any
    one-shot 35B forward longer than ~4K tokens."""
    if ids.shape[1] <= chunk:
        return T.forward(model, ids, T.Flow("exact"), False)[0]
    caches = model.language_model.make_cache() if hasattr(model.language_model, "make_cache") else model.make_cache()
    outs = []
    for a in range(0, ids.shape[1], chunk):
        h, _ = T.forward(model, ids[:, a:a + chunk], T.Flow("exact"), False, caches=caches)
        mx.eval(h); outs.append(h)
    return mx.concatenate(outs, axis=1)


# ---------------------------------------------------------------- tau by simulated passes
def heldout_sets(tok):
    out = []
    for r in json.load(open(os.path.expanduser("~/Chat1000/data/sot35/texts.json"))):
        p = tok.apply_chat_template([{"role": "user", "content": r["q"]}], add_generation_prompt=True,
                                    enable_thinking=False, tokenize=True)
        out.append((f"sot/{r['cls']}/{r['k']}", list(p), list(r["seq_ids"])))
    for n in ("edit00", "edit03", "edit07", "bash17", "bash20", "text23", "text26", "write11", "write14"):
        z = np.load(os.path.expanduser(f"~/SpecDecode/data/route4/{n}.npz"))
        out.append((f"agent/{n[:4]}/{n}", z["ids"].tolist(), z["toks"].tolist()))
    return out


def simulate(model, head, prompt, ans, Ks):
    """tau(K) for each K by simulated passes against the stored greedy answer; depth-1 alpha (exact, ratio 0.1)."""
    from mlx_lm.models import cache as cache_mod
    lm = model.language_model
    embed, lmh = lm.model.embed_tokens, lm.lm_head
    S = prompt + ans
    ids = mx.array(S)[None]
    h = chunked_forward(model, ids)                                   # post-norm; chunks keep gather rows < 32,767
    lg = lmh(h).astype(mx.float32)
    tlp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
    mx.eval(h, tlp)
    P, n = len(prompt), len(S)
    q0 = P - 1
    # depth-1 alpha over every answer position q (pair (h_q, S[q+1]) -> S[q+2]), teacher-forced
    feats = unrolled(head, embed, h[:, :n - 1], ids[:, :n - 1], 1)[0]
    d1 = np.array(mx.argmax(draft_logits(lmh, feats).astype(mx.float32), axis=-1))[0]
    tl = np.array(tlp[0])
    qs = range(q0, n - 2)
    hit = [d1[q] == S[q + 2] for q in qs]
    ratio = [tl[q + 1, d1[q]] >= tl[q + 1].max() + math.log(0.1) for q in qs]
    res = {"alpha1": float(np.mean(hit)), "alpha1_r01": float(np.mean(ratio)), "n": len(hit)}
    for K in Ks:
        mc = cache_mod.make_prompt_cache(head)
        if q0 > 0:                                                    # true pairs for prompt positions < q0
            head(h[:, :q0], embed(ids[:, 1:q0 + 1]), mc)
        q, toks, passes = q0, 0, 0
        while q + 1 < n - 1:
            x = head(h[:, q:q + 1], embed(ids[:, q + 1:q + 2]), mc)    # true pair (h_q, S[q+1])
            # chain until the first wrong draft (later drafts would be rejected anyway: same a, ~4x fewer steps)
            a, added = 0, 0
            while a < K and q + 2 + a < n:
                d = int(mx.argmax(draft_logits(lmh, x)[0, -1]))
                if d != S[q + 2 + a]:
                    break
                a += 1
                if a < K and q + 2 + a < n:
                    x = head(x, embed(mx.array([[d]])), mc); added += 1
            if added:
                mc[0].trim(added)
            if a > 0:                                                 # accepted positions get their true pairs
                head(h[:, q + 1:q + 1 + a], embed(ids[:, q + 2:q + 2 + a]), mc)
            toks += a + 1; passes += 1; q += a + 1
        res[f"tau{K}"] = toks / max(passes, 1)
    return res


def cmd_eval(a, loaded=None):
    rd = os.path.join(T.TL, "data/mtp_redirect.json")              # in-turn redirect: train first, then evaluate
    if a.run and os.path.exists(rd) and loaded is None:
        r = json.load(open(rd)); os.rename(rd, rd + ".used")
        ta = argparse.Namespace(**{**vars(a), **r["train"], "cmd": "train"})
        ta.time_limit = max(60.0, r["deadline"] - time.time() - r.get("eval_reserve", 420))
        print(f"redirect: training {ta.run} lr {ta.lr} for {ta.time_limit:.0f}s before the eval", flush=True)
        model, tok = cmd_train(ta)                    # the same 35B serves the eval: never two in one process
        a.run = ta.run
        loaded = (model, tok)
    model, tok = loaded if loaded is not None else T.setup()
    head = load_head(model, a.run)
    Ks = [int(k) for k in a.K.split(",")]
    budget = 1e9
    ov = os.path.join(T.TL, "data/mtp_eval_override.json")          # read at start: K and a time budget (turns)
    if os.path.exists(ov):
        o = json.load(open(ov))
        Ks = [int(k) for k in str(o.get("K", a.K)).split(",")]
        budget = float(o.get("max_seconds", budget))
    sets = heldout_sets(tok)
    t0 = time.time()
    agg = {}
    for name, p, ans in sets:
        if time.time() - t0 > budget:
            print(f"eval budget {budget:.0f}s reached: partial aggregates", flush=True)
            break
        r = simulate(model, head, p, ans, Ks)
        cls = name.split("/")[1]
        for k, v in r.items():
            agg.setdefault(cls, {}).setdefault(k, []).append(v)
            agg.setdefault("ALL", {}).setdefault(k, []).append(v)
        print(name, {k: round(v, 3) for k, v in r.items()}, flush=True)
    for cls, d in agg.items():
        w = np.array(d["n"], float)
        line = f"{time.strftime('%m-%d %H:%M')} mtp {a.run or 'stock'} {cls:6s} " + " ".join(
            f"{k} {np.average(v, weights=w):.3f}" for k, v in d.items() if k != "n") + f" (n {int(w.sum())})"
        print(line, flush=True)
        open(RES, "a").write(line + "\n")
    print(f"eval {time.time() - t0:.0f}s", flush=True)


def cmd_train(a):
    t0 = time.time()
    limit = a.time_limit or float(os.environ.get("TL_LIMIT", "1e9"))
    model, tok = T.setup()
    head = load_head(model, a.run if os.path.exists(os.path.join(T.TL, "ckpt", a.run + ".mtp.json")) else None)
    lm = model.language_model
    embed = lm.model.embed_tokens
    stream = np.load(os.path.join(T.TL, "data/train_tokens.npy"), mmap_mode="r")
    stp = os.path.join(T.TL, "ckpt", a.run + ".mtp.json")
    state = json.load(open(stp)) if os.path.exists(stp) else {"tokens": 0, "step": 0}
    opt = optim.Adam(learning_rate=lambda s: a.lr * mx.minimum(1.0, (s + 1) / 20.0))
    # fixed held-out probe: KL per depth on 4 held-out greedy answers (prompt + answer, first 512 tokens), never trained
    probe = []
    for name, p_, ans in heldout_sets(tok)[0:30:8]:
        ids_p = mx.array((p_ + ans)[:512])[None]
        hp, tp_, tl_ = target_features(model, ids_p)
        probe.append((ids_p, hp, tp_, tl_))

    def probe_kl():
        per = np.zeros(a.depth)
        for ids_p, hp, tp_, tl_ in probe:
            _, pr = kd_depths(head, lm.lm_head, unrolled(head, embed, hp, ids_p, a.depth), tp_, tl_)
            per += np.array([float(x) for x in pr])
        return per / len(probe)

    pk = probe_kl()
    print(f"probe step {state['step']} held-out KL per depth " + " ".join(f"{x:.3f}" for x in pk) + f" (mean {pk.mean():.3f})", flush=True)
    lossf = nn.value_and_grad(head, lambda hd, h, ids, top, tlp: kd_depths(hd, lm.lm_head, unrolled(hd, embed, h, ids, a.depth), top, tlp))
    hist = []
    while state["tokens"] < a.tokens and time.time() - t0 < limit - 120:
        t_step = time.time()
        offs = np.random.default_rng(2 * 10 ** 6 + state["step"]).integers(0, stream.shape[0] - a.seq - 1, size=a.batch)
        ids = mx.array(np.stack([np.asarray(stream[o:o + a.seq]) for o in offs]).astype(np.int32))
        h, top, tlp = target_features(model, ids)
        (loss, per), g = lossf(head, h, ids, top, tlp)
        g, gn = optim.clip_grad_norm(g, 1.0)
        mx.eval(loss, gn)
        if not np.isfinite(float(loss)):
            print(f"step {state['step']}: non-finite loss, skipped", flush=True); state["step"] += 1; continue
        opt.update(head, g)
        mx.eval(head.trainable_parameters(), opt.state)
        if a.duty < 1:                                  # quiet nights: idle (1/duty - 1) x the step time
            time.sleep((time.time() - t_step) * (1 / a.duty - 1))
        state["step"] += 1; state["tokens"] += a.seq * a.batch
        hist.append([float(x) for x in per])
        if state["step"] % 10 == 0:
            m = np.mean(hist, 0); hist = []
            print(f"step {state['step']} tokens {state['tokens']} KL per depth " + " ".join(f"{x:.3f}" for x in m)
                  + f" |g| {float(gn):.2f} {time.time() - t0:.0f}s", flush=True)
        if state["step"] % 50 == 0:
            pk = probe_kl()
            print(f"probe step {state['step']} held-out KL per depth " + " ".join(f"{x:.3f}" for x in pk)
                  + f" (mean {pk.mean():.3f})", flush=True)
        if state["step"] % 100 == 0:
            mx.save_safetensors(os.path.join(T.TL, "ckpt", a.run + ".mtp.safetensors"), dict(tree_flatten(head.trainable_parameters())))
            json.dump(state, open(stp, "w"))
    mx.save_safetensors(os.path.join(T.TL, "ckpt", a.run + ".mtp.safetensors"), dict(tree_flatten(head.trainable_parameters())))
    json.dump(state, open(stp, "w"))
    print(f"saved {a.run} at {state['tokens']} tokens", flush=True)
    del head, opt, lossf
    import gc; gc.collect(); mx.clear_cache()
    return model, tok


def cmd_smoke(a):
    """~2 min on the GPU: one training step (loss, time, memory) and two held-out items simulated at K = 16."""
    model, tok = T.setup()
    head = load_head(model)
    lm = model.language_model
    stream = np.load(os.path.join(T.TL, "data/train_tokens.npy"), mmap_mode="r")
    ids = mx.array(np.stack([np.asarray(stream[o:o + 1024]) for o in (1000, 900000)]).astype(np.int32))
    t = time.time(); h, top, tlp = target_features(model, ids); t1 = time.time() - t
    lossf = nn.value_and_grad(head, lambda hd: kd_depths(hd, lm.lm_head, unrolled(hd, lm.model.embed_tokens, h, ids, 6), top, tlp))
    for r in range(2):
        t = time.time(); (l, per), g = lossf(head); mx.eval(l, g); t2 = time.time() - t
    print(f"smoke: target features {t1:.2f}s, depth-6 step {t2:.2f}s, KL per depth " + " ".join(f"{float(x):.3f}" for x in per)
          + f", peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
    sets = heldout_sets(tok)
    for name, p, ans in (sets[0], sets[-1]):
        t = time.time(); r = simulate(model, head, p, ans, [16])
        print(f"smoke: {name} ({len(p)}+{len(ans)} tokens) {r} {time.time() - t:.1f}s", flush=True)


def main():
    ex = os.path.join(T.TL, "data/exec_redirect.json")              # in-turn: replace this process by another job
    if os.path.exists(ex):
        argv = json.load(open(ex)); os.rename(ex, ex + ".used")
        print("exec redirect:", " ".join(argv), flush=True)
        os.execv(sys.executable, [sys.executable] + argv)
    T.limit_memory()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("--run"); ap.add_argument("--K", default="8,16,32")
    ap.add_argument("--depth", type=int, default=6); ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seq", type=int, default=1024); ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=10 ** 9); ap.add_argument("--time-limit", type=float, default=0)
    ap.add_argument("--duty", type=float, default=1.0)
    a = ap.parse_args()
    {"eval": cmd_eval, "train": cmd_train, "smoke": cmd_smoke}[a.cmd](a)


if __name__ == "__main__":
    main()
