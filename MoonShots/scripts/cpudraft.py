"""M-19 CPUDRAFT (MoonShots, Mon 2026-11-02 21:0x; Kamen 21:01 via the orchestrator; pre-registered in ~/NewIdeas/QUEUE.md
M-19 before any timing): the CPU as the drafter beside the GPU check, in Leap's ANE-2 call shape (3 candidates x 16 rows
drafted while the check runs; ANE-2: 12.42 ms per batched call on the Neural Engine).
One call = 16 context rows (the 5 target taps -> fc -> hidden_norm -> each layer's k / v, as the ring write) + B x 16 block
rows (B = 1 a single draft, B = 3 ANE-2's batch) through DFlash2 L1-20 (Leap's l120_ft, 5 layers), attention over the
sliding ring + each candidate's own 16 rows (fp32), the head on rows 1-15, the two-stage top-16 and the selector chain.
Forms (MLX on the CPU device): B16 = all bf16 (BNNS), F32 = layers + fc fp32 (Accelerate sgemm). The D7b form (4-bit
layers) cannot run there: MLX's CPU quantized_matmul is a reference loop (micro, Mon 21:2x: 1,391 ms for one 2560 -> 7680
linear at 16 rows, 0.5 GFLOP/s), and the target's full 4-bit head likewise, so the CPU timings take Leap's FR-Spec top-32,768
head rows (ANE-2's head) in the form's precision. Threads: N CPU streams (one thread each): every linear split by output
rows, the attention by KV head; the norms / convs / gathers stay on one stream.
Lines (the orchestrator's): PASS = the B = 3 call <= 12 ms AND the GPU check slows <= 3 % beside it; KILL = > 20 ms OR > 5 %.
Modes: micro (GEMM rates per form) | pmicro (one linear split over N streams / threads) | check FORM (CPUDRAFT_DEVICE=gpu:
my batched path vs DFlash2's own propose on real taps, B = 1, the full head) | time FORM N (ms per call: B 1 / 3 x ring 512 /
2,047, N streams)
Usage: [BigMoE venv] scripts/cpudraft.py MODE [FORM [N]] -> data/cpudraft.txt"""
import glob, os, resource, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
mx.set_default_device(mx.gpu if os.environ.get("CPUDRAFT_DEVICE") == "gpu" else mx.cpu)
sys.path.insert(0, os.path.expanduser("~/Leap/third_party"))
import dflash_model_mlx as DF                    # Leap's DFlash2 MLX model (read-only import)

V, H, NH, NKV, HD, NL, FF = 248320, 2560, 32, 8, 128, 5, 7680
NB, NC, NFR, BK = 16, 16, 32768, 128
THR = os.environ.get("VECLIB_MAXIMUM_THREADS", "all")
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
LEAP = os.path.expanduser("~/Leap/data")
TAPS = sorted(glob.glob(os.path.expanduser("~/TrainLab/data/taps_raw/he_*.npz")))
OUT = os.path.expanduser("~/MoonShots/data/cpudraft.txt")
ST = []                                          # the CPU streams a split call fans out over (empty = MLX's one stream)


def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")


def gate():
    while os.path.exists(PAUSE):
        time.sleep(2)


def bench(f, n=30, warm=4):
    """median wall ms and CPU ms (process time: cores used = cpu / wall) of f(); restarts a round that PAUSE interrupted"""
    while True:
        gate()
        for _ in range(warm):
            f()
        w, c = [], []
        for _ in range(n):
            t0, c0 = time.perf_counter(), time.process_time()
            f()
            w.append(time.perf_counter() - t0); c.append(time.process_time() - c0)
        if not os.path.exists(PAUSE):
            return 1e3 * float(np.median(w)), 1e3 * float(np.median(c)), 1e3 * float(np.percentile(w, 90))


def micro():
    rng = np.random.default_rng(0)
    say(f"== micro (threads {THR}): one linear on the CPU device; ms median, GFLOP/s, weight GB/s, cores used")
    for n_in, n_out in ((H, FF), (FF, H)):
        W = mx.array(rng.standard_normal((n_out, n_in)).astype(np.float32) * 0.02)
        forms = {"F32": W, "B16": W.astype(mx.bfloat16), "F16": W.astype(mx.float16)}
        mx.eval(forms)
        for rows in (16, 64):                     # (MLX's CPU quantized_matmul, Mon 21:2x: 1,391 ms Q4 / 1,529 ms Q8 for
            x = mx.array(rng.standard_normal((rows, n_in)).astype(np.float32))   # 2560->7680 at 16 rows: a reference loop)
            arms = [(k, w.nbytes, (lambda xx, ww: lambda: mx.eval(xx @ ww.T))(x.astype(w.dtype), w)) for k, w in forms.items()]
            xn, wn = np.array(x), np.array(W).T.copy()
            arms.append(("numpy F32", wn.nbytes, lambda: xn @ wn))
            for k, nbytes, f in arms:
                ms, cpu, _ = bench(f)
                say(f"  {n_in}->{n_out} rows {rows:2d} {k:9s}: {ms:7.3f} ms  {2 * rows * n_in * n_out / ms / 1e6:7.1f} GFLOP/s  "
                    f"{nbytes / ms / 1e6:6.1f} GB/s  cores {cpu / ms:4.1f}")


def pmicro():
    """one linear split by output rows over N MLX CPU streams (one thread each) or N numpy threads (Accelerate per slice)"""
    from concurrent.futures import ThreadPoolExecutor
    rng = np.random.default_rng(0)
    say(f"== pmicro (VECLIB threads {THR}): one linear split over N streams / threads; ms median, weight GB/s, cores used")
    for n_in, n_out in ((H, FF), (FF, H)):
        W = mx.array(rng.standard_normal((n_out, n_in)).astype(np.float32) * 0.02)
        for dt, name in ((mx.float32, "F32"), (mx.bfloat16, "B16")):
            Wd = W.astype(dt)
            mx.eval(Wd)
            for rows in (16, 48):
                x = mx.array(rng.standard_normal((rows, n_in)).astype(np.float32)).astype(dt)
                mx.eval(x)
                for N in (1, 2, 3, 4, 6, 8):
                    st = [mx.new_stream(mx.cpu) for _ in range(N)]
                    parts = [mx.array(np.array(Wd[int(s[0]):int(s[-1]) + 1].astype(mx.float32))).astype(dt)
                             for s in np.array_split(np.arange(n_out), N)]
                    mx.eval(parts)
                    f = lambda: mx.eval(mx.concatenate([mx.matmul(x, p.T, stream=s) for p, s in zip(parts, st)], axis=-1))
                    ms, cpu, _ = bench(f)
                    say(f"  {n_in}->{n_out} {name} rows {rows:2d} streams {N}: {ms:6.3f} ms  {Wd.nbytes / ms / 1e6:6.1f} GB/s  "
                        f"cores {cpu / ms:4.1f}")
            if name == "F32":
                xn = np.array(mx.array(rng.standard_normal((48, n_in)).astype(np.float32)))
                for N in (1, 2, 4, 8):
                    wn = [np.ascontiguousarray(np.array(W)[s].T) for s in np.array_split(np.arange(n_out), N)]
                    with ThreadPoolExecutor(N) as ex:
                        f = lambda: list(ex.map(lambda w: xn @ w, wn))
                        ms, cpu, _ = bench(f)
                    say(f"  {n_in}->{n_out} numpy F32 rows 48 threads {N}: {ms:6.3f} ms  {W.nbytes / ms / 1e6:6.1f} GB/s  "
                        f"cores {cpu / ms:4.1f}")


class Target(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.QuantizedEmbedding(V, H, group_size=64, bits=4)
        self.lm_head = nn.QuantizedLinear(H, V, bias=False, group_size=64, bits=4)


def load(form):
    w = mx.load(os.path.expanduser("~/BigMoE/models/bare/model-00016-of-00017.safetensors"))
    tgt = Target()
    tgt.load_weights([(f"embed_tokens.{s}", w[f"language_model.model.embed_tokens.{s}"]) for s in ("weight", "scales", "biases")]
                     + [(f"lm_head.{s}", w[f"language_model.lm_head.{s}"]) for s in ("weight", "scales", "biases")])
    del w
    d = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
    d.load_weights(os.path.join(LEAP, "l120_ft.safetensors"), strict=False)
    if form in ("E", "Q8"):
        nn.quantize(d, group_size=64, bits=4 if form == "E" else 8,
                    class_predicate=lambda p, m: isinstance(m, nn.Linear) and p.startswith("layers"))
    elif form == "F32":
        for m in [d.fc, d.hidden_norm, d.norm, d.candidate_selector.hidden_projection] + list(d.layers):
            m.update(tree_map(lambda a: a.astype(mx.float32), m.parameters()))
    d.bind(tgt)
    mx.eval(d.parameters(), tgt.parameters())
    return d, tgt


class SplitLinear(nn.Module):
    """y = x @ W.T with W's output rows split over the CPU streams ST (one thread each)"""
    def __init__(self, w):
        super().__init__()
        self.parts = [mx.contiguous(w[int(c[0]):int(c[-1]) + 1]) for c in np.array_split(np.arange(w.shape[0]), len(ST))]

    def __call__(self, x):
        ys = []
        for p, s in zip(self.parts, ST):
            with mx.stream(s):
                ys.append(x @ p.T)
        return mx.concatenate(ys, axis=-1)


def split_model(d):
    for layer in d.layers:
        for parent, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")), (layer.mlp, ("gate_proj", "up_proj", "down_proj")),
                              (layer.attention_conv, ("kernel_projection",)), (layer.mlp_conv, ("kernel_projection",))):
            for nm in names:
                setattr(parent, nm, SplitLinear(getattr(parent, nm).weight))
    d.fc = SplitLinear(d.fc.weight)
    d.candidate_selector.hidden_projection = SplitLinear(d.candidate_selector.hidden_projection.weight)
    mx.eval(d.parameters())


def head_of(form, kind, tgt):
    """x -> logits over the head's rows, and the head-row -> token map (None = identity)"""
    lm = tgt.lm_head
    if kind == "full":
        return lambda x: lm(x.astype(mx.bfloat16)), None
    sub = np.sort(np.load(os.path.join(LEAP, "frspec_rank.npy"))[:NFR])
    idx = mx.array(sub)
    wq, sc, bi = lm.weight[idx], lm.scales[idx], lm.biases[idx]
    if form == "E":
        f = lambda x: mx.quantized_matmul(x.astype(mx.bfloat16), wq, sc, bi, transpose=True, group_size=64, bits=4)
        mx.eval(wq, sc, bi)
    else:
        wd = mx.dequantize(wq, sc, bi, group_size=64, bits=4)
        if form == "Q8":
            q8 = mx.quantize(wd, group_size=64, bits=8)
            mx.eval(q8)
            f = lambda x: mx.quantized_matmul(x.astype(mx.bfloat16), *q8, transpose=True, group_size=64, bits=8)
        else:
            wd = wd.astype(mx.float32 if form == "F32" else mx.bfloat16)
            mx.eval(wd)
            if ST:
                sl = SplitLinear(wd)
                mx.eval(sl.parameters())
                f = lambda x: sl(x.astype(wd.dtype))
            else:
                f = lambda x: x.astype(wd.dtype) @ wd.T
    return f, mx.array(sub.astype(np.int32))


def forward(d, head, tokmap, DT):
    rope, scale, G = d.rope, HD ** -0.5, NH // NKV
    cs, cfg = d.candidate_selector, d.config
    groups = [(int(c[0]), int(c[-1]) + 1) for c in np.array_split(np.arange(NKV), max(1, len(ST)))]

    def attn(sa, x, hc, rk, rv, off):
        B, L, _ = x.shape
        S = hc.shape[1]
        f32 = lambda a: a.astype(mx.float32)
        q = f32(sa.q_norm(sa.q_proj(x).reshape(B, L, NH, HD))).transpose(0, 2, 1, 3)
        kb = f32(sa.k_norm(sa.k_proj(x).reshape(B, L, NKV, HD))).transpose(0, 2, 1, 3)
        vb = f32(sa.v_proj(x).reshape(B, L, NKV, HD)).transpose(0, 2, 1, 3)
        kc = f32(sa.k_norm(sa.k_proj(hc).reshape(1, S, NKV, HD))).transpose(0, 2, 1, 3)
        vc = f32(sa.v_proj(hc).reshape(1, S, NKV, HD)).transpose(0, 2, 1, 3)
        q, kb, kc = rope(q, offset=off + S) * scale, rope(kb, offset=off + S), rope(kc, offset=off)
        outs = []
        for g, (a, b) in enumerate(groups):                                      # KV-head groups, one stream each
            n = b - a
            with mx.stream(ST[g] if ST else mx.default_stream(mx.default_device())):
                qh = q[:, a * G:b * G]
                qg = qh.reshape(B, n, G, L, HD).transpose(1, 0, 2, 3, 4).reshape(1, n, B * G * L, HD)
                s_r, s_c = qg @ rk[:, a:b].swapaxes(-1, -2), qg @ kc[:, a:b].swapaxes(-1, -2)   # the ring, the 16 new rows
                s_b = (qh.reshape(B, n, G * L, HD) @ kb[:, a:b].swapaxes(-1, -2)).transpose(1, 0, 2, 3).reshape(1, n, B * G * L, L)
                m = mx.maximum(mx.maximum(s_r.max(-1, keepdims=True), s_c.max(-1, keepdims=True)), s_b.max(-1, keepdims=True))
                p_r, p_c, p_b = mx.exp(s_r - m), mx.exp(s_c - m), mx.exp(s_b - m)
                den = p_r.sum(-1, keepdims=True) + p_c.sum(-1, keepdims=True) + p_b.sum(-1, keepdims=True)
                o_b = p_b.reshape(n, B, G * L, L).transpose(1, 0, 2, 3) @ vb[:, a:b]   # each candidate's own block rows
                outs.append((p_r @ rv[:, a:b] + p_c @ vc[:, a:b] + o_b.transpose(1, 0, 2, 3).reshape(1, n, B * G * L, HD)) / den)
        o = mx.concatenate(outs, axis=1) if len(outs) > 1 else outs[0]
        o = o.reshape(NKV, B, G, L, HD).transpose(1, 3, 0, 2, 4).reshape(B, L, NH * HD)
        return sa.o_proj(o.astype(x.dtype))

    def fwd(blk, taps, rks, rvs, off):
        h = (d.embed_tokens(blk) * d.embed_scale).astype(DT)                      # the form's residual dtype
        hc = d.hidden_norm(d.fc(taps.astype(DT)))
        for li, layer in enumerate(d.layers):
            r = h
            x, kern = layer.attention_conv.prepare(layer.input_layernorm(h))
            h = r + layer.attention_conv.finish(attn(layer.self_attn, x, hc, rks[li], rvs[li], off), kern)
            r = h
            x, kern = layer.mlp_conv.prepare(layer.post_attention_layernorm(h))
            h = r + layer.mlp_conv.finish(layer.mlp(x), kern)
        hid = d.norm(h)[:, 1:]                                                   # rows 1-15: the drafted positions
        lg = head(hid).astype(mx.float32) * cfg.output_multiplier
        if cfg.final_logit_softcapping:
            lg = mx.tanh(lg / cfg.final_logit_softcapping) * cfg.final_logit_softcapping
        s = lg.shape
        blkm = lg.reshape(*s[:-1], s[-1] // BK, BK)
        tb = mx.argpartition(blkm.max(axis=-1), -16, axis=-1)[..., -16:]
        sel = mx.take_along_axis(blkm, mx.broadcast_to(tb[..., None], (*s[:-1], 16, BK)), axis=-2).reshape(*s[:-1], 16 * BK)
        j = mx.argpartition(sel, -16, axis=-1)[..., -16:]
        cand = mx.take_along_axis(tb, j // BK, axis=-1) * BK + j % BK
        unary = mx.take_along_axis(lg, cand, axis=-1)
        tok = cand if tokmap is None else tokmap[cand]
        proj = cs.hidden_projection(hid.astype(DT)).astype(mx.float32)
        succ = cs.successor_codebook(tok).astype(mx.float32)                     # [B, 15, 16, 256]
        pred = cs.predecessor_codebook(tok[:, :-1]).astype(mx.float32)           # [B, 14, 16, 256]
        pa = cs.predecessor_codebook(blk[:, 0]).astype(mx.float32)               # [B, 256]
        m0 = (succ[:, 0] @ (pa * proj[:, 0])[..., None])[..., 0]                # [B, 16]
        M = (pred * proj[:, 1:, None, :]) @ succ[:, 1:].swapaxes(-1, -2)         # [B, 14, 16 pred, 16 succ]
        return tok, unary, m0, M
    return fwd


def chain(tok, unary, m0, M):
    mx.eval(tok, unary, m0, M)
    tok, unary, m0, M = (np.array(a) for a in (tok, unary, m0, M))
    out = np.zeros(tok.shape[:2], np.int64)
    for b in range(tok.shape[0]):
        s = int(np.argmax(unary[b, 0] + m0[b]))
        out[b, 0] = tok[b, 0, s]
        for p in range(1, tok.shape[1]):
            s = int(np.argmax(unary[b, p] + M[b, p - 1, s]))
            out[b, p] = tok[b, p, s]
    return out


def ring_of(d, taps, off, DT):
    """each layer's ring K (k_norm + rope) / V for context taps [1, R, 5H] at positions off.., fp32 (DF's cache contents)"""
    hc = d.hidden_norm(d.fc(taps.astype(DT)))
    R = taps.shape[1]
    rks, rvs = [], []
    for layer in d.layers:
        sa = layer.self_attn
        k = sa.k_norm(sa.k_proj(hc).reshape(1, R, NKV, HD)).transpose(0, 2, 1, 3)
        rks.append(d.rope(k.astype(mx.float32), offset=off))
        rvs.append(sa.v_proj(hc).reshape(1, R, NKV, HD).transpose(0, 2, 1, 3).astype(mx.float32))
    mx.eval(rks, rvs)
    return rks, rvs


def sample(i):
    z = np.load(TAPS[i])
    return z["f"].reshape(z["f"].shape[0], -1), z["ids"]


def check(form):
    """B = 1, full head: my ring + 16-new-row path vs DFlash2's own propose (whole context in one call, its own cache)"""
    d, tgt = load(form)
    DT = mx.float32 if form == "F32" else mx.bfloat16
    fwd = forward(d, *head_of(form, "full", tgt), DT)
    MASK = int(d.config.mask_token_id)
    same = tot = 0
    for i in range(0, len(TAPS), max(1, len(TAPS) // 8)):
        gate()
        f, ids = sample(i)
        for a in range(64, f.shape[0] - 16, 61):
            blk = mx.array([[int(ids[a])] + [MASK] * 15])
            cache = d.make_cache()
            ref = np.array(d.propose(blk, mx.array(f[None, :a]).astype(mx.bfloat16), cache, 0.0, 1)[0])[0]
            rks, rvs = ring_of(d, mx.array(f[None, :a - NC]), 0, DT)
            mine = chain(*fwd(blk, mx.array(f[None, a - NC:a]), rks, rvs, a - NC))[0]
            same += int((ref == mine).sum()); tot += ref.size
    say(f"== check form {form} on {mx.default_device()}: my batched path vs DFlash2's propose (B = 1, full head, he taps): "
        f"identical draft tokens {same} / {tot} ({same / tot:.2%})")


def timing(form, N):
    t0 = time.time()
    d, tgt = load(form)
    DT = mx.float32 if form == "F32" else mx.bfloat16
    if N > 1:
        ST[:] = [mx.new_stream(mx.cpu) for _ in range(N)]
        split_model(d)
    say(f"== time form {form}, {N} CPU stream(s) (loaded in {time.time() - t0:.0f} s; MLX peak {mx.get_peak_memory() / 1e9:.2f} GB)")
    f, ids = sample(0)
    MASK = int(d.config.mask_token_id)
    fwd = forward(d, *head_of(form, "fr32", tgt), DT)
    for R in (512, 2047 - NC):
        ctx = np.concatenate([f] * (R // f.shape[0] + 2))[:R + NC]               # stored taps, tiled to the ring size
        rks, rvs = ring_of(d, mx.array(ctx[None, :R]), 0, DT)
        new = mx.array(ctx[None, R:])
        for B in (1, 3):
            blk = mx.array([[int(ids[40 + 7 * b])] + [MASK] * 15 for b in range(B)])
            mx.eval(new, blk)
            ms, cpu, p90 = bench(lambda: chain(*fwd(blk, new, rks, rvs, R)), n=20)
            say(f"  form {form} streams {N} head fr32 ring {R + NC:4d} B {B}: {ms:6.2f} ms (p90 {p90:6.2f}), cores {cpu / ms:4.1f}")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    say(f"  form {form} streams {N}: MLX peak {mx.get_peak_memory() / 1e9:.2f} GB, max RSS {rss:.2f} GB")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "micro":
        micro()
    elif mode == "pmicro":
        pmicro()
    elif mode == "check":
        check(sys.argv[2])
    else:
        timing(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1)
