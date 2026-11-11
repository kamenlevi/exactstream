"""ANE-1 (Leap, Mon 2026-11-02 17:1x; Fable's mission 17 idea 2.3 in ~/FableGod/DRAFTER_WALL.md, the Neural Engine shadow
drafter; the orchestrator's order with Kamen's OK for coremltools 9.0 in ~/BigMoE/.venv): DFlash2 L1-20 (l120_ft) as a
CoreML int8 model on the Neural Engine (CPU_AND_NE), built with coremltools' MIL builder from the safetensors (no torch).
Pre-registered (the order's lines): step 1 KILL if one 16-row block (body + head) takes > 30 ms; step 2 KILL if the GPU's
warm check time grows > 10 % while the Neural Engine drafts (judged with the ANE drafting back to back, the worst case;
one and three drafts per day-time check (~85 ms: 12 / 36 per s) reported beside it).
Model, as DFlash2's hidden_states + compute_logits:
- body: fc + hidden_norm over 16 new context rows; 5 DFlash2 layers (grouped dynamic causal convs, sliding-window attention
  over 2,031 cached context rows + the 16 new ones + the 16-row block, q/k norms, RoPE from host tables, SwiGLU); norm on
  rows 1-15; the selector's hidden projection. Outputs the 15 hidden rows and the new context rows' keys / values.
- head: the target's lm_head (4-bit, dequantized, then int8 per channel) in 20 column chunks of 12,416 + the per-128 block
  max for the two-stage top-16; headm = the same head returning only the block max (measures the head without Python's
  fp32 copy of the 248,320-column output, which native code would read in place).
- CPU (numpy): the two-stage top-16 and the selector chain (15 steps over 16 candidates); the 16 embedding rows.
int8 = coremltools linear_quantize_weights, symmetric per output channel (W8A16), matrices only.
Modes: build (-> ~/Leap/data/ane/*.mlpackage) | check (vs the MLX drafter, bf16) | time (step 1) | gpu (step 2: a check
proxy on the GPU, the real NAX6 expert kernels on real layers 6/20/40 + real routing at 16 rows, 48 chained calls = the
expert part of a 16-row check, 7.4 GB; ANE worker process idle / back to back / 12 / 36 drafts per s, interleaved).
Usage: [BigMoE venv] ane_probe.py build|check|time|gpu -> ~/Leap/data/ane_probe.txt"""
import glob, json, os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np

D = os.path.expanduser("~/Leap/data")
AD = os.path.join(D, "ane")
SNAP = os.path.expanduser("~/.cache/huggingface/hub/models--incoai--Qwen3.8-Flash-Next-DFlash2/snapshots/"
                          "a22a1744e85c17364dcc54a6a420af7135d4c7f6/model.safetensors")
TGT = os.path.expanduser("~/BigMoE/models/bare/model-00016-of-00017.safetensors")
H, NH, NKV, HD, FF, NL, V = 2560, 32, 8, 128, 7680, 5, 248320
WC, NB, NC, CH = 2031, 16, 16, 12416          # cached ctx rows, block rows, new ctx rows, head chunk (97 x 128)
T = WC + NC + NB
OUT = os.path.join(D, "ane_probe.txt")
RATE = {2: 12.0, 3: 36.0}                     # drafts per s: one / three candidates per ~85 ms check
SCL = 32.0                                    # the body carries the residual stream / 32 (fp16 range)
PRE = ([2.0 ** 14, 1 / 64, 1 / 64, 1 / 64, 1 / 128],   # prescale before input_layernorm, layers 0-4 (layer 0 sees emb / 32)
       [1 / 64, 1 / 64, 1 / 64, 1 / 64, 1 / 128],      # before post_attention_layernorm
       1 / 128)                                         # before the final norm


def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")


INV = 1.0 / (10000.0 ** (np.arange(0, HD, 2) / HD))
def rope_tab(pos):
    a = np.asarray(pos, np.float64)[:, None] * INV[None]
    c, s = np.cos(a), np.sin(a)
    return (np.concatenate([c, c], -1)[None, None].astype(np.float16),
            np.concatenate([s, s], -1)[None, None].astype(np.float16))


def body_inputs(x, tg, p0, n_cache, rng=None):
    """x (1,16,H) embeddings, tg (1,16,5H) new ctx features, ctx at positions p0..p0+15, block at p0+16..p0+31."""
    cq, sq = rope_tab(range(p0 + NC, p0 + NC + NB))
    ck, sk = rope_tab(range(p0, p0 + NC + NB))
    m = np.zeros((1, 1, 4 * NB, T), np.float16)
    m[..., n_cache:WC] = -30000
    inp = {"x": x.astype(np.float16), "tg": tg.astype(np.float16), "cq": cq, "sq": sq, "ckb": ck, "skb": sk, "mask": m}
    for i in range(NL):
        if rng is None:
            inp[f"kc{i}"] = np.zeros((1, NKV, WC, HD), np.float16)
            inp[f"vc{i}"] = np.zeros((1, NKV, WC, HD), np.float16)
        else:
            inp[f"kc{i}"] = rng.standard_normal((1, NKV, WC, HD)).astype(np.float16)
            inp[f"vc{i}"] = rng.standard_normal((1, NKV, WC, HD)).astype(np.float16)
    return inp


def drafter_weights():
    import mlx.core as mx
    w = dict(mx.load(SNAP))
    w.update(mx.load(os.path.join(D, "l120_ft.safetensors")))
    return {k: np.array(v.astype(mx.float16)) for k, v in w.items() if "codebook" not in k}


def lm_head16():
    import mlx.core as mx
    w = mx.load(TGT)
    lm = mx.dequantize(w["language_model.lm_head.weight"], w["language_model.lm_head.scales"],
                       w["language_model.lm_head.biases"], group_size=64, bits=4)
    return np.array(lm.astype(mx.float16))


def build(fp16=False, body_only=False):
    import coremltools as ct
    import coremltools.optimize as cto
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    os.makedirs(AD, exist_ok=True)
    CFG = cto.coreml.OptimizationConfig(global_config=cto.coreml.OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8", granularity="per_channel", weight_threshold=100000))
    f16 = np.float16

    def save(prog, name):
        t0 = time.time()
        m = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                       compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True)
        q = m if fp16 else cto.coreml.linear_quantize_weights(m, config=CFG)
        q.save(os.path.join(AD, name + ("16" if fp16 else "") + ".mlpackage"))
        say(f"built {name}.mlpackage in {time.time() - t0:.0f} s")

    W = drafter_weights()
    # fp16 range: the drafter's residual stream reaches ~35,000 after layer 0 and ~196,000 in layer 4 (MLX fp32, diag),
    # past fp16's 65,504 (bf16 has fp32's range, so MLX never sees it). Every norm on the stream is an RMSNorm (scale-
    # invariant), so the body carries h / 32: embeddings x 1/32, o_proj and down_proj x 1/32 (exact powers of two,
    # absorbed by the per-channel int8 scales), and each norm's input is prescaled to ~10-50 so a layer norm that squares
    # in fp16 cannot overflow.
    for i in range(NL):
        for k_ in (f"layers.{i}.self_attn.o_proj.weight", f"layers.{i}.mlp.down_proj.weight"):
            W[k_] = (W[k_].astype(np.float32) / SCL).astype(f16)
    E = np.zeros((H, H // 16), f16)
    E[np.arange(H), np.arange(H) // 16] = 1
    Z = np.zeros((1, 1, H), f16)

    def rms(x, w, m=1.0, name=None):
        shp = list(x.shape)
        if m != 1.0:
            x = mb.mul(x=x, y=f16(m))
        xx = mb.concat(values=[x, mb.mul(x=x, y=f16(-1))], axis=-1)
        n = mb.layer_norm(x=xx, axes=[-1], epsilon=f16(1e-6))
        n = mb.slice_by_index(x=n, begin=[0] * len(shp), end=shp)
        return mb.mul(x=n, y=w, name=name) if name else mb.mul(x=n, y=w)

    def rope(t, c, s):
        shp = list(t.shape)
        x1 = mb.slice_by_index(x=t, begin=[0, 0, 0, 0], end=shp[:-1] + [HD // 2])
        x2 = mb.slice_by_index(x=t, begin=[0, 0, 0, HD // 2], end=shp)
        rot = mb.concat(values=[mb.mul(x=x2, y=f16(-1)), x1], axis=-1)
        return mb.add(x=mb.mul(x=t, y=c), y=mb.mul(x=rot, y=s))

    def heads(x, w, n):
        L = x.shape[1]
        return mb.transpose(x=mb.reshape(x=mb.linear(x=x, weight=w), shape=[1, L, n, HD]), perm=[0, 2, 1, 3])

    def shift(t):
        return mb.concat(values=[Z, mb.slice_by_index(x=t, begin=[0, 0, 0], end=[1, NB - 1, H])], axis=1)

    def gprep(n, p):
        dyn = mb.linear(x=n, weight=W[p + ".kernel_projection.weight"])                     # (1, 16, 640)
        e = mb.linear(x=mb.reshape(x=dyn, shape=[1, NB * 4, H // 16]), weight=E)            # (1, 64, H)
        e = mb.reshape(x=e, shape=[1, NB, 4, H])
        base = W[p + ".base_kernel"]
        ks = [mb.add(x=mb.reshape(x=mb.slice_by_index(x=e, begin=[0, 0, j, 0], end=[1, NB, j + 1, H]), shape=[1, NB, H]),
                     y=base[j // 2, j % 2]) for j in range(4)]
        return mb.add(x=mb.mul(x=ks[0], y=n), y=mb.mul(x=ks[1], y=shift(n))), ks[2:]

    def gfin(y, ks):
        return mb.add(x=mb.mul(x=ks[0], y=y), y=mb.mul(x=ks[1], y=shift(y)))

    def attn(xa, hctx, i, cq, sq, ckb, skb, mask, kc, vc):
        p = f"layers.{i}.self_attn."
        q = rope(rms(heads(xa, W[p + "q_proj.weight"], NH), W[p + "q_norm.weight"], 1 / 8), cq, sq)    # (1, 32, 16, HD)
        xk = mb.concat(values=[hctx, xa], axis=1)                                                    # ctx rows, block rows
        k = rope(rms(heads(xk, W[p + "k_proj.weight"], NKV), W[p + "k_norm.weight"], 1 / 8), ckb, skb)  # (1, 8, 32, HD)
        v = heads(xk, W[p + "v_proj.weight"], NKV)
        K = mb.concat(values=[kc, k], axis=2)
        Vv = mb.concat(values=[vc, v], axis=2)
        qg = mb.reshape(x=q, shape=[1, NKV, 4 * NB, HD])
        s = mb.add(x=mb.mul(x=mb.matmul(x=qg, y=K, transpose_y=True), y=f16(HD ** -0.5)), y=mask)
        o = mb.matmul(x=mb.softmax(x=s, axis=-1), y=Vv)
        o = mb.reshape(x=mb.transpose(x=mb.reshape(x=o, shape=[1, NH, NB, HD]), perm=[0, 2, 1, 3]), shape=[1, NB, NH * HD])
        kx = mb.slice_by_index(x=k, begin=[0, 0, 0, 0], end=[1, NKV, NC, HD], name=f"kx{i}")
        vx = mb.slice_by_index(x=v, begin=[0, 0, 0, 0], end=[1, NKV, NC, HD], name=f"vx{i}")
        return mb.linear(x=o, weight=W[p + "o_proj.weight"]), kx, vx

    def mlp(xm, i):
        p = f"layers.{i}.mlp."
        g = mb.silu(x=mb.linear(x=xm, weight=W[p + "gate_proj.weight"]))
        return mb.linear(x=mb.mul(x=g, y=mb.linear(x=xm, weight=W[p + "up_proj.weight"])), weight=W[p + "down_proj.weight"])

    fp = types.fp16
    specs = ([mb.TensorSpec(shape=(1, NB, H), dtype=fp), mb.TensorSpec(shape=(1, NC, 5 * H), dtype=fp),
              mb.TensorSpec(shape=(1, 1, NB, HD), dtype=fp), mb.TensorSpec(shape=(1, 1, NB, HD), dtype=fp),
              mb.TensorSpec(shape=(1, 1, NC + NB, HD), dtype=fp), mb.TensorSpec(shape=(1, 1, NC + NB, HD), dtype=fp),
              mb.TensorSpec(shape=(1, 1, 4 * NB, T), dtype=fp)]
             + [mb.TensorSpec(shape=(1, NKV, WC, HD), dtype=fp) for _ in range(2 * NL)])

    @mb.program(input_specs=specs, opset_version=ct.target.macOS15)
    def body(x, tg, cq, sq, ckb, skb, mask, kc0, kc1, kc2, kc3, kc4, vc0, vc1, vc2, vc3, vc4):
        kcs, vcs = [kc0, kc1, kc2, kc3, kc4], [vc0, vc1, vc2, vc3, vc4]
        hctx = rms(mb.linear(x=tg, weight=W["fc.weight"]), W["hidden_norm.weight"])
        h, ko, vo = mb.mul(x=x, y=f16(1 / SCL)), [], []
        for i in range(NL):
            p = f"layers.{i}."
            xa, ks = gprep(rms(h, W[p + "input_layernorm.weight"], PRE[0][i]), p + "attention_conv")
            a, kx, vx = attn(xa, hctx, i, cq, sq, ckb, skb, mask, kcs[i], vcs[i])
            h = mb.add(x=h, y=gfin(a, ks))
            xm, ks = gprep(rms(h, W[p + "post_attention_layernorm.weight"], PRE[1][i]), p + "mlp_conv")
            h = mb.add(x=h, y=gfin(mlp(xm, i), ks))
            ko.append(kx)
            vo.append(vx)
        hid = rms(mb.slice_by_index(x=h, begin=[0, 1, 0], end=[1, NB, H]), W["norm.weight"], PRE[2], name="hid")
        selh = mb.linear(x=hid, weight=W["candidate_selector.hidden_projection.weight"], name="selh")
        return (hid, selh, *ko, *vo)

    save(body, "body")
    del W, body
    if fp16 or body_only:
        return
    LM = lm_head16()

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, NB - 1, H), dtype=fp)], opset_version=ct.target.macOS15)
    def head(hid):
        lg, bm = [], []
        for c in range(V // CH):
            l_ = mb.linear(x=hid, weight=LM[c * CH:(c + 1) * CH])
            lg.append(l_)
            bm.append(mb.reduce_max(x=mb.reshape(x=l_, shape=[1, NB - 1, CH // 128, 128]), axes=[-1], keep_dims=False))
        return mb.concat(values=lg, axis=-1, name="logits"), mb.concat(values=bm, axis=-1, name="bmax")

    save(head, "head")
    del head

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, NB - 1, H), dtype=fp)], opset_version=ct.target.macOS15)
    def headm(hid):
        bm = [mb.reduce_max(x=mb.reshape(x=mb.linear(x=hid, weight=LM[c * CH:(c + 1) * CH]), shape=[1, NB - 1, CH // 128, 128]),
                            axes=[-1], keep_dims=False) for c in range(V // CH)]
        return mb.concat(values=bm, axis=-1, name="bmax")

    save(headm, "headm")


def load_models(names=("body", "head", "headm")):
    import coremltools as ct
    out = {}
    for n in names:
        t0 = time.time()
        out[n] = ct.models.MLModel(os.path.join(AD, n + ".mlpackage"), compute_units=ct.ComputeUnit.CPU_AND_NE)
        print(f"loaded {n} in {time.time() - t0:.1f} s", flush=True)
    return out


def placement(m):
    import coremltools as ct
    from collections import Counter
    from coremltools.models.compute_plan import MLComputePlan
    plan = MLComputePlan.load_from_path(path=m.get_compiled_model_path(), compute_units=ct.ComputeUnit.CPU_AND_NE)
    c, cpu_ops = Counter(), Counter()
    for op in plan.model_structure.program.functions["main"].block.operations:
        u = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if u is None:
            continue
        dev = type(u.preferred_compute_device).__name__[2:-13]
        c[dev] += 1
        if dev != "NeuralEngine":
            cpu_ops[op.operator_name.split(".")[-1]] += 1
    return dict(c), dict(cpu_ops)


def codebooks():
    import mlx.core as mx
    w = mx.load(os.path.join(D, "l120_ft.safetensors"))
    return (np.array(w["candidate_selector.predecessor_codebook.weight"].astype(mx.float16)),
            np.array(w["candidate_selector.successor_codebook.weight"].astype(mx.float16)))


def post(lg, bm, selh, anchor, PC, SC):
    """The two-stage top-16 (as r84's head2s) and the selector chain (temperature 0), on the CPU."""
    tb = np.argpartition(bm, -16, axis=-1)[:, -16:]                                  # (15, 16) top blocks
    sel = np.take_along_axis(lg.reshape(NB - 1, V // 128, 128), tb[:, :, None], axis=1).reshape(NB - 1, 16 * 128)
    j = np.argpartition(sel, -16, axis=-1)[:, -16:]
    cand = np.take_along_axis(tb, j // 128, axis=1) * 128 + j % 128
    un = np.take_along_axis(sel, j, axis=1).astype(np.float32)
    pred, path = int(anchor), []
    for p in range(NB - 1):
        e = (PC[pred].astype(np.float32) * selh[p]) @ SC[cand[p]].astype(np.float32).T
        pred = int(cand[p, int(np.argmax(un[p] + e))])
        path.append(pred)
    return path, cand


def mlx_drafter(fp32=False):
    """The stock MLX DFlash2 drafter with l120_ft, bound to the target's 4-bit embeddings and lm_head (as r84)."""
    import mlx.core as mx
    import mlx.nn as nn
    sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p4s/Leap_third_party"))
    import dflash_model_mlx as DF

    class Target(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.QuantizedEmbedding(V, H, group_size=64, bits=4)
            self.lm_head = nn.QuantizedLinear(H, V, bias=False, group_size=64, bits=4)
    w = mx.load(TGT)
    tgt = Target()
    tgt.load_weights([(f"embed_tokens.{s}", w[f"language_model.model.embed_tokens.{s}"]) for s in ("weight", "scales", "biases")]
                     + [(f"lm_head.{s}", w[f"language_model.lm_head.{s}"]) for s in ("weight", "scales", "biases")])
    del w
    mx.eval(tgt.parameters())
    d = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
    d.load_weights(os.path.join(D, "l120_ft.safetensors"), strict=False)
    if fp32:
        d.set_dtype(mx.float32)
    d.bind(tgt)
    mx.eval(d.parameters())
    return d, tgt


def check():
    import mlx.core as mx
    d, tgt = mlx_drafter()
    M = load_models(("body", "head"))
    PC, SC = codebooks()
    MASK = int(d.config.mask_token_id)
    res = []
    for fpath in sorted(glob.glob(os.path.join(D, "dfl2d_code", "*.npz")))[:3] + sorted(glob.glob(os.path.join(D, "dfl2d_math", "*.npz")))[:3]:
        z = np.load(fpath)
        f, ids = z["f"].reshape(z["f"].shape[0], -1), z["ids"]
        for a in (NC, NC + 40):                 # first call: empty cache, 16 new ctx rows at positions 0..15
            if a + NB >= len(ids):
                continue
            blk = mx.array([[int(ids[a])] + [MASK] * (NB - 1)])
            ctx = f[a - NC:a]
            cache = d.make_cache()
            hid_ref = d.hidden_states(blk, mx.array(ctx[None]).astype(mx.bfloat16), cache, 1)
            lg_ref = d.compute_logits(hid_ref)
            path_ref = np.array(d.candidate_selector.select(hid_ref, lg_ref, blk[:, 0], 0.0)[0][0]).tolist()
            emb = np.array((tgt.embed_tokens(blk) * d.embed_scale).astype(mx.float16))
            o = M["body"].predict(body_inputs(emb, ctx[None], 0, 0))
            hid = o["hid"].astype(np.float32)[0]
            hr = np.array(hid_ref.astype(mx.float32))[0]
            cos = (hid * hr).sum(-1) / np.linalg.norm(hid, axis=-1) / np.linalg.norm(hr, axis=-1)
            kr = np.array(cache[0].keys.astype(mx.float32))[0, :, :NC]
            kerr = np.abs(o["kx0"][0] - kr).max() / np.abs(kr).max()
            oh = M["head"].predict({"hid": o["hid"].astype(np.float16)})
            path, cand = post(oh["logits"][0], oh["bmax"][0], o["selh"][0].astype(np.float32), int(ids[a]), PC, SC)
            lr = np.array(lg_ref.astype(mx.float32))[0]
            top1 = float(np.mean(oh["logits"][0].argmax(-1) == lr.argmax(-1)))
            same = sum(int(p == q) for p, q in zip(path, path_ref))
            k = 0
            while k < NB - 1 and path[k] == path_ref[k]:
                k += 1
            res.append((float(cos.min()), float(cos.mean()), float(kerr), top1, same, k))
            say(f"  check {os.path.basename(os.path.dirname(fpath))}/{os.path.basename(fpath)} at {a}: hidden cos min "
                f"{cos.min():.4f} mean {cos.mean():.4f}; layer-0 ctx keys rel err {kerr:.4f}; logits top-1 = MLX's on "
                f"{top1 * 15:.0f}/15 rows; drafted tokens = MLX's on {same}/15 (equal prefix {k})")
    r = np.array(res)
    say(f"check vs MLX (bf16 drafter, 4-bit head): {len(r)} blocks, hidden cos min {r[:, 0].min():.4f} mean {r[:, 1].mean():.4f}, "
        f"top-1 agreement {r[:, 3].mean() * 100:.1f} %, drafted tokens equal {r[:, 4].sum():.0f}/{15 * len(r)}, "
        f"equal prefix mean {r[:, 5].mean():.1f}/15")


def diag():
    """Where the ANE body's distance to MLX comes from: MLX fp32 is the reference; MLX bf16 and the CoreML body (int8 /
    fp16 weights) on CPU_ONLY and on CPU_AND_NE against it, hidden cos per row, 6 blocks."""
    import mlx.core as mx
    import coremltools as ct
    d, tgt = mlx_drafter()
    d32, _ = mlx_drafter(fp32=True)
    MS = {}
    for n in [n for n in ("body", "body16") if os.path.isdir(os.path.join(AD, n + ".mlpackage"))]:
        for cu in ("CPU_ONLY", "CPU_AND_NE"):
            MS[f"{n} {cu}"] = ct.models.MLModel(os.path.join(AD, n + ".mlpackage"), compute_units=getattr(ct.ComputeUnit, cu))
    MASK = int(d.config.mask_token_id)
    cs = {k: [] for k in ["mlx bf16"] + list(MS)}

    def cosr(a, b):
        return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)
    for fpath in sorted(glob.glob(os.path.join(D, "dfl2d_code", "*.npz")))[:3] + sorted(glob.glob(os.path.join(D, "dfl2d_math", "*.npz")))[:3]:
        z = np.load(fpath)
        f, ids = z["f"].reshape(z["f"].shape[0], -1), z["ids"]
        a = NC + 40
        blk = mx.array([[int(ids[a])] + [MASK] * (NB - 1)])
        ctx = f[a - NC:a]
        r32 = np.array(d32.hidden_states(blk, mx.array(ctx[None]).astype(mx.float32), d32.make_cache(), 1))[0]
        r16 = np.array(d.hidden_states(blk, mx.array(ctx[None]).astype(mx.bfloat16), d.make_cache(), 1).astype(mx.float32))[0]
        cs["mlx bf16"].append(cosr(r16, r32))
        emb = np.array((tgt.embed_tokens(blk) * d.embed_scale).astype(mx.float16))
        inp = body_inputs(emb, ctx[None], 0, 0)
        for k, m in MS.items():
            cs[k].append(cosr(m.predict(inp)["hid"].astype(np.float32)[0], r32))
    for k, v in cs.items():
        v = np.concatenate(v)
        say(f"  diag {k:20s} vs MLX fp32: hidden cos mean {v.mean():.4f}, min {v.min():.4f}, rows < 0.99: {(v < 0.99).sum()}/{len(v)}")
    # activation magnitudes along DFlash2DecoderLayer.__call__ (MLX fp32, last block): fp16 tops out at 65,504, and a
    # layer norm that squares in fp16 overflows above ~256
    import mlx.nn as nn
    mxa = lambda t: float(mx.abs(t).max())
    h = (d32.embed_tokens(blk) * d32.embed_scale).astype(mx.float32)
    fco = d32.fc(mx.array(ctx[None]).astype(mx.float32))
    hc = d32.hidden_norm(fco)
    say(f"  magnitudes: ctx features max {np.abs(ctx).max():.1f}, fc out max {mxa(fco):.1f}, h_ctx max {mxa(hc):.1f}, embeddings max {mxa(h):.3f}")
    for i, (layer, c) in enumerate(zip(d32.layers, d32.make_cache())):
        res = h
        n1 = layer.input_layernorm(h)
        x, k1 = layer.attention_conv.prepare(n1)
        a = layer.self_attn(x, hc, d32.rope, c)
        h = res + layer.attention_conv.finish(a, k1)
        n2 = layer.post_attention_layernorm(h)
        xm, k2 = layer.mlp_conv.prepare(n2)
        g = nn.silu(layer.mlp.gate_proj(xm)) * layer.mlp.up_proj(xm)
        m = layer.mlp.down_proj(g)
        h2 = h + layer.mlp_conv.finish(m, k2)
        say(f"  layer {i}: norm1 {mxa(n1):.1f}, conv-in {mxa(x):.1f}, attn {mxa(a):.1f}, h {mxa(h):.1f}, norm2 {mxa(n2):.1f}, "
            f"mlp-in {mxa(xm):.1f}, gate*up {mxa(g):.1f}, mlp out {mxa(m):.1f}, h {mxa(h2):.1f}")
        h = h2


def walk():
    """Kept per check, teacher-forced along the 125B's own tokens (r84's replay: dfl2d_code 24 samples, dfl2d_math 60):
    the stock MLX drafter (bf16, 4-bit head, temperature 0) vs the Neural Engine drafter (int8 body + head, fp16; the
    host keeps the sliding-window cache: each call takes the <= 16 newest context rows, the prompt is fed in 16-row
    chunks; cached rows capped at 2,031, MLX's window is 2,046 + the new rows)."""
    import mlx.core as mx
    d, tgt = mlx_drafter()
    sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p4s/Leap_third_party"))
    import dflash_model_mlx as DF
    M = load_models(("body", "head"))
    PC, SC = codebooks()
    MASK = int(d.config.mask_token_id)
    emb_mask = np.array((tgt.embed_tokens(mx.array([MASK])) * d.embed_scale).astype(mx.float16))[0]

    def mlx_walk(sample):
        f, ids, off, gen = sample
        cache = d.make_cache()
        start = max(off, gen - (d.config.sliding_window - 1))
        for c in cache:
            c.offset = start
        fed, a, out, drafts, n = start, gen, [], [], len(ids)
        while a < off + n - 1:
            ctx = mx.array(f[fed - off:a - off][None]).astype(mx.bfloat16)
            blk = mx.array([[int(ids[a - off])] + [MASK] * 15])
            dr = np.array(d.propose(blk, ctx, cache, 0.0, 1)[0][0]).tolist()
            drafts.append(tuple(dr))
            fed = a
            true = ids[a - off + 1:a - off + 16]
            k = 0
            while k < len(true) and dr[k] == true[k]:
                k += 1
            out.append(k + 1)
            if (trim := cache[0].offset - a) > 0:
                DF._trim_recent_cache(cache, trim)
            a = a + k + 1
        return out, drafts

    def ane_walk(sample):
        f, ids, off, gen = sample
        n = len(ids)
        emb = np.array((tgt.embed_tokens(mx.array(ids.astype(np.int64))) * d.embed_scale).astype(mx.float16))
        K = [np.zeros((NKV, 0, HD), np.float16) for _ in range(NL)]
        Vh = [np.zeros((NKV, 0, HD), np.float16) for _ in range(NL)]
        bufk = [np.zeros((1, NKV, WC, HD), np.float16) for _ in range(NL)]
        bufv = [np.zeros((1, NKV, WC, HD), np.float16) for _ in range(NL)]

        def call(rows, p_ctx, tok_i, p_blk, draft):
            m = len(rows)
            x = np.concatenate([emb[tok_i][None], np.repeat(emb_mask[None], NB - 1, 0)])[None]
            tg = np.zeros((1, NC, 5 * H), np.float16)
            tg[0, :m] = rows
            nc = min(K[0].shape[1], WC)
            for i in range(NL):
                bufk[i][:] = 0
                bufv[i][:] = 0
                if nc:
                    bufk[i][0, :, WC - nc:] = K[i][:, -nc:]
                    bufv[i][0, :, WC - nc:] = Vh[i][:, -nc:]
            jt = np.full(T, -1)
            jt[WC - nc:WC] = np.arange(nc)
            jt[WC:WC + m] = nc + np.arange(m)
            ok = (jt[None] >= 0) & ((nc + m) + np.arange(NB)[:, None] - jt[None] < 2048)
            ok[:, WC + NC:] = True
            inp = {"x": x, "tg": tg, "mask": np.tile(np.where(ok, 0, -30000).astype(np.float16), (4, 1))[None, None]}
            inp["cq"], inp["sq"] = rope_tab(range(p_blk, p_blk + NB))
            inp["ckb"], inp["skb"] = rope_tab(list(range(p_ctx, p_ctx + NC)) + list(range(p_blk, p_blk + NB)))
            for i in range(NL):
                inp[f"kc{i}"], inp[f"vc{i}"] = bufk[i], bufv[i]
            o = M["body"].predict(inp)
            for i in range(NL):
                K[i] = np.concatenate([K[i], o[f"kx{i}"][0, :, :m].astype(np.float16)], axis=1)[:, -2047:]
                Vh[i] = np.concatenate([Vh[i], o[f"vx{i}"][0, :, :m].astype(np.float16)], axis=1)[:, -2047:]
            if not draft:
                return None
            q = M["head"].predict({"hid": o["hid"].astype(np.float16)})
            return post(q["logits"][0], q["bmax"][0], o["selh"][0].astype(np.float32), int(ids[tok_i]), PC, SC)[0]
        start = max(off, gen - 2047)
        fed, a, out, drafts = start, gen, [], []
        while a < off + n - 1:
            rows = f[fed - off:a - off]
            while len(rows) > NC:
                call(rows[:NC], fed, a - off, a, False)
                fed += NC
                rows = rows[NC:]
            dr = call(rows, fed, a - off, a, True)
            drafts.append(tuple(dr))
            fed = a
            true = ids[a - off + 1:a - off + 16]
            k = 0
            while k < len(true) and dr[k] == true[k]:
                k += 1
            out.append(k + 1)
            a = a + k + 1
        return out, drafts
    for folder, name in (("dfl2d_code", "code (HumanEval+)"), ("dfl2d_math", "math (GSM8K CoT)")):
        smp = []
        for fp_ in sorted(glob.glob(os.path.join(D, folder, "*.npz"))):
            z = np.load(fp_)
            smp.append((z["f"].reshape(z["f"].shape[0], -1), z["ids"], int(z["off"]), int(z["gen"])))
        res = {}
        for arm, fn in (("mlx", mlx_walk), ("ane", ane_walk)):
            t0 = time.time()
            kept, drafts = [], []
            for sm in smp:
                o_, dr = fn(sm)
                kept += o_
                drafts.append(dr)
            res[arm] = (kept, drafts, time.time() - t0)
        same = sum(int(x == y) for da, db in zip(res["mlx"][1], res["ane"][1]) for x, y in zip(da, db))
        tot = sum(min(len(da), len(db)) for da, db in zip(res["mlx"][1], res["ane"][1]))
        km, ka = np.mean(res["mlx"][0]), np.mean(res["ane"][0])
        say(f"walk {name}: kept per check MLX bf16 {km:.3f} ({len(res['mlx'][0])} checks, {res['mlx'][2]:.0f} s), Neural "
            f"Engine int8 {ka:.3f} ({len(res['ane'][0])} checks, {res['ane'][2]:.0f} s) -> {100 * (ka / km - 1):+.2f} %; identical "
            f"drafts on {same} / {tot} aligned checks")


def tm(f, n=40, warm=5):
    for _ in range(warm):
        f()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts)), 1e3 * float(np.percentile(ts, 90))


def timing():
    M = load_models()
    for n, m in M.items():
        c, other = placement(m)
        say(f"placement {n}: {c}; ops off the Neural Engine: {other}")
    rng = np.random.default_rng(0)
    inp = body_inputs(rng.standard_normal((1, NB, H)) * 0.05, rng.standard_normal((1, NC, 5 * H)), 4000, WC, rng)
    PC, SC = codebooks()
    st = {}
    st["body"] = tm(lambda: M["body"].predict(inp))
    hid = M["body"].predict(inp)["hid"].astype(np.float16)
    st["head"] = tm(lambda: M["head"].predict({"hid": hid}))
    st["headm"] = tm(lambda: M["headm"].predict({"hid": hid}))
    oh = M["head"].predict({"hid": hid})
    selh = M["body"].predict(inp)["selh"][0].astype(np.float32)
    st["post"] = tm(lambda: post(oh["logits"][0], oh["bmax"][0], selh, 1000, PC, SC))

    def full():
        o = M["body"].predict(inp)
        q = M["head"].predict({"hid": o["hid"].astype(np.float16)})
        return post(q["logits"][0], q["bmax"][0], o["selh"][0].astype(np.float32), 1000, PC, SC)

    def fullm():
        o = M["body"].predict(inp)
        return M["headm"].predict({"hid": o["hid"].astype(np.float16)})
    st["full"] = tm(full)
    st["fullm"] = tm(fullm)
    for k, (med, p90) in st.items():
        say(f"  {k}: median {med:.2f} ms, p90 {p90:.2f} ms")
    say(f"step 1: one 16-row block (body + head + CPU top-16 and chain, Python CoreML calls) {st['full'][0]:.2f} ms; "
        f"body + head without the Python copy of the logits {st['fullm'][0]:.2f} ms (+ CPU post ~{st['post'][0]:.2f} on the "
        f"block max path) -> line 30 ms: {'PASS' if st['full'][0] <= 30 else ('PASS on the native path' if st['fullm'][0] + st['post'][0] <= 30 else 'KILL')}")
    gb = {n: os.path.getsize(os.path.join(AD, n + ".mlpackage", "Data", "com.apple.CoreML", "weights", "weight.bin")) / 1e9
          for n in ("body", "headm")}
    say(f"  int8 weight bytes per block: body {gb['body']:.3f} GB + head {gb['headm']:.3f} GB -> body {gb['body'] / st['body'][0] * 1e3:.0f} "
        f"GB/s, headm {gb['headm'] / st['headm'][0] * 1e3:.0f} GB/s, body + headm {(gb['body'] + gb['headm']) / st['fullm'][0] * 1e3:.0f} "
        f"GB/s effective (Python call overhead included)")


# ---- step 2: the GPU check proxy beside the ANE worker
def ane_worker(mode, q, ready):
    M = load_models(("body", "headm"))
    rng = np.random.default_rng(1)
    inp = body_inputs(rng.standard_normal((1, NB, H)) * 0.05, rng.standard_normal((1, NC, 5 * H)), 4000, WC, rng)

    def draft():
        o = M["body"].predict(inp)
        M["headm"].predict({"hid": o["hid"].astype(np.float16)})
    for _ in range(5):
        draft()
    ready.set()
    lat = {1: [], 2: [], 3: []}
    while True:
        m = mode.value
        if m < 0:
            break
        if m == 0:
            time.sleep(0.001)
            continue
        t0 = time.perf_counter()
        draft()
        lat[m].append(time.perf_counter() - t0)
        if m in RATE:
            time.sleep(max(0.0, t0 + 1.0 / RATE[m] - time.perf_counter()))
    q.put({m: (len(v), 1e3 * float(np.median(v)) if v else 0.0) for m, v in lat.items()})


def gpu():
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    mode, q, ready = ctx.Value("i", 0), ctx.Queue(), ctx.Event()
    wp = ctx.Process(target=ane_worker, args=(mode, q, ready))
    wp.start()
    import mlx.core as mx
    sys.path.insert(0, os.path.expanduser("~/MoonShots/patches/planes_engine"))
    import tierglu as TG
    sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p4s/Leap_scripts"))
    import tierglu_nax6 as N6
    PLAN, R, R4, S = "4433333333", 10, 2, 16
    RD = os.path.expanduser("~/BigMoE/models/bare/records3")
    SH = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}

    def load(layer):                                    # k62p_floor.py's loader (records3 pools, slot = expert id)
        idx = json.load(open(f"{RD}/index.json"))["layers"][str(layer)]
        E, REC = idx["E"], idx["recsz"]
        raw = np.fromfile(f"{RD}/{idx['file']}", dtype=np.uint8).reshape(E, REC)
        msb, lsb = {}, {}
        for p, pr in enumerate(TG.PROJ):
            o, i = SH[pr]
            ow, os_, ob = idx["msb_piece_off"][3 * p:3 * p + 3]
            msb[pr] = {"w3": mx.array(np.ascontiguousarray(raw[:, ow:ow + o * i * 3 // 8]).view(np.uint32).reshape(E, o, i * 3 // 32)),
                       "s3": mx.array(np.ascontiguousarray(raw[:, os_:os_ + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16),
                       "b3": mx.array(np.ascontiguousarray(raw[:, ob:ob + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16)}
            lw, lb = idx["lsb_piece_off"][2 * p:2 * p + 2]
            lsb[pr] = {"l1": mx.array(np.ascontiguousarray(raw[:, lw:lw + o * i // 8]).view(np.uint32).reshape(E, o, i // 32)),
                       "b": mx.array(np.ascontiguousarray(raw[:, lb:lb + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16)}
        del raw
        mx.eval([v for dd in (msb, lsb) for pr in dd for v in dd[pr].values()])
        return msb, lsb
    LAY = (6, 20, 40)
    POOLS = {l: load(l) for l in LAY}
    rng = np.random.default_rng(2)
    TR = {l: [np.load(os.path.expanduser(f"~/Breakthrough/data/expA/traces125/{t}_800_g.npz"))[f"L{l}"].astype(np.int32)
              for t in ("code", "chat", "math", "bulgarian")] for l in LAY}
    plans = []
    for _ in range(4):
        pl = []
        for c in range(48):
            l = LAY[c % 3]
            tr = TR[l][int(rng.integers(0, 4))]
            t0 = int(rng.integers(0, tr.shape[0] - S))
            ids = tr[t0:t0 + S, :R]
            sm, sl = mx.array(ids), mx.array(ids[:, :R4].copy())
            pl.append((l, sm, sl, None, mx.softmax(mx.random.normal((S, R)), axis=-1).astype(mx.bfloat16)))
        plans.append(pl)
    ONES, ONESW = mx.ones((S, R), mx.int32), mx.ones((H,), mx.bfloat16)
    X0 = (mx.random.normal((S, H)) * 0.5).astype(mx.bfloat16)
    mx.eval(X0, *[a for pl in plans for (_, sm, sl, g, sc) in pl for a in (sm, sl, sc)])

    def check(pl):
        x = X0
        for (l, sm, sl, g, sc) in pl:
            msb, lsb = POOLS[l]
            y = N6.tier_glu_nax6(x, msb, lsb, sm, sl, plan=PLAN, grp=N6.gpu_grp(sm, sl, PLAN)).reshape(S * R, -1)
            x = mx.fast.rms_norm(TG.combine_phase(y, y, ONES, sc), ONESW, 1e-6)
        mx.eval(x)
    for i in range(8):
        check(plans[i % 4])
    ready.wait(600)
    say("step 2: GPU check proxy = the expert part of a 16-row check (48 chained NAX6 calls on real layers 6/20/40, real "
        "routing, 7.4 GB), ms per check; ANE worker modes: 0 idle, 1 back to back, 2 = 12 drafts/s, 3 = 36 drafts/s")
    seq = [0, 1, 0, 2, 0, 3] * 5 + [0]
    ms = {m: [] for m in (0, 1, 2, 3)}
    blocks = []
    for bi, m in enumerate(seq):
        mode.value = m
        time.sleep(0.08)
        ts = []
        for i in range(12):
            mx.synchronize()
            t0 = time.perf_counter()
            check(plans[i % 4])
            ts.append(1e3 * (time.perf_counter() - t0))
        blocks.append((m, float(np.median(ts))))
        ms[m] += ts
    mode.value = -1
    lat = q.get(timeout=60)
    wp.join(30)
    rat = {m: [] for m in (1, 2, 3)}
    for bi in range(1, len(blocks) - 1):
        m, v = blocks[bi]
        if m:
            rat[m].append(v / (0.5 * (blocks[bi - 1][1] + blocks[bi + 1][1])))
    idle = [v for m, v in blocks if m == 0]
    say(f"  idle blocks: median {np.median(idle):.1f} ms (spread {min(idle):.1f}-{max(idle):.1f})")
    names = {1: "back to back", 2: "12 drafts/s", 3: "36 drafts/s"}
    for m in (1, 2, 3):
        n, med = lat[m]
        say(f"  ANE {names[m]}: check {np.median(ms[m]):.1f} ms, vs the neighbouring idle blocks x{np.median(rat[m]):.3f} "
            f"(per block {', '.join(f'{r:.3f}' for r in rat[m])}); ANE draft (body + headm) median {med:.2f} ms over {n} drafts")
    r1 = float(np.median(rat[1]))
    say(f"step 2 verdict (line +10 %, back to back): {'PASS' if r1 <= 1.10 else 'KILL'} at x{r1:.3f}; at 36/s x{np.median(rat[3]):.3f}, "
        f"12/s x{np.median(rat[2]):.3f}")


if __name__ == "__main__":
    {"build": build, "build16": lambda: build(fp16=True), "buildbody": lambda: build(body_only=True), "check": check, "diag": diag, "walk": walk, "time": timing, "gpu": gpu}[sys.argv[1]]()
