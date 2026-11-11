"""ANE-2 v2 (Leap, Mon 2026-11-02; Fable mission 18 item 1, line: one batched 3-candidate call of the 1x drafter <= 14 ms):
DFlash2 L1-20 rebuilt for the Neural Engine as ONE stateful CoreML call that drafts three 16-row candidates at once.
- Layout: channels-first (1, C, 1, S) everywhere, every linear a 1x1 conv (Apple's ANE layout); S = 48 block rows
  (3 candidates x 16) + 16 context rows (block i's taps) on the k / v path.
- Attention: q_proj's rows and o_proj's columns are permuted to (kv group, head dim, head), so the four query heads of a
  group reshape straight into (1, 8, 128, 4 x 48); one small transpose each way; scores query-major (1, 8, 192, W + 48)
  as ANE-1 (q @ K^T via transpose_y); the 1 / sqrt(128) scale is folded into q_norm's weight.
- Context cache = CoreML states (macOS 15) on the Neural Engine: K (1, 8, W, 128) and V^T (1, 8, 128, W) per layer, W =
  2,064 ring slots (K and V both rows x 128: the V^T write form fails on the ANE at run time). Each call writes its 16 context rows at a host-chosen ring position (one-hot scatter: state * keep +
  onehot @ new; ANE-2c: on the Neural Engine) and attends over [ring | its own 16 block rows]; the host's mask carries
  validity (rows past each candidate's cut, unused padding, the 2,048 sliding window). The next call writes at
  p + kept + 1, over the rejected rows.
- DFlash2's grouped dynamic convs by broadcasting on (1, 160, 16, S) views (no 0/1 expansion matmuls).
- fp16 range: the residual stream carries h / 32 (ANE-1's exact fix).
- Head: the target's lm_head restricted to the top-N frequency-ranked tokens (FR-Spec; data/frspec_rank.npy), int8,
  + the per-128 block max; the selector's hidden projection. The CPU does the two-stage top-16 and the chain.
Usage: [BigMoE venv] ane_v2.py build NAME [rand=1] [N=32768] [parts=...]; time NAME; check NAME"""
import glob, json, os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np

D = os.path.expanduser("~/Leap/data")
AD = os.path.join(D, "ane")
H, NH, NKV, HD, FF, NL, V = 2560, 32, 8, 128, 7680, 5, 248320
NB, NCAND, NC = 16, 3, 16
W = int(os.environ.get("ANE_W", 2064))   # ring slots (env, so build and time agree)
S = NB * NCAND                       # 48 block rows
SK = NC + S                          # 64 rows on the k / v path
TK = W + S                           # keys per query: ring + the block rows
SCL = 32.0
PRE = ([2.0 ** 14, 1 / 64, 1 / 64, 1 / 64, 1 / 128], [1 / 64, 1 / 64, 1 / 64, 1 / 64, 1 / 128], 1 / 128)
OUT = os.path.join(D, "ane_v2.txt")


def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")


def weights(rand, N):
    """name -> fp16 numpy, in the layouts the graph wants; rand = random values of the real shapes (latency only)."""
    f16 = np.float16
    if rand:
        rng = np.random.default_rng(0)
        R = lambda *s: (rng.standard_normal(s) * 0.02).astype(f16)
        W_ = {"fc.weight": R(H, 5 * H), "hidden_norm.weight": np.ones(H, f16), "norm.weight": np.ones(H, f16),
              "candidate_selector.hidden_projection.weight": R(256, H)}
        for i in range(NL):
            p = f"layers.{i}."
            W_.update({p + "self_attn.q_proj.weight": R(NH * HD, H), p + "self_attn.k_proj.weight": R(NKV * HD, H),
                       p + "self_attn.v_proj.weight": R(NKV * HD, H), p + "self_attn.o_proj.weight": R(H, NH * HD),
                       p + "self_attn.q_norm.weight": np.ones(HD, f16), p + "self_attn.k_norm.weight": np.ones(HD, f16),
                       p + "mlp.gate_proj.weight": R(FF, H), p + "mlp.up_proj.weight": R(FF, H), p + "mlp.down_proj.weight": R(H, FF),
                       p + "attention_conv.kernel_projection.weight": R(640, H), p + "mlp_conv.kernel_projection.weight": R(640, H),
                       p + "attention_conv.base_kernel": R(2, 2, H), p + "mlp_conv.base_kernel": R(2, 2, H),
                       p + "input_layernorm.weight": np.ones(H, f16), p + "post_attention_layernorm.weight": np.ones(H, f16)})
        W_["head"] = R(N, H)
        return W_
    sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfb/Leap_scripts"))
    from ane_probe import drafter_weights, lm_head16
    W_ = drafter_weights()
    rank = np.load(os.path.join(D, "frspec_rank.npy"))[:N]
    W_["head"] = lm_head16()[np.sort(rank)]
    return W_


def build(name, rand=1, N=32768, parts="all", nl=NL):
    import coremltools as ct
    import coremltools.optimize as cto
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    f16, fp = np.float16, types.fp16
    os.makedirs(AD, exist_ok=True)
    Wt = weights(rand, N)
    # permute q_proj rows / o_proj columns from (head = g * 4 + r, d) to (g, d, r)
    perm = np.array([(g * 4 + r) * HD + d for g in range(NKV) for d in range(HD) for r in range(4)])
    for i in range(NL):
        p = f"layers.{i}."
        Wt[p + "self_attn.q_proj.weight"] = Wt[p + "self_attn.q_proj.weight"][perm]
        Wt[p + "self_attn.o_proj.weight"] = (Wt[p + "self_attn.o_proj.weight"][:, perm].astype(np.float32) / SCL).astype(f16)
        Wt[p + "mlp.down_proj.weight"] = (Wt[p + "mlp.down_proj.weight"].astype(np.float32) / SCL).astype(f16)
        Wt[p + "self_attn.q_norm.weight"] = (Wt[p + "self_attn.q_norm.weight"].astype(np.float32) * HD ** -0.5).astype(f16)
    cw = lambda w: w[:, :, None, None]
    bmask = np.ones((1, 1, 1, S), f16)
    bmask[..., ::NB] = 0                                  # the shifted row 0 of each candidate block is zero

    def conv(x, w):
        return mb.conv(x=x, weight=cw(w))

    def rms(x, w, m, axis=1, name=None):
        """RMSNorm over `axis` via layer_norm([x, -x]) (zero mean), x prescaled by m (scale-invariant)."""
        shp = list(x.shape)
        x = mb.mul(x=x, y=f16(m)) if m != 1 else x
        n = mb.layer_norm(x=mb.concat(values=[x, mb.mul(x=x, y=f16(-1))], axis=axis), axes=[axis], epsilon=f16(1e-6))
        n = mb.slice_by_index(x=n, begin=[0] * len(shp), end=shp)
        wshape = [1] * len(shp)
        wshape[axis] = shp[axis]
        return mb.mul(x=n, y=w.reshape(wshape), name=name) if name else mb.mul(x=n, y=w.reshape(wshape))

    def rope(t, c, s):                                    # t (1, n, 128, cols): rotate-half over axis 2
        shp = list(t.shape)
        x1 = mb.slice_by_index(x=t, begin=[0, 0, 0, 0], end=[shp[0], shp[1], HD // 2, shp[3]])
        x2 = mb.slice_by_index(x=t, begin=[0, 0, HD // 2, 0], end=shp)
        rot = mb.concat(values=[mb.mul(x=x2, y=f16(-1)), x1], axis=2)
        return mb.add(x=mb.mul(x=t, y=c), y=mb.mul(x=rot, y=s))

    def shift(t4):                                        # (1, 160, 16, S): one row later inside each candidate block
        sh = list(t4.shape)
        z = np.zeros((1, sh[1], sh[2], 1), f16)
        return mb.mul(x=mb.concat(values=[z, mb.slice_by_index(x=t4, begin=[0, 0, 0, 0], end=sh[:3] + [S - 1])], axis=3), y=bmask)

    def gprep(n, p):
        dyn = conv(n, Wt[p + ".kernel_projection.weight"])                          # (1, 640, 1, S)
        base = Wt[p + ".base_kernel"].reshape(2, 2, H // 16, 16)
        n4 = mb.reshape(x=n, shape=[1, H // 16, 16, S])
        s4 = shift(n4)
        ks = []
        for j in range(4):
            dj = mb.slice_by_index(x=dyn, begin=[0, j * 160, 0, 0], end=[1, (j + 1) * 160, 1, S])   # (1, 160, 1, S)
            ks.append(mb.add(x=dj, y=base[j // 2, j % 2][None, :, :, None]))                       # (1, 160, 16, S)
        out = mb.add(x=mb.mul(x=n4, y=ks[0]), y=mb.mul(x=s4, y=ks[1]))
        return mb.reshape(x=out, shape=[1, H, 1, S]), ks[2:]

    def gfin(y, ks):
        y4 = mb.reshape(x=y, shape=[1, H // 16, 16, S])
        return mb.reshape(x=mb.add(x=mb.mul(x=y4, y=ks[0]), y=mb.mul(x=shift(y4), y=ks[1])), shape=[1, H, 1, S])

    fpsp = lambda *s: mb.TensorSpec(shape=s, dtype=fp)
    native = "native" in parts          # the host writes the accepted rows into the ring (MLState); the model only reads it
    tk = W + SK if native else TK       # native: ring + block i's 16 context rows + the 48 block rows
    specs = [fpsp(1, H, 1, S), fpsp(1, 5 * H, 1, NC), fpsp(1, 1, HD, 4 * S), fpsp(1, 1, HD, 4 * S), fpsp(1, 1, HD, SK),
             fpsp(1, 1, HD, SK), fpsp(1, 1, 4 * S, tk)] + ([] if native else [fpsp(1, 1, W, NC), fpsp(1, 1, W, 1)])
    specs += [mb.StateTensorSpec(shape=(1, NKV, W, HD), dtype=fp) for _ in range(nl)]
    specs += [mb.StateTensorSpec(shape=(1, NKV, W, HD), dtype=fp) for _ in range(nl)]
    names = (["x", "tg", "cq", "sq", "ck", "sk", "mask"] + ([] if native else ["ohT", "keepT"])
             + [f"K{i}" for i in range(nl)] + [f"V{i}" for i in range(nl)])

    def body(a):
        hctx = rms(conv(a["tg"], Wt["fc.weight"]), Wt["hidden_norm.weight"], 1.0)              # (1, H, 1, 16)
        h = mb.mul(x=a["x"], y=f16(1 / SCL))
        kvo = []
        for i in range(nl):
            p = f"layers.{i}."
            if "noconv" not in parts:
                xa, ks = gprep(rms(h, Wt[p + "input_layernorm.weight"], PRE[0][i]), p + "attention_conv")
            else:
                xa, ks = rms(h, Wt[p + "input_layernorm.weight"], PRE[0][i]), None
            if "noattn" not in parts:
                q = mb.reshape(x=conv(xa, Wt[p + "self_attn.q_proj.weight"]), shape=[1, NKV, HD, 4 * S])  # (g, d, r x s)
                q = rope(rms(q, Wt[p + "self_attn.q_norm.weight"], 1 / 8, axis=2), a["cq"], a["sq"])
                xkv = mb.concat(values=[hctx, xa], axis=3)                                               # (1, H, 1, 64)
                k = mb.reshape(x=conv(xkv, Wt[p + "self_attn.k_proj.weight"]), shape=[1, NKV, HD, SK])
                k = rope(rms(k, Wt[p + "self_attn.k_norm.weight"], 1 / 8, axis=2), a["ck"], a["sk"])
                v = mb.reshape(x=conv(xkv, Wt[p + "self_attn.v_proj.weight"]), shape=[1, NKV, HD, SK])
                kT = mb.transpose(x=k, perm=[0, 1, 3, 2])                                               # (1, 8, 64, 128)
                vT = mb.transpose(x=v, perm=[0, 1, 3, 2])
                cache = []
                for st, rows in ((a[f"K{i}"], kT), (a[f"V{i}"], vT)):
                    if native:                      # read only; this call's 16 context rows join as extra keys
                        kvo.append(mb.slice_by_index(x=rows, begin=[0, 0, 0, 0], end=[1, NKV, NC, HD]))
                        cache.append(mb.concat(values=[mb.read_state(input=st), rows], axis=2))
                        continue
                    if "nowrite" in parts:          # read only: the host would write the 16 rows into a shared buffer
                        c = mb.read_state(input=st)
                    else:
                        c = mb.add(x=mb.mul(x=mb.read_state(input=st), y=a["keepT"]),
                                   y=mb.matmul(x=a["ohT"], y=mb.slice_by_index(x=rows, begin=[0, 0, 0, 0], end=[1, NKV, NC, HD])))
                        mb.coreml_update_state(state=st, value=c)
                    cache.append(mb.concat(values=[c, mb.slice_by_index(x=rows, begin=[0, 0, NC, 0], end=[1, NKV, SK, HD])], axis=2))
                qT = mb.transpose(x=q, perm=[0, 1, 3, 2])                                               # (1, 8, 192, 128)
                sc = mb.add(x=mb.matmul(x=qT, y=cache[0], transpose_y=True), y=a["mask"])               # (1, 8, 192, TK)
                o = mb.transpose(x=mb.matmul(x=mb.softmax(x=sc, axis=3), y=cache[1]), perm=[0, 1, 3, 2])  # (1, 8, 128, 192)
                att = conv(mb.reshape(x=o, shape=[1, NH * HD, 1, S]), Wt[p + "self_attn.o_proj.weight"])
                h = mb.add(x=h, y=gfin(att, ks) if ks is not None else att)
            if "noconv" not in parts:
                xm, ks = gprep(rms(h, Wt[p + "post_attention_layernorm.weight"], PRE[1][i]), p + "mlp_conv")
            else:
                xm, ks = rms(h, Wt[p + "post_attention_layernorm.weight"], PRE[1][i]), None
            m = conv(mb.mul(x=mb.silu(x=conv(xm, Wt[p + "mlp.gate_proj.weight"])), y=conv(xm, Wt[p + "mlp.up_proj.weight"])),
                     Wt[p + "mlp.down_proj.weight"])
            h = mb.add(x=h, y=gfin(m, ks) if ks is not None else m)
        hid = rms(h, Wt["norm.weight"], PRE[2], name="hid")
        outs = [mb.identity(x=conv(hid, Wt["candidate_selector.hidden_projection.weight"]), name="selh")]
        if "nohead" not in parts:
            lg = conv(hid, Wt["head"])                                                                    # (1, N, 1, S)
            if "bmaxonly" not in parts:                    # native code reads the logits in place; Python copies them
                outs.append(mb.identity(x=lg, name="logits"))
            outs.append(mb.reduce_max(x=mb.reshape(x=lg, shape=[1, N // 128, 128, S]), axes=[2], keep_dims=False, name="bmax"))
        else:
            outs.append(mb.identity(x=hid, name="hid_out"))
        if native:                                     # the CPU scores the session's extra tokens from the final hidden rows
            outs.append(mb.identity(x=hid, name="hido"))
        for j, t in enumerate(kvo):                    # native: K / V of block i's context rows, for the host's commit
            outs.append(mb.identity(x=t, name=f"{'kv'[j % 2]}x{j // 2}"))
        return tuple(outs)

    src = f"def _b({', '.join(names)}):\n    return body(dict({', '.join(f'{n}={n}' for n in names)}))\n"
    ns = {"body": body}
    exec(src, ns)
    prog = mb.program(input_specs=specs, opset_version=ct.target.macOS15)(ns["_b"])
    t0 = time.time()
    m = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                   compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True)
    m = cto.coreml.linear_quantize_weights(m, config=cto.coreml.OptimizationConfig(global_config=cto.coreml.OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8", granularity="per_channel", weight_threshold=100000)))
    path = os.path.join(AD, f"v2_{name}.mlpackage")
    m.save(path)
    json.dump({"N": N, "rand": rand, "parts": parts, "nl": nl, "W": W}, open(path + ".json", "w"))
    say(f"built v2_{name} (rand {rand}, N {N}, parts {parts}, layers {nl}) in {time.time() - t0:.0f} s")


def dummy_inputs(N):
    rng = np.random.default_rng(1)
    f16 = np.float16
    oh = np.zeros((1, 1, NC, W), f16)
    oh[0, 0, np.arange(NC), 100 + np.arange(NC)] = 1
    keep = np.ones((1, 1, 1, W), f16)
    keep[..., 100:100 + NC] = 0
    return {"x": (rng.standard_normal((1, H, 1, S)) * 0.02).astype(f16), "tg": rng.standard_normal((1, 5 * H, 1, NC)).astype(f16),
            "cq": np.ones((1, 1, HD, 4 * S), f16), "sq": np.zeros((1, 1, HD, 4 * S), f16),
            "ck": np.ones((1, 1, HD, SK), f16), "sk": np.zeros((1, 1, HD, SK), f16),
            "mask": np.zeros((1, 1, 4 * S, TK), f16), "ohT": oh.transpose(0, 1, 3, 2).copy(), "keepT": keep.transpose(0, 1, 3, 2).copy()}


def timing(name):
    import coremltools as ct
    from collections import Counter
    from coremltools.models.compute_plan import MLComputePlan
    path = os.path.join(AD, f"v2_{name}.mlpackage")
    meta = json.load(open(path + ".json"))
    m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    plan = MLComputePlan.load_from_path(path=m.get_compiled_model_path(), compute_units=ct.ComputeUnit.CPU_AND_NE)
    c, off = Counter(), Counter()
    for op in plan.model_structure.program.functions["main"].block.operations:
        u = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if u is not None:
            dev = type(u.preferred_compute_device).__name__[2:-13]
            c[dev] += 1
            if dev != "NeuralEngine":
                off[op.operator_name.split(".")[-1]] += 1
    say(f"  v2_{name}: placement {dict(c)}, off the Neural Engine {dict(off)}")
    st = m.make_state()
    inp = dummy_inputs(meta["N"])
    try:
        m.predict(inp, state=st)
    except Exception as e:
        say(f"  v2_{name} {meta}: predict FAILED: {str(e)[:160]}")
        return
    for _ in range(5):
        m.predict(inp, state=st)
    ts = []
    for _ in range(40):
        t1 = time.perf_counter()
        m.predict(inp, state=st)
        ts.append(time.perf_counter() - t1)
    gb = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs) / 1e9
    say(f"  v2_{name} {meta}: {1e3 * np.median(ts):.2f} ms (p90 {1e3 * np.percentile(ts, 90):.2f}) per batched 3-candidate call; "
        f"placement {dict(c)}, off the Neural Engine {dict(off)}; package {gb:.3f} GB")


def native_check(name, nsmp=4):
    """Real-weight v2 native model vs the stock MLX drafter (fp32 reference): one call from an empty ring, block i's 16
    context rows as the call's context, the same 16-row block in all three candidate slots; hidden cos per row of each
    candidate, and the selector input (selh) cos."""
    import mlx.core as mx
    import coremltools as ct
    sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfb/Leap_scripts"))
    from ane_probe import mlx_drafter, rope_tab
    d32, tgt = mlx_drafter(fp32=True)
    path = os.path.join(AD, f"v2_{name}.mlpackage")
    m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    MASK = int(d32.config.mask_token_id)
    cosr = lambda a, b: (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)
    allc = []
    for fpath in sorted(glob.glob(os.path.join(D, "dfl2d_code", "*.npz")))[:nsmp // 2] + sorted(glob.glob(os.path.join(D, "dfl2d_math", "*.npz")))[:nsmp // 2]:
        z = np.load(fpath)
        f, ids = z["f"].reshape(z["f"].shape[0], -1), z["ids"]
        a = NC + 40
        blk = mx.array([[int(ids[a])] + [MASK] * (NB - 1)])
        ctx = f[a - NC:a]
        ref = np.array(d32.hidden_states(blk, mx.array(ctx[None]).astype(mx.float32), d32.make_cache(), 0))[0]   # (16, H)
        emb = np.array((tgt.embed_tokens(blk) * d32.embed_scale).astype(mx.float32))[0]                         # (16, H)
        x = np.tile(emb.T[None, :, None, :], (1, 1, 1, NCAND)).astype(np.float16)                             # (1, H, 1, 48)
        tg = ctx.T[None, :, None, :].astype(np.float16)                                                        # (1, 5H, 1, 16)
        cq, sq = rope_tab(range(NC, NC + NB))                                                                  # (1, 1, 16, 128)
        cq = np.tile(cq[0, 0].T, (1, 4 * NCAND))[None, None]
        sq = np.tile(sq[0, 0].T, (1, 4 * NCAND))[None, None]
        ck, sk = rope_tab(list(range(NC)) + list(range(NC, NC + NB)) * NCAND)
        ck, sk = ck[0, 0].T[None, None], sk[0, 0].T[None, None]                                               # (1, 1, 128, 64)
        mask = np.full((4 * S, W + SK), -30000, np.float16)
        for c in range(NCAND):
            for r in range(4):
                rows = slice(r * S + c * NB, r * S + (c + 1) * NB)
                mask[rows, W:W + NC] = 0                                    # block i's context rows
                mask[rows, W + NC + c * NB:W + NC + (c + 1) * NB] = 0       # its own block rows
        st = m.make_state()
        o = m.predict({"x": x, "tg": tg, "cq": cq.astype(np.float16), "sq": sq.astype(np.float16), "ck": ck.astype(np.float16),
                       "sk": sk.astype(np.float16), "mask": mask[None, None]}, state=st)
        hid = o["hido"][0, :, 0, :].T.astype(np.float32)                                                       # (48, H)
        cs = [cosr(hid[c * NB:(c + 1) * NB], ref) for c in range(NCAND)]
        allc.append(np.concatenate(cs))
        say(f"  native check {os.path.basename(fpath)}: hidden cos per candidate min " + ", ".join(f"{c.min():.4f}" for c in cs)
            + f"; mean {np.mean(np.concatenate(cs)):.4f}")
    a_ = np.concatenate(allc)
    say(f"native check v2_{name} vs MLX fp32: hidden cos mean {a_.mean():.4f}, min {a_.min():.4f} over {a_.size} rows")


def native_walk(name, sets="code,math,agent"):
    """End-to-end kept per check on the Neural Engine (r84's replay protocol) with the v2 native model: the host keeps the
    K / V ring (written into the states before each call: Python's write_state takes whole arrays), passes the accepted
    rows since the last draft as the call's context rows (prompts in 16-row chunks), commits them after the call, and
    drafts candidate slot 0 (slots 1-2 carry the same block). Head: the static top-N rows on the Neural Engine + the
    session's own tokens scored on the CPU from the hidden rows (FR-Spec + session, as frspec_walk's arm); two-stage
    top-16 + the selector chain on the CPU. Reference: frspec_walk's MLX arm with the same head (code 9.484, math 7.136,
    agent 2.889 at 32K) and the full head (9.569 / 7.218 / 3.011)."""
    import mlx.core as mx
    import coremltools as ct
    sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfb/Leap_scripts"))
    from ane_probe import mlx_drafter, rope_tab, codebooks, lm_head16
    path = os.path.join(AD, f"v2_{name}.mlpackage")
    N = json.load(open(path + ".json"))["N"]
    m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    d, tgt = mlx_drafter()
    MASK = int(d.config.mask_token_id)
    emb_mask = np.array((tgt.embed_tokens(mx.array([MASK])) * d.embed_scale).astype(mx.float32))[0]
    PC, SC = codebooks()
    LM = lm_head16()
    sub = np.sort(np.load(os.path.join(D, "frspec_rank.npy"))[:N])           # head row j <-> token sub[j]
    insub = np.zeros(V, bool)
    insub[sub] = True
    K_ = 16

    def draft(o, anchor, sess_extra):
        lg = o["logits"][0, :, 0, :].T.astype(np.float32)[1:NB]              # candidate 0's rows 1..15: (15, N)
        bm = o["bmax"][0][:, 1:NB].T.astype(np.float32)                       # (15, N / 128)
        tb = np.argpartition(bm, -K_, axis=-1)[:, -K_:]
        sel = np.take_along_axis(lg.reshape(NB - 1, N // 128, 128), tb[:, :, None], axis=1).reshape(NB - 1, K_ * 128)
        j = np.argpartition(sel, -K_, axis=-1)[:, -K_:]
        cand = sub[np.take_along_axis(tb, j // 128, axis=1) * 128 + j % 128]
        un = np.take_along_axis(sel, j, axis=1)
        if sess_extra.size:                                                   # the session's tokens outside the static set
            hid = o["hido"][0, :, 0, 1:NB].T.astype(np.float32)                # (15, H)
            ex = hid @ LM[sess_extra].astype(np.float32).T                    # (15, n_extra)
            cand = np.concatenate([cand, np.broadcast_to(sess_extra, (NB - 1, sess_extra.size))], axis=1)
            un = np.concatenate([un, ex], axis=1)
            top = np.argpartition(un, -K_, axis=-1)[:, -K_:]
            cand, un = np.take_along_axis(cand, top, axis=1), np.take_along_axis(un, top, axis=1)
        selh = o["selh"][0, :, 0, 1:NB].T.astype(np.float32)
        pred, path_ = int(anchor), []
        for p_ in range(NB - 1):
            e = (PC[pred].astype(np.float32) * selh[p_]) @ SC[cand[p_]].astype(np.float32).T
            pred = int(cand[p_, int(np.argmax(un[p_] + e))])
            path_.append(pred)
        return path_

    def walk(sample):
        f, ids, off, gen = sample
        n = len(ids)
        emb = np.array((tgt.embed_tokens(mx.array(ids.astype(np.int64))) * d.embed_scale).astype(mx.float32))
        RK = [np.zeros((1, NKV, W, HD), np.float32) for _ in range(NL)]       # write_state takes float32 only
        RV = [np.zeros((1, NKV, W, HD), np.float32) for _ in range(NL)]
        rpos = np.full(W, -10 ** 9)                                           # absolute position held by each ring slot
        st = m.make_state()
        ring = [0]

        def call(rows, p_ctx, tok_i, p_blk):
            mm = len(rows)
            blk = np.concatenate([emb[tok_i][None], np.repeat(emb_mask[None], NB - 1, 0)])          # (16, H)
            x = np.tile(blk.T[None, :, None, :], (1, 1, 1, NCAND)).astype(np.float16)
            tg = np.zeros((1, 5 * H, 1, NC), np.float16)
            tg[0, :, 0, :mm] = rows.T
            qpos = p_blk + np.arange(NB)
            cq, sq = rope_tab(qpos)
            cq = np.tile(cq[0, 0].T, (1, 4 * NCAND))[None, None].astype(np.float16)
            sq = np.tile(sq[0, 0].T, (1, 4 * NCAND))[None, None].astype(np.float16)
            ck, sk = rope_tab(list(p_ctx + np.arange(NC)) + list(qpos) * NCAND)
            ok = np.zeros((NB, W + SK), bool)
            ok[:, :W] = (rpos[None] >= 0) & (qpos[:, None] - rpos[None] < 2048)
            ok[:, W:W + mm] = True
            ok[:, W + NC:W + NC + NB] = True
            mask = np.full((4 * S, W + SK), -30000, np.float16)
            for c in range(NCAND):
                okc = ok.copy()
                if c:
                    okc[:, W + NC:] = False
                    okc[:, W + NC + c * NB:W + NC + (c + 1) * NB] = True
                for r in range(4):
                    mask[r * S + c * NB:r * S + (c + 1) * NB] = np.where(okc, 0, -30000)
            for i in range(NL):
                st.write_state(f"K{i}", RK[i])
                st.write_state(f"V{i}", RV[i])
            o = m.predict({"x": x, "tg": tg, "cq": cq, "sq": sq, "ck": ck[0, 0].T[None, None].astype(np.float16),
                           "sk": sk[0, 0].T[None, None].astype(np.float16), "mask": mask[None, None]}, state=st)
            slots = (ring[0] + np.arange(mm)) % W
            for i in range(NL):
                RK[i][0, :, slots] = o[f"kx{i}"][0, :, :mm].transpose(1, 0, 2)
                RV[i][0, :, slots] = o[f"vx{i}"][0, :, :mm].transpose(1, 0, 2)
            rpos[slots] = p_ctx + np.arange(mm)
            ring[0] = (ring[0] + mm) % W
            return o
        start = max(off, gen - 2047)
        fed, a, out = start, gen, []
        seen = np.zeros(V, bool)
        while a < off + n - 1:
            rows = f[fed - off:a - off]
            while len(rows) > NC:
                call(rows[:NC], fed, a - off, a)
                fed += NC
                rows = rows[NC:]
            o = call(rows, fed, a - off, a)
            seen[ids[:a - off + 1]] = True
            dr = draft(o, ids[a - off], np.nonzero(seen & ~insub)[0])
            fed = a
            true = ids[a - off + 1:a - off + 16]
            k = 0
            while k < len(true) and dr[k] == true[k]:
                k += 1
            out.append(k + 1)
            a = a + k + 1
        return out
    TL = os.path.expanduser("~/TrainLab/data")
    SRC = {"code": (os.path.join(D, "dfl2d_code"), "*.npz", 24), "math": (os.path.join(D, "dfl2d_math"), "*.npz", 30),
           "agent": (os.path.join(TL, "d02a"), "agent*.npz", 12)}
    for set_ in sets.split(","):
        folder, pat, nmax = SRC[set_]
        t0 = time.time()
        kept = []
        for fp_ in sorted(glob.glob(os.path.join(folder, pat)))[:nmax]:
            z = np.load(fp_)
            kept += walk((z["f"].reshape(z["f"].shape[0], -1), z["ids"], int(z["off"]), int(z["gen"])))
        say(f"native walk v2_{name} {set_}: kept per check {np.mean(kept):.3f} ({len(kept)} checks, {time.time() - t0:.0f} s); "
            f"MLX with the same head (top {N} + session) {dict(code=9.484, math=7.136, agent=2.889)[set_]}, full head "
            f"{dict(code=9.569, math=7.218, agent=3.011)[set_]}")


if __name__ == "__main__":
    cmd, name = sys.argv[1], sys.argv[2]
    kw = {k: (int(v) if v.isdigit() else v) for k, v in (t.split("=") for t in sys.argv[3:])}
    {"build": lambda: build(name, **kw), "time": lambda: timing(name), "ncheck": lambda: native_check(name),
     "nwalk": lambda: native_walk(name, **kw)}[cmd]()
