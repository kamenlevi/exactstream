"""ANE-2b (Leap, Mon 2026-11-02; Fable mission 18 item 1): where the Neural Engine body's 15-16 ms go (it streams 0.48 GB
at 30-32 GB/s against the head's 75-79), by ablation, with random int8 weights of the real shapes (latency does not
depend on the values). Same graph as ane_probe.py's body (rank-3 layout, layer-norm RMSNorms, GQA attention over the
cached context + 16 new context rows + the block, DFlash2 grouped dynamic convs, SwiGLU), with switches:
  W (cached context rows, inputs copied per call), conv / attn / mlp on-off, NB block rows (16, or 48 = three batched
  16-row candidates, the convs' shift kept inside each 16-row block), layers.
Usage: [BigMoE venv] ane_body_bench.py VARIANT [VARIANT ...] -> ~/Leap/data/ane_body_bench.txt
VARIANT = name:key=val,...  e.g. full:  w15:W=15  noconv:conv=0  mlp:conv=0,attn=0"""
import os, sys, time
import numpy as np
import coremltools as ct
import coremltools.optimize as cto
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

D = os.path.expanduser("~/Leap/data")
AD = os.path.join(D, "ane")
H, NH, NKV, HD, FF, V = 2560, 32, 8, 128, 7680, 248320
NC = 16
OUT = os.path.join(D, "ane_body_bench.txt")
CFG8 = cto.coreml.OptimizationConfig(global_config=cto.coreml.OpLinearQuantizerConfig(
    mode="linear_symmetric", dtype="int8", granularity="per_channel", weight_threshold=100000))
f16, fp = np.float16, types.fp16


def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")


def build(name, W=2031, conv=1, attn=1, mlp=1, NB=16, NL=5):
    rng = np.random.default_rng(0)
    R = lambda *s: (rng.standard_normal(s) * 0.02).astype(f16)
    T = W + NC + NB
    E = np.zeros((H, H // 16), f16)
    E[np.arange(H), np.arange(H) // 16] = 1
    G = NB // 16

    def rms(x, w, m=1 / 64):
        shp = list(x.shape)
        x = mb.mul(x=x, y=f16(m))
        n = mb.layer_norm(x=mb.concat(values=[x, mb.mul(x=x, y=f16(-1))], axis=-1), axes=[-1], epsilon=f16(1e-6))
        return mb.mul(x=mb.slice_by_index(x=n, begin=[0] * len(shp), end=shp), y=w)

    def rope(t, c, s):
        shp = list(t.shape)
        x1 = mb.slice_by_index(x=t, begin=[0, 0, 0, 0], end=shp[:-1] + [HD // 2])
        x2 = mb.slice_by_index(x=t, begin=[0, 0, 0, HD // 2], end=shp)
        return mb.add(x=mb.mul(x=t, y=c), y=mb.mul(x=mb.concat(values=[mb.mul(x=x2, y=f16(-1)), x1], axis=-1), y=s))

    def heads(x, w, n):
        L = x.shape[1]
        return mb.transpose(x=mb.reshape(x=mb.linear(x=x, weight=w), shape=[1, L, n, HD]), perm=[0, 2, 1, 3])

    def shift(t):                                        # causal shift by one row inside each 16-row block
        t4 = mb.reshape(x=t, shape=[1, G, 16, H])
        z = np.zeros((1, G, 1, H), f16)
        return mb.reshape(x=mb.concat(values=[z, mb.slice_by_index(x=t4, begin=[0, 0, 0, 0], end=[1, G, 15, H])], axis=2),
                          shape=[1, NB, H])

    def gprep(n, wk, base):
        e = mb.reshape(x=mb.linear(x=mb.reshape(x=mb.linear(x=n, weight=wk), shape=[1, NB * 4, H // 16]), weight=E),
                       shape=[1, NB, 4, H])
        ks = [mb.add(x=mb.reshape(x=mb.slice_by_index(x=e, begin=[0, 0, j, 0], end=[1, NB, j + 1, H]), shape=[1, NB, H]),
                     y=base[j // 2, j % 2]) for j in range(4)]
        return mb.add(x=mb.mul(x=ks[0], y=n), y=mb.mul(x=ks[1], y=shift(n))), ks[2:]

    def gfin(y, ks):
        return mb.add(x=mb.mul(x=ks[0], y=y), y=mb.mul(x=ks[1], y=shift(y)))

    specs = ([mb.TensorSpec(shape=(1, NB, H), dtype=fp), mb.TensorSpec(shape=(1, NC, 5 * H), dtype=fp),
              mb.TensorSpec(shape=(1, 1, NB, HD), dtype=fp), mb.TensorSpec(shape=(1, 1, NB, HD), dtype=fp),
              mb.TensorSpec(shape=(1, 1, NC + NB, HD), dtype=fp), mb.TensorSpec(shape=(1, 1, NC + NB, HD), dtype=fp),
              mb.TensorSpec(shape=(1, 1, 4 * NB, T), dtype=fp)]
             + [mb.TensorSpec(shape=(1, NKV, W, HD), dtype=fp) for _ in range(2 * NL)])
    LW = [dict(q=R(NH * HD, H), k=R(NKV * HD, H), v=R(NKV * HD, H), o=R(H, NH * HD), g=R(FF, H), u=R(FF, H), d=R(H, FF),
               ka=R(640, H), km=R(640, H), ba=R(2, 2, H), bm=R(2, 2, H), n1=np.ones(H, f16), n2=np.ones(H, f16),
               qn=np.ones(HD, f16), kn=np.ones(HD, f16)) for _ in range(NL)]
    FC = R(H, 5 * H)

    def body(x, tg, cq, sq, ckb, skb, mask, *cache):
        kcs, vcs = cache[:NL], cache[NL:]
        hctx = rms(mb.linear(x=tg, weight=FC), np.ones(H, f16), 1.0)
        h, outs = mb.mul(x=x, y=f16(1 / 32)), []
        for i in range(NL):
            P = LW[i]
            if attn:
                n = rms(h, P["n1"])
                xa, ks = gprep(n, P["ka"], P["ba"]) if conv else (n, None)
                q = rope(rms(heads(xa, P["q"], NH), P["qn"], 1 / 8), cq, sq)
                xk = mb.concat(values=[hctx, xa], axis=1)
                k = rope(rms(heads(xk, P["k"], NKV), P["kn"], 1 / 8), ckb, skb)
                v = heads(xk, P["v"], NKV)
                K = mb.concat(values=[kcs[i], k], axis=2)
                Vv = mb.concat(values=[vcs[i], v], axis=2)
                s = mb.add(x=mb.mul(x=mb.matmul(x=mb.reshape(x=q, shape=[1, NKV, 4 * NB, HD]), y=K, transpose_y=True),
                                    y=f16(HD ** -0.5)), y=mask)
                o = mb.matmul(x=mb.softmax(x=s, axis=-1), y=Vv)
                o = mb.reshape(x=mb.transpose(x=mb.reshape(x=o, shape=[1, NH, NB, HD]), perm=[0, 2, 1, 3]), shape=[1, NB, NH * HD])
                a = mb.linear(x=o, weight=P["o"])
                h = mb.add(x=h, y=gfin(a, ks) if conv else a)
                outs += [mb.slice_by_index(x=k, begin=[0, 0, 0, 0], end=[1, NKV, NC, HD]),
                         mb.slice_by_index(x=v, begin=[0, 0, 0, 0], end=[1, NKV, NC, HD])]
            if mlp:
                n = rms(h, P["n2"])
                xm, ks = gprep(n, P["km"], P["bm"]) if conv else (n, None)
                m = mb.linear(x=mb.mul(x=mb.silu(x=mb.linear(x=xm, weight=P["g"])), y=mb.linear(x=xm, weight=P["u"])), weight=P["d"])
                h = mb.add(x=h, y=gfin(m, ks) if conv else m)
        hid = rms(mb.slice_by_index(x=h, begin=[0, 0, 0], end=[1, NB, H]), np.ones(H, f16), 1 / 128)
        return tuple([mb.identity(x=hid, name="hid")] + [mb.identity(x=t, name=f"o{j}") for j, t in enumerate(outs)])

    import inspect
    names = ["x", "tg", "cq", "sq", "ckb", "skb", "mask"] + [f"kc{i}" for i in range(NL)] + [f"vc{i}" for i in range(NL)]
    src = f"def _b({', '.join(names)}):\n    return body({', '.join(names)})\n"
    ns = {"body": body}
    exec(src, ns)
    prog = mb.program(input_specs=specs, opset_version=ct.target.macOS15)(ns["_b"])
    m = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                   compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True)
    m = cto.coreml.linear_quantize_weights(m, config=CFG8)
    p = os.path.join(AD, f"bb_{name}.mlpackage")
    m.save(p)
    inp = {"x": (rng.standard_normal((1, NB, H)) * 0.02).astype(f16), "tg": rng.standard_normal((1, NC, 5 * H)).astype(f16),
           "cq": np.ones((1, 1, NB, HD), f16), "sq": np.zeros((1, 1, NB, HD), f16),
           "ckb": np.ones((1, 1, NC + NB, HD), f16), "skb": np.zeros((1, 1, NC + NB, HD), f16),
           "mask": np.zeros((1, 1, 4 * NB, T), f16)}
    for i in range(NL):
        inp[f"kc{i}"] = rng.standard_normal((1, NKV, W, HD)).astype(f16)
        inp[f"vc{i}"] = rng.standard_normal((1, NKV, W, HD)).astype(f16)
    return p, inp


def run(spec):
    name, _, kv = spec.partition(":")
    kw = {k: int(v) for k, v in (t.split("=") for t in kv.split(",") if t)}
    t0 = time.time()
    p, inp = build(name, **kw)
    m = ct.models.MLModel(p, compute_units=ct.ComputeUnit.CPU_AND_NE)
    tb = time.time() - t0
    for _ in range(5):
        m.predict(inp)
    ts = []
    for _ in range(30):
        t1 = time.perf_counter()
        m.predict(inp)
        ts.append(time.perf_counter() - t1)
    gb = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(p) for f in fs) / 1e9
    from coremltools.models.compute_plan import MLComputePlan
    plan = MLComputePlan.load_from_path(path=m.get_compiled_model_path(), compute_units=ct.ComputeUnit.CPU_AND_NE)
    ops = [op for op in plan.model_structure.program.functions["main"].block.operations if op.operator_name.split(".")[-1] not in ("const",)]
    off = sum(1 for op in ops if (u := plan.get_compute_device_usage_for_mlprogram_operation(op)) is not None
              and "NeuralEngine" not in type(u.preferred_compute_device).__name__)
    say(f"  {name} {kw}: {1e3 * np.median(ts):.2f} ms (p90 {1e3 * np.percentile(ts, 90):.2f}); {len(ops)} ops, {off} off the "
        f"Neural Engine; package {gb:.3f} GB; built + loaded {tb:.0f} s")
    del m
    import shutil
    shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":
    say(f"ANE-2b {time.strftime('%a %H:%M')}")
    for spec in sys.argv[1:]:
        run(spec)
