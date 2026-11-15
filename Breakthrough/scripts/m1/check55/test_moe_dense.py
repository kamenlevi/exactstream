"""test_moe_dense.py (Breakthrough, CHECK55 lever 2; the bit test pre-registered in docs/CHECK55_LEVER2.md, ad35cd4).

moe_dense.dense_pre(blk, x) vs K-73's chain as the exact-45 engine runs it: Leap's router_fused(blk.gate(x), 10) and
shared_cat(blk, x) (imported read-only) under ref2.apply() and, at 17-32 rows, v32.install(ref2) in the decode_verify
phase (its 2 x 16 split, as V32_MOE1's _tier_block).
- Weights: layers 0 / 24 / 47 of the 125B, read from the dense shard (only those tensors).
- Rows 16 and 32 are the verdict; 1 / 8 / 17 / 24 are extras.
- 6 input draws per (layer, rows). Compared bitwise: inds, scores, g, s, and moe_tail(y, g, s).
- Unit checks: A's logits / swiglu rows / g vs MLX; my swiglu vs mlx_lm's compiled swiglu on all 65,536 bf16 gates
  x 8 ups.
- Sensitivity: (a) the ladder reversed must differ somewhere; (b) a 1-ulp flip in one x element must change an output.
usage: [BigMoE venv] python -I -B test_moe_dense.py [out stem]   (MEM < 2 GB; share lock)"""
import json
import os
import struct
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
sys.path.insert(0, os.path.expanduser("~/Leap/scripts/week98"))
import numpy as np
import mlx.core as mx
import mlx.nn as nn

import moe_dense as MD
import ref2
import v32
from router_fused import router_fused
from glue_fused import shared_cat, moe_tail
from mlx_lm.models.activations import swiglu
from mtplx.attention_context import attention_phase

STEM = sys.argv[1] if len(sys.argv) > 1 else "l2_bit"
KSPEC = sys.argv[2] if len(sys.argv) > 2 else "a"         # A kernel: "a" or "a2", optionally :rv:rr:rp:sgn
_ks = KSPEC.split(":")
ACFG = dict(zip(("rv", "rr", "rp", "sgn"), (int(v) for v in _ks[1:])))
KW = dict(a=_ks[0], **ACFG)
OUT = os.path.expanduser("~/Breakthrough/data/m1/check55/lever2")
os.makedirs(OUT, exist_ok=True)
SHARD = os.path.expanduser("~/BigMoE/models/bare/model-00016-of-00017.safetensors")
LAYERS = (0, 24, 47)
ROWS_V, ROWS_X = (16, 32), (1, 8, 17, 24)
LOG = []


def say(s):
    print(s, flush=True)
    LOG.append(s)


def st_read(path, keys):
    """only these tensors from a safetensors file (raw bytes; U32 packs, BF16 scales / biases)"""
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        for k in keys:
            h = hdr[k]
            b, e = h["data_offsets"]
            f.seek(8 + n + b)
            buf = f.read(e - b)
            if h["dtype"] == "U32":
                out[k] = mx.array(np.frombuffer(buf, np.uint32).reshape(h["shape"]))
            elif h["dtype"] == "BF16":
                out[k] = mx.array(np.frombuffer(buf, np.uint16).reshape(h["shape"])).view(mx.bfloat16)
            else:
                raise ValueError((k, h["dtype"]))
    return out


def qlin(W, pre, din, dout, bits):
    m = nn.QuantizedLinear(din, dout, bias=False, group_size=64, bits=bits)
    m.weight, m.scales, m.biases = W[pre + ".weight"], W[pre + ".scales"], W[pre + ".biases"]
    assert m.weight.shape == (dout, din * bits // 32), (pre, m.weight.shape)
    return m


class SE(nn.Module):
    def __init__(self, g, u, d):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = g, u, d


class Blk(nn.Module):
    def __init__(self, gate, se, seg):
        super().__init__()
        self.gate, self.shared_expert, self.shared_expert_gate = gate, se, seg
        self.top_k = 10
        self.norm_topk_prob = True


def block(L):
    p = f"language_model.model.layers.{L}.mlp."
    names = ["gate", "shared_expert.gate_proj", "shared_expert.up_proj", "shared_expert.down_proj", "shared_expert_gate"]
    W = st_read(SHARD, [p + n + s for n in names for s in (".weight", ".scales", ".biases")])
    mx.eval(list(W.values()))
    se = SE(qlin(W, p + "shared_expert.gate_proj", 2560, 640, 4), qlin(W, p + "shared_expert.up_proj", 2560, 640, 4),
            qlin(W, p + "shared_expert.down_proj", 640, 2560, 4))
    return Blk(qlin(W, p + "gate", 2560, 512, 8), se, qlin(W, p + "shared_expert_gate", 2560, 1, 4))


def bits(a):
    return a.view(mx.uint16) if a.dtype == mx.bfloat16 else a


def ndiff(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype, (a.shape, b.shape, a.dtype, b.dtype)
    return int(mx.sum(bits(a) != bits(b)).item())


def draws(rows, seed):
    """the pre-registered 6 inputs [1, rows, 2560] bf16"""
    ks = mx.random.split(mx.random.key(seed), 8)
    out = {}
    for i, sc in enumerate((0.5, 1.0, 2.0)):
        out[f"normal x{sc}"] = (mx.random.normal((1, rows, 2560), key=ks[i]) * sc).astype(mx.bfloat16)
    base = mx.random.normal((1, rows, 2560), key=ks[3])
    spikes = mx.random.uniform(shape=(1, rows, 2560), key=ks[4]) < 0.02
    out["heavy tail"] = mx.where(spikes, base * 30, base).astype(mx.bfloat16)
    one = mx.random.normal((1, 1, 2560), key=ks[5]).astype(mx.bfloat16)
    out["rows equal"] = mx.broadcast_to(one, (1, rows, 2560))
    t = mx.random.normal((1, rows, 2560), key=ks[6]).astype(mx.bfloat16)
    t = mx.concatenate([mx.zeros((1, 1, 2560), mx.bfloat16), mx.full((1, 1, 2560), 0.37, mx.bfloat16), t[:, 2:]], axis=1) \
        if rows >= 2 else mx.zeros((1, rows, 2560), mx.bfloat16)
    out["ties (zero + const rows)"] = t
    return out


def ref_chain(blk, x):
    with attention_phase("decode_verify"):
        inds, scores = router_fused(blk.gate(x), blk.top_k)
        g, s = shared_cat(blk, x)
        mx.eval(inds, scores, g, s)
    return inds, scores, g, s


def swiglu_mine(gate, up):
    """an elementwise kernel with moe_dense's swiglu expressions (MLX's Sigmoid functor, then two bf16 multiplies)"""
    k = mx.fast.metal_kernel(name="b2_swiglu_probe", input_names=["gt_", "up_"], output_names=["o_"], header=MD._SIG,
                             source="""
        const uint i = thread_position_in_grid.x;
        const T gt = gt_[i];
        const T up = up_[i];
        const T sg = SigmoidL()(gt);
        const T si = gt * sg;
        o_[i] = si * up;
    """)
    (o,) = k(inputs=[gate, up], template=[("T", gate.dtype)], grid=(gate.size, 1, 1), threadgroup=(256, 1, 1),
             output_shapes=[gate.shape], output_dtypes=[gate.dtype])
    return o


def main():
    t0 = time.time()
    say(f"LEVER2 BIT TEST {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; A kernel {KSPEC}; sources {MD.SOURCES}")
    ref2.apply()
    v32.install(ref2)
    say(f"  ref2 on {ref2.enabled()}; v32 on {v32.ST['on']}; layers {LAYERS}; rows {ROWS_V} (+ extras {ROWS_X})")
    res = {"unit": {}, "bit": [], "sens": {}}

    # unit 1: swiglu, all 65,536 bf16 gate values x 8 up values
    gates = mx.arange(65536, dtype=mx.uint32).astype(mx.uint16).view(mx.bfloat16)
    ups = mx.array([1.0, -1.0, 0.5, 3.1415, -0.007, 100.0, 1e-3, -2.5]).astype(mx.bfloat16)
    G = mx.broadcast_to(gates[None, :], (8, 65536))
    U = mx.broadcast_to(ups[:, None], (8, 65536))
    G, U = mx.contiguous(G).reshape(-1), mx.contiguous(U).reshape(-1)
    a, b = swiglu_mine(G, U), swiglu(G, U)
    gf = G.astype(mx.float32)
    fin = mx.isfinite(gf)
    d_fin = int(mx.sum((bits(a) != bits(b)) & fin).item())
    d_all = int(mx.sum(bits(a) != bits(b)).item())
    res["unit"]["swiglu_exhaustive"] = {"diff_finite_gate": d_fin, "diff_all": d_all, "n": int(G.size)}
    say(f"  unit swiglu: {d_fin} differ of {int(mx.sum(fin).item())} finite-gate pairs ({d_all} of {G.size} incl. inf / nan gates)")

    blocks = {}
    for L in LAYERS:
        blocks[L] = block(L)
    say(f"  weights read: layers {LAYERS} ({time.time() - t0:.1f} s)")

    # unit 2: A's logits / swiglu rows / g vs MLX's own 16-row calls (ref2's _qmm -> MLX's 16-row call)
    for L in LAYERS:
        blk = blocks[L]
        pk = MD.packs(blk)
        x = draws(16, 1000 + L)["normal x1.0"].reshape(16, 2560)
        lg, xs, g = {"a": MD.dense_a, "a2": MD.dense_a2, "a3": MD.dense_a3, "a4": MD.dense_a4, "a5": MD.dense_a5, "a2p": MD.dense_a2p}[KW["a"]](pk, x, **ACFG)
        (wr, sr, br), (wc, sc, bc), _ = pk
        ref_lg = mx.quantized_matmul(x, wr, sr, br, transpose=True, group_size=64, bits=8)
        y = mx.quantized_matmul(x, wc, sc, bc, transpose=True, group_size=64, bits=4)
        ref_xs = swiglu(y[:, :640], y[:, 640:1280])
        u = {"logits": ndiff(lg, ref_lg), "swiglu": ndiff(xs, ref_xs), "g": ndiff(g, y[:, 1280:1281])}
        res["unit"][f"A_L{L}"] = u
        say(f"  unit A L{L}: differing logits {u['logits']} / 8192, swiglu {u['swiglu']} / 10240, g {u['g']} / 16")

    # the bit test
    worst = 0
    for L in LAYERS:
        blk = blocks[L]
        for rows in ROWS_V + ROWS_X:
            for name, x in draws(rows, 17 * L + rows).items():
                ri, rs, rg, rsh = ref_chain(blk, x)
                mi, ms, mg, msh = MD.dense_pre(blk, x, **KW)
                mx.eval(mi, ms, mg, msh)
                yr = (mx.random.normal((1, rows, 2560), key=mx.random.key(rows + L)) * 0.3).astype(mx.bfloat16)
                tr, tm = moe_tail(yr, rg, rsh, exp="precise"), moe_tail(yr, mg, msh, exp="precise")
                d = {"inds": ndiff(mi, ri), "scores": ndiff(ms, rs), "g": ndiff(mg, rg), "s": ndiff(msh, rsh),
                     "tail": ndiff(tm, tr)}
                tot = sum(d.values())
                if rows in ROWS_V:
                    worst = max(worst, tot)
                res["bit"].append({"layer": L, "rows": rows, "draw": name, **d})
                say(f"  L{L:2d} rows {rows:2d} {name:26s} differing: inds {d['inds']} scores {d['scores']} g {d['g']} "
                    f"s {d['s']} tail {d['tail']}{'' if rows in ROWS_V else '  (extra)'}")

    # sensitivity (a): the ladder reversed
    sa = 0
    for L in LAYERS:
        pk = MD.packs(blocks[L])
        for rows in (16, 32):
            for name, x in draws(rows, 99 + L + rows).items():
                x2 = x.reshape(rows, 2560)
                ref = MD.dense_a(pk, x2)
                var = MD.dense_a(pk, x2, ladder=1)
                sa += sum(ndiff(p, q) for p, q in zip(var, ref))
    res["sens"]["ladder_reversed_outputs_differ"] = sa
    say(f"  sensitivity (a): the reversed ladder changes {sa} of A's outputs (must be > 0)")
    # sensitivity (b): one x element's lowest mantissa bit
    blk = blocks[24]
    x = draws(16, 7)["normal x1.0"]
    xb = x.reshape(-1).view(mx.uint16)
    flip = mx.concatenate([xb[:1234], (xb[1234:1235] ^ 1), xb[1235:]]).view(mx.bfloat16).reshape(x.shape)
    o0, o1 = MD.dense_pre(blk, x, **KW), MD.dense_pre(blk, flip, **KW)
    sb = sum(ndiff(p, q) for p, q in zip(o0, o1))
    res["sens"]["one_ulp_flip_outputs_differ"] = sb
    say(f"  sensitivity (b): a 1-ulp flip of x[0, 0, 1234] changes {sb} outputs (must be > 0)")

    unit_ok = (res["unit"]["swiglu_exhaustive"]["diff_finite_gate"] == 0
               and all(v == 0 for k, u in res["unit"].items() if k.startswith("A_") for v in u.values()))
    extras = sum(sum(r[k] for k in ("inds", "scores", "g", "s", "tail")) for r in res["bit"] if r["rows"] in ROWS_X)
    ok = worst == 0 and unit_ok and sa > 0 and sb > 0
    res["verdict"] = {"PASS": ok, "worst_verdict_case": worst, "unit_ok": unit_ok, "extras_differing": extras}
    say(f"VERDICT: {'PASS' if ok else 'FAIL'} (verdict rows {ROWS_V}: worst case {worst} differing elements; units "
        f"{'ok' if unit_ok else 'FAIL'}; extras {extras} differing; sensitivity {sa} / {sb}); "
        f"{time.time() - t0:.1f} s, MLX peak {mx.get_peak_memory() / 1e9:.2f} GB")
    with open(os.path.join(OUT, STEM + ".json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(OUT, STEM + ".txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
