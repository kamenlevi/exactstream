"""Unit test for planes_stream.py on the 125B's layer 0 (needs make_planes.py output for layer 0 in PLANE_DIR). Registers
the MSB and LSB virtual layers with BigMoE's ext (add_layer + ensure, no gate spin), then checks:
  - decode math (_glu over pool slots): ranks < R vs the original 4-bit experts, ranks >= R vs the MSB-only reference;
  - prefill math (_prefill: grouped ensure + scratch + compact ids) on 16 tokens, same references.
References are computed from ~/BigMoE/models/bare (read-only) with mx.gather_qmm at 4 bits.
usage: planes_unit.py PLANE_DIR   (GPU, ~3 GB)"""
import json, os, struct, sys
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import planes_stream as PS
from planes_stream import ext
from mlx_lm.models.switch_layers import SwiGLU

PD = sys.argv[1]
MD = os.path.expanduser("~/BigMoE/models/bare")
idx = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]
R, K, Cm, Cl = 4, 10, 64, 32


def tensor(name):
    f = f"{MD}/{idx[name]}"
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; m = json.loads(fh.read(n))[name]
    a, b = m["data_offsets"]
    dt = {"U32": np.uint32, "BF16": np.uint16}[m["dtype"]]
    arr = np.memmap(f, dt, "r", offset=8 + n + a, shape=((b - a) // np.dtype(dt).itemsize,)).reshape(m["shape"])
    return mx.array(np.array(arr)) if dt == np.uint32 else mx.array(np.array(arr)).view(mx.bfloat16)


# register layer 0's planes as virtual layers 0 (MSB) and 1 (LSB); ext.start via a 2-layer gate
st = PS.PlaneStreamer(PD, Cm, Cl, R)
st.gate = PS.PlaneGate(2)
st.nl, st.pools, st.act = 1, {}, SwiGLU()
for kind, C, vl_want in (("msb", Cm, 0), ("lsb", Cl, 1)):
    pieces = st.lay[kind]["pieces"]
    fd = os.open(os.path.join(PD, f"L0.{kind}"), os.O_RDONLY)
    pools = [mx.zeros((C, *pc["shape"]), PS.DTYPES[pc["dtype"]]) for pc in pieces]
    mx.eval(*pools)
    vl = ext.add_layer(C, 512, [fd] * len(pieces), [pc["offset"] for pc in pieces], [pc["bytes"] for pc in pieces], pools)
    assert vl == vl_want, vl
    st.pools[(kind, 0)] = {pc["name"]: p for pc, p in zip(pieces, pools)}

# references from the original 4-bit tensors of layer 0
W = {}
for p in ("gate_proj", "up_proj", "down_proj"):
    pre = f"language_model.model.layers.0.mlp.switch_mlp.{p}."
    W[p] = (tensor(pre + "weight"), tensor(pre + "scales"), tensor(pre + "biases"))
KEEP2 = mx.array(0xCCCCCCCC, dtype=mx.uint32)


def ref(x, ids, msb):
    """x [N, 1, D] rows, ids [N] expert ids -> [N, D] with the 4-bit (or MSB-2 masked) experts."""
    def proj(p, h):
        w, s, b = W[p]
        if msb:
            lsum = None
            for j in range(8):
                part = (w >> (4 * j)) & 3
                lsum = part if lsum is None else lsum + part
            off = lsum.reshape(*s.shape, 8).sum(-1).astype(mx.float32) / 64
            w, b = w & KEEP2, (b.astype(mx.float32) + s.astype(mx.float32) * off).astype(s.dtype)
        return mx.gather_qmm(h, w, s, b, rhs_indices=ids, transpose=True, group_size=64, bits=4)
    h = SwiGLU()(proj("up_proj", x), proj("gate_proj", x))
    return proj("down_proj", h).squeeze(-2)


def rel(a, b):
    return float((mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max() / mx.abs(b.astype(mx.float32)).max()).item())


rng = np.random.default_rng(0)
# decode: one token, 10 distinct experts in "gate-rank order"
ids = rng.choice(512, K, replace=False).astype(np.int32)
sm = mx.array(np.array(ext.ensure(0, mx.array(ids)))).reshape(1, K).astype(mx.uint32)
sl = mx.array(np.array(ext.ensure(1, mx.array(ids[:R])))).reshape(1, R).astype(mx.uint32)
x = (mx.random.normal((1, 1, 2560), key=mx.random.key(1)) * 0.5).astype(mx.bfloat16)
M, Lp = st.pools[("msb", 0)], st.pools[("lsb", 0)]
Lr = {k: mx.take(v, sl[0], axis=0) for k, v in Lp.items()}
ss = {p: mx.take(M[p + ".scales4"], sm[0, :R], axis=0) for p in ("gate_proj", "up_proj", "down_proj")}
Y = PS._glu(st.act, x, sm, M, mx.arange(R).astype(mx.uint32).reshape(1, R), Lr, ss, R, K)[0]      # [K, D]
xr = mx.repeat(x, K, axis=0)
Y4 = ref(xr, mx.array(ids).astype(mx.uint32), False)
Ym = ref(xr, mx.array(ids).astype(mx.uint32), True)
e4, em, ex = rel(Y[:R], Y4[:R]), rel(Y[R:], Ym[R:]), rel(Y[:R], Ym[:R])
print(f"decode: ranks<{R} vs 4-bit {e4:.2e} (vs MSB-only {ex:.2e}); ranks>={R} vs MSB-only {em:.2e}", flush=True)
assert e4 < 2e-2 and em < 2e-2 and ex > 3 * e4, (e4, em, ex)

# prefill: 16 tokens, random distinct experts per token, through _prefill (ensure groups + scratch + compact ids)
T = 16
idsT = np.stack([rng.choice(512, K, replace=False) for _ in range(T)]).astype(np.int32)
xT = (mx.random.normal((1, T, 2560), key=mx.random.key(2)) * 0.5).astype(mx.bfloat16)
YT = PS._prefill(st, 0, xT, mx.array(idsT)[None])[0]                                   # [T, K, D]
xrT = mx.repeat(xT.reshape(T, 1, 2560), K, axis=0)
fl = mx.array(idsT.reshape(-1)).astype(mx.uint32)
R4 = ref(xrT, fl, False).reshape(T, K, -1)
RM = ref(xrT, fl, True).reshape(T, K, -1)
p4, pm = rel(YT[:, :R], R4[:, :R]), rel(YT[:, R:], RM[:, R:])
print(f"prefill ({T} tokens, {len(np.unique(idsT))} experts > Cm={Cm}: grouped ensure): ranks<{R} vs 4-bit {p4:.2e}; "
      f"ranks>={R} vs MSB-only {pm:.2e}", flush=True)
assert p4 < 2e-2 and pm < 2e-2, (p4, pm)
print("planes_unit: OK")
