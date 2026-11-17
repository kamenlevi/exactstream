"""PF-B MOE-BLOCK INVARIANCE (Leap, Sun 2026-11-08 16:4x). BigMoE's PFB A/B (d44ec7a): every PF-B arm breaks the decode
logits hash vs PF-A (0/7, 0/24, 0/22), pb3n (no read-ahead) included, and even on edit10 (1,565 tokens: ONE 1,565-row MoE
call vs PF-A's 1,024 + 541). That is below qwen4_exp's 2,048-row DQ-GEMM lane (which only the fused gate/up modules use;
this engine's are unfused) and below BUG1. My pfb_fix test covered the SwitchGLU alone (bitwise ==).
This test runs the engine's whole prefill MoE block, mlx_lm 0.31.3 Qwen3NextSparseMoeBlock.__call__ (what qwen4_exp's
SparseMoeBlock calls at prefill with an unfused switch). Steps: router 8-bit g64 N 512, softmax precise, argpartition
top-10, take, norm, SwitchGLU, weighted sum, shared expert 4-bit g64 swiglu MLP, shared_expert_gate 4-bit g64 N 1,
sigmoid, add. It then runs the residual write (qwen4_exp._hyper_residual_write, hc 4).
Two call shapes:
  - PF-A: 1,024-token chunks, stock SwitchGLU.
  - PF-B: groups of whole chunks <= TOK, pflm's switch_glu_padded, Y sliced per chunk.
Synthetic weights at the real shapes and quantization; 1,565 / 4,730 / 7,932 tokens at TOK 1,024 / 4,096 / 8,192.
Pre-registered: the first op (forward order) whose output differs between the two shapes is the culprit. If all ops are
==, the MoE block is invariant and the break is elsewhere (next: an engine per-layer hidden-state diff). -> stdout"""
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU, _gather_sort, _scatter_unsort
import mtplx.models.qwen4_exp as Q

E, TOPK, D, F, HC = 512, 10, 2560, 640, 4
rng = np.random.default_rng(5)
mx.random.seed(5)


def qlin(i, o, bits, smin, smax):
    m = nn.QuantizedLinear(i, o, bias=False, group_size=64, bits=bits)
    m.weight = mx.array(rng.integers(0, 2 ** 32, (o, i * bits // 32), dtype=np.uint32))
    m.scales = mx.random.uniform(smin, smax, (o, i // 64)).astype(mx.bfloat16)
    m.biases = (-(2 ** bits - 1) / 2 * m.scales.astype(mx.float32)).astype(mx.bfloat16)
    mx.eval(m.weight, m.scales, m.biases)
    return m


def build_switch():
    m = SwitchGLU(D, F, E)
    for p, (o, i) in (("gate_proj", (F, D)), ("up_proj", (F, D)), ("down_proj", (D, F))):
        q = QuantizedSwitchLinear(i, o, E, bias=False, group_size=64, bits=4)
        q.weight = mx.array(rng.integers(0, 2 ** 32, (E, o, i // 8), dtype=np.uint32))
        q.scales = mx.random.uniform(0.002, 0.02, (E, o, i // 64)).astype(mx.bfloat16)
        q.biases = (-7.5 * q.scales.astype(mx.float32)).astype(mx.bfloat16)
        mx.eval(q.weight, q.scales, q.biases)
        setattr(m, p, q)
    return m


def switch_glu_padded(m, x, indices, align=64):
    """verbatim copy of ~/BigMoE/scripts/pflm.py switch_glu_padded (snapshot b5e0d1f)"""
    xe = mx.expand_dims(x, (-2, -3))
    xs, idx, inv = _gather_sort(xe, indices)
    R = idx.shape[0]
    pad = (-R) % align
    if pad:
        xs = mx.concatenate([xs, mx.repeat(xs[-1:], pad, axis=0)], axis=0)
        idx = mx.concatenate([idx, mx.repeat(idx[-1:], pad, axis=0)], axis=0)
    h = m.activation(m.up_proj(xs, idx, sorted_indices=True), m.gate_proj(xs, idx, sorted_indices=True))
    y = m.down_proj(h, idx, sorted_indices=True)
    if pad:
        y = y[:R]
    return _scatter_unsort(y, inv, indices.shape).squeeze(-2)


router = qlin(D, E, 8, 0.0005, 0.0015)
sh_gate, sh_up = qlin(D, F, 4, 0.002, 0.02), qlin(D, F, 4, 0.002, 0.02)
sh_down = qlin(F, D, 4, 0.002, 0.02)
seg = qlin(D, 1, 4, 0.002, 0.02)
swm = build_switch()
OPS = ("router", "gates", "inds", "scores", "switch", "combine", "shared", "seg", "shared_y", "out", "write")


def block(x, sw, sizes=None):
    """mlx_lm 0.31.3 Qwen3NextSparseMoeBlock.__call__ (sharding off), with its intermediates"""
    g = router(x)
    gates = mx.softmax(g, axis=-1, precise=True)
    inds = mx.argpartition(gates, kth=-TOPK, axis=-1)[..., -TOPK:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    y = sw(x, inds)
    mx.eval(y)                                       # as the tier wrapper (mx.eval(y) inside prefill_call)
    yc = (y * scores[..., None]).sum(axis=-2)
    se = sh_down(swiglu(sh_gate(x), sh_up(x)))
    sg = seg(x) if not sizes else mx.concatenate([seg(x[:, o:o + n]) for o, n in zip(np.cumsum([0] + sizes[:-1]), sizes)], axis=1)
    sy = mx.sigmoid(sg) * se
    return dict(router=g, gates=gates, inds=inds, scores=scores, switch=y, combine=yc, shared=se, seg=sg,
                shared_y=sy, out=yc + sy)


def pfa(xs, hy, inj):
    outs = []
    for x, h, j in zip(xs, hy, inj):
        r = block(x, swm)
        r["write"] = Q._hyper_residual_write(h, r["out"], j)
        mx.eval(list(r.values()))
        outs.append(r)
    return {k: mx.concatenate([o[k] for o in outs], axis=1) for k in OPS}


def pfb(xs, hy, inj, tok):
    groups, cur, n = [], [], 0                       # pflm._split_layers' grouping, verbatim
    for k, x in enumerate(xs):
        if cur and n + x.shape[1] > tok:
            groups.append(cur)
            cur, n = [], 0
        cur.append(k)
        n += x.shape[1]
    groups.append(cur)
    parts = {k: [] for k in OPS}
    for g in groups:
        M = xs[g[0]] if len(g) == 1 else mx.concatenate([xs[k] for k in g], axis=1)
        r = block(M, lambda x, i: switch_glu_padded(swm, x, i), [xs[k].shape[1] for k in g] if SEGC else None)
        off = 0
        for k in g:
            n_ = xs[k].shape[1]
            Yk = r["out"] if len(g) == 1 else r["out"][:, off:off + n_]
            w = Q._hyper_residual_write(hy[k], Yk, inj[k])
            mx.eval(w, *r.values())
            for op in OPS[:-1]:
                parts[op].append(r[op] if len(g) == 1 else r[op][:, off:off + n_])
            parts["write"].append(w)
            off += n_
    return {k: mx.concatenate(v, axis=1) for k, v in parts.items()}, [len(g) for g in groups]


def diff_rows(a, b):
    ne = a != b
    if ne.ndim > 2:
        ne = ne.reshape(ne.shape[0], ne.shape[1], -1).any(axis=-1)
    return int(ne.sum().item())


print(f"PF-B MOE-BLOCK + SEG-PER-CHUNK (pfb_seg.py) {time.strftime('%a %H:%M:%S')}: MLX {mx.__version__}; block = mlx_lm Qwen3NextSparseMoeBlock "
      f"(router q8 g64 N {E}, top-{TOPK}, SwitchGLU q4 g64 {D}->{F}->{D}, shared q4 g64, shared_expert_gate q4 g64 N 1) "
      f"+ qwen4_exp._hyper_residual_write (hc {HC}); opdiet resid {Q.qwen4_opdiet_enabled('resid')}", flush=True)
c = (mx.random.normal((D,)) * 0.8).astype(mx.float32)    # a shared component: skewed (popular-expert) routing
first = {}
SEGC = True
for T in (1565, 4730, 7932):
    x = ((mx.random.normal((1, T, D)) * 0.6) + c).astype(mx.bfloat16)
    hyper = (mx.random.normal((1, T, HC * D)) * 0.5).astype(mx.bfloat16)
    inject = (2.0 * mx.sigmoid(mx.random.normal((1, T, HC)))).astype(mx.bfloat16)
    mx.eval(x, hyper, inject)
    cut = list(range(0, T, 1024))
    xs = [x[:, a:a + 1024] for a in cut]
    hy = [hyper[:, a:a + 1024] for a in cut]
    inj = [inject[:, a:a + 1024] for a in cut]
    ref = pfa(xs, hy, inj)
    cnt = np.bincount(np.array(ref["inds"]).reshape(-1), minlength=E)
    for tok in (4096, 8192):
        got, gl = pfb(xs, hy, inj, tok)
        d = {op: diff_rows(ref[op], got[op]) for op in OPS}
        bad = [op for op in OPS if d[op]]
        first[(T, tok)] = bad[0] if bad else None
        print(f"  T {T:5d} TOK {tok:5d} (groups of {gl} chunks): " + ("ALL ==" if not bad else
              "DIFF rows: " + ", ".join(f"{op} {d[op]}" for op in bad) + f"; first: {bad[0]}"), flush=True)
    print(f"    routing skew at T {T}: rows per expert max {cnt.max()}, median {int(np.median(cnt))}, unused {int((cnt == 0).sum())}",
          flush=True)
culprits = sorted({v for v in first.values() if v})
print(f"SEG-PER-CHUNK verdict (pre-registered): " + (f"first differing op(s): {culprits}" if culprits else
      "ALL == (the MoE block + write are invariant: the break is elsewhere)") + f"; peak {mx.get_peak_memory() / 1e9:.2f} GB",
      flush=True)

x = ((mx.random.normal((1, 8192, D)) * 0.6) + c).astype(mx.bfloat16)
mx.eval(x)
for M in (541, 1024, 1025, 1100, 1565, 2048, 4096, 8192):
    one = seg(x[:, :M])
    ch = mx.concatenate([seg(x[:, a:min(a + 1024, M)]) for a in range(0, M, 1024)], axis=1)
    ne = np.array((one != ch).reshape(-1))
    idx = np.nonzero(ne)[0]
    print(f"  seg probe: M {M:5d} one call vs 1,024-row calls: {int(ne.sum())} rows differ" + (f" (rows {idx.min()}..{idx.max()})" if len(idx) else ""), flush=True)
sh = sh_gate
for M in (1565, 4096):
    one = sh(x[:, :M]); ch = mx.concatenate([sh(x[:, a:min(a + 1024, M)]) for a in range(0, M, 1024)], axis=1)
    print(f"  control (shared gate_proj N 640) M {M}: {int((one != ch).any(axis=-1).sum().item())} rows differ", flush=True)
