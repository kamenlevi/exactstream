#!/usr/bin/env python3
"""CPU tests for the G1 / G2 map arms (Breakthrough, Mon 16 Nov 2026 13:xx; pre-registered docs/CHECK_BYTES_900.md
section 2). Runs on MLX's CPU device; the GPU is not touched.
  1. numerics: _fq64 + _qmm32 == Leap's int8d formula y = sx (s sum q code + b sum q) in float64 from the raw 4-bit codes
     (rel err <= 1e-5); and the error profile vs fp32 on random data (w ~ N(0, 0.02) at 4-bit g64, x ~ N(0, 1) bf16,
     32 rows) is ~2x the bf16 path's, as Leap's int8d check (6.6-7.9e-3 vs 2.9-3.9e-3 of max |ref|).
  2. wiring on a toy model through the real int8() setup: prompt rows == the baseline lane bit for bit, answer rows ==
     the int8 path, chunks that straddle the answer start; the census; the per-window checks see no mismatch; a negative
     control (one module skipped in one chunk) is flagged; teardown restores the modules, run_window and the text call.
Usage: ~/BigMoE/.venv/bin/python -I -B test_g1g2_arms.py <path of the dm_bt.py to test>
"""
import importlib.util
import os
import sys
import tempfile

os.environ["DM_OUT"] = tempfile.mkdtemp(prefix="g1g2test_")
DM = os.path.expanduser("~/FableGod/decision_map")
sys.path.insert(0, DM)
import mlx.core as mx                                   # noqa: E402
import mlx.nn as nn                                     # noqa: E402
import numpy as np                                      # noqa: E402

mx.set_default_device(mx.cpu)
import dm_common as C                                   # noqa: E402
import dm_d2                                            # noqa: E402
import dm_ablations                                     # noqa: E402

spec = importlib.util.spec_from_file_location("dm_bt", sys.argv[1])
B = importlib.util.module_from_spec(spec)
sys.modules["dm_bt"] = B
spec.loader.exec_module(B)
fails = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        fails.append(msg)


rng = np.random.default_rng(1116)


def ql(N, K, bits=4):
    lin = nn.Linear(K, N, bias=False)
    lin.weight = mx.array((rng.standard_normal((N, K)) * 0.02).astype(np.float32)).astype(mx.bfloat16)
    return nn.QuantizedLinear.from_linear(lin, 64, bits)


def codes(m):
    """[N, K] 4-bit codes, scales, biases (float64) of a 4-bit g64 QuantizedLinear"""
    W = np.array(m["weight"]).view(np.uint32)
    N, KW = W.shape
    q = ((W[:, :, None] >> (4 * np.arange(8, dtype=np.uint32))) & 0xF).reshape(N, KW * 8).astype(np.float64)
    return q, np.array(m["scales"].astype(mx.float32), np.float64), np.array(m["biases"].astype(mx.float32), np.float64)


def int8_ref(x, m):
    """Leap's int8d math in float64: per (row, 64-group) sx = max|x| / 127 (1 if 0), qx = clip(round(x / sx)),
    y = sum_g sx_g (s_g sum q code + b_g sum q)"""
    xf = np.array(x.astype(mx.float32), np.float64).reshape(-1, x.shape[-1])
    q, s, b = codes(m)
    R, K = xf.shape
    G = K // 64
    xg = xf.reshape(R, G, 64).astype(np.float32)                    # the quantizer in float32, as MLX runs it
    mxa = np.abs(xg).max(-1, keepdims=True)
    sx32 = np.where(mxa > 0, mxa / np.float32(127.0), np.float32(1.0)).astype(np.float32)
    qx = np.clip(np.round(xg / sx32), -127, 127).astype(np.float64)
    sx = sx32.astype(np.float64)
    qg = q.reshape(q.shape[0], G, 64)
    dot = np.einsum("rgk,ngk->rng", qx, qg)                         # sum q code per (row, out, group)
    sq = qx.sum(-1)                                                  # sum q per (row, group)
    y = (sx[:, None, :, 0] * (s[None] * dot + b[None] * sq[:, None, :])).sum(-1)
    return y


# ---------------------------------------------------------------- 1. numerics
for N, K in ((1024, 2560), (2560, 1024), (1280, 2560), (2560, 640)):
    m = ql(N, K)
    e8 = ebf = rel = 0.0
    for seed in range(2):
        X = mx.array(np.random.default_rng(seed).standard_normal((32, K)).astype(np.float32)).astype(mx.bfloat16)
        q, s, b = codes(m)
        wd = (np.repeat(s, 64, axis=1) * q + np.repeat(b, 64, axis=1))
        ref = np.array(X.astype(mx.float32), np.float64) @ wd.T
        y8f = B._qmm32(B._fq64(X), m["weight"], m["scales"], m["biases"], 64, 4)
        y8 = np.array(y8f, np.float64)
        y8b = np.array(y8f.astype(mx.bfloat16).astype(mx.float32), np.float64)     # the wrapper's bf16 output
        # the bf16 path as the GPU runs it (MLX's CPU qmm loses precision in bf16): bf16 x, dequantized weights, float32
        # sums, bf16 output
        yb32 = (np.array(X.astype(mx.float32), np.float32) @ wd.astype(np.float32).T).astype(np.float32)
        yb = np.array(mx.array(yb32).astype(mx.bfloat16).astype(mx.float32), np.float64)
        r8 = int8_ref(X, m)
        rel = max(rel, float(np.abs(y8 - r8).max() / np.abs(r8).max()))
        e8 = max(e8, float(np.abs(y8b - ref).max() / np.abs(ref).max()))
        ebf = max(ebf, float(np.abs(yb - ref).max() / np.abs(ref).max()))
    check(rel <= 1e-5, f"numerics N {N} K {K}: fake-quant == int8d formula (float64), max rel {rel:.1e}")
    check(1.3 <= e8 / ebf <= 3.5 and 4e-3 <= e8 <= 1.2e-2,
          f"error profile N {N} K {K}: int8 {e8:.2e} vs bf16 {ebf:.2e} (x{e8 / ebf:.2f}; Leap: 6.6-7.9e-3 vs 2.9-3.9e-3)")

# ---------------------------------------------------------------- 2. wiring on a toy model through int8()
K, H = 128, 64


class ToyGDN(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj_qkv, self.in_proj_z, self.in_proj_b, self.in_proj_a = ql(192, K), ql(64, K), ql(64, K), ql(64, K)
        self.out_proj = ql(K, 64)

    def __call__(self, x):
        h = self.in_proj_qkv(x)[..., :64] + self.in_proj_z(x) + self.in_proj_b(x) + self.in_proj_a(x)
        return self.out_proj(h)


class ToyAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = ql(64, K), ql(64, K), ql(64, K), ql(K, 64)
        self.indexer = nn.Module()
        self.indexer.index_qk_proj = ql(64, K, bits=8)                # 8-bit: must NOT be wrapped

    def __call__(self, x):
        return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x)) + self.indexer.index_qk_proj(x)[..., :1]


class ToyMoE(nn.Module):
    def __init__(self):
        super().__init__()
        gu = ql(2 * H, K)
        self.shared_expert = C.Q._FusedGateUpMLP(ql(K, H), gu["weight"], gu["scales"], gu["biases"], 64, 4, "affine")
        self.shared_expert_gate = ql(64, K)
        self.gate = ql(32, K, bits=8)                                 # the router: never wrapped

    def __call__(self, x):
        return self.shared_expert(x) * mx.sigmoid(self.shared_expert_gate(x)[..., :1])


class ToyPLE(nn.Module):
    def __init__(self):
        super().__init__()
        self.key_proj, self.value_proj = ql(K, K), ql(K, K)

    def __call__(self, x):
        return self.key_proj(x) + self.value_proj(x)


SKIP = {"on": False}


class ToyText(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = mx.array(np.random.default_rng(7).standard_normal((512, K)).astype(np.float32)).astype(mx.bfloat16)
        self.gdn, self.attn, self.moe, self.ple = ToyGDN(), ToyAttn(), ToyMoE(), ToyPLE()
        self.mtp = nn.Module()
        self.mtp.q_proj = ql(64, K)                                   # MTP: never wrapped

    def __call__(self, inputs, cache=None, input_embeddings=None):
        x = self.emb[inputs]
        y = self.gdn(x) + self.attn(x) + self.moe(x)
        if not SKIP["on"]:
            y = y + self.ple(x)
        return y


ORIG_GDN, ORIG_TEXT, ORIG_LIN = C.Q.GatedDeltaNet, C.Q.Qwen4ExpTextModel, C.Q._linear
C.Q.GatedDeltaNet, C.Q.Qwen4ExpTextModel = ToyGDN, ToyText
C.Q._linear = lambda m, x: m(x)
toy = ToyText()
C.RT["rt"] = toy
before = {k: id(v) for k, v in (("qkv", toy.gdn.in_proj_qkv), ("o", toy.attn.o_proj), ("se", toy.moe.shared_expert),
                                ("kp", toy.ple.key_proj), ("idx", toy.attn.indexer.index_qk_proj), ("mtpq", toy.mtp.q_proj))}
ids1 = np.arange(37, dtype=np.int32) + 3                     # off 0, gen 20 -> a = 21; chunks of 16 straddle it
ids2 = np.arange(29, dtype=np.int32) + 100                   # off 5, gen 9 -> a = 5
WINS = [("code", "w1", ids1, 0, 20), ("code", "w2", ids2, 5, 9)]
C.load_windows = lambda n=12: WINS
CH = {"size": 16}
OUTS = []


def toy_run_window(model, ids):
    outs = []
    for i in range(0, len(ids), CH["size"]):
        outs.append(model(mx.array(ids[None, i:i + CH["size"]])))
    OUTS.append(mx.concatenate(outs, axis=1)[0])
    return None


C.run_window = toy_run_window
REF_BASE = {}
for nm, ids in (("w1", ids1), ("w2", ids2)):                 # the baseline outputs (no wrappers)
    REF_BASE[nm] = np.array(toy(mx.array(ids[None]))[0].astype(mx.float32))
B.I8_WANT = {"gdn": 5, "att": 4}
C.clear()
B.ABLATIONS["bt_int8"]()
cen = {}
for _, m in toy.named_modules():
    if isinstance(m, (B._I8Lin, B._I8FusedGU)):
        cen[type(m).__name__] = cen.get(type(m).__name__, 0) + 1
check(cen == {"_I8Lin": 5 + 4 + 1 + 2, "_I8FusedGU": 1},
      f"census: 5 GDN + 4 attention + shared_expert_gate + 2 PLE wrapped, 1 fused shared expert; 8-bit indexer, router, "
      f"MTP untouched ({cen})")
check(toy.attn.indexer.index_qk_proj is not None and id(toy.attn.indexer.index_qk_proj) == before["idx"]
      and id(toy.mtp.q_proj) == before["mtpq"], "8-bit indexer and MTP projections left as they were")
for (nm, ids), a in zip((("w1", ids1), ("w2", ids2)), (21, 5)):
    C.run_window(toy, ids)
    y = np.array(OUTS[-1].astype(mx.float32))
    same_prompt = np.array_equal(y[:a], REF_BASE[nm][:a])
    diff_answer = not np.array_equal(y[a:], REF_BASE[nm][a:])
    check(same_prompt and diff_answer, f"{nm}: prompt rows 0-{a - 1} == baseline bit for bit, answer rows differ (int8)")
check(not B.I8["bad"], f"per-window checks: no mismatch ({B.I8['bad']})")
# the answer rows equal a manual int8 forward of the toy (the same layers, int8 inputs, float32 sums)
x = toy.emb[mx.array(ids1[None])]


def lin8(m, x):
    return B._qmm32(B._fq64(x), m["weight"], m["scales"], m["biases"], 64, 4).astype(x.dtype)


g = toy.gdn
h = lin8(g.in_proj_qkv.inner, x)[..., :64] + lin8(g.in_proj_z.inner, x) + lin8(g.in_proj_b.inner, x) + lin8(g.in_proj_a.inner, x)
yg = lin8(g.out_proj.inner, h)
at = toy.attn
ya = lin8(at.o_proj.inner, lin8(at.q_proj.inner, x) + lin8(at.k_proj.inner, x) + lin8(at.v_proj.inner, x)) + \
    at.indexer.index_qk_proj(x)[..., :1]
se = toy.moe.shared_expert.inner
gu = B._qmm32(B._fq64(x), se.gu_weight, se.gu_scales, se.gu_biases, 64, 4).astype(x.dtype)
gt, up = mx.split(gu, 2, axis=-1)
ys = lin8(se.down_proj, nn.silu(gt) * up) * mx.sigmoid(lin8(toy.moe.shared_expert_gate.inner, x)[..., :1])
yp = lin8(toy.ple.key_proj.inner, x) + lin8(toy.ple.value_proj.inner, x)
man = np.array((yg + ya + ys + yp)[0].astype(mx.float32))
y1 = np.array(OUTS[0].astype(mx.float32))
check(np.array_equal(y1[21:], man[21:]), "w1 answer rows == a manual all-int8 forward bit for bit")
# negative control: one module skipped in one chunk -> flagged
SKIP["on"] = True
n_bad = len(B.I8["bad"])
C.run_window(toy, ids2)
SKIP["on"] = False
check(len(B.I8["bad"]) == n_bad + 1, "negative control: a skipped module (PLE) in a window is flagged")
C.clear()
after_ok = (id(toy.gdn.in_proj_qkv) == before["qkv"] and id(toy.attn.o_proj) == before["o"]
            and id(toy.moe.shared_expert) == before["se"] and id(toy.ple.key_proj) == before["kp"])
check(after_ok and C.run_window is toy_run_window and ToyText.__call__ is not None
      and "text_call" not in getattr(ToyText.__call__, "__qualname__", ""),
      "teardown: modules, run_window and the text call restored")
rep = os.path.join(os.environ["DM_OUT"], "bt_int8_census.jsonl")
check(os.path.exists(rep), f"census report written ({rep})")
C.Q.GatedDeltaNet, C.Q.Qwen4ExpTextModel, C.Q._linear = ORIG_GDN, ORIG_TEXT, ORIG_LIN
for nm in ("bt_g1h8", "bt_l20g2", "bt_g1", "bt_c6comb", "bt_l20comb", "bt_dense3", "bt_restore"):
    check(dm_ablations.ablation(nm) is not None, f"arm {nm} resolves")
print(f"{'ALL PASS' if not fails else str(len(fails)) + ' FAIL'}", flush=True)
sys.exit(1 if fails else 0)
