"""WEEK98 PILLAR 3, M0 / M1 / M2 (MoonShots, Mon 16 Nov 2026; pre-registered in LEVERS_A_MOONSHOTS.md "PILLAR 3",
c12c192, before any code ran). The t30b-shaped level step: the 125B's MTP head (t30b's architecture) built exactly as
Leap P2 builds it (Qwen4ExpMTP from ~/BigMoE/models/bare/mtp.safetensors, Leap's ScreenedHead, P2's synthetic states;
all imported read-only). Timed three ways:
  M0  one host sync per level. (i) P2's own chain verbatim (as stored; 4-bit + head 8K): must reproduce P2's 1.147 /
      0.864 ms within 15 %. (ii) the tree level (top-K candidates per row, frontier = top-F by path score), F = 1/2/3.
  M1  sync-free: a whole L-level block as one lazy graph (the frontier chosen on the GPU, token ids gathered by index,
      no host sync until the block ends): 'free' = one eval at the end; 'async' = async_eval per level, one sync at the
      end. ms per level -> the fixed cost f = ms - MB / BW.
  M2  the byte-cut ladder (experts top-10 -> 4 / 2, frontier-shared routing for F > 1, q / o and the bf16 Linears at 3 /
      2 bits, head 8K -> 4K / 2K), sync-free; fit ms per level = f + MB / BW.
Weights' values don't matter for time; kept (acceptance) is M3's (TRUE-STATE, dd_tree.py). Timings under a share lock
beside other GPU jobs are INDICATIVE (arms interleaved, order alternating, medians; the GPU's busy % is logged per
round). Self-checks: sync / free / async give the same tokens (same graph); the shared-route MoE equals the stock MoE
at one row.
usage: ~/BigMoE/.venv/bin/python -I -B p3_m012.py [--smoke] [--rounds 30] [--L 21] (MEM ~3.5 GB; share lock)"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np                                          # noqa: E402
import mlx.core as mx                                       # noqa: E402
import mlx.nn as nn                                         # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten          # noqa: E402

H = os.path.expanduser
sys.path.insert(0, H("~/Leap/scripts"))                     # screened_head.py (read-only import)
from mtplx.models import qwen4_exp as Q                     # noqa: E402
from screened_head import ScreenedHead                      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--set", choices=("stock", "fused"), default="stock",
                help="stock: C0-C8 (P2's two points + the cut ladder); fused: E1-E3 (the engine's fusions), a separate "
                     "process so each stays <= 4 GB (the fused gate|up bank replaces the stock one)")
ap.add_argument("--rounds", type=int, default=30)
ap.add_argument("--L", type=int, default=21)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--out", default=None)
A = ap.parse_args()
A.out = A.out or H(f"~/MoonShots/week98/p3_m012_{A.set}.json")
t00 = time.time()


def say(s):
    print(s, flush=True)


def gpu_busy():
    try:
        o = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"], capture_output=True,
                           text=True, timeout=5).stdout
        m = re.search(r'"Device Utilization %"=(\d+)', o)
        return int(m.group(1)) if m else -1
    except Exception:
        return -1


# ---- the MTP head exactly as Leap P2 builds it (p2_chainbytes.py, read-only copy of its build) ----
D = H("~/BigMoE/models/bare")
cfg = json.load(open(f"{D}/config.json"))
args = Q.TextArgs.from_dict(cfg.get("text_config", cfg))
raw = mx.load(f"{D}/mtp.safetensors")
ST0 = {k[len("mtp."):]: v for k, v in raw.items() if k.startswith("mtp.")}
del raw                                                     # ST0 holds the only references (the fused set frees some)
for n in ("pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight"):
    if n in ST0 and ST0[n].ndim == 1:
        ST0[n] = (ST0[n].astype(mx.float32) + 1.0).astype(ST0[n].dtype)


class _Fake:
    pass


FST, SHAPES = None, {}


def build(requant=None, fused=False, hc_bf16=False):
    """P2's build: the MTP head as stored; requant = (bits, group) re-quantizes every remaining bf16 Linear (>= 64K).
    fused: the engine's sanitize-time fusions (MTPLX_FUSED_GATE_UP: routed + shared gate|up; MTPLX_FUSED_QSA_QKV: q|k|v
    (+ indexer qk) as one GEMV), run on the MTP's tensors under a 'mtp.' prefix. hc_bf16: leave the hyper-connection
    mixes in bf16 (the fused-HC kernel reads bf16 only)."""
    global FST
    mtp = Q.Qwen4ExpMTP(args)
    st = ST0
    if fused:
        old = {k: os.environ.get(k) for k in ("MTPLX_FUSED_GATE_UP", "MTPLX_FUSED_QSA_QKV")}
        os.environ.update(MTPLX_FUSED_GATE_UP="1", MTPLX_FUSED_QSA_QKV="1")
        try:
            fake = _Fake()
            fake.layers = mtp.layers
            if FST is None:                                  # first fused build: fuse the real packs once
                pref = {f"mtp.{k}": v for k, v in ST0.items()}
                SHAPES.update({k: (v.shape, v.dtype) for k, v in pref.items()})
                pref = Q._fuse_gate_up_sanitize(fake, pref)
                pref = Q._fuse_qsa_qkv_sanitize(fake, pref)
                FST = {k[len("mtp."):]: v for k, v in pref.items()}
            else:                                            # later builds: module swaps only (lazy placeholders)
                ph = {k: mx.zeros(shp, dtype=dt) for k, (shp, dt) in SHAPES.items()}
                Q._fuse_gate_up_sanitize(fake, ph)
                Q._fuse_qsa_qkv_sanitize(fake, ph)
            st = FST
        finally:
            for k, x in old.items():
                if x is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = x
    module_in = {p[:-len(".weight")]: int(a.shape[-1]) for p, a in tree_flatten(mtp.parameters())
                 if p.endswith(".weight") and a.ndim >= 2}
    qmap = {}
    mods = dict(mtp.named_modules())
    for mod, in_f in module_in.items():
        if not hasattr(mods.get(mod), "to_quantized"):       # the fused modules hold their packs already
            continue
        w, s = st.get(f"{mod}.weight"), st.get(f"{mod}.scales")
        if w is None or s is None or w.dtype != mx.uint32:
            continue
        qmap[mod] = {"bits": int(w.shape[-1]) * 32 // in_f, "group_size": in_f // int(s.shape[-1])}
    nn.quantize(mtp, group_size=64, bits=8, class_predicate=lambda p, m: qmap.get(p, False))
    mtp.load_weights(list(st.items()), strict=True)
    if requant:
        b_, g_ = requant
        nn.quantize(mtp, group_size=g_, bits=b_,
                    class_predicate=lambda p, m: (isinstance(m, nn.Linear) and not isinstance(m, nn.QuantizedLinear)
                                                  and m.weight.size >= 65536 and m.weight.shape[-1] % g_ == 0
                                                  and not (hc_bf16 and "input_mix_weight" in p)))
    mtp.eval()
    mx.eval(mtp.parameters())
    if fused:                                                # free the stock packs the fused ones replace
        for k in [k for k in ST0 if k not in FST]:
            del ST0[k]
    return mtp


def requant_quantized(mtp, pred, bits, gs=64):
    """re-quantize the already-quantized Linears whose path matches pred (dequantize, then quantize at `bits`)"""
    new = []
    for path, m in mtp.named_modules():
        if type(m) is nn.QuantizedLinear and pred(path) and m.bits != bits:
            w = mx.dequantize(m["weight"], m["scales"], m.get("biases"), group_size=m.group_size, bits=m.bits,
                              mode=getattr(m, "mode", "affine"))
            lin = nn.Linear(int(w.shape[1]), int(w.shape[0]), bias="bias" in m)
            lin.weight = w.astype(mx.bfloat16)
            if "bias" in m:
                lin.bias = m["bias"]
            new.append((path, nn.QuantizedLinear.from_linear(lin, gs, bits)))
    if new:
        mtp.update_modules(tree_unflatten(new))
    mx.eval(mtp.parameters())
    return sorted(p for p, _ in new)


wmap = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]


def get(k):
    return mx.load(f"{D}/{wmap[k]}")[k]


V = args.vocab_size
emb = nn.QuantizedEmbedding(V, args.hidden_size, group_size=64, bits=4)
head = nn.QuantizedLinear(args.hidden_size, V, bias=False, group_size=64, bits=4)
for m_, p_ in ((emb, "language_model.model.embed_tokens"), (head, "language_model.lm_head")):
    m_.weight, m_.scales, m_.biases = get(p_ + ".weight"), get(p_ + ".scales"), get(p_ + ".biases")
mx.eval(emb.parameters(), head.parameters())
RANK = np.load(H("~/Leap/data/frspec_rank.npy"))
HCW = args.hc_count * args.hidden_size
TOPK = int(getattr(args, "num_experts_per_tok", 10))
NE = int(getattr(args, "num_experts", 512))


class SharedRoute(nn.Module):
    """frontier-shared routing: the level's F rows use ONE expert set (top-k of the rows' summed gate probabilities);
    each row weighs that set by its own gates (renormalized as the block does). One batched GEMM per expert, so the
    experts' bytes don't grow with F. The stock block's math otherwise (shared expert, sigmoid gate)."""

    def __init__(self, blk):
        super().__init__()
        object.__setattr__(self, "_blk", blk)               # not a child: no double-counted parameters

    def __call__(self, x):
        b = self._blk
        sw, k = b.switch_mlp, b.top_k
        g2 = mx.softmax(b.gate(x), axis=-1, precise=True).reshape(-1, NE)     # [S, NE]
        idx = mx.argpartition(g2.sum(axis=0), kth=NE - k)[NE - k:]             # [k]
        w = mx.take(g2, idx, axis=1)                                           # [S, k]
        if b.norm_topk_prob:
            w = w / w.sum(axis=-1, keepdims=True)
        x2 = x.reshape(-1, x.shape[-1])
        xb = mx.broadcast_to(x2[None], (k,) + tuple(x2.shape))                # [k, S, D]

        def gq(lin, a):
            return mx.gather_qmm(a, lin["weight"], lin["scales"], lin.get("biases"), rhs_indices=idx, transpose=True,
                                 group_size=lin.group_size, bits=lin.bits, mode=getattr(lin, "mode", "affine"))
        y = gq(sw.down_proj, sw.activation(gq(sw.up_proj, xb), gq(sw.gate_proj, xb)))   # [k, S, D]
        y = (y * w.T[:, :, None]).sum(axis=0).reshape(x.shape)
        sy = mx.sigmoid(b.shared_expert_gate(x)) * b.shared_expert(x)
        return y + sy


class Compiled(nn.Module):
    """mx.compile of a pure sub-block (its weights captured as constants), with the wrapped module's call signature
    (x) or (x, pending=(block_out, inject)). Holds the module outside the Module tree (no parameters)."""

    def __init__(self, mod):
        super().__init__()
        object.__setattr__(self, "_m", mod)
        object.__setattr__(self, "_f1", mx.compile(lambda x: mod(x)))
        object.__setattr__(self, "_f2", mx.compile(lambda x, b, i: mod(x, pending=(b, i))))

    def __call__(self, x, pending=None):
        return self._f1(x) if pending is None else self._f2(x, *pending)


class Var:
    pass


FUSED_ENV = {"MTPLX_FUSED_MOE_DECODE": "1", "MTPLX_FUSED_MOE_VERIFY": "1", "MTPLX_FUSED_HC": "1"}


class envflags:
    def __init__(self, on):
        self.on = on

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in FUSED_ENV}
        if self.on:
            os.environ.update(FUSED_ENV)

    def __exit__(self, *a):
        for k, x in self.old.items():
            if x is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = x


def make(name, requant=(4, 64), qo=None, top=TOPK, nh=8192, low=None, fused=False, hc_bf16=False, compiled=False):
    v = Var()
    v.name, v.top, v.nh, v.fused, v.compiled = name, top, nh, fused, compiled
    v.tails = {}
    v.mtp = build(requant, fused, hc_bf16)
    v.cut = []
    if qo:
        v.cut += requant_quantized(v.mtp, lambda p: p.endswith("self_attn.q_proj") or p.endswith("self_attn.o_proj"), qo)
    if low:
        v.cut += requant_quantized(v.mtp, lambda p: (p.endswith("self_attn.k_proj") or p.endswith("self_attn.v_proj")
                                                     or ".shared_expert." in p), low)
    v.mtp.layers[0].mlp.top_k = top
    v.stock_mlp = v.mtp.layers[0].mlp
    v.shared_mlp = None if fused else SharedRoute(v.stock_mlp)
    v.sh = ScreenedHead(head, RANK[:nh], extra_cap=4096)
    v.cache = [Q.QSACache(args.indexer_compress_ratio or 4)]
    rng = np.random.default_rng(5)
    Wh = mx.array((rng.standard_normal((1, 400, HCW)) * 0.6).astype(np.float32)).astype(mx.bfloat16)
    ids0 = mx.array(rng.integers(0, 200000, (1, 400)).astype(np.int32))
    with envflags(fused):
        mx.eval(v.mtp.fuse_and_run(Wh, emb(ids0), v.cache))
    dense, ex_row, parts = 0.0, 0.0, {}
    for p, a in tree_flatten(v.mtp.parameters()):
        if a.ndim >= 3 and a.shape[0] == NE:
            ex_row += a.nbytes / NE
            key = "routed experts (per expert)"
        else:
            dense += a.nbytes
            key = p.split(".")[0] if not p.startswith("layers.") else ".".join(p.split(".")[:4])
        parts[key] = parts.get(key, 0.0) + (a.nbytes / NE if key.startswith("routed") else a.nbytes)
    v.dense, v.ex_row = dense / 1e6, ex_row / 1e6
    v.head_mb = sum(x.nbytes for x in (v.sh.w, v.sh.s, v.sh.b)) / 1e6
    v.parts = {k: round(x / 1e6, 2) for k, x in sorted(parts.items(), key=lambda t: -t[1])}
    if compiled:                                             # the pure sub-blocks compiled; attention + cache stay eager
        L0 = v.mtp.layers[0]
        L0.attn_hyper_connection = Compiled(L0.attn_hyper_connection)
        L0.mlp_hyper_connection = Compiled(L0.mlp_hyper_connection)
        v.stock_mlp = Compiled(v.stock_mlp)
        v.shared_mlp = Compiled(v.shared_mlp) if v.shared_mlp is not None else None
        L0.mlp = v.stock_mlp
        with envflags(fused):
            v.mtp.install_compiled_prepare()
    return v


def level_mb(v, F, shared=False):
    """weight bytes one level reads: dense + head once; routed experts: top per row, F x top (upper bound) for F rows
    with their own routing, top with frontier-shared routing"""
    nexp = v.top if (F == 1 or shared) else min(NE, F * v.top)
    return v.dense + v.head_mb + v.ex_row * nexp


mx.random.seed(0)
h0 = (mx.random.normal((1, 1, HCW)) * 0.6).astype(mx.bfloat16)          # P2's synthetic state
mx.eval(h0)


def p2_chain(v, tok0, n=32):
    """Leap P2's chain() verbatim: one sync per step, argmax pick (M0 (i))"""
    mtp, sh, cache = v.mtp, v.sh, v.cache
    h, tok, toks, ts = h0, tok0, [], []
    with envflags(v.fused):
        return _p2_chain(mtp, sh, cache, h, tok, toks, ts, n)


def _p2_chain(mtp, sh, cache, h, tok, toks, ts, n):
    for d in range(n):
        t0 = time.perf_counter()
        hh = mtp.fuse_and_run(h, emb(mx.array([[tok]])), cache)
        lg = sh.logits(mtp.hyper_connection_mixer(hh))[0]
        j = mx.argmax(lg)
        tk = mx.take(sh.ids, j)
        mx.eval(tk)
        ts.append(time.perf_counter() - t0)
        h = hh
        toks.append(int(tk.item()))
        tok = toks[-1]
    cache[0].trim(n)
    return ts, toks


def level(v, h, tok, sc, F, K):
    """one tree level: F frontier rows -> top-K candidates per row -> the next frontier = top-F by path log-score.
    All on the GPU (no host read)."""
    hh = v.mtp.fuse_and_run(h, emb(tok), v.cache)                           # [1, F, HCW]
    if v.compiled:
        if (F, K) not in v.tails:
            v.tails[(F, K)] = mx.compile(lambda a, b: _tail(v, a, b, F, K))
        o = v.tails[(F, K)](hh, sc)
    else:
        o = _tail(v, hh, sc, F, K)
    return o[0], o[1], o[2], tuple(o[3:])


def _tail(v, hh, sc, F, K):
    lg = v.sh.logits(v.mtp.hyper_connection_mixer(hh))                      # [F, nh] fp32
    lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
    nh = lg.shape[-1]
    kidx = mx.argpartition(lp, kth=nh - K, axis=-1)[:, nh - K:]             # [F, K] (unsorted)
    klp = mx.take_along_axis(lp, kidx, axis=-1)
    cand = (sc[:, None] + klp).reshape(-1)                                  # [F * K]
    sel = mx.argpartition(cand, kth=F * K - F)[F * K - F:]                  # [F]
    par = sel // K
    tk = mx.take(v.sh.ids, mx.take(kidx.reshape(-1), sel))                  # [F] token ids
    return mx.take(hh, par, axis=1), tk[None, :], mx.take(cand, sel), kidx, klp, par


def run_block(v, L, F, K, tok0, mode, shared=False):
    """mode 'sync': one host sync per level (the host reads the frontier tokens and re-uploads them, as a host-driven
    tree does); 'free': one eval per block; 'async': async_eval per level, one sync at the end"""
    v.mtp.layers[0].mlp = v.shared_mlp if (shared and F > 1) else v.stock_mlp
    fl = envflags(v.fused)
    fl.__enter__()
    try:
        t0 = time.perf_counter()
        h = mx.broadcast_to(h0, (1, F, HCW)) if F > 1 else h0
        tok = mx.array([[tok0 + i for i in range(F)]], dtype=mx.int32)
        sc = mx.zeros((F,), mx.float32)
        outs, lv = [], []
        for _ in range(L):
            tl = time.perf_counter()
            h, tok, sc, rec = level(v, h, tok, sc, F, K)
            outs.append((tok,) + rec)
            if mode == "sync":
                mx.eval(h, tok, sc, *rec)
                tok = mx.array([tok.tolist()[0]], dtype=mx.int32)
                lv.append(time.perf_counter() - tl)
            elif mode == "async":
                mx.async_eval(h, tok, sc, *rec)
        tb = time.perf_counter()
        mx.eval(outs, h, sc)
        t1 = time.perf_counter()
    finally:
        v.mtp.layers[0].mlp = v.stock_mlp
        fl.__exit__()
    v.cache[0].trim(L * F)
    return dict(total=t1 - t0, build=tb - t0, levels=lv, toks=[o[0].tolist()[0] for o in outs])


# ---- the variants: P2's two reference points, then the cut ladder from P2's best (4-bit + head 8K) ----
LADDER = [
    dict(name="C0 as stored (P2 base)", requant=None, nh=32768),
    dict(name="C1 4-bit + head 8K (P2 best)"),
    dict(name="C2 C1 + experts top-4", top=4),
    dict(name="C3 C2 + head 4K", top=4, nh=4096),
    dict(name="C4 C3 + q/o 3-bit", top=4, nh=4096, qo=3),
    dict(name="C5 C4 + experts top-2", top=2, nh=4096, qo=3),
    dict(name="C6 C5 with q/o 2-bit", top=2, nh=4096, qo=2),
    dict(name="C7 C6 + bf16 Linears 2-bit", requant=(2, 64), top=2, nh=4096, qo=2),
    dict(name="C8 C7 + head 2K + k/v/shared 2-bit", requant=(2, 64), top=2, nh=2048, qo=2, low=2),
    dict(name="K1 C1 compiled (hc reads, MoE, tail, prepare)", compiled=True),
    dict(name="K3 C3 compiled", top=4, nh=4096, compiled=True),
    dict(name="K8 C8 compiled", requant=(2, 64), top=2, nh=2048, qo=2, low=2, compiled=True),
    dict(name="E1 C1 + engine fusions (gate|up, MoE kernel, QSA qkv)", fused=True),
    dict(name="E2 E1 + bf16 hc mixes + fused HC", fused=True, hc_bf16=True),
    dict(name="E3 C3 + engine fusions", top=4, nh=4096, fused=True),
    dict(name="EK1 E1 compiled", fused=True, compiled=True),
]
LADDER = [x for x in LADDER if (x["name"][0] == "E") == (A.set == "fused")]
if A.smoke:
    LADDER = [LADDER[0], LADDER[-1]] if A.set == "fused" else [LADDER[1], LADDER[8], LADDER[9]]
VS = []
for spec in LADDER:
    v = make(**spec)
    VS.append(v)
    say(f"built {v.name}: {level_mb(v, 1):.1f} MB per level at F=1 (dense {v.dense:.1f}, head {v.head_mb:.1f}, "
        f"experts {v.top} x {v.ex_row:.2f}); F=3: {level_mb(v, 3):.1f} own routing / {level_mb(v, 3, True):.1f} shared"
        + (f"; re-quantized {len(v.cut)} Linears" if v.cut else ""))
    if A.smoke or spec is LADDER[1]:
        say("  parts (MB): " + ", ".join(f"{k} {x}" for k, x in list(v.parts.items())[:14]))
say(f"all built at {time.time() - t00:.0f} s; active {mx.get_active_memory() / 1e9:.2f} GB; GPU busy {gpu_busy()} %")

# ---- self-checks ----
chk = {}
v1 = VS[0] if (A.smoke or A.set == "fused") else VS[1]
for F in (1, 3):
    for sh_ in ((False, True) if (F > 1 and v1.shared_mlp is not None) else (False,)):
        T = {m: run_block(v1, A.L, F, A.K, 1000, m, sh_)["toks"] for m in ("sync", "free", "async")}
        chk[f"F{F}{' shared' if sh_ else ''} sync == free == async"] = (T["sync"] == T["free"] == T["async"])
_, p2t = p2_chain(v1, 1000, A.L)
chk["F1 tree level == P2 chain (argmax)"] = ([t[0] for t in run_block(v1, A.L, 1, A.K, 1000, "free")["toks"]] == p2t)
TWIN = {"K1": "C1", "K3": "C3", "K8": "C8", "EK": "E1"}
byname = {v.name[:2]: v for v in VS}
for v in VS:
    t = TWIN.get(v.name[:2])
    if t and t in byname:
        for F in (1, 3):
            a = run_block(v, A.L, F, A.K, 1000, "free")["toks"]
            b = run_block(byname[t], A.L, F, A.K, 1000, "free")["toks"]
            chk[f"{v.name[:2]} F{F} tokens == {t}'s (share)"] = round(float(np.mean([x == y for x, y in zip(a, b)])), 3)
REFTOK = {f"{v.name[:2]} F{F}": run_block(v, A.L, F, A.K, 1000, "free")["toks"] for v in VS for F in (1, 3)}
x1 = (mx.random.normal((1, 1, args.hidden_size)) * 0.5).astype(mx.bfloat16)
if v1.shared_mlp is not None:
    a_, b_ = v1.stock_mlp(x1), v1.shared_mlp(x1)
    chk["shared-route MoE == stock MoE at one row (max abs diff)"] = float(
        mx.abs(a_.astype(mx.float32) - b_.astype(mx.float32)).max())
say("self-checks: " + json.dumps(chk))
# graph size: primitives in one level's lazy graph (the dispatch count before any fusion)
try:
    with envflags(v1.fused):
        hs, ts_, ss = level(v1, h0, mx.array([[7]], dtype=mx.int32), mx.zeros((1,), mx.float32), 1, A.K)[:3]
    dot = H(f"~/MoonShots/week98/p3_level_graph_{A.set}.dot")
    with envflags(v1.fused):
        mx.export_to_dot(dot, hs, ts_, ss)
    VIEW = {"Reshape", "ExpandDims", "Squeeze", "Broadcast", "Flatten", "Unflatten", "Depends", "Split"}
    ops = [m.group(1) for m in (re.search(r'label ="([A-Za-z]+)", shape=rectangle', ln) for ln in open(dot)) if m]
    chk["level graph primitives (F=1, dot export)"] = len(ops)
    chk["level graph primitives that can dispatch (not views)"] = sum(1 for o in ops if o not in VIEW)
    v1.cache[0].trim(1)
    say(f"level graph: {len(ops)} primitives, {chk['level graph primitives that can dispatch (not views)']} not views "
        f"(F=1; {dot})")
except Exception as e:                                      # diagnostic only
    say(f"level graph count skipped: {e!r}")

# ---- timing: arms interleaved, order alternating per round; the first 2 rounds are warm-up ----
ARMS = []
for vi, v in enumerate(VS):
    if v.name[:3] in ("C0 ", "C1 ", "E1 "):
        ARMS.append(("p2", vi, 1, "sync", False))
    for F in (1, 2, 3):
        for mode in ("sync", "free", "async"):
            ARMS.append(("tree", vi, F, mode, False))
        if F > 1 and not v.fused:
            for mode in ("free", "async"):
                ARMS.append(("tree", vi, F, mode, True))
R = 3 if A.smoke else A.rounds
TT = {i: [] for i in range(len(ARMS))}
BUSY = []
for r in range(R):
    BUSY.append(gpu_busy())
    order = list(range(len(ARMS))) if r % 2 == 0 else list(range(len(ARMS)))[::-1]
    for i in order:
        kind, vi, F, mode, sh_ = ARMS[i]
        v = VS[vi]
        if kind == "p2":
            ts, _ = p2_chain(v, 1000 + r)
            if r >= 2:
                TT[i] += [1e3 * t for t in ts[2:]]
        else:
            o = run_block(v, A.L, F, A.K, 1000 + r, mode, sh_)
            if r >= 2:
                if mode == "sync":
                    TT[i] += [1e3 * t for t in o["levels"][2:]]
                else:
                    TT[i].append(1e3 * o["total"] / A.L)
                    TT.setdefault(("build", i), []).append(1e3 * o["build"] / A.L)
    if r % 5 == 4:
        say(f"round {r + 1}/{R} done at {time.time() - t00:.0f} s (GPU busy at round start {BUSY[-1]} %)")

res = []
for i, (kind, vi, F, mode, sh_) in enumerate(ARMS):
    v = VS[vi]
    x = np.array(TT[i])
    d = dict(variant=v.name, kind=kind, F=F, mode=mode, shared=sh_, mb=round(level_mb(v, F, sh_), 2),
             ms=round(float(np.median(x)), 4), p10=round(float(np.percentile(x, 10)), 4),
             p90=round(float(np.percentile(x, 90)), 4), n=int(len(x)))
    if ("build", i) in TT:
        d["build_ms_per_level"] = round(float(np.median(TT[("build", i)])), 4)
    res.append(d)

say("")
say("M0 (i): P2's chain, one sync per step (P2's published: as stored 1.147, 4-bit + head 8K 0.864 ms)")
for d in res:
    if d["kind"] == "p2":
        pub = 1.147 if d["variant"].startswith("C0") else 0.864
        d["vs_p2"] = round(d["ms"] / pub, 3)
        say(f"  {d['variant']}: {d['ms']:.3f} ms [{d['p10']:.3f}-{d['p90']:.3f}] at {d['mb']:.1f} MB -> x{d['vs_p2']:.3f} "
            f"of P2 ({'within' if abs(d['vs_p2'] - 1) <= 0.15 else 'OUTSIDE'} 15 %)")
say("")
say("ms per level (median [p10-p90]); MB per level; build = CPU graph-building ms per level (free / async)")
say("variant | F | routing | MB | sync | free | async | build (free)")
for vi, v in enumerate(VS):
    for F in (1, 2, 3):
        for sh_ in ((False, True) if F > 1 else (False,)):
            row = {d["mode"]: d for d in res if d["kind"] == "tree" and d["variant"] == v.name and d["F"] == F
                   and d["shared"] == sh_}
            if not row:
                continue
            cell = lambda m: f"{row[m]['ms']:.3f} [{row[m]['p10']:.3f}-{row[m]['p90']:.3f}]" if m in row else "-"
            mb = next(iter(row.values()))["mb"]
            say(f"{v.name} | {F} | {'shared' if sh_ else 'own'} | {mb:.1f} | {cell('sync')} | {cell('free')} | "
                f"{cell('async')} | {row['free'].get('build_ms_per_level', float('nan')):.3f}")

fits = {}
FAM = {"C": "eager ladder C0-C8", "K": "compiled sub-blocks K1/K3/K8", "E": "engine fusions E1/E3"}
for fam, lab in FAM.items():
    for mode in ("sync", "free", "async"):
        pts = [(d["mb"], d["ms"]) for d in res if d["kind"] == "tree" and d["F"] == 1 and d["mode"] == mode
               and d["variant"][0] == fam and not d["variant"].startswith(("E2", "EK"))]
        if len(pts) >= 2:
            mb, ms = np.array(pts).T
            slope, f0 = np.polyfit(mb, ms, 1)
            fits[f"{lab} | {mode}"] = dict(f_ms=round(float(f0), 4),
                                           bw_gbs=round(float(1.0 / slope), 1) if slope > 0 else None,
                                           resid_ms=round(float(np.sqrt(np.mean((ms - (f0 + slope * mb)) ** 2))), 4),
                                           n=len(pts))
say("")
say("M2 fits at F=1, ms per level = f + MB / BW (BW in GB/s):")
for k, x in fits.items():
    say(f"  {k}: {json.dumps(x)}")
LEVELS = {"HE+ tree-32": 20.8, "MBPP tree-32": 14.6, "HE+ tree-16": 12.6, "MBPP tree-16": 10.3}
say("")
say("per-block ms = levels x ms per level (best sync-free mode per arm); PASS needs <= 4.0 at HE+ tree-32")
proj = []
for vi, v in enumerate(VS):
    for F in (1, 2, 3):
        for sh_ in ((False, True) if F > 1 else (False,)):
            cand = [d for d in res if d["kind"] == "tree" and d["variant"] == v.name and d["F"] == F
                    and d["shared"] == sh_ and d["mode"] in ("free", "async")]
            if not cand:
                continue
            b = min(cand, key=lambda d: d["ms"])
            row = dict(variant=v.name, F=F, shared=sh_, mode=b["mode"], ms_level=b["ms"], mb=b["mb"],
                       **{k: round(L * b["ms"], 2) for k, L in LEVELS.items()})
            proj.append(row)
            say(f"{v.name} | F={F} {'shared' if sh_ else 'own'} | {b['mode']} {b['ms']:.3f} ms/level | "
                + " | ".join(f"{k} {row[k]:.1f}" for k in LEVELS))
out = dict(when=time.strftime("%Y-%m-%d %H:%M"), set=A.set, L=A.L, K=A.K, rounds=R, smoke=A.smoke, selfchecks=chk,
           reference_tokens=REFTOK,
           gpu_busy_per_round=BUSY, variants=[dict(name=v.name, mb_f1=round(level_mb(v, 1), 2), dense=round(v.dense, 2),
                                                   head=round(v.head_mb, 2), expert_mb=round(v.ex_row, 3), top=v.top,
                                                   nh=v.nh, recut=v.cut, parts=v.parts) for v in VS],
           arms=res, fits=fits, per_block=proj,
           labels="TIMING, INDICATIVE (share lock; other GPU jobs running); synthetic states; no kept here (M3)")
if not A.smoke:
    json.dump(out, open(A.out, "w"), indent=1)
say(f"done in {time.time() - t00:.0f} s; peak {mx.get_peak_memory() / 1e9:.2f} GB; GPU busy per round {BUSY}")
