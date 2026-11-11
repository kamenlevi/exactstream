"""MTP-STEP-2 (Leap, Wed 2026-11-04 16:3x; TABLE_350 'MTP-STEP-2'): where the chat switch's MTP step (R57_CHATSW: mtp_forward at 1
row + the pick) spends its ~1.28 ms of layer + mixer (MTPSCREEN), standalone: the 125B's MTP head built as
qwen4_exp.Model.attach_mtp builds it (Qwen4ExpMTP from mtp.safetensors, per-module quant inferred, pre_fc norm +1 shift),
the real 4-bit embed, the screened head (FR-Spec 32K). No trunk, no engine (~2.5 GB).
(1) the step as the chain runs it (mtp_forward's body + argmax / softmax max + one sync) on a 400-position QSA cache;
(2) each piece alone at 1 row, 20 independent calls per eval, median of 9 rounds: fc prep, attn HC read, QSA attention
    (+ cache trim), mlp HC read (with the pending write), the 512-expert MoE block, the residual write, the mixer, the
    screened head + argmax / softmax.
usage: [BigMoE venv] mtp_step_profile.py -> ~/Leap/data/mtp_step_profile.txt (GPU, MEM ~2.5 GB; share lock)"""
import os, sys, json, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_fu/Leap_scripts"))
from mtplx.models import qwen4_exp as Q
from mlx.utils import tree_flatten
from screened_head import ScreenedHead

OUT = os.path.expanduser("~/Leap/data/mtp_step_profile.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
t00 = time.time()
D = os.path.expanduser("~/BigMoE/models/bare")
cfg = json.load(open(f"{D}/config.json"))
args = Q.TextArgs.from_dict(cfg.get("text_config", cfg))
mtp = Q.Qwen4ExpMTP(args)
raw = mx.load(f"{D}/mtp.safetensors")
st = {k[len("mtp."):]: v for k, v in raw.items() if k.startswith("mtp.")}
for n in ("pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight"):
    if n in st and st[n].ndim == 1:
        st[n] = (st[n].astype(mx.float32) + 1.0).astype(st[n].dtype)
module_in = {p[:-len(".weight")]: int(a.shape[-1]) for p, a in tree_flatten(mtp.parameters()) if p.endswith(".weight") and a.ndim >= 2}
qmap = {}
for mod, in_f in module_in.items():
    w, s = st.get(f"{mod}.weight"), st.get(f"{mod}.scales")
    if w is None or s is None or w.dtype != mx.uint32:
        continue
    qmap[mod] = {"bits": int(w.shape[-1]) * 32 // in_f, "group_size": in_f // int(s.shape[-1])}
nn.quantize(mtp, group_size=64, bits=8, class_predicate=lambda p, m: qmap.get(p, False))
mtp.load_weights(list(st.items()), strict=True)
mtp.eval()
mx.eval(mtp.parameters())
wmap = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
def get(k):
    return mx.load(f"{D}/{wmap[k]}")[k]
V = args.vocab_size
emb = nn.QuantizedEmbedding(V, args.hidden_size, group_size=64, bits=4)
head = nn.QuantizedLinear(args.hidden_size, V, bias=False, group_size=64, bits=4)
for m, p in ((emb, "language_model.model.embed_tokens"), (head, "language_model.lm_head")):
    m.weight, m.scales, m.biases = get(p + ".weight"), get(p + ".scales"), get(p + ".biases")
mx.eval(emb.parameters(), head.parameters())
sh = ScreenedHead(head, np.load(os.path.expanduser("~/Leap/data/frspec_rank.npy"))[:32768], extra_cap=4096)
qbits = sorted({(v["bits"], v["group_size"]) for v in qmap.values()})
say(f"\nMTP-STEP-2 {time.strftime('%a %H:%M:%S')}: MLX {mx.__version__}; MTP head built standalone ({len(qmap)} quantized modules "
    f"{qbits}), {sum(a.nbytes for _, a in tree_flatten(mtp.parameters())) / 1e9:.2f} GB; loaded in {time.time() - t00:.0f} s")

HCW = args.hc_count * args.hidden_size
rng = np.random.default_rng(5)
cache = [Q.QSACache(args.indexer_compress_ratio or 4)]
NH = 400
Wh = mx.array((rng.standard_normal((1, NH, HCW)) * 0.6).astype(np.float32)).astype(mx.bfloat16)
ids = mx.array(rng.integers(0, 200000, (1, NH)).astype(np.int32))
mx.eval(mtp.fuse_and_run(Wh, emb(ids), cache))
if os.environ.get("MTP_REF2") == "1":                   # MTP-STEP-4: REF2's kernels during the step
    import ref2
    ref2.apply()
    say(f"REF2 applied: {ref2.CFG.get('experts')=}")
mx.random.seed(0)
h0 = (mx.random.normal((1, 1, HCW)) * 0.6).astype(mx.bfloat16)
mx.eval(h0)
layer = mtp.layers[0]
allids = sh.ids

def step(h, tok):
    hh = mtp.fuse_and_run(h, emb(mx.array([[tok]])), cache)
    lg = sh.logits(mtp.hyper_connection_mixer(hh))[0]
    j = mx.argmax(lg)
    p = mx.exp(lg[j] - mx.logsumexp(lg))
    return hh, mx.take(allids, j), p

# (1) chained steps as the chain runs them (one sync per step), cache trimmed after each chain
ts = []
for chain in range(12):
    h, tok, n = h0, 1000 + chain, 0
    for d in range(8):
        t0 = time.perf_counter()
        h, tk, p = step(h, tok)
        mx.eval(tk, p)
        ts.append(time.perf_counter() - t0)
        tok = int(tk.item()); n += 1
    cache[0].trim(n)
say(f"(1) MTP step (fuse_and_run + mixer + screened head + argmax / p + one sync), 12 chains x 8 depths: median "
    f"{1e3 * np.median(ts[16:]):.3f} ms (p10 {1e3 * np.percentile(ts[16:], 10):.3f}, p90 {1e3 * np.percentile(ts[16:], 90):.3f})")

# MTP-STEP-5: the chain in chunks on the GPU (the token stays an array: embedding of take(allids, argmax)); one eval per
# chunk vs one per step. Same ops per step, so the same tokens (checked), only fewer host round trips.
if os.environ.get("CHUNK_TEST") == "1":
    def step_gpu(h, tk_arr):
        hh = mtp.fuse_and_run(h, emb(tk_arr.reshape(1, 1)), cache)
        lg = sh.logits(mtp.hyper_connection_mixer(hh))[0]
        j = mx.argmax(lg)
        return hh, mx.take(allids, j), mx.exp(lg[j] - mx.logsumexp(lg))
    def run_chain(chunk, depth=8, tok0=777):
        h, tk, toks, ps = h0, mx.array(tok0, mx.int32), [], []
        t0 = time.perf_counter()
        for d0 in range(0, depth, chunk):
            for d in range(chunk):
                h, tk, p = step_gpu(h, tk)
                toks.append(tk); ps.append(p)
            mx.eval(h, *toks[-chunk:], *ps[-chunk:])
        dt = time.perf_counter() - t0
        cache[0].trim(depth)
        return dt / depth, [int(t.item()) for t in toks]
    res, seqs = {}, {}
    for r in range(7):
        for ch in ((1, 2, 4, 8) if r % 2 == 0 else (8, 4, 2, 1)):
            dt, sq = run_chain(ch)
            res.setdefault(ch, []).append(dt); seqs[ch] = sq
    same = all(seqs[c] == seqs[1] for c in seqs)
    say(f"MTP-STEP-5 chunked chain (8 depths, ms per step, median of 7): " + ", ".join(f"chunk {c}: {1e3 * np.median(v):.3f}" for c, v in sorted(res.items()))
        + f"; tokens identical across chunk sizes: {same}")
    sys.exit(0)

# STEP_ONLY: the flag screen (MTP-STEP-3) stops here with a hash of 8 chained steps' outputs (bits across flag settings)
if os.environ.get("STEP_ONLY") == "1":
    import hashlib
    h, tok, outs = h0, 4321, []
    for d in range(8):
        h, tk, p = step(h, tok)
        mx.eval(h, tk, p)
        outs.append(np.array(h.view(mx.uint16)).tobytes()); tok = int(tk.item())
    cache[0].trim(8)
    flags = {k: os.environ.get(k, "") for k in ("MTPLX_FUSED_MOE_DECODE", "MTPLX_FUSED_HC", "MTPLX_FUSED_HC_V3", "MTPLX_FUSED_QSA_INDEXER")}
    flags["MTP_REF2"] = os.environ.get("MTP_REF2", "")
    if os.environ.get("MTP_REF2") == "1":
        flags["counts"] = {k: v for k, v in ref2.COUNTS.items() if v}
    say(f"STEP_ONLY flags {flags}: step median {1e3 * np.median(ts[16:]):.3f} ms; 8-step output hash {hashlib.md5(b''.join(outs)).hexdigest()[:12]}")
    sys.exit(0)

# (2) pieces alone at 1 row (inputs computed once, outside the timing)
e1 = emb(mx.array([[1234]]))
x_in = mtp._prepare_inputs_eager(h0, e1)
mixed, hyper, inject = layer.attn_hyper_connection(x_in)
mx.eval(x_in, mixed, hyper, inject)
blk = layer.self_attn(mixed, cache[0]); mx.eval(blk); cache[0].trim(1)
mixed2, hyper2, inject2 = layer.mlp_hyper_connection(hyper, pending=(blk, inject))
mx.eval(mixed2, hyper2, inject2)
mo = layer.mlp(mixed2); mx.eval(mo)
hout = Q._hyper_residual_write(hyper2, mo, inject2); mx.eval(hout)
mixo = mtp.hyper_connection_mixer(hout); mx.eval(mixo)

def attn_once():
    o = layer.self_attn(mixed, cache[0])
    return o
PIECES = {
    "fc prep (norms + fc_hidden x4 + fc_embedding)": lambda: mtp._prepare_inputs_eager(h0, e1),
    "attn HC read": lambda: layer.attn_hyper_connection(x_in)[0],
    "mlp HC read (+ pending write)": lambda: layer.mlp_hyper_connection(hyper, pending=(blk, inject))[0],
    "MoE block (router + 10 experts + shared)": lambda: layer.mlp(mixed2),
    "residual write": lambda: Q._hyper_residual_write(hyper2, mo, inject2),
    "mixer": lambda: mtp.hyper_connection_mixer(hout),
    "screened head + argmax / p": lambda: (lambda lg: mx.exp(lg[mx.argmax(lg)] - mx.logsumexp(lg)))(sh.logits(mixo)[0]),
}
res = {k: [] for k in PIECES}
for r in range(9):
    for k in (list(PIECES) if r % 2 == 0 else list(PIECES)[::-1]):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval([PIECES[k]() for _ in range(20)]); mx.synchronize()
        res[k].append((time.perf_counter() - t0) / 20)
ta = []
for r in range(9):                                       # attention: one call + eval + trim (the cache grows by one)
    mx.synchronize(); t0 = time.perf_counter()
    for _ in range(10):
        mx.eval(attn_once())
        cache[0].trim(1)
    mx.synchronize(); ta.append((time.perf_counter() - t0) / 10)
say("(2) pieces alone at 1 row (us per call; 20 per eval, median of 9): " + "; ".join(f"{k} {1e6 * np.median(v):.0f}" for k, v in res.items())
    + f"; QSA attention (one per eval, incl. sync) {1e6 * np.median(ta):.0f}")
say(f"  sum of pieces {1e6 * (sum(np.median(v) for v in res.values()) + np.median(ta)):.0f} us; mlx peak {mx.get_peak_memory() / 2**30:.2f} GB; {time.time() - t00:.0f} s")

# (3) ablation inside the step: each piece replaced by its precomputed output (the rest runs on the same values)
ORIG = {"attn": layer["self_attn"], "moe": layer["mlp"], "hc1": layer["attn_hyper_connection"],
        "hc2": layer["mlp_hyper_connection"], "mix": mtp["hyper_connection_mixer"]}
ORIG_W, ORIG_P = Q._hyper_residual_write, mtp._mtp_prepare_inputs_impl
c_hc1, c_hc2 = (mixed, hyper, inject), (mixed2, hyper2, inject2)
STUB = {"attn": lambda on: layer.__setitem__("self_attn", (lambda x, c: blk) if on else ORIG["attn"]),
        "moe": lambda on: layer.__setitem__("mlp", (lambda x: mo) if on else ORIG["moe"]),
        "hc": lambda on: (layer.__setitem__("attn_hyper_connection", (lambda x: c_hc1) if on else ORIG["hc1"]),
                          layer.__setitem__("mlp_hyper_connection", (lambda x, pending=None: c_hc2) if on else ORIG["hc2"]),
                          setattr(Q, "_hyper_residual_write", (lambda a, b, c: hout) if on else ORIG_W)),
        "mixer": lambda on: mtp.__setitem__("hyper_connection_mixer", (lambda x: mixo) if on else ORIG["mix"]),
        "fc": lambda on: object.__setattr__(mtp, "_mtp_prepare_inputs_impl", (lambda a, b: x_in) if on else ORIG_P)}
HEADC = [False]
def step_ab(h, tok):
    hh = mtp.fuse_and_run(h, emb(mx.array([[tok]])), cache)
    if HEADC[0]:
        return hh, mx.array(7), mx.array(0.5)
    lg = sh.logits(mtp.hyper_connection_mixer(hh))[0]
    j = mx.argmax(lg)
    return hh, mx.take(allids, j), mx.exp(lg[j] - mx.logsumexp(lg))
AB = {"full": [], "no-attn": ["attn"], "no-moe": ["moe"], "no-hc (reads + write)": ["hc"], "no-mixer": ["mixer"],
      "no-fc": ["fc"], "no-head": ["head"], "skeleton": ["attn", "moe", "hc", "mixer", "fc", "head"]}
def run_ab(arm):
    for k in STUB:
        STUB[k](k in AB[arm])
    HEADC[0] = "head" in AB[arm]
    try:
        ts = []
        for chain in range(6):
            h, tok, n = h0, 1000 + chain, 0
            for d in range(8):
                t0 = time.perf_counter()
                h, tk, p = step_ab(h, tok)
                mx.eval(tk, p, h)
                ts.append(time.perf_counter() - t0)
                n += 1
            cache[0].trim(n)
        return float(np.median(ts[8:]))
    finally:
        for k in STUB:
            STUB[k](False)
        HEADC[0] = False
rab = {k: [] for k in AB}
for r in range(3):
    for k in (list(AB) if r % 2 == 0 else list(AB)[::-1]):
        rab[k].append(run_ab(k))
b0 = np.median(rab["full"])
say("(3) ablation inside the step (ms, median of 3 interleaved rounds): " + ", ".join(
    f"{k} {1e3 * np.median(v):.3f} ({100 * (np.median(v) / b0 - 1):+.0f} %)" for k, v in rab.items()))
