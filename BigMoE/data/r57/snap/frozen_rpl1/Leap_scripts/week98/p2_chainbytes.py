"""P2 CHAIN-BYTES (WEEK98 pillar 3; Leap, Sun 15 Nov 2026 01:5x; data/week98/plan_1115.md): the chain head's step at
fewer bytes. The 125B's MTP head (the shape of the WEEK98 chain drafter t30b) built standalone as mtp_step_profile.py
builds it; its bf16 Linear modules re-quantized (8-bit / 4-bit, g64) and its screened head cut (32K -> 16K / 8K rows).
Per variant: the weight bytes one step reads (MoE experts at top-k / E of their bytes; the screened head's rows; the
rest in full) and the step's time as the chain runs it (one sync per step; 12 chains x 32 steps per round, variants
interleaved, median of steps after a warm-up). Drafts are verified, so the output stays exact whatever the step reads;
only acceptance can move, which these synthetic hidden states cannot measure (the owners' bank replay does).
Pre-registered (plan_1115.md P2, written Sun 00:4x): a step <= 0.5 ms AND <= 60 MB -> hand the forms to BigMoE / TrainLab
for an acceptance replay on the bank; else close the chain-step diet (block drafting is the route).
usage: [venv] p2_chainbytes.py (MEM ~10 GB; quiet ticket for the times)"""
import os, sys, json, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rpl1/Leap_scripts"))
from mtplx.models import qwen4_exp as Q
from mlx.utils import tree_flatten
from screened_head import ScreenedHead
t00 = time.time()


def say(s):
    print(s, flush=True)


D = os.path.expanduser("~/BigMoE/models/bare")
cfg = json.load(open(f"{D}/config.json"))
args = Q.TextArgs.from_dict(cfg.get("text_config", cfg))
raw = mx.load(f"{D}/mtp.safetensors")
ST0 = {k[len("mtp."):]: v for k, v in raw.items() if k.startswith("mtp.")}
for n in ("pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight"):
    if n in ST0 and ST0[n].ndim == 1:
        ST0[n] = (ST0[n].astype(mx.float32) + 1.0).astype(ST0[n].dtype)


def build(requant=None):
    """the MTP head as stored; requant = (bits, group) re-quantizes every remaining bf16 Linear (>= 64K weights)"""
    mtp = Q.Qwen4ExpMTP(args)
    module_in = {p[:-len(".weight")]: int(a.shape[-1]) for p, a in tree_flatten(mtp.parameters())
                 if p.endswith(".weight") and a.ndim >= 2}
    qmap = {}
    for mod, in_f in module_in.items():
        w, s = ST0.get(f"{mod}.weight"), ST0.get(f"{mod}.scales")
        if w is None or s is None or w.dtype != mx.uint32:
            continue
        qmap[mod] = {"bits": int(w.shape[-1]) * 32 // in_f, "group_size": in_f // int(s.shape[-1])}
    nn.quantize(mtp, group_size=64, bits=8, class_predicate=lambda p, m: qmap.get(p, False))
    mtp.load_weights(list(ST0.items()), strict=True)
    if requant:
        b_, g_ = requant
        nn.quantize(mtp, group_size=g_, bits=b_,
                    class_predicate=lambda p, m: (isinstance(m, nn.Linear) and not isinstance(m, nn.QuantizedLinear)
                                                  and m.weight.size >= 65536 and m.weight.shape[-1] % g_ == 0))
    mtp.eval()
    mx.eval(mtp.parameters())
    return mtp


wmap = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
def get(k):
    return mx.load(f"{D}/{wmap[k]}")[k]
V = args.vocab_size
emb = nn.QuantizedEmbedding(V, args.hidden_size, group_size=64, bits=4)
head = nn.QuantizedLinear(args.hidden_size, V, bias=False, group_size=64, bits=4)
for m, p in ((emb, "language_model.model.embed_tokens"), (head, "language_model.lm_head")):
    m.weight, m.scales, m.biases = get(p + ".weight"), get(p + ".scales"), get(p + ".biases")
mx.eval(emb.parameters(), head.parameters())
RANK = np.load(os.path.expanduser("~/Leap/data/frspec_rank.npy"))
HCW = args.hc_count * args.hidden_size
TOPK = int(getattr(args, "num_experts_per_tok", 10))
NE = int(getattr(args, "num_experts", 512))


def step_bytes(mtp, sh):
    """weight bytes one step reads: expert banks (leading dim NE) at TOPK / NE, everything else in full; + the head rows"""
    tot, parts = 0.0, {}
    for p, a in tree_flatten(mtp.parameters()):
        b = a.nbytes * (TOPK / NE if a.ndim >= 3 and a.shape[0] == NE else 1.0)
        key = p.split(".")[0] if not p.startswith("layers.") else ".".join(p.split(".")[:3])
        parts[key] = parts.get(key, 0.0) + b
        tot += b
    hb = sum(x.nbytes for x in (sh.w, sh.s, sh.b))
    parts["screened head"] = hb
    return (tot + hb) / 1e6, {k: v / 1e6 for k, v in parts.items()}


VARIANTS = [("base (as stored)", None, 32768), ("bf16 Linears -> 8-bit", (8, 64), 32768),
            ("bf16 Linears -> 4-bit", (4, 64), 32768), ("head 16K rows", None, 16384),
            ("head 8K rows", None, 8192), ("4-bit + head 8K", (4, 64), 8192)]
M = []
for name, rq, nh in VARIANTS:
    mtp = build(rq)
    sh = ScreenedHead(head, RANK[:nh], extra_cap=4096)
    rng = np.random.default_rng(5)
    cache = [Q.QSACache(args.indexer_compress_ratio or 4)]
    Wh = mx.array((rng.standard_normal((1, 400, HCW)) * 0.6).astype(np.float32)).astype(mx.bfloat16)
    ids0 = mx.array(rng.integers(0, 200000, (1, 400)).astype(np.int32))
    mx.eval(mtp.fuse_and_run(Wh, emb(ids0), cache))
    mb, parts = step_bytes(mtp, sh)
    M.append(dict(name=name, mtp=mtp, sh=sh, cache=cache, mb=mb, parts=parts))
    say(f"built {name}: {mb:.1f} MB per step ({', '.join(f'{k} {v:.1f}' for k, v in sorted(parts.items(), key=lambda t: -t[1])[:8])})")
mx.random.seed(0)
h0 = (mx.random.normal((1, 1, HCW)) * 0.6).astype(mx.bfloat16)
mx.eval(h0)
say(f"all built at {time.time() - t00:.0f} s; active {mx.get_active_memory() / 1e9:.1f} GB")


def chain(v, tok0, n=32, forced=None):
    mtp, sh, cache = v["mtp"], v["sh"], v["cache"]
    h, tok, toks, ts = h0, tok0, [], []
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
        tok = forced[d] if forced is not None else toks[-1]
    cache[0].trim(n)
    return ts, toks


T = {i: [] for i in range(len(M))}
base_toks = {}
for r in range(14):
    order = list(range(len(M))) if r % 2 == 0 else list(range(len(M)))[::-1]
    for i in order:
        ts, toks = chain(M[i], 1000 + r)
        if r >= 2:
            T[i] += ts[2:]
        if i == 0:
            base_toks[r] = toks
agree = {}
for i in range(1, len(M)):                          # teacher-forced on the base chain's tokens (synthetic states: info only)
    a_, n_ = 0, 0
    for r in (2, 3, 4, 5):
        _, toks = chain(M[i], 1000 + r, forced=base_toks[r])
        a_ += sum(int(x == y) for x, y in zip(toks, base_toks[r])); n_ += len(toks)
    agree[i] = a_ / n_
say("")
say("variant | MB per step | step ms (median) | p10-p90 | top-1 == base (synthetic states, info)")
res = []
for i, v in enumerate(M):
    ms = 1e3 * float(np.median(T[i]))
    res.append(dict(name=v["name"], mb=v["mb"], ms=ms, p10=1e3 * float(np.percentile(T[i], 10)),
                    p90=1e3 * float(np.percentile(T[i], 90)), agree=agree.get(i, 1.0)))
    say(f"{v['name']} | {v['mb']:.1f} | {ms:.3f} | {res[-1]['p10']:.3f}-{res[-1]['p90']:.3f} | {agree.get(i, 1.0):.3f}")
best = min(res, key=lambda d: d["ms"])
ok = [d for d in res if d["ms"] <= 0.5 and d["mb"] <= 60]
say(f"P2 verdict: {'HAND TO OWNERS: ' + ', '.join(d['name'] for d in ok) if ok else 'CLOSE the chain-step diet (no variant <= 0.5 ms and <= 60 MB)'}; "
    f"fastest {best['name']} {best['ms']:.3f} ms at {best['mb']:.1f} MB (base {res[0]['ms']:.3f} ms, {res[0]['mb']:.1f} MB); "
    f"32 steps: base {32 * res[0]['ms']:.1f} ms -> fastest {32 * best['ms']:.1f} ms")
json.dump(res, open(os.path.expanduser("~/Leap/data/week98/screens/p2_chainbytes.json"), "w"), indent=1)
say(f"done in {time.time() - t00:.0f} s; peak {mx.get_peak_memory() / 1e9:.1f} GB")
