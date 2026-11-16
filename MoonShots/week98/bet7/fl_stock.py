"""WEEK98 BET 7: the STOCK level, as P2 builds it (C0 "as stored") and as M0-M2 timed it (p3_m012.py; its build
copied here, C0 only, so the fused-level tests and timings import one definition).

The shipped MTP head (~/BigMoE/models/bare/mtp.safetensors, Qwen4ExpMTP, read-only), the 125B's 4-bit embedding and
lm_head (from the model shards), Leap's ScreenedHead over FR-Spec's top-nh ids (read-only import), a QSACache
prefilled with T0 synthetic widened states (P2's), and M0's tree level: fuse_and_run -> mixer -> screened logits ->
log-softmax -> top-K per row -> frontier top-F by path score. MLX eager, engine flags off (MTPLX_* unset here).
Memory: ~2.5 GB (the 512 experts are 1.57 GB of it)."""
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
for _k in [k for k in os.environ if k.startswith("MTPLX_")]:   # the stock level: no engine fusions
    os.environ.pop(_k)
import numpy as np                                          # noqa: E402
import mlx.core as mx                                       # noqa: E402
import mlx.nn as nn                                         # noqa: E402
from mlx.utils import tree_flatten                          # noqa: E402

H = os.path.expanduser
sys.path.insert(0, H("~/Leap/scripts"))                     # screened_head.py (read-only import)
from mtplx.models import qwen4_exp as Q                     # noqa: E402
from screened_head import ScreenedHead                      # noqa: E402

D = H("~/BigMoE/models/bare")
cfg = json.load(open(f"{D}/config.json"))
args = Q.TextArgs.from_dict(cfg.get("text_config", cfg))
HCW = args.hc_count * args.hidden_size
NE = int(args.num_experts)
V = int(args.vocab_size)


def load_mtp():
    """P2's build, as stored: the MTP's quantized Linears keep their stored bits / group sizes"""
    raw = mx.load(f"{D}/mtp.safetensors")
    st = {k[len("mtp."):]: v for k, v in raw.items() if k.startswith("mtp.")}
    del raw
    for n in ("pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight"):
        if n in st and st[n].ndim == 1:
            st[n] = (st[n].astype(mx.float32) + 1.0).astype(st[n].dtype)
    mtp = Q.Qwen4ExpMTP(args)
    module_in = {p[:-len(".weight")]: int(a.shape[-1]) for p, a in tree_flatten(mtp.parameters())
                 if p.endswith(".weight") and a.ndim >= 2}
    qmap = {}
    mods = dict(mtp.named_modules())
    for mod, in_f in module_in.items():
        if not hasattr(mods.get(mod), "to_quantized"):
            continue
        w, s = st.get(f"{mod}.weight"), st.get(f"{mod}.scales")
        if w is None or s is None or w.dtype != mx.uint32:
            continue
        qmap[mod] = {"bits": int(w.shape[-1]) * 32 // in_f, "group_size": in_f // int(s.shape[-1])}
    nn.quantize(mtp, group_size=64, bits=8, class_predicate=lambda p, m: qmap.get(p, False))
    mtp.load_weights(list(st.items()), strict=True)
    mtp.eval()
    mx.eval(mtp.parameters())
    return mtp, qmap


_wmap = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]


def _get(k):
    return mx.load(f"{D}/{_wmap[k]}")[k]


def load_emb_head():
    emb = nn.QuantizedEmbedding(V, args.hidden_size, group_size=64, bits=4)
    head = nn.QuantizedLinear(args.hidden_size, V, bias=False, group_size=64, bits=4)
    for m_, p_ in ((emb, "language_model.model.embed_tokens"), (head, "language_model.lm_head")):
        m_.weight, m_.scales, m_.biases = _get(p_ + ".weight"), _get(p_ + ".scales"), _get(p_ + ".biases")
    mx.eval(emb.parameters(), head.parameters())
    return emb, head


RANK = np.load(H("~/Leap/data/frspec_rank.npy"))


class Stock:
    """the stock level: build once, then level() / block() as M0-M2's 'tree' arm (eager, sync-free)"""

    def __init__(self, nh=32768, T0=400, seed=5, full_head=False):
        self.mtp, self.qmap = load_mtp()
        self.emb, self.head = load_emb_head()
        self.sh = ScreenedHead(self.head, np.arange(V) if full_head else RANK[:nh], extra_cap=16)
        self.cache = [Q.QSACache(args.indexer_compress_ratio or 4)]
        rng = np.random.default_rng(seed)
        if T0 > 0:
            Wh = mx.array((rng.standard_normal((1, T0, HCW)) * 0.6).astype(np.float32)).astype(mx.bfloat16)
            ids0 = mx.array(rng.integers(0, 200000, (1, T0)).astype(np.int32))
            mx.eval(self.mtp.fuse_and_run(Wh, self.emb(ids0), self.cache))
        mx.random.seed(0)
        self.h0 = (mx.random.normal((1, 1, HCW)) * 0.6).astype(mx.bfloat16)          # P2's synthetic state
        mx.eval(self.h0)

    @property
    def T(self):
        return int(self.cache[0].offset)

    def level(self, h, tok, sc, F, K):
        """one tree level: F frontier rows -> top-K candidates per row -> the next frontier (top-F by path score)"""
        hh = self.mtp.fuse_and_run(h, self.emb(tok), self.cache)                # [1, F, HCW]
        lg = self.sh.logits(self.mtp.hyper_connection_mixer(hh))               # [F, nh] fp32
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        nh = lg.shape[-1]
        kidx = mx.argpartition(lp, kth=nh - K, axis=-1)[:, nh - K:]             # [F, K]
        klp = mx.take_along_axis(lp, kidx, axis=-1)
        cand = (sc[:, None] + klp).reshape(-1)                                  # [F * K]
        sel = mx.argpartition(cand, kth=F * K - F)[F * K - F:]                  # [F]
        par = sel // K
        tk = mx.take(self.sh.ids, mx.take(kidx.reshape(-1), sel))               # [F] token ids
        return dict(h=mx.take(hh, par, axis=1), tok=tk[None, :], sc=mx.take(cand, sel), hh=hh, kidx=kidx,
                    klp=klp, cand=cand, sel=sel, par=par)

    def block(self, L, F=1, K=8, tok0=1000, mode="async", keep=False):
        """L levels from the synthetic state; mode 'async' (async_eval per level, one sync) or 'free' (one eval);
        returns the per-level outputs; the cache is trimmed back unless keep"""
        h = mx.broadcast_to(self.h0, (1, F, HCW)) if F > 1 else self.h0
        tok = mx.array([[tok0 + i for i in range(F)]], dtype=mx.int32)
        sc = mx.zeros((F,), mx.float32)
        outs = []
        for _ in range(L):
            o = self.level(h, tok, sc, F, K)
            h, tok, sc = o["h"], o["tok"], o["sc"]
            outs.append(o)
            if mode == "async":
                mx.async_eval(h, tok, sc, o["hh"], o["kidx"], o["klp"])
        mx.eval([list(o.values()) for o in outs])
        if not keep:
            self.cache[0].trim(L * F)
        return outs
