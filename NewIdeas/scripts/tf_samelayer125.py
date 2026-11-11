"""Same-layer pre-attention forecast on the 125B (NewIdeas 2026-10-22; IDEAS I8 "next if alive").
Exact teacher-forced decode calls (the Eval suite's tf items, MTPLX's step path, exact streaming, nothing changed in the
forward). Per 1-row decode call and layer l it records:
  TRUE  l's own router on its MLP input (post-attention): top-10 ids by logit + their softmax probabilities
  SAME  l's router on its MLP hyper-connection read of the PRE-attention streams (available at the layer's start): top-16
  D1    layer l+1's router on layer l's MLP input (BigMoE's d=1 tap, available at l's router point): top-16
  D1C   the same + MoEPrefetch's learned correction (corr125_r128.npz: + ((x - mu) A) B + delta; BIGMOE_PF_CORR): top-16
and reports the recall of the true ranks 0-1 and of the true top-10 inside each forecast's top-k.
usage: tf_samelayer125.py FRAC SETS EVERY   -> ~/NewIdeas/data/samelayer/<SETS>_e<EVERY>.npz + results.txt
"""
import json, os, sys, time
os.environ["EVALSUITE_FRUGAL"] = "1"
os.environ["BIGMOE_SPARE"] = "0"
os.environ["BIGMOE_PF_NOPUBLISH"] = "1"
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
sys.path.insert(0, os.path.expanduser("~/Breakthrough/fork125bt"))
sys.path.insert(1, os.path.expanduser("~/MoEPrefetch/scripts"))
import numpy as np
import mlx.core as mx
from bigmoe.loader import load_streamed
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
from evalsuite import stages as SG, engine as EN

frac, SETS, EVERY = float(sys.argv[1]), sys.argv[2].split(","), int(sys.argv[3])
OUT = os.path.expanduser("~/NewIdeas/data/samelayer"); os.makedirs(OUT, exist_ok=True)
MD = os.path.expanduser("~/BigMoE/models/bare")
cfg = json.load(open(os.path.join(MD, "config.json"))); cfg = cfg.get("text_config", cfg)
E, K, NL = cfg["num_experts"], cfg["num_experts_per_tok"], cfg["num_hidden_layers"]
C = max(16, round(frac * E))
rt, st = load_streamed(MD, C, mtp=False)
from mtplx.models import qwen4_exp
from mtplx.models.qwen4_exp import _hyper_residual_write

LAYERS = rt.model.language_model.model.layers if hasattr(rt.model, "language_model") else rt.model.model.layers
REC = {"T": [[] for _ in range(NL)], "P": [[] for _ in range(NL)], "S": [[] for _ in range(NL)],
       "D": [[] for _ in range(NL)], "DC": [[] for _ in range(NL)]}
_z = np.load(os.path.expanduser("~/MoEPrefetch/data/corr125_r128.npz"))
CORR = [(mx.array(_z["A"][i].astype(np.float32)), mx.array(_z["B"][i].astype(np.float32)), mx.array(_z["mu"][i]),
         mx.array(_z["delta"][i])) for i in range(_z["A"].shape[0])]
LAZY = []


def _mixed(r):
    return r[0] if isinstance(r, tuple) else r


def _top(logits, k):                                  # ids of the k largest logits, by logit (descending)
    lg = logits.reshape(-1).astype(mx.float32)
    idx = mx.argpartition(-lg, k - 1)[:k]
    return idx[mx.argsort(-lg[idx])]


def layer_call(self, hidden, *, input_ids, ssm_mask, cache):
    if "ple" in self:
        hidden = hidden + self.ple(hidden, input_ids, cache)
    mixed, hyper, inject = self.attn_hyper_connection(hidden)
    L = self._ni_L
    one = hidden.shape[1] == 1
    if one:                                           # forecast from the pre-attention streams (no attention write)
        pre = _mixed(self.mlp_hyper_connection(hyper, pending=None))
        s_ids = _top(self.mlp.gate(pre), 16)
    if self.is_linear:
        block_out = self.linear_attn(mixed, ssm_mask, cache)
    else:
        block_out = self.self_attn(mixed, cache)
    mixed, hyper, inject = self.mlp_hyper_connection(hyper, pending=(block_out, inject))
    if one:
        lg = self.mlp.gate(mixed).reshape(-1).astype(mx.float32)
        t_ids = _top(lg, K)
        t_p = mx.softmax(lg, precise=True)[t_ids]
        if L + 1 < NL:
            dl = LAYERS[L + 1].mlp.gate(mixed).reshape(-1).astype(mx.float32)
            d_ids = _top(dl, 16)
            if L < len(CORR):
                A_, B_, mu_, de_ = CORR[L]
                xc = mixed.reshape(1, -1).astype(mx.float32) - mu_
                dc_ids = _top(dl + ((xc @ A_) @ B_).reshape(-1) + de_, 16)
            else:
                dc_ids = d_ids
        else:
            d_ids = dc_ids = mx.zeros((16,), mx.uint32)
        LAZY.append((L, t_ids, t_p, s_ids, d_ids, dc_ids))
    block_out = self.mlp(mixed)
    return _hyper_residual_write(hyper, block_out, inject)


for i, l in enumerate(LAYERS):
    l._ni_L = i
qwen4_exp.DecoderLayer.__call__ = layer_call
items = SG.tf_items()
head = EN.var_head(SG.load_ref_tf(items), SG._targets(items))
seqs = [i for i, it in enumerate(items) if it["set"] in SETS][::EVERY]


def score(i):
    def f(pos, lg):
        r = head(i, pos, lg)
        if LAZY:
            mx.eval(*[a for row in LAZY for a in row[1:]])
            for L, t, p, s, d, dc in LAZY:
                REC["T"][L].append(np.array(t)); REC["P"][L].append(np.array(p))
                REC["S"][L].append(np.array(s)); REC["D"][L].append(np.array(d)); REC["DC"][L].append(np.array(dc))
            LAZY.clear()
        return r
    return f


t0 = time.time(); kl = []
for i in seqs:
    it = items[i]
    r = SG.step_forward(rt, it["ids"], it["pos"], score(i))
    st.gate.check()
    kl.append(r["kld"])
T = {L: np.stack(REC["T"][L]) for L in range(NL)}
P = {L: np.stack(REC["P"][L]) for L in range(NL)}
S = {L: np.stack(REC["S"][L]) for L in range(NL)}
D = {L: np.stack(REC["D"][L]) for L in range(NL - 1)}          # D[L] forecasts layer L+1
DC = {L: np.stack(REC["DC"][L]) for L in range(NL - 1)}
n = T[0].shape[0]
lines = [f"tf_samelayer125 C={C} sets={','.join(SETS)} every={EVERY}: {n} decode calls, {len(seqs)} seqs, "
         f"KLD vs reference {np.concatenate(kl).mean():.4f} (exact streaming: numerics floor expected), {time.time() - t0:.0f}s"]
for name, F, off in (("SAME (pre-attention, lead = attention half)", S, 0), ("D1 (BigMoE tap, lead = MoE half + attention)", D, 1),
                    ("D1C (tap + MoEPrefetch correction)", DC, 1)):
    for k in (2, 3, 4, 8, 10, 12, 16):
        r01 = []; r10 = []; m10 = []
        for L in range(1, NL):
            src = L - off
            if src < 0 or src not in F: continue
            f = F[src][:, :k]
            hit = (T[L][:, :, None] == f[:, None, :]).any(-1)          # [n, 10]
            r01.append(hit[:, :2].mean()); r10.append(hit.mean())
            m10.append((hit * P[L]).sum(-1).mean() / P[L].sum(-1).mean())
        lines.append(f"  {name} top-{k:2d}: recall ranks 0-1 {np.mean(r01):.3f}  top-10 {np.mean(r10):.3f}  "
                     f"top-10 mass {np.mean(m10):.3f}")
print("\n".join(lines), flush=True)
tag = "".join(x[3] for x in SETS)
np.savez(os.path.join(OUT, f"S{tag}_e{EVERY}.npz"), **{f"T{L}": T[L] for L in T}, **{f"P{L}": P[L] for L in P},
         **{f"S{L}": S[L] for L in S}, **{f"D{L}": D[L] for L in D}, **{f"DC{L}": DC[L] for L in DC})
open(os.path.join(OUT, "results.txt"), "a").write("\n".join(lines) + "\n")
