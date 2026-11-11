"""Look-ahead recall on Qwen3.8-Flash-Next (qwen4_exp): does layer L+1's router applied to layer L's
MoE input (a hyper-connection mix, no shared norm to swap) predict layer L+1's experts?
Runs the streamed model (exact) for N decode tokens via MTPLX, records per layer the real top-10
and the top-B prediction for the next layer. Prints recall@B per depth band. -> data/recall_target.txt
usage: recall_target.py [FRAC] [N]"""
import os, sys, json, time
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")   # MTPLX would pre-read up to ~18 GiB of the n-gram table into page cache
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import numpy as np, mlx.core as mx
from bigmoe.loader import load_streamed
from bigmoe import stream as S

MD = os.path.expanduser("~/BigMoE/models/bare")
frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
N = int(sys.argv[2]) if len(sys.argv) > 2 else 64
rt, st = load_streamed(MD, round(frac * 512))
layers = rt.model.language_model.model.layers
NL = len(layers)
rec = {"act": [[] for _ in range(NL)], "pred": [[] for _ in range(NL)]}
Bmax = 16
# wrap the streamed class once more to record (decode width only)
for L, layer in enumerate(layers):
    sw = layer.mlp.switch_mlp
    base = type(sw)
    def make(base, L):
        def __call__(self, x, indices, *a, **kw):
            if indices.size <= 10:
                rec["act"][L].append(indices)
                if L + 1 < NL:
                    lg = layers[L + 1].mlp.gate(x.reshape(-1, x.shape[-1]))
                    rec["pred"][L + 1].append(mx.argpartition(-lg, kth=Bmax - 1, axis=-1)[:, :Bmax])
            return base.__call__(self, x, indices, *a, **kw)
        return type(base.__name__ + "Rec", (base,), {"__call__": __call__})
    sw.__class__ = make(base, L)
from mtplx.sampling import SamplerConfig
from mtplx import generation as G
ids = rt.tokenizer.apply_chat_template([{"role": "user", "content": open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read()}],
                                       add_generation_prompt=True, enable_thinking=False)
ids = list(ids if isinstance(ids[0], int) else ids["input_ids"])
G.generate_ar(rt, ids, max_tokens=N, sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0))
lines = []
for B in (10, 12, 16):
    bands = {}
    for L in range(1, NL):
        A = [np.array(a).reshape(-1) for a in rec["act"][L]]
        P = [np.array(p).reshape(-1)[:B] for p in rec["pred"][L]]
        m = min(len(A), len(P))
        if not m: continue
        r = np.mean([len(set(A[i]) & set(P[i])) / len(set(A[i])) for i in range(m)])
        bands.setdefault(L * 4 // NL, []).append(r)
    lines.append(f"recall@{B}: " + ", ".join(f"layers {b*NL//4}-{(b+1)*NL//4-1}: {np.mean(v):.2f}" for b, v in sorted(bands.items())))
print("\n".join(lines))
open(os.path.expanduser("~/BigMoE/data/recall_target.txt"), "w").write("\n".join(lines) + "\n")
