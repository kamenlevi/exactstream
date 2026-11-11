"""Round 32g: what does tree WIDTH cost in experts? (125B on BigMoE's streamed runtime, greedy.)
For a few prompts, generate G tokens greedily; at branch points t, take the model's top-4 next-token candidates at
position t (siblings: same context, different token) and record each sibling's top-10 routing per layer (a hook on the MoE
block computes the router on the last row itself, so any expert path works). Compare per layer:
  siblings  union of the 4 candidates' experts at one position (what 4-wide tree nodes cost)
  chain     union of 4 consecutive committed tokens ending at t (what a depth-4 chain costs; trace value ~26.9)
  single    10
If siblings' union is far below the chain's, wide trees are byte-cheap and tokens/pass can grow at little SSD cost.
Each probe is a cache-free forward of prompt + prefix (+ candidate), ~150-300 tokens.
usage: sibling_union125.py [FRAC] [G] [NPOS]   (GPU, BigMoE-size memory: exclusive daytime window, ~3-5 min)"""
import json, os, sys
for k, v in (("MTPLX_NGRAM_PREWARM", "off"), ("MTPLX_COMPILED_VERIFY", "0"), ("MTPLX_QWEN4EXP_COMPILE", "0"),
             ("MTPLX_COMPILE_AR_FORWARD", "0"), ("MTPLX_COMPILED_TARGET_PREFIX", "0")):
    os.environ.setdefault(k, v)
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import numpy as np
import mlx.core as mx
from bigmoe.loader import load_streamed
import mtplx.models.qwen4_exp as Q

frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
G = int(sys.argv[2]) if len(sys.argv) > 2 else 48
NPOS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
rt, st = load_streamed(os.path.expanduser("~/BigMoE/models/bare"), round(frac * 512))
model, tok = rt.model, rt.tokenizer
L0 = lambda o: o[0] if isinstance(o, tuple) else o
layers = model.language_model.model.layers if hasattr(model, "language_model") else model.model.layers
LIDX = {id(l.mlp): i for i, l in enumerate(layers) if hasattr(l, "mlp")}
REC = {}                                                   # layer -> ids [rows, 10] of the last NR rows of the call
NR = 4
_orig = Q.SparseMoeBlock.__call__


def hook(self, x):
    L = LIDX.get(id(self))
    if L is not None and x.ndim >= 2:
        xs = x.reshape(-1, x.shape[-1])[-NR:]
        g = self.gate(xs)
        REC[L] = np.array(mx.argpartition(g, kth=-10, axis=-1)[..., -10:])
    return _orig(self, x)


Q.SparseMoeBlock.__call__ = hook
probs = [json.loads(l) for l in open(os.path.expanduser("~/MoEPrefetch/data/gsm8k/test.jsonl"))]
PROMPTS = {"code": open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read(),
           "math": "Solve the problem step by step, briefly. End with a line of the form 'Answer: <number>'.\n\n" + probs[0]["question"],
           "chat": "Explain to a curious teenager why the sky is blue, in a few short paragraphs."}


def fwd(ids):
    REC.clear()
    lg = L0(model(mx.array([ids]), cache=model.make_cache()))[0, -1].astype(mx.float32)   # fresh cache per probe
    mx.eval(lg)
    return lg, {L: v.copy() for L, v in REC.items()}


out = {}
for name, text in PROMPTS.items():
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    ids = list(ids if isinstance(ids[0], int) else ids["input_ids"])
    gen = []
    cache = model.make_cache()
    lg = L0(model(mx.array([ids]), cache=cache))[0, -1]
    for _ in range(G):                                     # greedy continuation
        y = int(mx.argmax(lg).item())
        gen.append(y)
        lg = L0(model(mx.array([[y]]), cache=cache))[0, -1]
    del cache
    rows = []
    for t in np.linspace(8, G - 1, NPOS).astype(int):
        base = ids + gen[:t]
        lg, rec_chain = fwd(base)                          # last 4 rows = 4 consecutive committed tokens ending at t-1
        cands = np.argsort(-np.array(lg))[:4].tolist()
        sib = []
        for c in cands:
            _, r = fwd(base + [c])
            sib.append(r)
        su = [len(np.unique(np.concatenate([s[L][-1] for s in sib]))) for L in sorted(sib[0])]
        cu = [len(np.unique(rec_chain[L].reshape(-1))) for L in sorted(rec_chain)]
        p = np.exp(np.array(lg) - np.array(lg).max()); p /= p.sum()
        rows.append(dict(t=int(t), sib_union=float(np.mean(su)), chain_union=float(np.mean(cu)),
                         cand_p=[float(p[c]) for c in cands], greedy_is_top=bool(cands[0] == gen[t])))
        print(f"{name} t={t}: siblings union {np.mean(su):5.1f} | chain-4 union {np.mean(cu):5.1f} | cand p "
              f"{' '.join(f'{p[c]:.2f}' for c in cands)}", flush=True)
    out[name] = rows
allr = [r for v in out.values() for r in v]
print(f"ALL: siblings union {np.mean([r['sib_union'] for r in allr]):.1f} vs chain-4 union "
      f"{np.mean([r['chain_union'] for r in allr]):.1f} experts/layer (single token 10)", flush=True)
json.dump(out, open(os.path.expanduser("~/MoonShots/data/sibling_union125.json"), "w"), indent=1)
