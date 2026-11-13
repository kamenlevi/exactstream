"""QUEUE L1-13 + L1-14 (Leap, Sat 2026-10-24), Qwen3.6-35B-A3B 4-bit fully in RAM, one quiet lock.
L1-13: where the 16-row DFlash block goes. One real code context (EvalSuite tf_code item 0 + 48 greedy tokens), then
one S-row verify pass (S = 16, 8, 1) timed whole and by component on captured per-layer inputs (SpecDecode
roofline.py's method, extended to S rows), each against its bytes floor at 280 GB/s.
- The whole pass is timed plain, with the drafter's 8 feature taps, and with taps + the GDN rollback capture, as
  z-lab's loop runs it.
- Also timed: the DFlash drafter's 16-block forward.
L1-14: z-lab's loop at block 16 with confidence truncation (DFL35_TRUNC = t; 0 = as is) on 12 HumanEval+ prompts.
Usage: [SpecDecode venv] small35_l113.py  -> ~/Leap/data/small35_l113.txt"""
import json, os, sys, time
os.environ["HF_HUB_OFFLINE"] = "1"                   # local weights only: no downloads
import numpy as np
import mlx.core as mx
mx.set_cache_limit(4 << 30)
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_third_party/sd35"))
import dflash35_model_mlx as dfm
from mlx_lm import stream_generate as lm_stream
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

MODEL = os.path.expanduser("~/.cache/huggingface/hub/models--Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed/snapshots/"
                           "95460516fbfe34c48f577f67c9ed191f5d61fbaf")
BW = 280e9
FLOOR_GB = {"GDN mixers (30)": 0.570, "full-attn mixers (10)": 0.153, "router + top-8 (40)": 0.022,
            "shared expert (40)": 0.071, "lm_head + argmax": 0.286}
dfm.snapshot_download = lambda *a, **k: os.path.expanduser("~/SpecDecode/data/dflash")
model, tok = dfm.load(MODEL)
draft = dfm.load_draft("z-lab/Qwen3.6-35B-A3B-DFlash")
draft.bind(model)                                    # the loop binds it too; calling the drafter directly needs it first
mx.eval(model.parameters(), draft.parameters())
LM = model.language_model; TM = LM.model; HEAD = LM.lm_head
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

def code_prompts(n):
    out = []
    for l in open(os.path.expanduser("~/EvalSuite/cache/ref/tf_gen.jsonl")):
        r = json.loads(l)
        if r["key"].startswith("tf_code") and len(out) < n:
            user = tok.decode(r["prompt_ids"]).split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
            ids = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True,
                                          enable_thinking=False)
            out.append((r["key"], list(ids if isinstance(ids[0], int) else ids["input_ids"])))
    return out

def greedy(ids, n):
    return [int(r.token) for r in lm_stream(model, tok, prompt=mx.array(ids), max_tokens=n, sampler=make_sampler(0.0))]

def timeit(fn, reps=12):
    ts = []
    for _ in range(reps):
        mx.synchronize(); t = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t)
    return sorted(ts)[reps // 2] * 1e3

# ---------------------------------------------------------------- L1-13
key, ids = code_prompts(1)[0]
out = greedy(ids, 64)
ctx, blk = ids + out[:48], out[48:64]
cache = make_prompt_cache(model)
mx.eval(model(mx.array([ctx[:-1]]), cache=cache))
L = TM.layers
lin = [i for i, l in enumerate(L) if l.is_linear]; full = [i for i, l in enumerate(L) if not l.is_linear]
say(f"L1-13 profile: {key}, context {len(ctx)} tokens, {len(L)} layers ({len(lin)} GDN + {len(full)} attention)")
dfm._patch_model(model, draft.config.target_layer_ids)
for S in (16, 8, 1):
    inp = mx.array([[ctx[-1]] + blk[:S - 1]])
    h = TM.embed_tokens(inp)
    fa_mask = create_attention_mask(h, cache[TM.fa_idx]); ssm_mask = create_ssm_mask(h, cache[TM.ssm_idx])
    xs, hs = [], []
    for layer, c in zip(L, cache):
        xs.append(h)
        mix = layer.linear_attn if layer.is_linear else layer.self_attn
        h = h + mix(layer.input_layernorm(h), ssm_mask if layer.is_linear else fa_mask, c)
        hs.append(layer.post_attention_layernorm(h))
        h = h + layer.mlp(hs[-1])
    final = TM.norm(h); mx.eval(xs, hs, final)
    K = L[0].mlp.top_k if hasattr(L[0].mlp, "top_k") else 8
    def router(i):
        g = mx.softmax(L[i].mlp.gate(hs[i]).astype(mx.float32), axis=-1)
        return mx.argpartition(g, kth=-K, axis=-1)[..., -K:]
    inds = [router(i) for i in range(len(L))]; mx.eval(inds)
    union = float(np.mean([len(np.unique(np.array(x).reshape(-1))) for x in inds]))
    whole = lambda: [mx.argmax(model(inp, cache=cache), -1)] + [x for x in model._hidden_states if x is not None]
    res = {"whole pass + 8 taps (as the loop)": timeit(whole)}
    cap = dfm._GDNStateCapture()
    try:
        res["whole pass + taps + GDN rollback capture"] = timeit(lambda: (cap.clear(), whole())[1])
    finally:
        cap.close()
    parts = {
        "GDN mixers (30)": lambda: [L[i].linear_attn(L[i].input_layernorm(xs[i]), ssm_mask, cache[i]) for i in lin],
        "full-attn mixers (10)": lambda: [L[i].self_attn(L[i].input_layernorm(xs[i]), fa_mask, cache[i]) for i in full],
        "router + top-8 (40)": lambda: [router(i) for i in range(len(L))],
        "routed experts (40)": lambda: [L[i].mlp.switch_mlp(hs[i], inds[i]) for i in range(len(L))],
        "shared expert (40)": lambda: [L[i].mlp.shared_expert(hs[i]) for i in range(len(L))],
        "lm_head + argmax": lambda: mx.argmax(HEAD(final), -1),
    }
    for k, f in parts.items():
        res[k] = timeit(f)
    fl = dict(FLOOR_GB); fl["routed experts (40)"] = union * len(L) * 1.769e-3
    say(f"S = {S:2d} rows: expert union {union:.1f} per layer of 256")
    for k, v in res.items():
        f_ms = fl[k] * 1e9 / BW * 1e3 if k in fl else None
        say(f"  {k:42s} {v:7.2f} ms" + (f"   floor {f_ms:5.2f} ms ({f_ms / v:4.0%} of floor speed)" if f_ms else ""))
    say(f"  {'sum of components':42s} {sum(v for k, v in res.items() if k in parts):7.2f} ms (rest of the whole pass = norms, residual, glue)")
# the drafter's 16-block forward, with the context features in its cache as the loop has them
mx.eval(model(mx.array([ctx[-17:-1]]), cache=cache))           # any 16 rows: refresh the taps
hid = mx.concatenate(model._hidden_states, axis=-1)
dcache = make_prompt_cache(draft)
mx.eval(draft(mx.array([[ctx[-1]] + [int(draft.config.mask_token_id)] * 15]), hid, dcache, logits_start=1))
blk16 = mx.array([[ctx[-1]] + [int(draft.config.mask_token_id)] * 15])
t = timeit(lambda: mx.argmax(draft(blk16, hid[:, -4:], dcache, logits_start=1), -1))
say(f"drafter 16-block forward (+ argmax): {t:.2f} ms   floor {(0.772 + 0.286) * 1e9 / BW * 1e3:.2f} ms (its 0.77 GB + the target head)")

# ---------------------------------------------------------------- L1-14
say("L1-14: z-lab DFlash block 16 with confidence truncation, 12 HumanEval+ prompts (decode tok/s from the first token)")
P = code_prompts(12)
for t_ in ("0", "0.3", "0.5", "0.7"):
    os.environ["DFL35_TRUNC"] = t_
    n_tok, dt, nb, na = 0, 0.0, 0, 0
    for key, ids in P:
        toks, t_first, acc = [], None, []
        for r in dfm.stream_generate(model, draft, tok, ids, block_size=16, max_tokens=400, temperature=0.0):
            if t_first is None:
                t_first = time.perf_counter()
            toks += list(getattr(r, "tokens", []) or [])
            a = getattr(r, "accepted", None)
            if a is not None:
                acc.append(a)
        n_tok += len(toks) - 1; dt += time.perf_counter() - t_first; nb += len(acc); na += sum(acc)
    say(f"  t = {t_:3s}: {n_tok / dt:6.1f} tok/s, {na / max(1, nb):.2f} accepted + 1 per block, {nb} blocks, {n_tok} tokens")
os.environ["DFL35_TRUNC"] = "0"
say(f"peak {mx.get_peak_memory() / 1e9:.1f} GB")
open(os.path.expanduser("~/Leap/data/small35_l113.txt"), "w").write("\n".join(lines) + "\n")
