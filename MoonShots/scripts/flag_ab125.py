"""Round 29: per-flag A/B of MTPLX's own kernel-cutting switches on BigMoE's streamed 125B (BigMoE ledger A2 tested four
together: slower, output changed from token 1; R2 asks for a per-flag A/B). The flag under test is set in the environment
by the runner (MTPLX reads them at import). One process per configuration:
  kernels   kernel-launching graph nodes of a 4-token forward under attention_phase("decode_verify"), per layer type
  exact     sha1 of that forward's logits (bitwise) and of each prompt's greedy tokens (committed MTP depth 3)
  speed     decode tok/s, 3 prompts x MAXTOK tokens
Prints one JSON line.   usage: flag_ab125.py LABEL [FRAC] [MAXTOK]   (GPU, BigMoE-size memory: quiet lock)"""
import collections, hashlib, json, os, re, sys, time
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for _k, _v in (("MTPLX_FAMILY_CAPTURE_COMMIT", "1"), ("MTPLX_DROP_EVENTS", "1"), ("MTPLX_LAZY_MTP_HISTORY_APPEND", "1"),
               ("MTPLX_DEFER_REPAIR_EVAL", "1")):
    os.environ.setdefault(_k, _v)
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import numpy as np
import mlx.core as mx
from bigmoe.loader import load_streamed
from bigmoe import fuse
LABEL = sys.argv[1]
frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 128
rt, st = load_streamed(os.path.expanduser("~/BigMoE/models/bare"), round(frac * 512))
from mtplx.sampling import SamplerConfig
from mtplx import generation as G
from mtplx.attention_context import attention_phase
model = rt.model
TM = model.language_model
FREE = {"Reshape", "Broadcast", "ExpandDims", "Squeeze", "Flatten", "Unflatten", "Slice", "Split", "Transpose", "AsStrided", "View"}
# 1) kernels + bitwise logits of a fixed 4-row forward
ids = rt.tokenizer.encode("The quick brown fox jumps over the lazy dog. " * 4)[:32]
fuse.reset_generation_state(model)
cache = model.make_cache()
o = model(mx.array([ids[:28]]), cache=cache)
mx.eval(o if not isinstance(o, tuple) else o[0])
with attention_phase("decode_verify"):
    y = model(mx.array([ids[28:32]]), cache=cache)
y = y[0] if isinstance(y, tuple) else y
path = os.path.expanduser(f"~/MoonShots/data/_flag_{LABEL}.dot")
mx.export_to_dot(path, y)
mx.eval(y)
txt = open(path).read()
os.remove(path)
labels = [re.sub(r"[\s(].*$", "", l) for l in re.findall(r'\{\s*\d+\s*\[label\s*=\s*"([^"]*)"', txt)]
kern = collections.Counter(l for l in labels if l not in FREE)
logits_sha = hashlib.sha1(np.array(y.astype(mx.float32)).tobytes()).hexdigest()[:12]
# 2) greedy decode, committed MTP depth 3
g8 = json.loads(open(os.path.expanduser("~/MoEPrefetch/data/gsm8k/test.jsonl")).readline())["question"]
PROMPTS = {"code": open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read(),
           "math": "Solve the problem step by step, briefly. End with a line of the form 'Answer: <number>'.\n\n" + g8,
           "prose": "Explain how a refrigerator works to a curious 10 year old, in about 150 words."}
greedy = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)
shas, tps = {}, {}
for name, text in PROMPTS.items():
    t = rt.tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    t = list(t if isinstance(t[0], int) else t["input_ids"])
    stamps = []
    cb = lambda tk: stamps.append((time.perf_counter(), len(tk) if isinstance(tk, list) else 1))
    fuse.reset_generation_state(model)
    res = G.generate_mtpk(rt, t, max_tokens=MAXTOK, sampler=greedy, speculative_depth=3, seed=0,
                          mtp_history_policy="committed", token_callback=cb)
    shas[name] = hashlib.sha1(str(list(res.tokens)).encode()).hexdigest()[:10]
    n = sum(k for _, k in stamps) - stamps[0][1]
    tps[name] = round(n / (stamps[-1][0] - stamps[0][0]), 2)
st.gate.check()
env = {k: v for k, v in os.environ.items() if (k.startswith("MTPLX_") and ("FUSED" in k or "OPDIET" in k or "GLUE" in k))
       or k.startswith("MLX_")}
print(json.dumps(dict(label=LABEL, env=env, kernels_total=sum(kern.values()), kernels_per_layer=round(sum(kern.values()) / 48, 1),
                      top=dict(kern.most_common(12)), logits_sha=logits_sha, shas=shas, tok_s=tps,
                      tok_s_mean=round(sum(tps.values()) / len(tps), 2), peak_gb=round(mx.get_peak_memory() / 1e9, 1))), flush=True)
