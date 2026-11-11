"""Round 28 smoke test: the 125B with and without the 8-bit HC-mixer + PLE patch (hc8.py), one load at C=256:
greedy eager decode, 96 tokens on BigMoE's code prompt (tokens compared: first divergence), then decode tok/s with
committed MTP depth 3 over 192 tokens (code, math, prose).   usage: hc8_smoke.py   (GPU, 42 GB, quiet lock)"""
import hashlib, json, os, sys, time
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for _k, _v in (("MTPLX_FAMILY_CAPTURE_COMMIT", "1"), ("MTPLX_DROP_EVENTS", "1"), ("MTPLX_LAZY_MTP_HISTORY_APPEND", "1"),
               ("MTPLX_DEFER_REPAIR_EVAL", "1")):
    os.environ.setdefault(_k, _v)
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
from bigmoe.loader import load_streamed
from bigmoe import fuse
import hc8
rt, st = load_streamed(os.path.expanduser("~/BigMoE/models/bare"), 256)
from mtplx.sampling import SamplerConfig
from mtplx import generation as G
greedy = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)
g8 = json.loads(open(os.path.expanduser("~/MoEPrefetch/data/gsm8k/test.jsonl")).readline())["question"]
PROMPTS = {"code": open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read(),
           "math": "Solve the problem step by step, briefly. End with a line of the form 'Answer: <number>'.\n\n" + g8,
           "prose": "Explain how a refrigerator works to a curious 10 year old, in about 150 words."}
IDS = {}
for n, t in PROMPTS.items():
    ids = rt.tokenizer.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, enable_thinking=False)
    IDS[n] = list(ids if isinstance(ids[0], int) else ids["input_ids"])


def arm():
    fuse.reset_generation_state(rt.model)
    toks = list(G.generate_ar(rt, IDS["code"], max_tokens=96, sampler=greedy).tokens)
    tps = []
    for n, ids in IDS.items():
        stamps = []
        cb = lambda t: stamps.append((time.perf_counter(), len(t) if isinstance(t, list) else 1))
        fuse.reset_generation_state(rt.model)
        G.generate_mtpk(rt, ids, max_tokens=192, sampler=greedy, speculative_depth=3, seed=0,
                        mtp_history_policy="committed", token_callback=cb)
        k = sum(c for _, c in stamps) - stamps[0][1]
        tps.append(k / (stamps[-1][0] - stamps[0][0]))
    return toks, tps


ta, pa = arm()
n, mb0, mb1 = hc8.install(rt.model)
print(f"hc8: {n} modules swapped, {mb0:.0f} MB bf16 -> {mb1:.0f} MB 8-bit", flush=True)
tb, pb = arm()
st.gate.check()
div = next((i for i, (a, b) in enumerate(zip(ta, tb)) if a != b), None)
print(f"eager greedy 96 tokens: bf16 sha {hashlib.sha1(str(ta).encode()).hexdigest()[:10]}, hc8 sha "
      f"{hashlib.sha1(str(tb).encode()).hexdigest()[:10]}, first divergence at token {div}", flush=True)
print("committed MTP depth 3 decode tok/s (code, math, prose): bf16 " + " ".join(f"{x:.1f}" for x in pa)
      + " | hc8 " + " ".join(f"{x:.1f}" for x in pb) + f" | mean {sum(pb) / sum(pa) - 1:+.1%}", flush=True)
print("hc8 text:", repr(rt.tokenizer.decode(tb)[:300]), flush=True)
