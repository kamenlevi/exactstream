"""MTP acceptance per draft position on the 125B (Qwen3.8-Flash-Next) through MTPLX on BigMoE's exact streamed runtime
(acceptance is a property of the computation, not of streaming; C only changes speed). For three prompt kinds (code: BigMoE's
prompt_code.txt; math: GSM8K problem 0; prose: a short explanation request), depth 3 and depth 1, greedy and temperature 1.0
(oMLX reports 96.8 % at temperature 1.0 / top-p 1.0, adaptive depth <= 3, oQ4e), 128 new tokens, MTPLX server defaults.
Reports MTPLX's own accepted_by_depth / drafted_by_depth, mean accept probability by depth, and tokens per verify.
usage: mtp_accept125.py [FRAC]   (GPU, BigMoE-size memory: exclusive window)"""
import json, os, sys, time
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for _k, _v in (("MTPLX_FAMILY_CAPTURE_COMMIT", "1"), ("MTPLX_DROP_EVENTS", "1"), ("MTPLX_LAZY_MTP_HISTORY_APPEND", "1"),
               ("MTPLX_DEFER_REPAIR_EVAL", "1")):
    os.environ.setdefault(_k, _v)                 # as BigMoE's mtp_diag.py (MTPLX server defaults)
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
from bigmoe.loader import load_streamed
from bigmoe import fuse
frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
rt, st = load_streamed(os.path.expanduser("~/BigMoE/models/bare"), round(frac * 512))
from mtplx.sampling import SamplerConfig
from mtplx import generation as G

g8 = json.loads(open(os.path.expanduser("~/MoEPrefetch/data/gsm8k/test.jsonl")).readline())["question"]
PROMPTS = {
    "code": open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read(),
    "math": "Solve the problem step by step, briefly. End with a line of the form 'Answer: <number>'.\n\n" + g8,
    "prose": "Explain how a refrigerator works to a curious 10 year old, in about 150 words.",
}


def chat_ids(text):
    ids = rt.tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    return list(ids if isinstance(ids[0], int) else ids["input_ids"])


SAMPLERS = {"greedy": SamplerConfig(temperature=0.0, top_p=1.0, top_k=0), "t1.0": SamplerConfig(temperature=1.0, top_p=1.0, top_k=0)}
rows = []
for name, text in PROMPTS.items():
    ids = chat_ids(text)
    for sname, smp in SAMPLERS.items():
        for depth in (3, 1):
            fuse.reset_generation_state(rt.model)
            t0 = time.perf_counter()
            res = G.generate_mtpk(rt, ids, max_tokens=128, sampler=smp, speculative_depth=depth, seed=0)
            dt = time.perf_counter() - t0
            toks, S = list(res.tokens), res.stats                  # GenerationOutput.stats is MTPLX's GenerationStats
            acc, dr = list(S.accepted_by_depth), list(S.drafted_by_depth)
            rate = [round(a / d, 3) if d else None for a, d in zip(acc, dr)]
            mp = [None if p is None else round(p, 3) for p in (S.mean_accept_probability_by_depth or [])]
            tpv = len(toks) / max(1, S.verify_calls)
            row = dict(prompt=name, sampler=sname, depth=depth, tokens=len(toks), verify_calls=S.verify_calls,
                       tokens_per_verify=round(tpv, 2), accepted_by_depth=acc, drafted_by_depth=dr, accept_rate_by_depth=rate,
                       mean_accept_prob_by_depth=mp, bonus=S.bonus_tokens, correction=S.correction_tokens,
                       skipped=S.skipped_drafts, seconds=round(dt, 1))
            rows.append(row)
            print(json.dumps(row), flush=True)
st.gate.check()
json.dump(rows, open(os.path.expanduser("~/MoonShots/data/mtp_accept125.json"), "w"), indent=1)
