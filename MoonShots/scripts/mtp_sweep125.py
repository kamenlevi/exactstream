"""MTP sweeps on the 125B over BigMoE's streamed runtime with MTPLX's product history setting ("committed", round 21):
decode tok/s, acceptance by draft position and the loader's work per token, for given draft depths at one expert-cache
size (C = FRAC x 512 slots per layer; C=128 is 17 GB of experts, the 24 GB-total target). Depth 0 = eager decode (MTP
off, generate_ar). Prompts: BigMoE's code prompt, GSM8K #0, a prose request; greedy; MAXTOK tokens each.
Night jobs (round 22): the depth sweep at 17 GB (depths 0-6), and the speed-vs-memory curve (depth 3 and eager at
C = 96 .. 256). Utilization per Kamen's rule: SSD GB/s (misses x 2.77 MB / time) and the loader-read share of wall time.
usage: mtp_sweep125.py FRAC DEPTHS [MAXTOK]   e.g. mtp_sweep125.py 0.25 0,1,2,3,4,5,6 192
(GPU, BigMoE-size memory: exclusive window)"""
import hashlib, json, os, sys, time
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for _k, _v in (("MTPLX_FAMILY_CAPTURE_COMMIT", "1"), ("MTPLX_DROP_EVENTS", "1"), ("MTPLX_LAZY_MTP_HISTORY_APPEND", "1"),
               ("MTPLX_DEFER_REPAIR_EVAL", "1")):
    os.environ.setdefault(_k, _v)
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import mlx.core as mx
from bigmoe.loader import load_streamed
from bigmoe import fuse
frac = float(sys.argv[1])
DEPTHS = [int(x) for x in sys.argv[2].split(",")]
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 192
MD = os.path.expanduser("~/BigMoE/models/bare")
C = round(frac * 512)
t0 = time.perf_counter()
rt, st = load_streamed(MD, C)
print(f"C={C} ({C * 2.77 * 48 / 1e3:.0f} GB of experts) loaded in {time.perf_counter() - t0:.0f}s", flush=True)
from mtplx.sampling import SamplerConfig
from mtplx import generation as G
from bigmoe.gated import ext
g8 = json.loads(open(os.path.expanduser("~/MoEPrefetch/data/gsm8k/test.jsonl")).readline())["question"]
PROMPTS = {
    "code": open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read(),
    "math": "Solve the problem step by step, briefly. End with a line of the form 'Answer: <number>'.\n\n" + g8,
    "prose": "Explain how a refrigerator works to a curious 10 year old, in about 150 words.",
}
if os.environ.get("PROMPTS") == "6":                     # night runs: three more kinds (explainer, 2nd math, code edit)
    import mtplx
    _src = "".join(open(os.path.join(os.path.dirname(mtplx.__file__), "sampling.py")).readlines()[:60])
    _g8b = json.loads(open(os.path.expanduser("~/MoEPrefetch/data/gsm8k/test.jsonl")).readlines()[1])["question"]
    PROMPTS.update({
        "tcp": "Explain in detail how TCP congestion control works: slow start, congestion avoidance, fast retransmit and "
               "fast recovery, and how CUBIC and BBR differ. Use headings and a short example.",
        "math2": "Solve the problem step by step, briefly. End with a line of the form 'Answer: <number>'.\n\n" + _g8b,
        "edit": "Add type hints and one-line docstrings to every function in this Python code, and return the full "
                "edited code only:\n\n```python\n" + _src + "```",
    })
IDS = {}
for name, text in PROMPTS.items():
    ids = rt.tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    IDS[name] = list(ids if isinstance(ids[0], int) else ids["input_ids"])
greedy = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)
rows = []
for depth in DEPTHS:
    for name, ids in IDS.items():
        stamps = []

        def cb(toks):
            if not stamps:
                ext.reset_stats()                   # decode only: the prefill is done by the first callback
            stamps.append((time.perf_counter(), len(toks) if isinstance(toks, list) else 1))
        fuse.reset_generation_state(rt.model)
        if depth == 0:
            res = G.generate_ar(rt, ids, max_tokens=MAXTOK, sampler=greedy, token_callback=cb)
        else:
            res = G.generate_mtpk(rt, ids, max_tokens=MAXTOK, sampler=greedy, speculative_depth=depth, seed=0,
                                  mtp_history_policy="committed", token_callback=cb)
        x = ext.stats()
        toks = list(getattr(res, "tokens", None) or [])
        S = getattr(res, "stats", None)
        acc = list(S.accepted_by_depth) if depth and S else []
        dr = list(S.drafted_by_depth) if depth and S else []
        n_dec = max(1, sum(k for _, k in stamps) - stamps[0][1]) if stamps else 1
        dec = stamps[-1][0] - stamps[0][0] if len(stamps) > 1 else float("nan")
        tps = n_dec / dec if dec == dec and dec > 0 else float("nan")
        row = dict(C=C, depth=depth, prompt=name, tokens=len(toks), decode_tok_s=round(tps, 2),
                   verify_calls=getattr(S, "verify_calls", None) if depth else None,
                   accepted_by_depth=acc, drafted_by_depth=dr,
                   misses_per_tok=round(x["misses"] / n_dec, 2), stalled_per_tok=round(x["stalled"] / n_dec, 2),
                   io_ms_per_tok=round(1e3 * x.get("io_s", 0.0) / n_dec, 2),
                   sha=hashlib.sha1(str(toks).encode()).hexdigest()[:10])
        rows.append(row)
        print(json.dumps(row), flush=True)
st.gate.check()
out = os.path.expanduser(f"~/MoonShots/data/mtp_sweep125_C{C}{os.environ.get('TAG', '')}.json")
json.dump(rows, open(out, "w"), indent=1)
print(f"\nC={C} ({C * 2.77 * 48 / 1e3:.0f} GB), committed history, greedy, {MAXTOK} tokens x {len(IDS)} prompts, pooled:")
print("depth  cond. accept by position              MTP tok/round  tok/s   loader-read share  SSD GB/s  stalled layers/tok")
for depth in DEPTHS:
    rs = [r for r in rows if r["depth"] == depth]
    tps = sum(r["decode_tok_s"] for r in rs) / len(rs)
    ms = 1e3 / tps
    io = sum(r["io_ms_per_tok"] for r in rs) / len(rs)
    miss = sum(r["misses_per_tok"] for r in rs) / len(rs)
    stl = sum(r["stalled_per_tok"] for r in rs) / len(rs)
    if depth:
        acc = [sum(r["accepted_by_depth"][i] for r in rs) for i in range(depth)]
        dr = [sum(r["drafted_by_depth"][i] for r in rs) for i in range(depth)]
        cond = [acc[0] / max(1, dr[0])] + [acc[i] / acc[i - 1] if acc[i - 1] else 0 for i in range(1, depth)]
        tpr = (sum(acc) + dr[0]) / max(1, dr[0])
        cs = " ".join(f"{c:.2f}" for c in cond)
    else:
        cs, tpr = "(eager)", 1.0
    print(f"{depth:5d}  {cs:40s} {tpr:5.2f}         {tps:5.1f}   {io / ms:5.0%}              {miss * 2.77 / ms:5.2f}     {stl:5.1f}",
          flush=True)
