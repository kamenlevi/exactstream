"""MTP acceptance at long context on the 125B (round 21 follow-up): MTPLX's "cycle" (BigMoE's scripts until round 21)
vs "committed" history (the product setting) with ~4K-token prompts, where the draft head's own history should matter
most (oMLX quotes 96.8 % acceptance at 4K). Prompts: a ~4K-token Python module (MTPLX's sampling.py, read-only) with a
code task, and a ~4K-token prose document (MoonShots' PRIOR_ART.md) with a summary task. Depth 3, greedy, MAXTOK tokens.
usage: mtp_longctx125.py [FRAC] [MAXTOK] [CTX_TOKENS]   (GPU, BigMoE-size memory: exclusive window)"""
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
frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 192
CTX = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
MD = os.path.expanduser("~/BigMoE/models/bare")
C = round(frac * 512)
rt, st = load_streamed(MD, C)
from mtplx.sampling import SamplerConfig
from mtplx import generation as G
import mtplx
from bigmoe.gated import ext
tok = rt.tokenizer


def head_tokens(text, n):
    ids = tok.encode(text)
    return tok.decode(ids[:n])


src = open(os.path.join(os.path.dirname(mtplx.__file__), "sampling.py")).read()
doc = open(os.path.expanduser("~/MoonShots/PRIOR_ART.md")).read()
PROMPTS = {
    "code4k": "Here is a Python module:\n\n```python\n" + head_tokens(src, CTX) + "\n```\n\nWrite a new function "
              "`describe_sampler(cfg)` for this module that returns a one-line human-readable summary of a SamplerConfig, "
              "reusing the module's own field names. Then list three edge cases it must handle.",
    "prose4k": "Here is a research note:\n\n" + head_tokens(doc, CTX) + "\n\nSummarize the note in about 200 words for an "
               "engineer who has not read it, keeping its own terms and numbers.",
}
IDS = {}
for name, text in PROMPTS.items():
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    IDS[name] = list(ids if isinstance(ids[0], int) else ids["input_ids"])
greedy = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)
rows = []
for name, ids in IDS.items():
    for policy in ("cycle", "committed"):
        stamps = []

        def cb(toks):
            if not stamps:
                ext.reset_stats()
            stamps.append((time.perf_counter(), len(toks) if isinstance(toks, list) else 1))
        fuse.reset_generation_state(rt.model)
        t0 = time.perf_counter()
        res = G.generate_mtpk(rt, ids, max_tokens=MAXTOK, sampler=greedy, speculative_depth=3, seed=0,
                              mtp_history_policy=policy, token_callback=cb)
        S = res.stats
        acc, dr = list(S.accepted_by_depth), list(S.drafted_by_depth)
        cond = [round(acc[0] / dr[0], 3) if dr[0] else None] + [round(acc[i] / acc[i - 1], 3) if acc[i - 1] else None
                                                                for i in range(1, len(acc))]
        x = ext.stats()
        n_dec = max(1, sum(k for _, k in stamps) - stamps[0][1]) if stamps else 1
        dec = stamps[-1][0] - stamps[0][0] if len(stamps) > 1 else float("nan")
        row = dict(C=C, prompt=name, prompt_tokens=len(ids), policy=policy, tokens=len(res.tokens),
                   verify_calls=S.verify_calls, accepted_by_depth=acc, drafted_by_depth=dr, conditional_accept=cond,
                   mtp_tokens_per_round=round((sum(acc) + dr[0]) / max(1, dr[0]), 2),
                   tokens_per_verify=round(len(res.tokens) / max(1, S.verify_calls), 2),
                   decode_tok_s=round(n_dec / dec, 2) if dec == dec and dec > 0 else None,
                   misses_per_tok=round(x["misses"] / n_dec, 2), io_ms_per_tok=round(1e3 * x.get("io_s", 0.0) / n_dec, 2),
                   sha=hashlib.sha1(str(list(res.tokens)).encode()).hexdigest()[:10],
                   seconds_total=round(time.perf_counter() - t0, 1))
        rows.append(row)
        print(json.dumps(row), flush=True)
st.gate.check()
json.dump(rows, open(os.path.expanduser(f"~/MoonShots/data/mtp_longctx125_C{C}.json"), "w"), indent=1)
