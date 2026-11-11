"""CPU-side profile of the streamed 125B decode (MTPLX's own cProfile lane, steps 8..24) plus the PLE n-gram
gather time per token. Prints the top functions by own time and cumulative time. -> data/profile125.txt"""
import os, sys, time, io, pstats
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
PATH = os.path.expanduser("~/BigMoE/data/profile125.pstats")
os.environ["MTPLX_AR_PROFILE_TOKENS"] = "16"; os.environ["MTPLX_AR_PROFILE_PATH"] = PATH
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
from bigmoe.loader import load_streamed
rt, st = load_streamed(os.path.expanduser("~/BigMoE/models/bare"), round(float(sys.argv[1] if len(sys.argv) > 1 else 0.55) * 512))
from mtplx.sampling import SamplerConfig
from mtplx import generation as G
from mtplx.models import qwen4_exp as Q
ids = list(rt.tokenizer.apply_chat_template([{"role": "user", "content": open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read()}],
                                            add_generation_prompt=True, enable_thinking=False))
ids = ids if isinstance(ids[0], int) else ids["input_ids"]
p0 = Q.ple_stage_seconds(); c0 = Q.ple_stage_calls(); t0 = time.perf_counter()
G.generate_ar(rt, ids, max_tokens=32, sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0))
dt = time.perf_counter() - t0
out = io.StringIO()
out.write(f"PLE n-gram gather: {(Q.ple_stage_seconds()-p0)/max(1,Q.ple_stage_calls()-c0)*1e3:.2f} ms per call over {Q.ple_stage_calls()-c0} calls (incl. prefill)\n")
s = pstats.Stats(PATH, stream=out); s.sort_stats("tottime").print_stats(14); s.sort_stats("cumulative").print_stats(14)
txt = out.getvalue()
open(os.path.expanduser("~/BigMoE/data/profile125.txt"), "w").write(txt)
print(txt[:6000])
