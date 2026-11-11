"""Where does spec125 add ~2 GB of MLX peak over plain decode (24 GB goal)? Peak after load, after spec125's prefill,
and over its decode, separately; then plain greedy for comparison. Usage: mem_probe.py [frac=0.25] [N=64]"""
import os, sys
for k, v in (("MTPLX_NGRAM_PREWARM", "off"), ("MTPLX_COMPILED_VERIFY", "0"), ("MTPLX_QWEN4EXP_COMPILE", "0"),
             ("MTPLX_COMPILE_AR_FORWARD", "0"), ("MTPLX_COMPILED_TARGET_PREFIX", "0"), ("MTPLX_FAMILY_CAPTURE_COMMIT", "1")):
    os.environ.setdefault(k, v)
sys.path.insert(0, os.path.expanduser("~/BigMoE")); sys.path.insert(0, os.path.expanduser("~/SpecDecode"))
import mlx.core as mx
from bigmoe.loader import load_streamed
from bigmoe.decode import greedy
import spec125
frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
N = int(sys.argv[2]) if len(sys.argv) > 2 else 64
rt, st = load_streamed(os.path.expanduser("~/BigMoE/models/bare"), round(frac * 512))
GB = lambda b: f"{b / 1e9:.2f} GB"
print("after load: active", GB(mx.get_active_memory()), "cache", GB(mx.get_cache_memory()), "peak", GB(mx.get_peak_memory()), flush=True)
ids = rt.tokenizer.apply_chat_template([{"role": "user", "content": open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read()}],
                                       add_generation_prompt=True, enable_thinking=False)
ids = ids if isinstance(ids[0], int) else ids["input_ids"]
for name, fn in (("greedy", lambda: greedy(rt.model, ids, N)),
                 ("spec125 k=3", lambda: spec125.generate(rt.model, ids, N, k_mtp=3, max_draft=5, lookup=False))):
    mx.clear_cache(); mx.reset_peak_memory()
    seen = {"first": None}
    def cb(t, _s=seen):
        if _s["first"] is None:
            _s["first"] = mx.get_peak_memory(); mx.reset_peak_memory()
    if name == "greedy":
        greedy(rt.model, ids, N, on_token=cb)
    else:
        spec125.generate(rt.model, ids, N, k_mtp=3, max_draft=5, lookup=False, on_token=cb)
    print(f"{name}: peak through prefill + first token {GB(seen['first'] or 0)}, peak over decode {GB(mx.get_peak_memory())}, "
          f"active after {GB(mx.get_active_memory())}, cache {GB(mx.get_cache_memory())}", flush=True)
st.gate.check()
