"""L1-15 check (Leap, Sat 2026-10-24): the 16-row verify pass under SUSTAINED load, stock vs the GDN kernel program.
In the DFlash loop the kernel pass looks ~3.4 ms slower than in short isolated bursts (test_pass16: 37.0 ms), while stock
matches its burst time; if sustained load lowers the GPU clock, compute-heavier kernels (NAX + dequant) slow more than
memory-bound ones. 240 passes per mode back to back (stock, kernels, stock, kernels), medians of the first 20 and last 60.
Usage: [SpecDecode venv] test_sustain.py  -> ~/Leap/data/test_sustain.txt"""
import json, os, sys, time
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np
import mlx.core as mx
mx.set_cache_limit(4 << 30)
sys.path.insert(0, os.path.expanduser("~/Leap/third_party/sd35"))
import dflash35_model_mlx as dfm
from mlx_lm import stream_generate as lm_stream
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.cache import make_prompt_cache, ArraysCache
MODEL = os.path.expanduser("~/.cache/huggingface/hub/models--Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed/snapshots/"
                           "95460516fbfe34c48f577f67c9ed191f5d61fbaf")
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
model, tok = dfm.load(MODEL)
mx.eval(model.parameters())
for l in open(os.path.expanduser("~/EvalSuite/cache/ref/tf_gen.jsonl")):
    r = json.loads(l)
    if r["key"].startswith("tf_code"):
        user = tok.decode(r["prompt_ids"]).split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
        ids = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, enable_thinking=False)
        ids = list(ids if isinstance(ids[0], int) else ids["input_ids"])
        break
out = [int(r.token) for r in lm_stream(model, tok, prompt=mx.array(ids), max_tokens=64, sampler=make_sampler(0.0))]
ctx, blk = ids + out[:48], out[48:64]
cache = make_prompt_cache(model)
mx.eval(model(mx.array([ctx[:-1]]), cache=cache))
dfm._patch_model(model, [1, 6, 11, 16, 22, 27, 32, 37])
snap = [list(c.cache) if isinstance(c, ArraysCache) else c.offset for c in cache]
def restore():
    for c, s in zip(cache, snap):
        if isinstance(c, ArraysCache):
            c.cache = list(s)
        else:
            c.offset = s
inp = mx.array([[ctx[-1]] + blk[:15]])
for gdnk in ("0", "1", "0", "1"):
    os.environ["DFL35_GDNK"] = gdnk
    cap = dfm._GDNStateCapture()
    ts = []
    try:
        t_start = time.perf_counter()
        for k in range(240):
            cap.clear()
            mx.synchronize(); t = time.perf_counter()
            lg = model(inp, cache=cache)
            mx.eval(mx.argmax(lg, -1), [x for x in model._hidden_states if x is not None])
            ts.append(time.perf_counter() - t)
            restore()
        wall = time.perf_counter() - t_start
    finally:
        cap.close()
    say(f"GDN kernels {'on ' if gdnk == '1' else 'off'}: first 20 passes {np.median(ts[:20]) * 1e3:5.2f} ms, last 60 "
        f"{np.median(ts[-60:]) * 1e3:5.2f} ms (240 passes, {wall:.1f} s)")
open(os.path.expanduser("~/Leap/data/test_sustain.txt"), "w").write("\n".join(lines) + "\n")
