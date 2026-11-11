"""L1-15 check (Leap, Sat 2026-10-24): the z-lab loop's GDN rollback replay after a 16-row verify pass, stock vs the GDN
kernel program (DFL35_GDNK=1), timed alone. In the loop the target + rollback window shrank only ~3 ms while the pass alone
shrank ~7 ms (small35_prof.txt vs test_pass16.txt), so the replay may cost more in kernel mode.
Replays for accepted = 3 / 7 / 11 (n = accepted + 1 tokens re-run), median of 15, caches restored each time.
Usage: [SpecDecode venv] test_rollback.py  -> ~/Leap/data/test_rollback.txt"""
import json, os, sys, time
os.environ["HF_HUB_OFFLINE"] = "1"
import mlx.core as mx
mx.set_cache_limit(4 << 30)
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_e5/Leap_third_party/sd35"))
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
dfm._patch_model(model, [1, 6, 11, 16, 22, 27, 32, 37])
inp = mx.array([[ctx[-1]] + blk[:15]])
for gdnk in ("0", "1"):
    os.environ["DFL35_GDNK"] = gdnk
    cache = make_prompt_cache(model)
    mx.eval(model(mx.array([ctx[:-1]]), cache=cache))
    cap = dfm._GDNStateCapture()
    try:
        cap.clear()
        lg = model(inp, cache=cache)
        mx.eval(lg, [x for x in model._hidden_states if x is not None], [c.state for c in cache])
        post = [(list(c.cache) if isinstance(c, ArraysCache) else c.offset) for c in cache]
        def restore():
            for c, s in zip(cache, post):
                if isinstance(c, ArraysCache):
                    c.cache = list(s)
                else:
                    c.offset = s
        res = []
        for acc in (3, 7, 11):
            ts = []
            for k in range(17):
                restore()
                mx.synchronize(); t = time.perf_counter()
                cap.rollback(cache, acc, 15 - acc)
                mx.eval([x for c in cache if isinstance(c, ArraysCache) for x in c.cache])
                if k >= 2:
                    ts.append(time.perf_counter() - t)
            res.append(f"accepted {acc:2d}: {sorted(ts)[7] * 1e3:5.2f} ms")
        say(f"rollback replay, GDN kernels {'on ' if gdnk == '1' else 'off'}: " + ";  ".join(res))
    finally:
        cap.close()
open(os.path.expanduser("~/Leap/data/test_rollback.txt"), "w").write("\n".join(lines) + "\n")
