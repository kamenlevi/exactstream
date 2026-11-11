"""R-38 PF24 step 1 probe (NewIdeas Tue ~21:25): where spec125's prefill peak (24.26 GB vs greedy's 22.16, ram24_probe)
comes from, on the code prompt, shipped v3 stack. Peak MLX memory (reset before each) of:
  spec_as_is   spec125._prefill: trunk chunk with return_hidden, then the MTP head over the chunk, evaluated as ONE graph
  eval_first   the same, but the trunk's hidden is evaluated (and the cache cleared) before the MTP head runs
  trunk_only   spec125._prefill without the MTP head
  greedy_like  the trunk chunk without return_hidden (as decode.greedy), states evaluated
and whether eval_first is exact: head logits, head hidden and every cache state bitwise equal to spec_as_is.
usage: r38_probe.py OUT.json  (the stack's env from the runner)"""
import json, os, sys, time
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for _k, _v in (("MTPLX_FAMILY_CAPTURE_COMMIT", "1"), ("MTPLX_DROP_EVENTS", "1"), ("MTPLX_LAZY_MTP_HISTORY_APPEND", "1"),
               ("MTPLX_DEFER_REPAIR_EVAL", "1")):
    os.environ.setdefault(_k, _v)
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
import mlx.core as mx
from bigmoe.loader import load_streamed
import spec125

OUT = sys.argv[1]
MD = os.path.expanduser("~/BigMoE/models/bare")
cfg = json.load(open(os.path.join(MD, "config.json"))); cfg = cfg.get("text_config", cfg)
rt, st = load_streamed(MD, max(16, round(0.25 * cfg["num_experts"])), mtp=True)
tok, model = rt.tokenizer, rt.model
P = list(tok.apply_chat_template([{"role": "user", "content": open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read()}],
                                 add_generation_prompt=True, enable_thinking=False))
P = list(P if isinstance(P[0], int) else P["input_ids"])
spec125.generate(model, list(tok.encode("Hi")), 8, k_mtp=1, max_draft=7, lookup=1)   # warm-up
st.gate.check()
res = {"prompt_tokens": len(P), "peak_gb": {}}


def prefill(eval_first, with_head=True, hidden=True):
    """spec125._prefill's chunk loop (one chunk here), with the evaluation order switchable"""
    cache = model.make_cache()
    head = spec125._MTPHead(model) if with_head else None
    n = len(P) - 1
    for i in range(0, n, 2048):
        j = min(i + 2048, n)
        if hidden:
            _, hid = spec125._split(spec125._prefill_call(model, P[i:j], cache, False))
            if eval_first:
                mx.eval(hid); mx.clear_cache()
            if head is not None:
                head.append(hid, P[i + 1:j + 1])
                mx.eval(head.logits)
            mx.eval(hid)
        else:
            model(mx.array([P[i:j]]), cache=cache, emit_logits=False)
            mx.eval([c.state for c in cache if hasattr(c, "state")])
        mx.clear_cache()
    return cache, head


def measure(tag, **kw):
    mx.clear_cache(); mx.reset_peak_memory()
    t0 = time.perf_counter()
    out = prefill(**kw)
    st.gate.check()
    res["peak_gb"][tag] = round(mx.get_peak_memory() / 1e9, 3)
    print(f"[{tag}] peak {res['peak_gb'][tag]:.2f} GB, active {mx.get_active_memory() / 1e9:.2f} GB, "
          f"{time.perf_counter() - t0:.1f} s", flush=True)
    return out


def arrays(cache, head):
    from mlx.utils import tree_flatten
    flat = lambda cs: [v for c in cs for _, v in tree_flatten(getattr(c, "state", None) or []) if isinstance(v, mx.array)]
    return flat(cache) + ([head.logits, head.h] + flat(head.cache) if head else [])


c1, h1 = measure("spec_as_is", eval_first=False)
a1 = [mx.array(x) for x in arrays(c1, h1)]; mx.eval(a1); del c1, h1
c2, h2 = measure("eval_first", eval_first=True)
a2 = arrays(c2, h2)
res["eval_first_exact"] = bool(len(a1) == len(a2) and all(x.shape == y.shape and mx.array_equal(x, y).item() for x, y in zip(a1, a2)))
res["n_arrays_compared"] = len(a1)
print(f"eval_first vs spec_as_is: {'BITWISE EQUAL' if res['eval_first_exact'] else 'DIFFERENT'} over {len(a1)} arrays", flush=True)
del c2, h2, a1, a2
measure("trunk_only", eval_first=False, with_head=False)
measure("greedy_like", eval_first=False, with_head=False, hidden=False)
json.dump(res, open(OUT, "w"), indent=1)
print("saved", OUT, flush=True)
