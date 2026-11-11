"""Demand-miss split (orchestrator): per generated token, for plain greedy and spec125 k=3 (verify passes):
  on-time  - prefetched and ready when the layer was served (not a stall)
  late     - prefetched, read still in flight when the layer was served
  pred-unissued - in L+1's look-ahead set but never issued (L did not stall, cap, or no free slot)
  unpred   - not in the look-ahead set at all
(a) late -> more lead time; (b) unpred/unissued -> over-fetch (larger B into victim slots).
Usage: miss_split.py [frac=0.50] [N=256]   (env: reference + prefetch / victim settings under test)"""
import os, sys, time
for k, v in (("MTPLX_NGRAM_PREWARM", "off"), ("MTPLX_COMPILED_VERIFY", "0"), ("MTPLX_QWEN4EXP_COMPILE", "0"),
             ("MTPLX_COMPILE_AR_FORWARD", "0"), ("MTPLX_COMPILED_TARGET_PREFIX", "0"), ("MTPLX_FAMILY_CAPTURE_COMMIT", "1")):
    os.environ.setdefault(k, v)
sys.path.insert(0, os.path.expanduser("~/BigMoE")); sys.path.insert(0, os.path.expanduser("~/SpecDecode"))
import mlx.core as mx
from bigmoe.loader import load_streamed
from bigmoe.gated import ext
from bigmoe.decode import greedy
import spec125
frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.50
N = int(sys.argv[2]) if len(sys.argv) > 2 else 256
rt, st = load_streamed(os.path.expanduser("~/BigMoE/models/bare"), round(frac * 512))
ids = rt.tokenizer.apply_chat_template([{"role": "user", "content": open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read()}],
                                       add_generation_prompt=True, enable_thinking=False)
ids = ids if isinstance(ids[0], int) else ids["input_ids"]
spec125.generate(rt.model, ids, 16, k_mtp=3, max_draft=5, lookup=False)       # warm-up (first spec run is slow)
K = ("pf_ontime", "pf_late", "miss_pred_unissued", "miss_unpred", "stalled", "pf_issued", "pf_victims")
for mode in ("plain", "spec3"):
    stamps = []
    cb = lambda t: (stamps.append(time.perf_counter()), ext.reset_stats() if len(stamps) == 1 else None)
    if mode == "plain":
        greedy(rt.model, ids, N, on_token=cb); extra = ""
    else:
        _, sst = spec125.generate(rt.model, ids, N, k_mtp=3, max_draft=5, lookup=False, on_token=cb)
        extra = f", {sst['passes']} passes"
    s = ext.stats(); n = len(stamps) - 1
    dem = s["pf_late"] + s["miss_pred_unissued"] + s["miss_unpred"]
    print(f"{mode} ({n} tok{extra}): {n / (stamps[-1] - stamps[0]):.1f} tok/s | per token: " +
          " ".join(f"{k}={s[k] / n:.2f}" for k in K) +
          f" | demand misses {dem / n:.2f}: late {s['pf_late'] / max(dem, 1):.0%}, pred-unissued "
          f"{s['miss_pred_unissued'] / max(dem, 1):.0%}, unpred {s['miss_unpred'] / max(dem, 1):.0%}", flush=True)
st.gate.check()
