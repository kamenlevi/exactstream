"""Speed prototype (round 18 #1, deterministic): decode tok/s of the 125B with plane-granular expert streaming
(planes_stream.py) vs BigMoE's 4-bit streamer at the same expert-cache bytes. BigMoE's code prompt, greedy, eager decode,
MTP off; a 32-token warm-up generation, then 128 tokens timed from the first decoded token. ext.stats() gives stalled
layers and misses (summed over virtual layers for planes: an MSB miss and an LSB miss count separately).
  bench_planes125.py planes PLANE_DIR CM CL R      (MSB slots, LSB slots, top-R ranks exact)
  bench_planes125.py bigmoe C                      (BigMoE's ExpertStreamer, C slots of whole 4-bit experts)
Same expert bytes: C * 2.77 MB = CM * 1.536 MB + CL * 1.38 MB per layer.   GPU, BigMoE-size memory: exclusive window."""
import hashlib, os, sys, time
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
os.environ["BIGMOE_PF_NOPUBLISH"] = "1"
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx

MD = os.path.expanduser("~/BigMoE/models/bare")
mode = sys.argv[1]
if mode == "planes":
    import planes_stream as PS
    pdir, Cm, Cl, R = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    t0 = time.perf_counter()
    rt, st = PS.load_planes(MD, pdir, Cm, Cl, R)
    cfg = f"planes Cm={Cm} Cl={Cl} R={R} ({(Cm * 1.536 + Cl * 1.38) / 1e3:.3f} GB/layer)"
else:
    from bigmoe.loader import load_streamed
    C = int(sys.argv[2])
    t0 = time.perf_counter()
    rt, st = load_streamed(MD, C, mtp=False)
    cfg = f"bigmoe C={C} ({C * 2.77 / 1e3:.3f} GB/layer)"
print(f"{cfg}: loaded in {time.perf_counter() - t0:.0f}s", flush=True)
from bigmoe.gated import ext
from bigmoe import fuse
from mtplx.sampling import SamplerConfig
from mtplx import generation as G
greedy = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)


def chat_ids(text):
    ids = rt.tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    return list(ids if isinstance(ids[0], int) else ids["input_ids"])


def run(ids, n):
    stamps = []

    def cb(toks):
        if not stamps:
            ext.reset_stats()                       # count decode only (the prefill is done by the first callback)
        stamps.append((time.perf_counter(), len(toks) if isinstance(toks, list) else 1))
    fuse.reset_generation_state(rt.model)
    res = G.generate_ar(rt, ids, max_tokens=n, sampler=greedy, token_callback=cb)
    toks = getattr(res, "tokens", None) or getattr(res, "token_ids", None) or (res.get("tokens") if isinstance(res, dict) else None)
    if toks is None and isinstance(res, tuple):
        toks = res[0]
    toks = list(toks or [])
    n_dec = max(1, sum(k for _, k in stamps) - stamps[0][1]) if stamps else 1
    dec = stamps[-1][0] - stamps[0][0] if len(stamps) > 1 else float("nan")
    return toks, n_dec / dec if dec == dec and dec > 0 else float("nan"), n_dec


ids = chat_ids(open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read())
run(ids, 32)                                         # warm-up (prefill path, caches, compiled kernels)
toks, tps, n = run(ids, 128)
s = ext.stats()
st.gate.check()
print(f"{cfg}: decode {tps:.1f} tok/s over {n} tokens, stalled layers/tok {s['stalled'] / n:.1f}, misses/tok "
      f"{s['misses'] / n:.1f}, uses/tok {s['uses'] / n:.1f}, sha {hashlib.sha1(str(toks).encode()).hexdigest()[:10]}, "
      f"peak {mx.get_peak_memory() / 1e9:.1f} GB, prompt {len(ids)} tokens", flush=True)
# stall decomposition: per decoded token, ms of loader I/O (pread), gate processing, CPU idle waiting for ids
ms = {k: 1e3 * v / n for k, v in s.items() if k.endswith("_s")}
print(f"{cfg}: per token ms: wall {1e3 / tps:.1f}, " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(ms.items()))
      + f"; per stalled layer ms: io {ms.get('io_s', 0) / max(s['stalled'] / n, 1e-9):.3f}; raw {dict(s)}", flush=True)
