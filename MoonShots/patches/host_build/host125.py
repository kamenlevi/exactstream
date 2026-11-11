"""BigMoE: how much of the per-pass host graph build is exposed in the real 125B loop, and what ext.gate_glue saves.
usage (BigMoE's venv, quiet lock, ~3-4 min at N=48):  host125.py FRAC N DELAY_US MODE[,MODE]   e.g. 0.50 48 100 mtp3,ar2
For each mode (mtp3 = MTPLX generate_mtpk k=3, ar2 = bigmoe.decode.greedy) it runs 3 rounds of
  base | +DELAY_US busy-wait inside every streamed MoE call (= that much more graph build per layer) | BIGMOE_CGLUE path (if built)
and prints ms per trunk pass, the exposed fraction of host build (slope), the ms per pass saved by the C++ glue, and the
build-time split (streamed MoE glue vs SwitchGLU). Sha must be identical in every run. Read-only on BigMoE (imports only)."""
import hashlib, json, os, sys, time
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for _k, _v in (("MTPLX_FAMILY_CAPTURE_COMMIT", "1"), ("MTPLX_DROP_EVENTS", "1"), ("MTPLX_LAZY_MTP_HISTORY_APPEND", "1"),
               ("MTPLX_DEFER_REPAIR_EVAL", "1")):
    os.environ.setdefault(_k, _v)                     # as scripts/bench.py
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import mlx.core as mx
from bigmoe.loader import load_streamed
from bigmoe.gated import ext
import bigmoe.stream as S
frac, N, D, modes = float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]) * 1e-6, sys.argv[4].split(",")
MD = os.path.expanduser("~/BigMoE/models/bare")
cfg = json.load(open(os.path.join(MD, "config.json"))); cfg = cfg.get("text_config", cfg)
rt, st = load_streamed(MD, max(16, round(frac * cfg["num_experts"])), mtp=True)
from mtplx.sampling import SamplerConfig
from mtplx import generation as G
from bigmoe import decode as Dd
greedy = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)
ids = rt.tokenizer.apply_chat_template([{"role": "user", "content": open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read()}],
                                       add_generation_prompt=True, enable_thinking=False)
ids = list(ids if isinstance(ids[0], int) else ids["input_ids"])

T = dict(moe=0.0, swiglu=0.0, calls=0)
DELAY = [0.0]
Streamed = next(type(m) for m in rt.model.modules() if type(m).__name__.endswith("Streamed"))
Base = Streamed.__mro__[1]
s_call, b_call = Streamed.__call__, Base.__call__


def streamed_call(self, *a, **kw):
    t = time.perf_counter()
    if DELAY[0]:
        while time.perf_counter() - t < DELAY[0]: pass
    r = s_call(self, *a, **kw)
    T["moe"] += time.perf_counter() - t; T["calls"] += 1
    return r


def base_call(self, *a, **kw):
    t = time.perf_counter(); r = b_call(self, *a, **kw); T["swiglu"] += time.perf_counter() - t; return r


Streamed.__call__, Base.__call__ = streamed_call, base_call
NL = sum(1 for m in rt.model.modules() if type(m) is Streamed)


def run(mode, delay, cglue):
    S._CGLUE = cglue
    ext.reset_stats()
    for k in T: T[k] = 0
    stamps = []
    cb = lambda toks: stamps.append((time.perf_counter(), len(toks) if isinstance(toks, list) else 1))
    DELAY[0] = delay
    if mode == "ar2":
        res = {"tokens": Dd.greedy(rt.model, ids, N, on_token=lambda t: cb([t]))}
    else:
        res = G.generate_mtpk(rt, ids, max_tokens=N, sampler=greedy, speculative_depth=int(mode[3:]), token_callback=cb)
    DELAY[0] = 0.0
    toks = getattr(res, "tokens", None) or getattr(res, "token_ids", None) or (res.get("tokens") if isinstance(res, dict) else None)
    toks = list(toks or [])
    passes = ext.stats()["gated"] / NL                  # trunk passes (each serves every streamed layer once)
    dec = stamps[-1][0] - stamps[0][0]                  # decode only (first token's prefill excluded)
    p_dec = passes * (1 - 1 / max(1, len(stamps)))      # ~passes after the first token
    return dict(mode=mode, delay=delay * 1e6, cglue=cglue, sha=hashlib.sha1(str(toks).encode()).hexdigest()[:10],
                ms_pass=dec / max(1, p_dec) * 1e3, tok_s=(sum(n for _, n in stamps) - stamps[0][1]) / dec,
                moe_ms=T["moe"] / max(1, passes) * 1e3, swiglu_ms=T["swiglu"] / max(1, passes) * 1e3, calls=T["calls"] / max(1, passes))


has_glue = hasattr(ext, "gate_glue") and hasattr(S, "_CGLUE")
for mode in modes:
    run(mode, 0.0, False)                                # warm-up
    res = [run(mode, d, c) for _ in range(3) for d, c in ((0.0, False), (D, False)) + (((0.0, True),) if has_glue else ())]
    for r in res:
        print(f"{r['mode']} delay {r['delay']:3.0f} us cglue {int(r['cglue'])}: {r['tok_s']:5.1f} tok/s, {r['ms_pass']:6.2f} ms/pass, "
              f"sha {r['sha']} | build/pass: streamed MoE {r['moe_ms']:.2f} ms ({r['calls']:.0f} calls), SwitchGLU {r['swiglu_ms']:.2f}", flush=True)
    med = lambda f: sorted(r["ms_pass"] for r in res if f(r))[1]
    base, slow = med(lambda r: r["delay"] == 0 and not r["cglue"]), med(lambda r: r["delay"] > 0)
    added = res[0]["calls"] * D * 1e3
    line = f"{mode}: ms/pass {base:.2f}; +{added:.2f} ms build -> {slow:.2f}: exposed fraction {(slow - base) / added:.2f}"
    if has_glue:
        cg = med(lambda r: r["cglue"]); line += f"; C++ glue {cg:.2f} ({base - cg:+.2f} ms/pass)"
    print(line, flush=True)
