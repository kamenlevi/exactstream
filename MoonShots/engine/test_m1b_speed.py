"""M1b speed: 4-row verify passes, the engine (own dense path + BigMoE's tier MoE block on its GPU gate) vs BigMoE's own
model, same env (BigMoE reference flags + 24 GB tier mode: records3 tier pools, 3c GLU with fused combine, OoO), same tokens,
fresh process each (identical cold expert cache). Teacher-forced chunks of 4 tokens: WARM chunks untimed, then timed passes.
Prints ms per pass (median / mean), SSD misses, and saves top-1 tokens for an agreement check.
usage: test_m1b_speed.py ref|eng [frac=0.25] [warm_tokens=32] [timed_tokens=64]   (GPU, ~22 GB; quiet lock)"""
import os, sys, time, json
ENV = dict(BIGMOE_DUPROW="1", BIGMOE_COMPILE_HC="1", BIGMOE_HC_VREAD="1", BIGMOE_ASYNC_EVERY="1", BIGMOE_PF_JIT="1",
           BIGMOE_PF_QD="4", BIGMOE_PREFETCH="12", BIGMOE_DUPROW_CHUNK="5", BIGMOE_MAXK="80", BIGMOE_SCOUT_BUNDLE="1",
           MTPLX_QWEN4_VERIFY_GLUE="1", BIGMOE_RECORDS="3", BIGMOE_TIER="4433333333", BIGMOE_TIER_GLU="3c",
           BIGMOE_TIER_COMBINE="1", BIGMOE_OOO="1", MTPLX_NGRAM_PREWARM="off", MTPLX_FAMILY_CAPTURE_COMMIT="1",
           MTPLX_DROP_EVENTS="1", MTPLX_COMPILED_VERIFY="0", MTPLX_QWEN4EXP_COMPILE="0", MTPLX_COMPILE_AR_FORWARD="0",
           MTPLX_COMPILED_TARGET_PREFIX="0", BIGMOE_GDN_EXACT="1")
if os.environ.get("M1B_ENV", "tier") != "exact":                # b13's v3 env: tier pools without BIGMOE_RECORDS=3
    ENV.pop("BIGMOE_RECORDS", None)
if os.environ.get("M1B_ENV") == "exact":                      # exact 24 GB: no tier plan (4-bit pools, records3 CPU rebuild)
    for k in ("BIGMOE_TIER", "BIGMOE_TIER_GLU", "BIGMOE_TIER_COMBINE", "BIGMOE_OOO"):
        ENV.pop(k)
for k, v in ENV.items():
    os.environ.setdefault(k, v)
S_ = int(os.environ.get("M1B_S", "4"))                          # rows per verify pass
MODE = sys.argv[1]
FRAC = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
WARM = int(sys.argv[3]) if len(sys.argv) > 3 else 32
NT = int(sys.argv[4]) if len(sys.argv) > 4 else 64
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.expanduser("~/BigMoE"))
import numpy as np
import mlx.core as mx
from bigmoe.loader import load_streamed
from bigmoe.gated import ext
from mtplx.attention_context import attention_phase

MD = os.path.expanduser("~/BigMoE/models/bare")
t0 = time.time()
rt, st = load_streamed(MD, round(FRAC * 512), mtp=False)
model = rt.model
ids = rt.tokenizer.encode(open(os.path.expanduser("~/MoonShots/FINDINGS.md")).read()[:12000])[:WARM + NT + 8]
if MODE == "eng":
    from rle.model import Engine
    MOE = os.environ.get("M1B_MOE", "bigmoe")
    eng = Engine(model, ext, moe_mode="bigmoe" if MOE == "oracle" else MOE)   # oracle: warm up through the gate first
    state = eng.new_state()
    fwd = lambda c: eng.forward(c, state)
else:
    cache = model.make_cache()
    fwd = lambda c: (lambda o: (o[0] if isinstance(o, tuple) else o)[0])(model(c, cache=cache))
print(f"{MODE}: loaded in {time.time() - t0:.0f} s", flush=True)
REPEAT = os.environ.get("M1B_REPEAT", "0") == "1"           # compute-only: time the SAME chunk again (all experts resident)
FIXED = os.environ.get("M1B_FIXED", "0") == "1"             # ... from a restored state: identical routing every pass (true all hits)
from mtplx.models.qwen4_exp import QSACache


def snap():
    caches = state["cache"] if MODE == "eng" else cache
    s_ = [("qsa", c.offset) if isinstance(c, QSACache) else ("arr", list(c.state)) for c in caches]
    g_ = {k: v for k, v in state.items() if k != "cache"} if MODE == "eng" else None
    return s_, g_


def restore(sv):
    s_, g_ = sv
    caches = state["cache"] if MODE == "eng" else cache
    for c, (kind, v) in zip(caches, s_):
        if kind == "qsa":
            c.trim(int(c.offset) - v)
        else:
            c.state = list(v)
    if g_ is not None:
        state.update(g_)
times, builds, tops = [], [], []
PROF = None
if os.environ.get("M1B_PROFILE") == "1":
    import cProfile, pstats
    PROF = cProfile.Profile()
with attention_phase("decode_verify"):
    SV = None
    for s in range(0, WARM + NT, S_):
        if FIXED and s == WARM - S_:
            SV = snap()                                         # state before the last warm chunk
        if FIXED and s >= WARM:
            restore(SV)
        c = mx.array([ids[s:s + S_]]) if not (REPEAT and s >= WARM) else mx.array([ids[WARM - S_:WARM]])
        if PROF is not None and s == WARM:
            PROF.enable()
        if MODE == "eng" and s == WARM and os.environ.get("M1B_MOE") == "oracle":
            eng.build_residency()
            eng.moe_mode = "oracle"
        a = time.perf_counter()
        lg = fwd(c)
        top = mx.argmax(lg, axis=-1)
        b = time.perf_counter()
        mx.eval(top)
        if s >= WARM:
            times.append((time.perf_counter() - a) * 1e3)
            builds.append((b - a) * 1e3)
        tops += np.array(top).tolist()
if PROF is not None:
    PROF.disable()
    ps = pstats.Stats(PROF)
    print(f"--- cProfile over {NT // 4} timed passes (cumulative)", flush=True)
    ps.sort_stats("cumulative").print_stats(28)
    print("--- (tottime)", flush=True)
    ps.sort_stats("tottime").print_stats(22)
if MODE == "eng" and getattr(eng, "miss", None):
    print(f"oracle path: misses in the timed passes {int(mx.stack(eng.miss).sum().item())} of {len(eng.miss) * 4 * 12} slot lookups", flush=True)
stats = ext.stats() if hasattr(ext, "stats") else {}
t = np.array(times)
print(f"{MODE}{('/' + os.environ.get('M1B_MOE', 'bigmoe')) if MODE == 'eng' else ''} frac {FRAC}{' REPEAT' + (' FIXED-STATE (true all hits)' if FIXED else ' (routing drifts)') if REPEAT else ''}: 4-row verify pass median {np.median(t):.1f} ms, mean {t.mean():.1f} ms (p10 {np.percentile(t, 10):.1f}, "
      f"p90 {np.percentile(t, 90):.1f}) over {len(t)} passes; graph build (until the last eval) median {np.median(builds):.1f} ms; ext stats {json.dumps(stats)[:300]}", flush=True)
out = os.path.expanduser(f"~/MoonShots/data/day15/m1b_tops_{MODE}{'_rep' if REPEAT else ''}.json")
json.dump(dict(tops=tops, times=times), open(out, "w"))
