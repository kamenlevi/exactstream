"""LC1 (in-process state leak at 32K): two generations in ONE process on the frozen prompt (data/lc1/ids_32768.json),
with an optional reset of MTPLX's long-lived QSA state between them (MoonShots' discriminator 2):
  plain2      plain -> plain                 (does plain alone drift?)
  reset       plain -> reset all -> spec3    (indexer cores + every functools.lru_cache in qwen4_exp / kernels.qsa_*)
  reset-core  plain -> reset cores -> spec3
  reset-lru   plain -> clear lru caches -> spec3
Fresh-process shas for the reference env: plain = spec3 = a67851027e (BigMoE LEDGER LC1).
Usage: lc1_reset.py MODE [N=128]   (reference env, quiet lock; ~2 x (prefill 32K + N tokens))"""
import os, sys, json, time, hashlib, functools, importlib, pkgutil
for k, v in (("MTPLX_NGRAM_PREWARM", "off"), ("MTPLX_COMPILED_VERIFY", "0"), ("MTPLX_QWEN4EXP_COMPILE", "0"),
             ("MTPLX_COMPILE_AR_FORWARD", "0"), ("MTPLX_COMPILED_TARGET_PREFIX", "0"), ("MTPLX_FAMILY_CAPTURE_COMMIT", "1")):
    os.environ.setdefault(k, v)
mode = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 128
sys.path.insert(0, os.path.expanduser("~/BigMoE")); sys.path.insert(0, os.path.expanduser("~/SpecDecode"))
import mlx.core as mx
from bigmoe.loader import load_streamed
from bigmoe.decode import greedy
import spec125
rt, st = load_streamed(os.path.expanduser("~/BigMoE/models/bare"), 256)
ids = json.load(open(os.path.expanduser("~/BigMoE/data/lc1/ids_32768.json")))
sha = lambda t: hashlib.sha1(str([int(x) for x in t]).encode()).hexdigest()[:10]


def reset_cores():
    n = 0
    for _, m in rt.model.named_modules():
        if "_compiled_indexer_core" in m.__dict__ or hasattr(type(m), "_get_compiled_indexer_core"):
            object.__setattr__(m, "_compiled_indexer_core", None)
            object.__setattr__(m, "_compiled_indexer_parameter_signature", None)
            n += 1
    return n


def reset_lru():
    import mtplx.models.qwen4_exp as Q, mtplx.kernels as K
    mods = [Q] + [importlib.import_module(f"mtplx.kernels.{x.name}") for x in pkgutil.iter_modules(K.__path__)
                  if x.name.startswith("qsa_") and f"mtplx.kernels.{x.name}" in sys.modules]
    n = 0
    for mod in mods:
        for v in vars(mod).values():
            if isinstance(v, functools._lru_cache_wrapper):
                v.cache_clear(); n += 1
    return n


def gen(m):
    t0 = time.perf_counter()
    out = greedy(rt.model, ids, N) if m == "plain" else spec125.generate(rt.model, ids, N, k_mtp=3, max_draft=5, lookup=False)[0]
    print(f"  {m}: sha {sha(out)} ({time.perf_counter() - t0:.0f}s)", flush=True)


print(f"T={len(ids)} mode {mode}", flush=True)
gen("plain")
if mode == "plain2":
    gen("plain")
else:
    if mode in ("reset", "reset-core"):
        print(f"  reset {reset_cores()} indexer cores", flush=True)
    if mode in ("reset", "reset-lru"):
        print(f"  cleared {reset_lru()} lru caches", flush=True)
    mx.clear_cache()
    gen("spec3")
st.gate.check()
