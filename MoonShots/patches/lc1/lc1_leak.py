"""LC1 leak hunt (in-process state): does a recycled, never-written buffer carry the difference?
usage (BigMoE reference env + LC1 flags; prompt frozen by lc1_fresh.py at the same T; quiet lock):
  lc1_leak.py clear [T] [N]          plain, mx.clear_cache(), spec3   -> spec3 == fresh sha?  (yes = recycled-buffer class)
  lc1_leak.py noclear [T] [N]        plain, spec3 (control; known to differ at 32K)
  lc1_leak.py poison-nan-spec3 [T] [N]   one generation after filling the free-buffer cache with NaN; try T=4096 first:
      if a kernel reads unwritten memory, this fails deterministically in ONE generation (also try poison-ff-plain)."""
import os, sys, time, glob, hashlib, json
for k, v in (("MTPLX_NGRAM_PREWARM", "off"), ("MTPLX_COMPILED_VERIFY", "0"), ("MTPLX_QWEN4EXP_COMPILE", "0"),
             ("MTPLX_COMPILE_AR_FORWARD", "0"), ("MTPLX_COMPILED_TARGET_PREFIX", "0"), ("MTPLX_FAMILY_CAPTURE_COMMIT", "1")):
    os.environ.setdefault(k, v)                       # as scripts/longctx.py
mode = sys.argv[1]; T = int(sys.argv[2]) if len(sys.argv) > 2 else 32768; N = int(sys.argv[3]) if len(sys.argv) > 3 else 128
OUT = os.environ.get("LC1_OUT", os.path.expanduser("~/MoonShots/data/lc1"))
IDS = f"{OUT}/ids_{T}.json"                       # the frozen prompt from lc1_fresh.py (same T)
sys.path.insert(0, os.path.expanduser("~/BigMoE")); sys.path.insert(0, os.path.expanduser("~/SpecDecode"))
import mlx.core as mx
import numpy as np
from bigmoe.loader import load_streamed
from bigmoe.decode import greedy
import spec125
rt, st = load_streamed(os.path.expanduser("~/BigMoE/models/bare"), round(float(os.environ.get("LC1_FRAC", "0.50")) * 512))
ids = json.load(open(IDS))
sha = lambda t: hashlib.sha1(str(list(t)).encode()).hexdigest()[:10]


def gen(m):
    t0 = time.perf_counter()
    if m == "plain":
        out = greedy(rt.model, ids, N)
    else:
        out, _ = spec125.generate(rt.model, ids, N, k_mtp=3, max_draft=5, lookup=False)
    out = [int(t) for t in out]
    print(f"  {m}: sha {sha(out)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    return out


def poison(pattern):
    """Fill MLX's free-buffer cache with buffers of every size class holding NaN (or 0xFF) bytes, then free them: any
    kernel that reads memory it never wrote now sees poison instead of the zero pages a fresh process gets."""
    word = {"nan": 0x7FC00000, "ff": 0xFFFFFFFF, "big": 0x7F7FFFFF}[pattern]
    keep = []
    budget = int(os.environ.get("LC1_POISON_MB", "900")) << 20
    total = 0
    for size in [4096 << i for i in range(17)]:      # 4 KB .. 256 MB, two of each while the budget lasts
        for _ in range(2):
            if total + size <= budget:
                a = mx.full((size // 4,), word, dtype=mx.uint32); mx.eval(a); keep.append(a); total += size
    del keep                                        # back to MLX's cache, contents intact
    print(f"  poisoned {total >> 20} MB of free buffers with {pattern}", flush=True)


print(f"T={len(ids)} mode {mode}", flush=True)
if mode == "clear":            # plain, then mx.clear_cache(), then spec3: matches the fresh sha if recycled buffers carry it
    gen("plain"); mx.clear_cache(); gen("spec3")
elif mode == "noclear":        # control: plain then spec3 (longctx.py's order)
    gen("plain"); gen("spec3")
elif mode.startswith("poison"):  # poison-<pattern>-<mode>: one generation on a poisoned allocator, e.g. poison-nan-spec3
    _, pat, m = mode.split("-"); poison(pat); gen(m)
