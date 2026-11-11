"""Fit12 R-42 GROWFIX (Wed 28 Oct 2026; NewIdeas QUEUE R-42 + 18:43 message; PLAN.md 'R-42', written before any run): can a
trim between prompts remove the 12 GB setup's per-prompt memory creep (GROW, Tue: ~0.2-0.3 GB per 32K prompt outside MLX's
active memory, levelling off after 4-5 prompts)? Breakthrough's bt_b12 at its 31b63af defaults (engine v3m, B12_FOLD=1
fold-exact, B12_PFR_GRID=keep), imported read-only and not patched, driven as its spec CLI does: per prompt
reset_generation_state + STASH / PENDH cleared, b12_prefill_state(model, ids, 512, 4096), spec125.generate(k_mtp 3,
max_draft 3, lookup 0). Environment = fit_cmp.py's MTP-path set without the CLI-only BIGMOE_ASYNC_EVERY and without
B12_PFR=0. Prompts F12_PROMPTS (default c32k_1..c32k_5, conf_prompts.py) in one process, F12_NDEC (32) tokens each.
F12_TRIM=1: after each prompt mx.clear_cache() and libSystem's malloc_zone_pressure_relief(NULL, 0). Per prompt: token sha,
the prompt's peak footprint (sampled every 20 ms), and after it (after the trim if on) footprint, MLX active / cache and the
lifetime peak (rusage u28). Usage: [BigMoE venv] fit_gfix.py TAG"""
import ctypes, gc, hashlib, os, sys, threading, time

ENV = {"MTPLX_PREFILL_CHUNK_SIZE": "512", "BIGMOE_TIER_PF_GROUP": "64", "BIGMOE_RESET_GEN": "0", "BIGMOE_PF_GROUP": "64",
       "B12_LATE3": "2,10", "B12_NORM": "mass", "BIGMOE_MTP_STREAM": "1", "BIGMOE_MTP_C": "64", "BIGMOE_MAXK": "80",
       "BIGMOE_FRSPEC": "1", "MTPLX_FAMILY_CAPTURE_COMMIT": "1", "MTPLX_DROP_EVENTS": "1", "MTPLX_LAZY_MTP_HISTORY_APPEND": "1",
       "MTPLX_DEFER_REPAIR_EVAL": "1", "BIGMOE_LOOP_GUARD": "0", "B12_CACHE_MB": "64"}
for k, v in ENV.items():
    os.environ.setdefault(k, v)
TAG = sys.argv[1] if len(sys.argv) > 1 else "gfix"
ND = int(os.environ.get("F12_NDEC", "32"))
PR = os.environ.get("F12_PROMPTS", "c32k_1,c32k_2,c32k_3,c32k_4,c32k_5").split(",")
TRIM = os.environ.get("F12_TRIM") == "1"


class _RU(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(40)]


_LIBC = ctypes.CDLL(None)
_LIBC.malloc_zone_pressure_relief.restype = ctypes.c_size_t
_LIBC.malloc_zone_pressure_relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]


def fp():                                            # (now, lifetime peak) physical footprint, GB
    ru = _RU()
    return (ru.u7 / 1e9, ru.u28 / 1e9) if _LIBC.proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru)) == 0 else (float("nan"),) * 2


PEAK = [0.0]


def _sampler():
    while True:
        PEAK[0] = max(PEAK[0], fp()[0])
        time.sleep(0.02)


threading.Thread(target=_sampler, daemon=True).start()
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Breakthrough/scripts"))
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
sys.path.insert(2, os.path.dirname(os.path.abspath(__file__)))
import bt_b12 as B
rt, st = B.load_b12(0.10, mtp=True)
import spec125
import conf_prompts as CP
from bigmoe_v3f.fuse import reset_generation_state
model, tok = rt.model, rt.tokenizer
print(f"[{TAG}] engine {B.V3F}, B12_FOLD {int(B.FOLD[0])}, PFR grid {os.environ.get('B12_PFR_GRID', 'keep')}, C_m {B.CFG['cm']} "
      f"C_l {B.CFG['cl']}, trim {int(TRIM)}; after load: footprint {fp()[0]:.3f} GB, MLX active "
      f"{mx.get_active_memory() / 1e9:.3f} GB", flush=True)


def chat_ids(text):
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    return list(ids if isinstance(ids[0], int) else ids["input_ids"])


for pos, P in enumerate(PR, 1):
    ids = chat_ids(CP.conf_text(P))
    PEAK[0] = fp()[0]
    t0 = time.perf_counter()
    reset_generation_state(model); B.STASH.clear(); B.PENDH.clear()
    state = B.b12_prefill_state(model, ids, 512, 4096)
    out, s = spec125.generate(model, ids, ND, k_mtp=3, max_draft=3, lookup=0, prefill_chunk=512, prefilled=state, stats={})
    sha = hashlib.sha1(str(list(out)).encode()).hexdigest()[:10]      # fit_cmp.py's sha (c32k_1 x 32 was 22451f8c02 on Tue)
    del state
    gc.collect()
    pk, rel = PEAK[0], -1
    if TRIM:
        mx.clear_cache()
        rel = _LIBC.malloc_zone_pressure_relief(None, 0)
    now, life = fp()
    print(f"[{TAG}] {P} RESULT pos {pos} | n {len(ids)} | sha{len(out)} {sha} | prompt peak {pk:.3f} GB | after: footprint "
          f"{now:.3f} GB, MLX active {mx.get_active_memory() / 1e9:.3f} GB, cache {mx.get_cache_memory() / 1e9:.3f} GB, lifetime "
          f"peak {life:.3f} GB | relief {rel / 1e9:.3f} GB | {time.perf_counter() - t0:.0f} s", flush=True)
print(f"[{TAG}] done", flush=True)
