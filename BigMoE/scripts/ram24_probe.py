"""R-34 RAM24 (MoonShots for NewIdeas, Tue ~20:50): where the shipped v3 24 GB engine's memory goes, one code prompt.
Sizes as loaded (every parameter grouped by its name with layer indices folded), mx active / peak / cache memory and
the process's phys_footprint and resident size (rusage) after load, after warm-up, after a prefill (1-token generate),
at the spec decode peak (400 tokens, adaptive width as b33) and at the AR decode peak (bigmoe.decode.greedy, 128 tokens).
resident - phys_footprint ~ clean file-backed pages (the n-gram table's hot cache, mapped files), which the footprint
does not count. Run with the stack's env from the runner (b33 settings, no S10b trace). usage: ram24_probe.py OUT.json"""
import ctypes, json, os, re, sys, time
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for _k, _v in (("MTPLX_FAMILY_CAPTURE_COMMIT", "1"), ("MTPLX_DROP_EVENTS", "1"), ("MTPLX_LAZY_MTP_HISTORY_APPEND", "1"),
               ("MTPLX_DEFER_REPAIR_EVAL", "1")):
    os.environ.setdefault(_k, _v)
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
import mlx.core as mx
from mlx.utils import tree_flatten
from bigmoe.loader import load_streamed
from bigmoe.decode import greedy
import spec125

OUT = sys.argv[1]
res = {"env": {k: v for k, v in os.environ.items() if k.startswith(("BIGMOE_", "MTPLX_"))}, "mem": {}, "sizes_gb": {}}


class _RU(ctypes.Structure):                                   # rusage_info_v4 (as Fit12's fit_pf.py fp(): u7, u28)
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(35)]


def mem(tag):
    """the 24 GB line (NewIdeas 10-28) is the LIFETIME peak phys_footprint (ri_lifetime_max_phys_footprint, u28),
    the max over load + prefill + decode so far; u7 = phys_footprint now, u6 = resident"""
    ru = _RU()
    ctypes.CDLL(None).proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru))
    d = dict(active=mx.get_active_memory(), peak=mx.get_peak_memory(), cache=mx.get_cache_memory(),
             phys_footprint=ru.u7, lifetime_peak_footprint=ru.u28, resident=ru.u6)
    res["mem"][tag] = {k: round(v / 1e9, 3) for k, v in d.items()}
    print(f"[{tag}] " + ", ".join(f"{k} {v:.2f} GB" for k, v in res["mem"][tag].items()), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)


MD = os.path.expanduser("~/BigMoE/models/bare")
cfg = json.load(open(os.path.join(MD, "config.json"))); cfg = cfg.get("text_config", cfg)
rt, st = load_streamed(MD, max(16, round(0.25 * cfg["num_experts"])), mtp=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtphead                                          # R-60: BIGMOE_MTP_HEAD=<safetensors> overlays a trained MTP head
res["mtp_head"] = mtphead.from_env(rt)
tok = rt.tokenizer
mem("after load")
grp = {}
for name, a in tree_flatten(rt.model.parameters()):
    g = re.sub(r"\.\d+\.", ".*.", name)
    grp[g] = grp.get(g, 0) + a.nbytes
res["sizes_gb"] = {g: round(b / 1e9, 4) for g, b in sorted(grp.items(), key=lambda kv: -kv[1])}
res["sizes_total_gb"] = round(sum(grp.values()) / 1e9, 3)
res["tier_pools_gb"] = round(st.resident_gb(), 3) if hasattr(st, "resident_gb") else None
print(f"parameters as loaded {res['sizes_total_gb']:.2f} GB; tier pools {res['tier_pools_gb']} GB; largest groups:")
for g, b in list(res["sizes_gb"].items())[:30]:
    print(f"  {b:8.4f} GB  {g}")


def ids_of(t):
    i = tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, enable_thinking=False)
    return list(i if isinstance(i[0], int) else i["input_ids"])


V0 = {1: 42, 2: 59, 3: 74, 4: 87, 5: 94, 6: 107, 7: 117, 8: 123}
spec125.generate(rt.model, ids_of("Hi"), 8, k_mtp=1, max_draft=7, lookup=1)   # warm-up (as s10b_trace.py)
st.gate.check()
mem("after warm-up")
P = ids_of(open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read())
res["prompt_tokens"] = len(P)
mx.reset_peak_memory()
spec125.generate(rt.model, P, 1, k_mtp=1, max_draft=7, lookup=1)
st.gate.check()
mem("prefill (1-token generate), peak since reset")
mx.reset_peak_memory()
t0 = time.perf_counter()
out, s = spec125.generate(rt.model, P, 400, k_mtp=1, max_draft=7, lookup=1, adapt=spec125.AdaptiveWidth(V0, step_ms=2.2, alpha=2.0))
st.gate.check()
res["spec"] = dict(n_tok=len(out), tok_s=s["tok_s"], wall=time.perf_counter() - t0)
mem("spec decode 400 tok (incl. its prefill), peak since reset")
if os.environ.get("RAM24_MODE") != "spec":          # RAM24_MODE=spec: one mode per process (the 24 GB line's rule)
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    o2 = greedy(rt.model, P, 128)
    st.gate.check()
    res["ar"] = dict(n_tok=len(o2) if hasattr(o2, "__len__") else None, wall=time.perf_counter() - t0)
    mem("ar decode 128 tok (incl. its prefill), peak since reset")
print("saved", OUT, flush=True)
