"""R-52 SCRATCH24 screen (NewIdeas 10-29 21:19 order (3), after the 45 GB items; tier.py ce73009): BIGMOE_PF_REUSE=1 keeps
the tier prefill's double buffers (_prefill_grouped) across layer calls instead of a fresh mx.zeros pair + clear_cache per
call; =2 also skips the per-call clear_cache. R-52's audit put the 24 GB prefill's fresh scratch at ~18-35 GB per 2048-token
chunk (~4-8 % of TTFT). One process per arm (the 24 GB line is a per-process lifetime peak): the shipped 24 GB stack (the
runner's env, ext/build_m01), a warm-up, then TTFT of an ~8K-token prompt (2x) and of the code prompt (3x), each a 1-token
spec125.generate from a fresh cache, then a 400-token spec decode of the code prompt (tok/s, sha), and the lifetime peak
phys_footprint (rusage v4 u28). PASS (NewIdeas): every sha equal across arms AND 8K TTFT -5 % or better AND lifetime peak
within +0.1 GB. usage: [venv] BIGMOE_PF_REUSE=<0|1|2> r52_ttft.py OUT.json"""
import ctypes, hashlib, json, os, sys, time
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
ARM = int(os.environ.get("BIGMOE_PF_REUSE", "0") or 0)
res = {"arm": ARM, "env": {k: v for k, v in os.environ.items() if k.startswith(("BIGMOE_", "MTPLX_"))}}


class _RU(ctypes.Structure):                                   # rusage_info_v4: u7 = phys_footprint now, u28 = lifetime peak
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(35)]


def fp():
    ru = _RU()
    ctypes.CDLL(None).proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru))
    return ru.u7 / 1e9, ru.u28 / 1e9


def save():
    json.dump(res, open(OUT, "w"), indent=1)


MD = os.path.expanduser("~/BigMoE/models/bare")
cfg = json.load(open(os.path.join(MD, "config.json"))); cfg = cfg.get("text_config", cfg)
t0 = time.perf_counter()
rt, st = load_streamed(MD, max(16, round(0.25 * cfg["num_experts"])), mtp=True)
tok = rt.tokenizer
from bigmoe import tier as T
assert T._PF_REUSE == ARM, (T._PF_REUSE, ARM)
res["load_s"] = round(time.perf_counter() - t0, 1)


def ids_of(t):
    i = tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, enable_thinking=False)
    return list(i if isinstance(i[0], int) else i["input_ids"])


GEN = dict(k_mtp=1, max_draft=7, lookup=1)
spec125.generate(rt.model, ids_of("Hi"), 8, **GEN)                # warm-up (as ram24_probe / s10b_trace)
st.gate.check()
code = open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read()
P = ids_of(code)
parts, P8 = [code], P
while len(P8) < 8000:                                            # ~8K tokens: the code prompt's text repeated, one user turn
    parts.append(code)
    P8 = ids_of("\n\n".join(parts) + "\n\nSummarize the code above in one sentence.")
res["prompt_tokens"], res["prompt8k_tokens"] = len(P), len(P8)


def ttft(ids):
    mx.synchronize()
    t = time.perf_counter()
    out, _ = spec125.generate(rt.model, ids, 1, **GEN)
    st.gate.check()
    return time.perf_counter() - t, out


res["ttft8k_s"], res["ttft_s"] = [], []
for _ in range(2):
    s, o8 = ttft(P8)
    res["ttft8k_s"].append(round(s, 3))
    print(f"[arm {ARM}] TTFT {len(P8)} tokens: {s:.2f} s", flush=True); save()
for _ in range(3):
    s, o1 = ttft(P)
    res["ttft_s"].append(round(s, 3))
    print(f"[arm {ARM}] TTFT {len(P)} tokens: {s:.3f} s", flush=True); save()
res["first8k"], res["first"] = list(map(int, o8))[:2], list(map(int, o1))[:2]
V0 = {1: 42, 2: 59, 3: 74, 4: 87, 5: 94, 6: 107, 7: 117, 8: 123}
t = time.perf_counter()
out, s = spec125.generate(rt.model, P, 400, adapt=spec125.AdaptiveWidth(V0, step_ms=2.2, alpha=2.0), **GEN)
st.gate.check()
res["decode"] = dict(n_tok=len(out), tok_s=round(s["tok_s"], 2), wall=round(time.perf_counter() - t, 2),
                     sha=hashlib.sha1(json.dumps(list(map(int, out))).encode()).hexdigest()[:10])
res["fp_now_gb"], res["lifetime_peak_gb"] = (round(x, 3) for x in fp())
print(f"[arm {ARM}] decode {res['decode']}; footprint now {res['fp_now_gb']} GB, lifetime peak {res['lifetime_peak_gb']} GB",
      flush=True)
save()
