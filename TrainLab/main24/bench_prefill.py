"""TrainLab main24 (NewIdeas 05:45): BigMoE's exact 24 GB main track, prefill cost per prompt size, patched vs not.
Same setup as ~/SpecDecode/scripts/test_spec125.py (BigMoE's reference driver), but the bigmoe package comes from this
copy (~/TrainLab/main24/bigmoe: D1i's bitwise prefill parts behind env switches; ~/BigMoE untouched). For each prompt:
one decode.greedy(prompt, N) (the ar2 reference path, tokens -> sha); time to the first output token = prefill + one
decode step. A 16-token warm-up generate runs first (as test_spec125).
Prompts: prose (~53 tokens), code (~327, BigMoE's prompt_code.txt), edit (copy-heavy rewrite), f2k / f4k (SpecDecode
FINDINGS.md cut to ~2K / ~4K tokens + a summary request).
usage: [BigMoE env] bench_prefill.py FRAC N PROMPT[,PROMPT...]"""
import hashlib, json, os, sys, time

os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for _k, _v in (("MTPLX_FAMILY_CAPTURE_COMMIT", "1"), ("MTPLX_DROP_EVENTS", "1"), ("MTPLX_LAZY_MTP_HISTORY_APPEND", "1"),
               ("MTPLX_DEFER_REPAIR_EVAL", "1")):
    os.environ.setdefault(_k, _v)
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
sys.path.insert(0, os.path.expanduser("~/TrainLab/" + os.environ.get("TL_PKG", "main24")))   # a patched copy of bigmoe
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
import ctypes, resource
import mlx.core as mx  # noqa: E402
from bigmoe.loader import load_streamed  # noqa: E402
from bigmoe import decode as D  # noqa: E402
from bigmoe import tier as T  # noqa: E402

MD = os.path.expanduser(os.environ.get("BIGMOE_MD", "~/BigMoE/models/bare"))
frac, N = float(sys.argv[1]), int(sys.argv[2])
names = sys.argv[3].split(",")
F = open(os.path.expanduser("~/SpecDecode/FINDINGS.md")).read()
PROMPTS = {
    "code": open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read(),
    "edit": "Here is a file:\n\n```python\n" + open(os.path.expanduser("~/SpecDecode/scripts/inventory_sample.py")).read()
            + "```\n\nRewrite the whole file with total_value renamed to total everywhere. Output only the full file in one code block.",
    "prose": "Explain, in plain prose for a curious 14-year-old, how a CPU cache works and why it makes programs faster. "
             "Use one everyday analogy and keep it to about 300 words.",
    "f2k": F[:7600] + "\n\nSummarize the key results above in 5 bullet points.",
    "f4k": F[:15500] + "\n\nSummarize the key results above in 5 bullet points.",
}
tc = json.load(open(os.path.join(MD, "config.json"))); tc = tc.get("text_config", tc)
C = max(16, round(frac * tc["num_experts"]))
rt, st = load_streamed(MD, C, mtp=True)
sha = lambda t: hashlib.sha1(str(list(t)).encode()).hexdigest()[:10]


class _RU(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(40)]


def footprint():
    ru = _RU()
    return ru.u28 / 1e9 if ctypes.CDLL(None).proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru)) == 0 else float("nan")


def ids_of(text):
    ids = rt.tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                           enable_thinking=False)
    return list(ids if isinstance(ids[0], int) else ids["input_ids"])


D.greedy(rt.model, ids_of(PROMPTS["prose"]), 16)                  # warm-up (as test_spec125)
st.gate.check()
tag = " ".join(f"{k}={os.environ[k]}" for k in ("TL_PKG", "BIGMOE_TIER_PF_GROUP", "BIGMOE_TIER_PF_RESIDENT", "BIGMOE_TIER_PF_GPU", "D1I_RB2", "D1I_NOLOGITS",
                                               "BIGMOE_PF_NOLOGITS")
               if os.environ.get(k)) or "unpatched"
for name in names:
    ids = ids_of(PROMPTS[name])
    PFD = getattr(T, "PF", {})                                     # the minimal patch keeps no counters
    for k in ("grouped", "single", "fallback", "groups"):
        PFD[k] = 0
    for k in ("t_read", "t_sync", "t_moe"):
        PFD[k] = 0.0
    for k in ("res_cold", "res_msb", "res_both"):
        PFD[k] = 0
    stamps = []
    t0 = time.perf_counter()
    out = D.greedy(rt.model, ids, N, on_token=lambda t: stamps.append(time.perf_counter()))
    st.gate.check()
    ttft, dec = stamps[0] - t0, (len(out) - 1) / (stamps[-1] - stamps[0])
    pf = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in PFD.items()
          if k in ("grouped", "single", "fallback", "groups", "t_read", "t_sync", "t_moe", "res_cold", "res_msb", "res_both")}
    print(f"[{name}] C={C} {tag}: {len(ids)} prompt tokens, time to first token {ttft:.2f} s "
          f"({(len(ids) - 1) / ttft:.0f} tok/s), decode {dec:.1f} tok/s, sha {sha(out)} | footprint {footprint():.1f} GB, "
          f"peak active {mx.get_peak_memory() / 1e9:.1f} GB | pf {pf}", flush=True)
