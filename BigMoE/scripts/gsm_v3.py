"""v3 reference validation: GSM8K problems 0..N-1 on BigMoE's engine as deployed (exact stack env from the runner, the
streamed 24 GB budget, BIGMOE_TIER=4433333333 or not), decoded with SpecDecode's spec125 adaptive width (spec == plain
greedy, checked in B5) with an EOS stop and a 600-token cap. Prompt and answer extraction = MoonShots'
gsm_planes125.py (same INSTR, regex, test.jsonl). Compares with the exact 4-bit answers (BigMoE eager 0-149, MoonShots
mix10 150-499) and MoonShots' plan4433333333 answers (tiered 1-row decode, masked 4-bit kernel). With
BIGMOE_TIER_NEARTIE=1 it also counts rows with a near tie at the 4/3-bit rank boundary.
Resumable: checkpoint data/v3/gsm_<TAG>.json after every problem; CHUNK problems per call.
usage: gsm_v3.py FRAC N   (env TAG, CHUNK)   GPU: one quiet lock per chunk"""
import json, os, re, sys, time
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
from bigmoe import tier
import spec125

FRAC, N = float(sys.argv[1]), int(sys.argv[2])
TAG = os.environ.get("TAG", ("v3_" + tier.PLAN) if tier.ON else "exact")
MD = os.path.expanduser("~/BigMoE/models/bare")
cfg = json.load(open(os.path.join(MD, "config.json"))); cfg = cfg.get("text_config", cfg)
C = max(16, round(FRAC * cfg["num_experts"]))
rt, st = load_streamed(MD, C, mtp=True)
tok = rt.tokenizer
stop = {i for i in (getattr(tok, "eos_token_id", None), tok.convert_tokens_to_ids("<|im_end|>") if hasattr(tok, "convert_tokens_to_ids") else None,
                    tok.convert_tokens_to_ids("<|endoftext|>") if hasattr(tok, "convert_tokens_to_ids") else None)
        if isinstance(i, int) and i >= 0}
V0 = {1: 42, 2: 59, 3: 74, 4: 87, 5: 94, 6: 107, 7: 117, 8: 123}


def chat_ids(text):
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    return list(ids if isinstance(ids[0], int) else ids["input_ids"])


probs = [json.loads(l) for l in open(os.path.expanduser("~/MoEPrefetch/data/gsm8k/test.jsonl"))][:N]
INSTR = "Solve the problem step by step, briefly. End with a line of the form 'Answer: <number>'.\n\n"
os.makedirs(os.path.expanduser("~/BigMoE/data/v3"), exist_ok=True)
ck = os.path.expanduser(f"~/BigMoE/data/v3/gsm_{TAG}.json")
state = json.load(open(ck)) if os.path.exists(ck) else {"answers": [], "ok": [], "ntok": [], "t": []}
budget = int(os.environ.get("CHUNK", "0")) or N
spec125.generate(rt.model, chat_ids("Hi"), 8, k_mtp=1, max_draft=7, lookup=1)   # warm-up
t0, done = time.perf_counter(), 0
for p in probs[len(state["answers"]):]:
    if done >= budget:
        break
    done += 1
    aw = spec125.AdaptiveWidth(V0, step_ms=2.2, alpha=2.0)
    ts = time.perf_counter()
    out, s = spec125.generate(rt.model, chat_ids(INSTR + p["question"]), 600, k_mtp=1, max_draft=7, lookup=1, stats={},
                              adapt=aw, stop_ids=stop)
    text = tok.decode([t for t in out if t not in stop])
    m = re.findall(r"Answer:\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)", text) or re.findall(r"(-?[\d,]*\.?\d+)", text)
    gold = p["answer"].split("####")[-1].strip().replace(",", "")
    a = m[-1].replace(",", "").rstrip(".") if m else None
    try:
        good = int(a is not None and abs(float(a) - float(gold)) < 1e-6)
    except ValueError:
        good = 0
    state["answers"].append(a); state["ok"].append(good); state["ntok"].append(len(out))
    state["t"].append(round(time.perf_counter() - ts, 2))
    json.dump(state, open(ck, "w"))
st.gate.check()
dt = time.perf_counter() - t0
n = len(state["answers"])
print(f"[{TAG}] C={C}: GSM8K {sum(state['ok'])}/{n} ({done} this call, {dt:.0f}s, "
      f"{sum(state['ntok'][-done:]) / max(dt, 1e-9) if done else 0:.1f} tok/s incl. prefill)", flush=True)
if tier.ON and tier._NEARTIE:
    near, rows = tier.neartie_summary()
    print(f"[{TAG}] near ties at the 4/3-bit rank boundary: {near}/{rows} layer-rows ({100 * near / max(rows, 1):.3f} %)", flush=True)
D = os.path.expanduser("~/MoonShots/data")
def load_ref(parts):
    ans, ok = [], []
    for f, off in parts:
        d = json.load(open(f)); assert d.get("offset", 0) == off, f
        ans += d["answers"]; ok += d["ok"]
    return ans, ok
refs = {"exact 4-bit": load_ref([(os.path.expanduser("~/BigMoE/gsmroot/data/w125_gsm_eager.json"), 0)]
                                + [(f"{D}/w125_gsm_planes_mix10_off{o}.json", o) for o in (150, 250, 450)]),
        "MoonShots plan4433333333": load_ref([(f"{D}/w125_gsm_planes_plan4433333333{s}.json", o)
                                              for s, o in (("", 0), ("_off150", 150), ("_off250", 250), ("_off450", 450))])}
for name, (ra, rk) in refs.items():
    m_ = min(n, len(ra))
    print(f"  vs {name} (first {m_}): answers equal {sum(x == y for x, y in zip(ra[:m_], state['answers']))}/{m_}; "
          f"correct {name} {sum(rk[:m_])} vs {TAG} {sum(state['ok'][:m_])}", flush=True)
