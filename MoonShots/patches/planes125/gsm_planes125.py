"""GSM8K on the 125B with nested bit-plane experts (planes125.py) over BigMoE's streamed runtime (its package, read-only).
Same load (C = 0.50 x 512 slots, MTP off, exact gate), chat template, greedy decoding, 600-token cap, prompt and answer
extraction as BigMoE's gsmroot/scripts/waittop125.py "exact" mode, which produced BigMoE's eager and v2a GSM8K-150 answers.
R >= 10 runs the unpatched model (a sanity arm). Checkpoints every problem to ~/MoonShots/data/w125_gsm_planes_mix<R>.json
(resumable; CHUNK problems per call; OFFSET > 0 adds _off<OFFSET> to the name), then compares with BigMoE's eager and v2a
answers at OFFSET=0 (v2a vs eager = the noise arm), or with our unpatched mix10 arm at the same offset.
usage: gsm_planes125.py R N   (env CHUNK, OFFSET)   GPU, BigMoE-size memory: exclusive quiet window"""
import json, os, re, sys, time
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
os.environ["BIGMOE_SPARE"] = "0"                         # waittop125 exact mode
os.environ["BIGMOE_PF_NOPUBLISH"] = "1"
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import mlx.core as mx
from bigmoe.loader import load_streamed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import planes125

ARG, N = sys.argv[1], int(sys.argv[2])     # ARG: R (top-R exact, rest MSB-2 packed) or a plan of bits per rank, e.g. 4333322222
PLAN = ARG if len(ARG) > 2 else None
R = 0 if PLAN else int(ARG)
TAG = f"plan{PLAN}" if PLAN else f"mix{R}"
MD = os.path.expanduser("~/BigMoE/models/bare")
cfg = json.load(open(os.path.join(MD, "config.json"))); cfg = cfg.get("text_config", cfg)
E, K = cfg["num_experts"], cfg["num_experts_per_tok"]
C = max(16, round(0.50 * E))
rt, st = load_streamed(MD, C, mtp=False)
if PLAN or R < K:
    base = planes125.install(R, rt.model if hasattr(rt, "model") else rt, plan=PLAN)
    print(f"planes125: {TAG} ({PLAN or 'top-%d exact, rest MSB-2' % R}; patched {base.__name__})", flush=True)

from mtplx.sampling import SamplerConfig
from mtplx import generation as G
greedy = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)


def chat_ids(text):
    ids = rt.tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    return list(ids if isinstance(ids[0], int) else ids["input_ids"])


def run(ids, n):
    res = G.generate_ar(rt, ids, max_tokens=n, sampler=greedy)
    toks = getattr(res, "tokens", None) or getattr(res, "token_ids", None) or (res.get("tokens") if isinstance(res, dict) else None)
    if toks is None and isinstance(res, tuple):
        toks = res[0]
    return list(toks or [])


OFF = int(os.environ.get("OFFSET", "0"))
probs = [json.loads(l) for l in open(os.path.expanduser("~/MoEPrefetch/data/gsm8k/test.jsonl"))][OFF:OFF + N]
INSTR = "Solve the problem step by step, briefly. End with a line of the form 'Answer: <number>'.\n\n"
SUF = "" if OFF == 0 else f"_off{OFF}"                    # offset-specific checkpoints (OFFSET=0 keeps the old name)
ck = os.path.expanduser(f"~/MoonShots/data/w125_gsm_planes_{TAG}{SUF}.json")
state = json.load(open(ck)) if os.path.exists(ck) else {"offset": OFF, "answers": [], "ok": []}
assert state.get("offset", 0) == OFF, f"{ck} holds offset {state.get('offset')}, not {OFF}"
ok, answers, t0 = state["ok"], state["answers"], time.perf_counter()
budget = int(os.environ.get("CHUNK", "0")) or len(probs)
done_now = ntok = 0
for p in probs[len(answers):]:
    if done_now >= budget:
        break
    done_now += 1
    toks = run(chat_ids(INSTR + p["question"]), 600)
    ntok += len(toks)
    text = rt.tokenizer.decode(toks)
    m = re.findall(r"Answer:\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)", text) or re.findall(r"(-?[\d,]*\.?\d+)", text)
    gold = p["answer"].split("####")[-1].strip().replace(",", "")
    answers.append(m[-1].replace(",", "").rstrip(".") if m else None)
    try:
        ok.append(int(bool(m) and abs(float(m[-1].replace(",", "").rstrip(".")) - float(gold)) < 1e-6))
    except ValueError:
        ok.append(0)
    json.dump({"offset": OFF, "answers": answers, "ok": ok}, open(ck, "w"))
st.gate.check()
dt = time.perf_counter() - t0
print(f"125B C={C} {TAG}: GSM8K {sum(ok)}/{len(ok)} of {len(probs)} target ({dt:.0f}s for {done_now} problems, "
      f"{ntok / max(dt, 1e-9):.1f} tok/s incl. prefill; mixed calls {planes125.STATS['mixed']}) per-problem {''.join(map(str, ok))}", flush=True)
n = len(answers)
if OFF == 0:                                              # BigMoE's eager / v2a answers cover problems 0-149 only
    ref = {f"BigMoE {t}": json.load(open(os.path.expanduser(f"~/BigMoE/gsmroot/data/w125_gsm_{t}.json"))) for t in ("eager", "v2a")}
else:                                                     # our own unpatched 4-bit arm at the same offset
    rp = os.path.expanduser(f"~/MoonShots/data/w125_gsm_planes_mix10{SUF}.json")
    ref = {"4-bit ref (mix10)": json.load(open(rp))} if os.path.exists(rp) and TAG != "mix10" else {}
for t, d in ref.items():
    assert d.get("offset", 0) == OFF
    m_ = min(n, len(d["answers"]))
    a = d["answers"][:m_]
    print(f"  vs {t}: answers equal {sum(x == y for x, y in zip(a, answers))}/{m_}; correct {t} {sum(d['ok'][:m_])} vs {TAG} "
          f"{sum(ok[:m_])}; ref-only {sum(1 for x, y in zip(d['ok'][:m_], ok) if x and not y)}, {TAG}-only "
          f"{sum(1 for x, y in zip(d['ok'][:m_], ok) if y and not x)}", flush=True)
if OFF == 0:
    e, v = ref["BigMoE eager"], ref["BigMoE v2a"]
    print(f"  noise arm (BigMoE v2a vs eager, first {n}): answers equal {sum(x == y for x, y in zip(e['answers'][:n], v['answers'][:n]))}/{n}")
