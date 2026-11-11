"""Queue L2-10 (Breakthrough, 2026-10-25; NewIdeas' question): is the prefill lane the REFERENCE behaviour?
MTPLX's generate_ar (the Eval suite's reference path, chunk 512) runs its prefill inside attention_phase("prefill"),
so past the 32K crossover the reference itself takes the sparse lane (metal selector + NAX flash); b12_prefill never
entered the phase, so its dense path may be the deviation. Cheap test at a ~4K prompt with both crossovers lowered to
2049 (MTPLX_QSA_PREFILL_MIN_CONTEXT, MTPLX_QSA_PREFILL_FLASH_MIN_CONTEXT), B12LL (every top-10 expert computed, so
residency cannot move the logits), one process, the same prompt (first PK_N tokens of ctx64k.txt):
  A0 generate_ar as the Eval suite runs it on B12: no MTPLX_SUSTAINED_PREFILL, so ONE forward over the whole prompt
     from history 0 (the lane never engages; dense [24, T, T] plane)
  A  generate_ar, MTPLX_SUSTAINED_PREFILL=1 + MTPLX_PREFILL_CHUNK_SIZE 512 (MTPLX's profile reference: lane past 2049)
  B  b12_prefill(512, 512) inside attention_phase("prefill"), greedy decode (MoE per 512 rows = A's MoE rows/groups)
  C  b12_prefill(512, 512) outside it (b12_prefill as it ships: dense)
  D  generate_ar with MTPLX_QSA_PREFILL=0 (the reference with the lane off)
  E  B with the opt-in flash-skip DECODE lane (MTPLX_QSA_FLASH=1; the suite never sets it, so dense-mask decode is the
     reference and flash-skip the change to verify)
Compared: the greedy tokens and the first token's top-20 log-probabilities (mtplx first_token_logprobs, same function
for all arms). A == B and C == D, A != D -> b12_prefill + lane reproduces the reference; the dense path is the deviation.
B vs E: is flash-skip decode bitwise equal to the dense-mask decode?
env: PK_N (4200), PK_TOK (16) -> data/pk64/lane_ref.txt"""
import hashlib, json, os, sys, time

os.environ.setdefault("B12_LATE3", "2,10")
os.environ.setdefault("B12_NORM", "mass")
os.environ["MTPLX_QSA_PREFILL_MIN_CONTEXT"] = "2049"
os.environ["MTPLX_QSA_PREFILL_FLASH_MIN_CONTEXT"] = "2049"
os.environ["MTPLX_PREFILL_CHUNK_SIZE"] = "512"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HOME = os.path.expanduser("~/Breakthrough")
N, NT = int(os.environ.get("LR_N", "4200")), int(os.environ.get("LR_TOK", "16"))   # (not PK_N: day_pk64b exports 65000)
import bt_b12 as B
rt, st = B.load_b12(0.10)
import mlx.core as mx
import mtplx.models.qwen4_exp as Q
from mtplx import generation as G
from mtplx.sampling import SamplerConfig
from mtplx.attention_context import attention_phase
from bigmoe_v3f.fuse import reset_generation_state
model, tok = rt.model, rt.tokenizer
ids = tok.encode(open(os.path.join(HOME, "data/prompts/ctx64k.txt")).read())[:N]
GREEDY = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)


def fresh():
    reset_generation_state(model)
    B.STASH.clear(); B.PENDH.clear()
    mx.clear_cache()
    Q._QSA_PREFILL_COUNTS.clear()


def arm_ar(lane_off, sustained=True):
    """generate_ar chunks its prefill only with MTPLX_SUSTAINED_PREFILL=1 (MTPLX's profiles set it; load_b12 and the
    Eval suite don't): without it the whole prompt is ONE forward from history 0, so the lane never engages and the
    dense plane is [24, T, T] (the 10-25 14:04 crash: 203 GB at T = 64,999)."""
    fresh()
    if lane_off:
        os.environ["MTPLX_QSA_PREFILL"] = "0"
    if sustained:
        os.environ["MTPLX_SUSTAINED_PREFILL"] = "1"
    before = G._runtime_counter_snapshot(rt)
    try:
        res = G.generate_ar(rt, list(ids), max_tokens=NT, sampler=GREEDY, first_token_logprobs_top_k=20)
    finally:
        os.environ.pop("MTPLX_QSA_PREFILL", None)
        os.environ.pop("MTPLX_SUSTAINED_PREFILL", None)
    tiers = dict(Q._QSA_PREFILL_COUNTS)
    tiers["prefill_chunks"] = G._runtime_counter_delta(rt, before).get("prefill_chunks")
    return [int(x) for x in res.tokens], res.first_token_logprobs, tiers


def arm_b12(lane, flash=False):
    fresh()
    cache = model.make_cache()
    with (attention_phase("prefill") if lane else __import__("contextlib").nullcontext()):
        B.b12_prefill(model, mx.array([ids[:-1]]), cache, 512, 512)
    if flash:                                                # E: the opt-in flash-skip decode lane (the suite never
        os.environ["MTPLX_QSA_FLASH"] = "1"                  # sets it: dense-mask decode is the reference)
    y, out, ftl = ids[-1], [], None
    try:
        for i in range(NT):
            lg = model(mx.array([[y]]), cache=cache)
            lg = (lg[0] if isinstance(lg, tuple) else lg)[0, -1]
            y = int(mx.argmax(lg).item())
            if i == 0:
                ftl = G.first_token_logprobs(lg, token_id=y, top_k=20)
            out.append(y)
    finally:
        os.environ.pop("MTPLX_QSA_FLASH", None)
    return out, ftl, dict(Q._QSA_PREFILL_COUNTS)


res = {}
for name, f in (("A0_ar_onespan", lambda: arm_ar(False, sustained=False)),
                ("A_ar", lambda: arm_ar(False)), ("B_b12_lane", lambda: arm_b12(True)),
                ("C_b12_dense", lambda: arm_b12(False)), ("D_ar_laneoff", lambda: arm_ar(True)),
                ("E_b12_lane_flashdec", lambda: arm_b12(True, flash=True))):
    t0 = time.perf_counter()
    toks, ftl, tiers = f()
    res[name] = dict(toks=toks, top=[(t, v) for t, v in (ftl.top if ftl else ())], tiers=tiers,
                     s=round(time.perf_counter() - t0, 1))
    print(f"lane_ref {name}: {res[name]['s']} s, tiers {tiers}, sha {hashlib.sha1(json.dumps(toks).encode()).hexdigest()[:10]}"
          f" | {tok.decode(toks)[:60]!r}", flush=True)
st.gate.check()


def cmp(a, b):
    ta, tb = res[a]["toks"], res[b]["toks"]
    n = min(len(ta), len(tb))
    same = next((i for i in range(n) if ta[i] != tb[i]), n)
    pa, pb = dict(res[a]["top"]), dict(res[b]["top"])
    common = set(pa) & set(pb)
    dmax = max((abs(pa[t] - pb[t]) for t in common), default=float("nan"))
    return f"{a} vs {b}: tokens equal for {same}/{n}, top-20 ids {'same' if set(pa) == set(pb) else 'DIFFER'}, max |dlogprob| {dmax:.2e}"


lines = [cmp("A_ar", "B_b12_lane"), cmp("C_b12_dense", "D_ar_laneoff"), cmp("A_ar", "D_ar_laneoff"), cmp("B_b12_lane", "C_b12_dense"),
         cmp("B_b12_lane", "E_b12_lane_flashdec"), cmp("A0_ar_onespan", "C_b12_dense"), cmp("A0_ar_onespan", "D_ar_laneoff")]
for l in lines:
    print("lane_ref " + l, flush=True)
with open(os.path.join(HOME, "data/pk64/lane_ref.txt"), "a") as f:
    f.write(time.strftime("%m-%d %H:%M ") + f"N {N}, crossovers 2049, B12LL, env TIER_PF_GROUP "
            f"{os.environ.get('BIGMOE_TIER_PF_GROUP', '-')}: " + " | ".join(lines) + "\n")
json.dump(res, open(os.path.join(HOME, "data/pk64/lane_ref.json"), "w"))
