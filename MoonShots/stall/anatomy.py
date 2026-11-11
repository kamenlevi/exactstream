"""Stall anatomy (NewIdeas 2026-10-22): BigMoE's real streamed MTP decode (SpecDecode spec125, the deployed adaptive config:
k_mtp=1, max_draft=7, lookup=1, AdaptiveWidth(V0, step 2.2, alpha 2.0)) with a per-pass hook recording BigMoE ext counter
deltas (demand uses/misses, stalled layer-events, demand I/O seconds, loader processing, prefetch issued/used/on-time).
Writes one JSON per prompt: per-pass records (S, accepted, verify ms, misses, stalled, io ms, ...) for stall/report.py.
Env: exact 24 GB by default (BigMoE reference flags + MTP_STREAM + RECORDS=3, C = round(FRAC x 512)); STALL_TIER=1 adds the v3
tier flags (4433333333, 3c, fused combine, OoO).
usage: anatomy.py PROMPT [N=160] [FRAC=0.25]   (GPU, ~22 GB; quiet lock)"""
import ast, json, os, re, sys, time
ENV = dict(BIGMOE_DUPROW="1", BIGMOE_COMPILE_HC="1", BIGMOE_HC_VREAD="1", BIGMOE_ASYNC_EVERY="1", BIGMOE_PF_JIT="1",
           BIGMOE_PF_QD="4", BIGMOE_PREFETCH="12", BIGMOE_DUPROW_CHUNK="5", BIGMOE_MAXK="80", BIGMOE_FRSPEC="1",
           BIGMOE_SCOUT_BUNDLE="1", MTPLX_QWEN4_VERIFY_GLUE="1", BIGMOE_MTP_STREAM="1", BIGMOE_RECORDS="3",
           MTPLX_NGRAM_PREWARM="off", MTPLX_FAMILY_CAPTURE_COMMIT="1", MTPLX_DROP_EVENTS="1", MTPLX_LAZY_MTP_HISTORY_APPEND="1",
           MTPLX_DEFER_REPAIR_EVAL="1", MTPLX_COMPILED_VERIFY="0", MTPLX_QWEN4EXP_COMPILE="0", MTPLX_COMPILE_AR_FORWARD="0",
           MTPLX_COMPILED_TARGET_PREFIX="0", BIGMOE_GDN_EXACT="1")
if os.environ.get("STALL_TRACE") == "1":                    # ext per-layer trace: (L, pass, kind, t_seen, t_rel)
    ENV["BIGMOE_TRACE"] = "1"
if os.environ.get("STALL_TIER") == "1":                     # BigMoE b13's "v3" env: tiers + 3c + fused combine + OoO, no RECORDS=3
    ENV.pop("BIGMOE_RECORDS")
    ENV.update(BIGMOE_TIER="4433333333", BIGMOE_TIER_GLU="3c", BIGMOE_TIER_COMBINE="1", BIGMOE_OOO="1")
for k, v in ENV.items():
    os.environ.setdefault(k, v)
PROMPT = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 160
FRAC = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
import mlx.core as mx
from bigmoe.loader import load_streamed
from bigmoe.gated import ext
import spec125

PROMPTS = {                                                    # verbatim from SpecDecode's test_spec125.py
    "code": open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read(),
           "edit": "Here is a file:\n\n```python\n" + open(os.path.expanduser("~/SpecDecode/scripts/inventory_sample.py")).read()
                   + "```\n\nRewrite the whole file with total_value renamed to total everywhere. Output only the full file in one code block.",
           "prose": "Explain, in plain prose for a curious 14-year-old, how a CPU cache works and why it makes programs faster. "
                    "Use one everyday analogy and keep it to about 300 words."}
MD = os.path.expanduser("~/BigMoE/models/bare")
C = max(16, round(FRAC * 512))
t0 = time.time()
rt, st = load_streamed(MD, C, mtp=True)
ids = rt.tokenizer.apply_chat_template([{"role": "user", "content": PROMPTS[PROMPT]}], add_generation_prompt=True,
                                       enable_thinking=False)
ids = list(ids if isinstance(ids[0], int) else ids["input_ids"])
KEYS = ("uses", "misses", "stalled", "gated", "io_s", "cpu_wait_s", "proc_s", "slow_calls", "slow_s", "pf_issued", "pf_used",
        "pf_ontime", "pf_late", "pf_waited_s", "miss_unpred", "miss_pred_unissued")
last = {}


def hook():
    s = ext.stats()
    d = {k: s[k] - last.get(k, 0) for k in KEYS}
    last.update({k: s[k] for k in KEYS})
    return {"x_" + k: v for k, v in d.items()}


spec125.generate(rt.model, ids, 16, k_mtp=3, max_draft=5, lookup=1)                    # warm-up, as SpecDecode's harness
st.gate.check()
V0 = {1: 42, 2: 59, 3: 74, 4: 87, 5: 94, 6: 107, 7: 117, 8: 123}
aw = spec125.AdaptiveWidth(V0, step_ms=2.2, alpha=2.0)
hook()
if os.environ.get("STALL_TRACE") == "1":
    ext.trace()                                             # clear the warm-up's events
stats = {"trace": [], "pass_hook": hook}
t1 = time.perf_counter()
out, s = spec125.generate(rt.model, ids, N, k_mtp=1, max_draft=7, lookup=1, stats=stats, adapt=aw)
wall = time.perf_counter() - t1
st.gate.check()
recs = [{k: (v if isinstance(v, (int, float)) else None) for k, v in r.items() if k in ("S", "n", "t_verify") or k.startswith("x_")}
        for r in stats["trace"]]
summ = {k: s[k] for k in ("passes", "tok_s", "tok_per_pass", "t_verify", "t_draft", "t_build", "t_wait", "t_commit", "t_head",
                          "t_dwait", "lookup_used") if k in s}
res = dict(prompt=PROMPT, N=N, frac=FRAC, C=C, tier=os.environ.get("STALL_TIER") == "1", wall_s=wall, summary=summ, passes=recs,
           prompt_tokens=len(ids), ext_trace=ext.trace() if os.environ.get("STALL_TRACE") == "1" else None)
tag = f"{PROMPT}_{'tier' if res['tier'] else 'exact'}_f{FRAC}" + ("_trace" if os.environ.get("STALL_TRACE") == "1" else "")
os.makedirs(os.path.expanduser("~/MoonShots/data/stall"), exist_ok=True)
json.dump(res, open(os.path.expanduser(f"~/MoonShots/data/stall/anatomy_{tag}.json"), "w"))
print(f"[{tag}] {s['tok_s']:.1f} tok/s, {s['tok_per_pass']:.2f} tok/pass, {s['passes']} passes; verify {s['t_verify']:.2f} s, "
      f"draft {s['t_draft']:.2f} s; misses {sum(r['x_misses'] for r in recs)}, stalled {sum(r['x_stalled'] for r in recs)}, "
      f"io {sum(r['x_io_s'] for r in recs):.2f} s (loaded {time.time() - t0:.0f} s)", flush=True)
