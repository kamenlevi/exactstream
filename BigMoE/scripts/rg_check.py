"""Repetition-guard check on the main engine (NewIdeas GAPS.md G): the 14 rows the shipped rule (R0) flags in the
replay of the plan4433333333 run (10 long math, 4 IFEval incl. ifeval/1219), decoded with spec125 adaptive (= greedy)
under guard off (math rows), R0 (shipped rule) and R1 (LaTeX math masked). Per row: tokens, finish, guard arms/hits,
and whether R0 / R1 change the tokens vs off. Eval suite items (prompt_ids, max_tokens); EOS stop.
usage: rg_check.py FRAC   (env as the v3 stack; writes data/rg/rg_check.json)"""
import json, os, sys, time
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
from bigmoe import repguard
import spec125

FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
MD = os.path.expanduser("~/BigMoE/models/bare")
cfg = json.load(open(os.path.join(MD, "config.json"))); cfg = cfg.get("text_config", cfg)
rt, st = load_streamed(MD, max(16, round(FRAC * cfg["num_experts"])), mtp=True)
tok = rt.tokenizer
stop = {i for i in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>"))
        if isinstance(i, int) and i >= 0}
rep = json.load(open(os.path.expanduser(
    "~/NewIdeas/data/rg/replay_plan4433333333_floor_w2f_r10_b12_noguard_plan4433332222.json")))["plan4433333333"]
keys = [r["key"] for r in rep if r["R0"][1] is not None]
items = {}
for f in ("dec_math.jsonl", "dec_ifeval.jsonl"):
    for l in open(os.path.expanduser(f"~/EvalSuite/data/items/{f}")):
        d = json.loads(l)
        if d["key"] in keys:
            items[d["key"]] = d
V0 = {1: 42, 2: 59, 3: 74, 4: 87, 5: 94, 6: 107, 7: 117, 8: 123}
spec125.generate(rt.model, items[keys[0]]["prompt_ids"][:32], 8, k_mtp=1, max_draft=7, lookup=1)   # warm-up
out_p = os.path.expanduser("~/BigMoE/data/rg/rg_check.json")
os.makedirs(os.path.dirname(out_p), exist_ok=True)
res = json.load(open(out_p)) if os.path.exists(out_p) else {}
ARMS = {"off": {"BIGMOE_LOOP_GUARD": "0"}, "R0": {"BIGMOE_LOOP_GUARD": "1", "BIGMOE_LOOP_GUARD_MATH": "0"},
        "R1": {"BIGMOE_LOOP_GUARD": "1", "BIGMOE_LOOP_GUARD_MATH": "1"}}
for key in keys:
    it = items[key]
    for arm, env in ARMS.items():
        if arm == "off" and it["set"] != "math":
            continue
        if f"{key}|{arm}" in res:
            continue
        os.environ.update(env)
        s0 = dict(repguard.STATS)
        t0 = time.perf_counter()
        out, s = spec125.generate(rt.model, it["prompt_ids"], it["max_tokens"], k_mtp=1, max_draft=7, lookup=1, stats={},
                                  adapt=spec125.AdaptiveWidth(V0, step_ms=2.2, alpha=2.0), stop_ids=stop)
        dt = time.perf_counter() - t0
        res[f"{key}|{arm}"] = dict(n=len(out), finish="eos" if out and out[-1] in stop else "length", toks=out,
                                   armed=repguard.STATS["armed"] - s0["armed"],
                                   penalized=repguard.STATS["penalized"] - s0["penalized"], s=round(dt, 1),
                                   tail=tok.decode(out[-200:]))
        json.dump(res, open(out_p, "w"))
        print(f"{key} {arm}: {len(out)} tok {res[f'{key}|{arm}']['finish']}, armed {res[f'{key}|{arm}']['armed']}, "
              f"penalized steps {res[f'{key}|{arm}']['penalized']} ({dt:.0f}s)", flush=True)
st.gate.check()
print("\nsummary (tokens identical to guard-off on math rows):")
for key in keys:
    r = {a: res.get(f"{key}|{a}") for a in ARMS}
    if items[key]["set"] == "math":
        print(f"  {key}: R0 {'same' if r['R0']['toks'] == r['off']['toks'] else 'CHANGED'} ({r['R0']['n']} vs {r['off']['n']}), "
              f"R1 {'same' if r['R1']['toks'] == r['off']['toks'] else 'CHANGED'} ({r['R1']['n']})")
    else:
        print(f"  {key}: R0 {r['R0']['n']} tok armed {r['R0']['armed']} / R1 {r['R1']['n']} tok armed {r['R1']['armed']}, "
              f"R0 == R1 {r['R0']['toks'] == r['R1']['toks']}")
