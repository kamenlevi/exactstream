"""Eval suite hook for the 24 GB OPT-IN FAST MODE (NON-EXACT): the v3 main-track stack + the per-pass expert budget with the
budget-aware JIT (TrainLab; bigmoe/budget.py). Same contract as Breakthrough's bt_b12.load_b12: load_fast24(...) -> (rt, st),
st.gate.check() after each generation.

Ships as:
- engine: ~/BigMoE at the pinned commit in FAST24_COMMIT below (bigmoe package itself, no copy), MLX 0.32.2, records3.
- load: 24 GB budget = frac 0.25 (C = 128 slot-equivalents: 17.07 GB of tier pools, C_m 147 + C_l 46 per layer), MTP on
  (streamed MTP experts), env = FAST24_ENV (set here, before bigmoe is imported).
- decode: spec125.generate(model, ids, max_tokens, k_mtp=3, max_draft=3, lookup=0, stop_ids={eos, <|im_end|>,
  <|endoftext|>}) = greedy MTP speculation with FIXED widths. Fixed, not adaptive: under the budget a row's routing depends
  on its pass-mates (the drafts), so timing-driven adaptive widths would make outputs vary run to run; with fixed widths
  the outputs are a deterministic function of the prompt.
- prefill: spec125's chunk-major prefill (2,048-token chunks) through the exact grouped tier prefill (BIGMOE_TIER_PF_GROUP).
- repetition guard: OFF for the verdict (BIGMOE_LOOP_GUARD=0, like-for-like with the guard-off anchors); the main track
  ships R1.
- footprint: ~24-25.5 GB whole process (B5/M24g/PFG1: 23.5-25.3 GB peak memory footprint at 256 tokens; +~1 GB at 5K+
  prompts); use 26 GB for the memory gate.
- state across rows: none that changes outputs (expert residency / LFU / JIT prefetch only change speed); BIGMOE_RESET_GEN
  resets QSA state per generation.
- anchor: the speed-up is measured against the v3 tier plan (plan4433333333, the main track's fast exact-to-v3 mode), not
  the 4-bit reference; quality anchors stay the suite's ref / floor.
usage: from fast24 import load_fast24, generate_fast24 (sys.path: ~/BigMoE/scripts)"""
import os, sys

FAST24_COMMIT = "af0cdcb"
FAST24_ENV = {
    # v3 main-track stack (HANDOVER.md "v3 fast mode")
    "BIGMOE_DUPROW": "1", "BIGMOE_COMPILE_HC": "1", "BIGMOE_HC_VREAD": "1", "BIGMOE_ASYNC_EVERY": "1",
    "BIGMOE_PF_JIT": "1", "BIGMOE_PF_QD": "4", "BIGMOE_PREFETCH": "12", "BIGMOE_DUPROW_CHUNK": "5", "BIGMOE_MAXK": "80",
    "BIGMOE_FRSPEC": "1", "BIGMOE_SCOUT_BUNDLE": "1", "MTPLX_QWEN4_VERIFY_GLUE": "1", "BIGMOE_MTP_STREAM": "1",
    "BIGMOE_TIER": "4433333333", "BIGMOE_TIER_GLU": "3c", "BIGMOE_TIER_COMBINE": "1", "BIGMOE_OOO": "1",
    "BIGMOE_GDN_EXACT": "1", "BIGMOE_TIER_PF_GROUP": "64", "BIGMOE_PF_NOLOGITS": "1", "BIGMOE_EVICT_LFU": "0.99",
    "BIGMOE_VERIFY_ROWS_SDPA": "1",
    # the fast mode itself
    "BIGMOE_BUDGET": "b16", "BIGMOE_BUDGET_JIT": "1",
    # verdict: guard off
    "BIGMOE_LOOP_GUARD": "0", "MTPLX_LOOP_GUARD": "0",
    # spec125 / MTPLX defaults the main-track measurements use (test_spec125.py)
    "MTPLX_NGRAM_PREWARM": "off", "MTPLX_FAMILY_CAPTURE_COMMIT": "1", "MTPLX_DROP_EVENTS": "1",
    "MTPLX_LAZY_MTP_HISTORY_APPEND": "1", "MTPLX_DEFER_REPAIR_EVAL": "1", "MTPLX_COMPILED_VERIFY": "0",
    "MTPLX_QWEN4EXP_COMPILE": "0", "MTPLX_COMPILE_AR_FORWARD": "0", "MTPLX_COMPILED_TARGET_PREFIX": "0",
}
_RT = {}


def load_fast24(frac=0.25, mtp=True, budget=None):
    """-> (rt, st). budget: "b16" (default), "b18", "b20" (the fallbacks), or "" for the v3 natural arm (exact-to-v3)."""
    env = dict(FAST24_ENV)
    if budget is not None:
        env["BIGMOE_BUDGET"] = budget
        if not budget:
            env.pop("BIGMOE_BUDGET"); env.pop("BIGMOE_BUDGET_JIT")
    for k, v in env.items():
        os.environ[k] = v
    for p in (os.path.expanduser("~/BigMoE"), os.path.expanduser("~/SpecDecode")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import json
    from bigmoe.loader import load_streamed
    md = os.path.expanduser("~/BigMoE/models/bare")
    cfg = json.load(open(os.path.join(md, "config.json"))); cfg = cfg.get("text_config", cfg)
    rt, st = load_streamed(md, max(16, round(frac * cfg["num_experts"])), mtp=mtp)
    tok = rt.tokenizer
    _RT["stop"] = {i for i in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                               tok.convert_tokens_to_ids("<|endoftext|>")) if isinstance(i, int) and i >= 0}
    return rt, st


def generate_fast24(rt, prompt_ids, max_tokens):
    """The decode as it ships: greedy MTP speculation, fixed widths 3:3:0, EOS stop. -> (tokens, stats)"""
    import spec125
    return spec125.generate(rt.model, list(prompt_ids), max_tokens, k_mtp=3, max_draft=3, lookup=0, stats={},
                            stop_ids=_RT["stop"])
