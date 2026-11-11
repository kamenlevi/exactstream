"""Frugal wait-top-2 runtime hook for Qwen3.8-Flash-Next 125B (Breakthrough D1, 2026-10-20), for the Eval suite.

    import os, sys; sys.path.insert(0, os.path.expanduser("~/Breakthrough/scripts"))
    from bt_frugal import load_frugal
    rt, st = load_frugal(0.10)      # 10 % of each layer's 512 experts resident (C = 51); 0.05 -> C = 26
    # then use rt as an ordinary MTPLX runtime: mtplx.generation.generate_ar(rt, ids, ...), rt.model(...), rt.tokenizer
    # after a run: st.gate.check() raises if any gate timed out (that pass would be invalid)

What it does (the setup measured in data/expA: GSM8K-150 146/150 at 10 %, 143/150 at 5 %, vs 147 exact):
- MoEPrefetch's fork125 streamer (read-only import from ~/MoEPrefetch/fork125) with BIGMOE_SPARE=0, so its background
  prefetch has zero capacity: a missing expert at rank >= 2 is never read from the SSD.
- Every decode step (one token per call): the router's top-20 are computed; ranks 0-1 are waited for (read if missing);
  the K = 10 experts actually used are the first 10 RESIDENT ones in rank order (so missing ranks 2-9 are replaced by
  resident ranks 10-19, else dropped), gates renormalised over the ones used. Resident ranks 2-9 are LRU-touched.
- IMPORTANT: multi-token calls (prefill, or teacher-forcing a sequence in one call) run the EXACT streamed MoE. Frugal
  routing depends on the expert cache, which evolves token by token, so a prefill-only / teacher-forced test of this
  variant must feed the reference tokens one at a time (decode-style calls) to measure frugal behaviour. Prompts can be
  prefilled normally (exact), as in the GSM8K runs.
- Decode is eager (MTP off), on the fork's older BigMoE base (no MTP, no PF_JIT): speed numbers are not the main track's.
"""
import json, os, sys


def load_frugal(frac, nwait=2, mtp=False):
    os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
    for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
        os.environ.setdefault(k, "0")
    os.environ["BIGMOE_SPARE"] = "0"          # frugal: no background reads of missing low ranks
    os.environ["BIGMOE_PF_NOPUBLISH"] = "1"
    root = os.path.expanduser("~/MoEPrefetch")
    sys.path.insert(0, os.path.join(root, "fork125"))
    sys.path.insert(1, os.path.join(root, "scripts"))
    import mlx.core as mx
    from bigmoe_wt.loader import load_streamed
    from fused_route import topk_probs, select_resident
    md = os.path.expanduser("~/BigMoE/models/bare")
    cfg = json.load(open(os.path.join(md, "config.json")))
    cfg = cfg.get("text_config", cfg)
    E, K = cfg["num_experts"], cfg["num_experts_per_tok"]
    C = max(16, round(frac * E))
    rt, st = load_streamed(md, C, mtp=mtp)
    from mtplx.models import qwen4_exp
    from mtplx.attention_math import attention_gate
    orig_call = qwen4_exp.SparseMoeBlock.__call__
    KC = 20

    def block(self, x):
        sw = self.switch_mlp
        bm = getattr(sw, "_bm", None)
        if bm is None or x.shape[1] != 1 or bm["full"]:
            return orig_call(self, x)
        if bm["L"] == bm["first"]:
            bm["gate"].new_pass()
        ids, pk = topk_probs(self.gate(x)[0], KC)
        slots = bm["gate"].route_and_gate(bm["L"], ids[0], nw=nwait, nk=K).reshape(1, KC).astype(mx.int32)
        sl, w = select_resident(slots, pk, topk=K)
        base = type(sw).__mro__[1]
        y = (base.__call__(sw, x, sl[None].astype(mx.uint32)) * w[None][..., None].astype(x.dtype)).sum(axis=-2)
        return y + attention_gate(self.shared_expert(x), self.shared_expert_gate(x))

    qwen4_exp.SparseMoeBlock.__call__ = block
    print(f"bt_frugal: C={C} of {E} experts per layer ({100 * C / E:.0f} %), wait for top-{nwait}, BIGMOE_SPARE=0", flush=True)
    return rt, st
