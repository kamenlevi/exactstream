"""Load Qwen3.8-Flash-Next through MTPLX with the routed experts streamed from SSD.

MTPLX's loader evaluates every weight (72 GB); here mlx_lm.utils.load_model is forced
lazy and the expert tensors are replaced by slot pools before anything is evaluated,
so the full expert tensors are never read."""
from __future__ import annotations
import mlx.core as mx
import mlx_lm.utils as mu
from .stream import ExpertStreamer


def load_streamed(model_dir, C, mtp=True):
    import os
    # Repetition guard (release must-have; Eval suite ifeval/1219 loops to the cap in the 3-bit plan): MTPLX's loop_guard,
    # on by default in every BigMoE mode. Disarmed it emits no penalties (bit-identical); it arms only on a 12-gram seen
    # 4x in the last 2,048 tokens past 256 tokens. BIGMOE_LOOP_GUARD=0 (or MTPLX_LOOP_GUARD=0) turns it off.
    if os.environ.get("BIGMOE_LOOP_GUARD", "1") == "1":
        os.environ.setdefault("MTPLX_LOOP_GUARD", "1")
    if os.environ.get("BIGMOE_GDN_EXACT") == "1":    # MoonShots: exact one-dispatch GDN conv+silu+l2norm (+ gated norm),
        import sys                                    # bitwise == the eager chain; routes MTPLX's CONVNORM branches to it
        sys.path.insert(0, os.path.expanduser("~/MoonShots/patches/planes_engine"))
        import gdn_exact_patch
        gdn_exact_patch.install()
        os.environ["MTPLX_FUSED_GDN_CONVNORM"] = "1"; os.environ["MTPLX_FUSED_CONVNORM_VERIFY"] = "1"
    streamer = ExpertStreamer(model_dir, C)
    orig = mu.load_model

    def lazy_load_model(path, *a, **kw):
        a = list(a)
        if a: a[0] = True                    # load_model(path, lazy, ...) positional
        else: kw["lazy"] = True
        model, cfg = orig(path, *a, **kw)
        if hasattr(model, "language_model"):
            streamer.install(model)
            mx.eval(model.parameters())        # everything else (dense, MTP) becomes resident now
        return model, cfg

    mu.load_model = lazy_load_model
    try:
        from mtplx.runtime import load
        rt = load(model_dir, mtp=mtp)
    finally:
        mu.load_model = orig
    import os
    if os.environ.get("BIGMOE_MTP_STREAM") == "1":
        Lm = streamer.install_mtp(rt.model, model_dir, int(os.environ.get("BIGMOE_MTP_C", "64")))
        print(f"[bigmoe] MTP experts streamed as gate layer {Lm}", flush=True)
    from . import fuse
    applied = fuse.apply(rt.model)       # optional fusions (env switches), A/B'd for identical tokens
    import os, sys
    if os.environ.get("BIGMOE_FRSPEC") == "1":   # Scout SC1: MTP draft head over a 64K frequent-token subset (drafts only)
        sys.path.insert(0, os.path.expanduser("~/Scout"))
        from patches.frspec4 import FRSpec
        FRSpec(rt.model).on(); applied["frspec"] = True
        # SC1b: per request, FRSpec(rt.model, extra_ids=prompt_ids).on() adds the prompt's own tokens (edit +3.4 %)
    if os.environ.get("BIGMOE_SCOUT_BUNDLE") == "1":   # Scout SC2 + SC3 + SC3b (exact): shared expert ahead of the gate,
        sys.path.insert(0, os.path.expanduser("~/Scout"))   # fused QSA verify rope (needs MTPLX_QWEN4_VERIFY_GLUE=1 before
        from patches.bundle import install                   # mtplx import), chunked HC_VREAD at S=6..8
        applied["scout_bundle"] = install(rt.model) or True
    if os.environ.get("BIGMOE_PKD1") == "1":     # PrefillKernels PKD1: grouped input projections, 1..8 rows (exact)
        sys.path.insert(0, os.path.expanduser("~/PrefillKernels/patches/fusedproj"))
        import fusedproj
        fusedproj.apply(rt.model); applied["pkd1"] = True
    if applied:
        print(f"[bigmoe] fusions: {applied}", flush=True)
    streamer.install_sidecars()          # BIGMOE_TIER only: LSB sidecars + rank-ordered MoE block (after every patch)
    return rt, streamer
