"""Load Qwen3.8-Flash-Next through MTPLX with the routed experts streamed from SSD.

MTPLX's loader evaluates every weight (72 GB); here mlx_lm.utils.load_model is forced
lazy and the expert tensors are replaced by slot pools before anything is evaluated,
so the full expert tensors are never read."""
from __future__ import annotations
import mlx.core as mx
import mlx_lm.utils as mu
from .stream import ExpertStreamer


def _frs_nocycle():
    """BIGMOE_FRS_NOCYCLE (default 1; Breakthrough's B12_FRS_NOCYCLE, Fit12 R-45), once per process. spec125 rebuilds the
    FR-Spec draft head per request (SC1b), and mtplx's _full_vocab_head builds it in a local class that super()'s
    __class__ cell keeps in a cycle; its __init__ closure held the pruned head (~0.095 GB) and ids until a gen-2 GC.
    The wrapper empties those two cells once the head is built (only __init__ reads them; the methods use self.head /
    self._ids), so a replaced head's arrays go with it. Memory only: the head computes the same thing."""
    import os
    if os.environ.get("BIGMOE_FRS_NOCYCLE", "1") == "0":
        return
    from mtplx import frspec_draft as fd
    if getattr(fd._full_vocab_head, "_nocycle", False):
        return
    f0 = fd._full_vocab_head

    def _full_vocab_head(head, ids, vocab_rows):
        full = f0(head, ids, vocab_rows)
        for c in type(full).__init__.__closure__ or ():
            if c.cell_contents is head or c.cell_contents is ids:
                c.cell_contents = None
        return full
    _full_vocab_head._nocycle = True
    fd._full_vocab_head = _full_vocab_head


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
        _frs_nocycle()
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
    if os.environ.get("BIGMOE_BUDGET"):  # opt-in NON-EXACT fast mode: per-pass expert budget (TrainLab; bigmoe/budget.py)
        from . import budget
        n = budget.install(rt.model, os.environ["BIGMOE_BUDGET"], stats=os.environ.get("BIGMOE_BUDGET_STATS") == "1")
        nj = budget.install_jit(rt.model, os.environ["BIGMOE_BUDGET"]) if os.environ.get("BIGMOE_BUDGET_JIT") == "1" else 0
        print(f"[bigmoe] FAST MODE (non-exact): expert budget {os.environ['BIGMOE_BUDGET']} on {n} routers, "
              f"budget-aware JIT on {nj} look-ahead routers", flush=True)
    return rt, streamer
