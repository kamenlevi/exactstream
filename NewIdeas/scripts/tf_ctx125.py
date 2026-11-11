"""NewIdeas I5 on the 125B: frugal on the CURRENT token only, over an EXACT context (2026-10-21).
Per teacher-forced position (the Eval suite's one-token step scorer and cached reference, as tf_frugal_var.py):
snapshot the cache -> frugal step (wait ranks 0-1, drop missing ranks >= 2, no renormalisation) -> score it ->
rollback_after_verify(1) -> exact step (all top-10 waited) -> score it, and the cache advances exactly. So every frugal
token sees an exact past: the KLD is what frugal costs when its context is repaired (I5's best case, repair lag 0).
Compare with noren (frugal everywhere: 0.1001 on the every-8 ifeval+multi subset, data/tfvar) and exact (0.0041).
Caveat: the exact steps load the full top-10, so frugal steps see a warmer expert cache than a pure frugal run (in I5's
real engine the repair passes warm it too); reads per frugal step are reported.
usage: tf_ctx125.py FRAC SETS EVERY      -> ~/NewIdeas/data/tfctx/results.txt
"""
import json, os, sys, time
os.environ["EVALSUITE_FRUGAL"] = "1"
os.environ["BIGMOE_SPARE"] = "0"
os.environ["BIGMOE_PF_NOPUBLISH"] = "1"
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
sys.path.insert(0, os.path.expanduser("~/Breakthrough/fork125bt"))
sys.path.insert(1, os.path.expanduser("~/MoEPrefetch/scripts"))
import numpy as np
import mlx.core as mx
from bigmoe.loader import load_streamed
from fused_route import topk_probs
from bigmoe.gated import ext
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
from evalsuite import stages as SG, engine as EN

frac, SETS, EVERY = float(sys.argv[1]), sys.argv[2].split(","), int(sys.argv[3])
OUT = os.path.expanduser("~/NewIdeas/data/tfctx"); os.makedirs(OUT, exist_ok=True)
MD = os.path.expanduser("~/BigMoE/models/bare")
cfg = json.load(open(os.path.join(MD, "config.json"))); cfg = cfg.get("text_config", cfg)
E, K = cfg["num_experts"], cfg["num_experts_per_tok"]
C = max(16, round(frac * E))
rt, st = load_streamed(MD, C, mtp=False)
from mtplx.models import qwen4_exp
from mtplx.attention_math import attention_gate
from mtplx import generation as G
from mtplx.attention_context import attention_phase
from mtplx.cache_state import rollback_after_verify, snapshot_untrimmable_cache_lazy
orig_call = qwen4_exp.SparseMoeBlock.__call__
MODE = {"exact": False}
STAT = {"ld": 0, "calls": 0}


def block(self, x):
    sw = self.switch_mlp
    bm = getattr(sw, "_bm", None)
    if bm is None or x.shape[1] != 1 or bm["full"]:
        return orig_call(self, x)
    if bm["L"] == bm["first"]:
        bm["gate"].new_pass()
    L = bm["L"]
    ids, pk = topk_probs(self.gate(x)[0], K)
    p = np.array(pk[0, :K].astype(mx.float32))
    idl = [int(v) for v in np.array(ids[0])]
    pre = np.array(ext.lookup(L, idl))
    nw = K if MODE["exact"] else 2
    need = idl[:nw] + [e for r, e in enumerate(idl) if r >= nw and pre[r] >= 0]
    ext.ensure(L, mx.array(np.array(need, np.int32)))
    slots = np.array(ext.lookup(L, idl))
    if not MODE["exact"]:
        STAT["ld"] += int((pre[:nw] < 0).sum()); STAT["calls"] += 1
    use = slots >= 0
    w = (p / p.sum()) * use
    safe = np.where(use, slots, slots[0]).astype(np.uint32)
    base = type(sw).__mro__[1]
    yk = base.__call__(sw, x, mx.array(safe).reshape(1, 1, K))
    y = (yk * mx.array(w.astype(np.float32)).astype(x.dtype)[None, None, :, None]).sum(axis=-2)
    return y + attention_gate(self.shared_expert(x), self.shared_expert_gate(x))


def run_item(ids, pos, score):
    p0 = int(pos[0])
    ps = G.restore_or_prefill_prompt_state(rt, [int(x) for x in ids[:p0 + 1]], base_hidden_variant=None,
                                           mtp_history_policy="cycle", capture_hidden=False)
    cache, logits, out_f, out_e = ps.trunk_cache, ps.logits, [], []
    for t in pos:
        if t > p0:
            x = mx.array([[int(ids[t])]])
            snap = snapshot_untrimmable_cache_lazy(cache)
            MODE["exact"] = False
            with attention_phase("ar_decode"):
                lf = rt.forward_ar(x, cache=cache)[:, -1, :]
            rf = score(np.array([t]), lf); mx.eval(rf)
            rollback_after_verify(cache, snap, verified_tokens=1)
            MODE["exact"] = True
            try:
                with attention_phase("ar_decode"):
                    logits = rt.forward_ar(x, cache=cache)[:, -1, :]
                mx.eval(logits)
            finally:
                MODE["exact"] = False
            out_f.append({k: np.array(v) for k, v in rf.items()})
        r = score(np.array([t]), logits); mx.eval(r)
        out_e.append({k: np.array(v) for k, v in r.items()})
    cat = lambda o: {k: np.concatenate([d[k] for d in o]) for k in o[0]}
    return cat(out_f), cat(out_e)


qwen4_exp.SparseMoeBlock.__call__ = block
items = SG.tf_items()
head = EN.var_head(SG.load_ref_tf(items), SG._targets(items))
seqs = [i for i, it in enumerate(items) if it["set"] in SETS][::EVERY]
kf = {s: [] for s in SETS}; ke = {s: [] for s in SETS}; af = {s: [] for s in SETS}
t0 = time.time()
for i in seqs:
    it = items[i]
    rf, re_ = run_item(it["ids"], it["pos"], lambda pos, lg, i=i: head(i, pos, lg))
    st.gate.check()
    kf[it["set"]].append(rf["kld"]); af[it["set"]].append(rf["agree"]); ke[it["set"]].append(re_["kld"])
line = (f"ctx C={C}: frugal-on-exact-context KLD {np.concatenate([np.concatenate(v) for v in kf.values() if v]).mean():.4f}"
        f" | exact steps {np.concatenate([np.concatenate(v) for v in ke.values() if v]).mean():.4f} | "
        + " ".join(f"{s} {np.concatenate(kf[s]).mean():.4f} (top-1 {np.concatenate(af[s]).mean():.3f})" for s in SETS if kf[s])
        + f" | {48 * STAT['ld'] / max(1, STAT['calls']):.1f} waited reads per frugal step | {len(seqs)} seqs every {EVERY}"
        f" | {time.time() - t0:.0f}s")
print(line, flush=True)
open(os.path.join(OUT, "results.txt"), "a").write(line + "\n")
