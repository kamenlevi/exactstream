"""Round 26b: where a 125B verify pass spends its time (orchestrator 20:12: per component ms, GB/s, % of the 310-326 GB/s
read ceiling, CPU-GPU sync points, and why). BigMoE's spin gate leaves no command-buffer boundary at the gates; with
BIGMOE_TRACE=1 its loader logs, per (layer L, pass), when it saw the GPU's routing row (t_seen: the GPU reached L's gate)
and when it released the slots (t_rel). On that clock (CLOCK_UPTIME_RAW, shared with Python's) a pass decomposes into:
  stall_L     t_rel[L] - t_seen[L]           loader read of L's misses (the GPU spins)
  between_L   t_seen[L+1] - t_rel[L]         L's MoE (routed + shared + HC write) and L+1's attention + HC read + router
  inter-pass  t_seen[0] of the next pass - t_rel[47]: layer 47's MoE, final mixer + lm_head, sampling and acceptance
              (CPU syncs), MTP history append, the draft steps, layer 0's attention
Bytes per interval come from the modules' resident weights (attention side of L+1, MoE side of L) plus L's unique
routed experts (2.765 MB each, counted by a route_and_gate hook) and the GDN recurrent state (fp32, read + write).
Python-side timestamps of the MTP draft steps and token callbacks are taken on the same clock.
usage: pass_profile125.py [FRAC] [MAXTOK]   (GPU, BigMoE-size memory: exclusive window)"""
import json, os, sys, time
os.environ["BIGMOE_TRACE"] = "1"
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for _k, _v in (("MTPLX_FAMILY_CAPTURE_COMMIT", "1"), ("MTPLX_DROP_EVENTS", "1"), ("MTPLX_LAZY_MTP_HISTORY_APPEND", "1"),
               ("MTPLX_DEFER_REPAIR_EVAL", "1")):
    os.environ.setdefault(_k, _v)
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from bigmoe.loader import load_streamed
from bigmoe import fuse
from bigmoe.gated import ext
frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 160
MD = os.path.expanduser("~/BigMoE/models/bare")
rt, st = load_streamed(MD, round(frac * 512))
from mtplx.sampling import SamplerConfig
from mtplx import generation as G
TM = rt.model.language_model
LAYERS = TM.model.layers
NL = len(LAYERS)
EXPERT_MB = 2.765


def clk():
    return time.clock_gettime(time.CLOCK_UPTIME_RAW)


# resident bytes per layer, split into the attention side and the MoE side (routed pools excluded)
def nbytes(mod):
    return sum(a.nbytes for _, a in tree_flatten(mod.parameters()))


att_b, moe_b, kinds = [], [], []
for L, layer in enumerate(LAYERS):
    names = dict(layer.children()) if hasattr(layer, "children") else {}
    a = m = 0
    for n, mod in names.items():
        if not hasattr(mod, "parameters"):
            continue
        b = nbytes(mod)
        if n == "mlp":
            b -= sum(a_.nbytes for k_, a_ in tree_flatten(mod.switch_mlp.parameters()))
            m += b
        elif n in ("mlp_hyper_connection",):
            m += b
        else:
            a += b
    att_b.append(a); moe_b.append(m)
    kinds.append("gdn" if getattr(layer, "is_linear", False) else "qsa")
print("resident bytes per layer (MB): attention side gdn %.1f / qsa %.1f, MoE side %.1f" % (
    np.mean([a for a, k in zip(att_b, kinds) if k == "gdn"]) / 1e6, np.mean([a for a, k in zip(att_b, kinds) if k == "qsa"]) / 1e6,
    np.mean(moe_b) / 1e6), flush=True)
print("non-layer resident MB: " + ", ".join(f"{n} {nbytes(mod) / 1e6:.0f}" for n, mod in dict(TM.model.children()).items()
                                            if n != "layers" and hasattr(mod, "parameters"))
      + f", lm_head {nbytes(TM.lm_head) / 1e6:.0f}" if hasattr(TM, "lm_head") else "", flush=True)
# GDN recurrent state per layer (fp32) from the args
A = TM.args
gdn_state = getattr(A, "linear_num_value_heads", 0) * getattr(A, "linear_key_head_dim", 0) * getattr(A, "linear_value_head_dim", 0) * 4
print(f"GDN recurrent state per layer: {gdn_state / 1e6:.2f} MB (fp32)", flush=True)

# hooks: unique routed experts per (pass, layer); MTP draft steps; token callbacks
gate = st.gate
UNI = {}
_orig = gate.route_and_gate


def hook(L, ids, seq=None, preds=None, counts=None):
    UNI[(int(gate.seq if seq is None else seq), int(L))] = ids
    return _orig(L, ids, seq=seq, preds=preds, counts=counts)


gate.route_and_gate = hook
DRAFT = []
_fr = TM.mtp.fuse_and_run


DRAFT_SYNC = os.environ.get("DRAFT_SYNC") == "1"           # round 32f: time each draft step's GPU work (serializes)
DRAFT_DONE = []


def fr_hook(widened, tok_emb, cache):
    t0 = clk()
    h = _fr(widened, tok_emb, cache)
    DRAFT.append((int(gate.seq), t0))
    if DRAFT_SYNC:
        mx.eval(h)
        DRAFT_DONE.append((int(gate.seq), t0, clk()))
    return h


object.__setattr__(TM.mtp, "fuse_and_run", fr_hook)
VERIFY_TIMES = os.environ.get("VERIFY_TIMES") == "1"       # round 32n: Python-side time to build each trunk forward graph
VT = []
if VERIFY_TIMES:
    _tm_cls = type(TM)
    _tm_call = _tm_cls.__call__

    def _tm_hook(self, *a, **k):
        t0 = clk()
        out = _tm_call(self, *a, **k)
        VT.append((int(gate.seq), t0, clk(), int(a[0].shape[-1]) if a and hasattr(a[0], "shape") else -1))
        return out

    _tm_cls.__call__ = _tm_hook
lines = [json.loads(l) for l in open(os.path.expanduser("~/MoEPrefetch/data/gsm8k/test.jsonl"))]
PROMPTS = {"code": open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read(),
           "math": "Solve the problem step by step, briefly. End with a line of the form 'Answer: <number>'.\n\n" + lines[0]["question"]}
smp = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)
EV, CB = [], []
for name, text in PROMPTS.items():
    ids = rt.tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    ids = list(ids if isinstance(ids[0], int) else ids["input_ids"])
    fuse.reset_generation_state(rt.model)
    first = []

    def cb(toks):
        if not first:
            ext.trace(); ext.reset_stats(); UNI.clear(); DRAFT.clear(); DRAFT_DONE.clear(); VT.clear(); first.append(clk())
        CB.append((name, int(gate.seq), clk(), len(toks) if isinstance(toks, list) else 1))
    res = G.generate_mtpk(rt, ids, max_tokens=MAXTOK, sampler=smp, speculative_depth=3, seed=0,
                          mtp_history_policy="committed", token_callback=cb)
    ev = [e for e in ext.trace() if int(e[2]) == 1]
    uni = {k: int(len(set(np.array(v).reshape(-1).tolist()))) for k, v in UNI.items()}
    EV.append(dict(prompt=name, events=ev, uni={f"{s}:{L}": n for (s, L), n in uni.items()}, drafts=list(DRAFT), draft_done=list(DRAFT_DONE), verify_times=list(VT),
                   tokens=len(res.tokens), verify_calls=res.stats.verify_calls))
    print(f"{name}: {len(ev)} loader events, {len(uni)} gated (pass, layer) calls, {len(DRAFT)} draft steps", flush=True)
st.gate.check()
json.dump(dict(EV=EV, CB=CB, att_b=att_b, moe_b=moe_b, kinds=kinds, gdn_state=gdn_state, frac=frac),
          open(os.path.expanduser(os.environ.get("PROFILE_OUT", "~/MoonShots/data/pass_profile125.json")), "w"))

# ---- analysis
rows = []
for run in EV:
    by = {}
    for L, seq, kind, ts, tr in run["events"]:
        by.setdefault(int(seq), {})[int(L)] = (ts, tr)
    seqs = sorted(s for s in by if len(by[s]) == NL)
    for a, b in zip(seqs, seqs[1:]):
        if b != a + 1:
            continue
        p, q = by[a], by[b]
        stall = sum(p[L][1] - p[L][0] for L in range(NL))
        betw = {L: p[L + 1][0] - p[L][1] for L in range(NL - 1)}
        inter = q[0][0] - p[NL - 1][1]
        uni = [run["uni"].get(f"{a}:{L}", 0) for L in range(NL)]
        rows.append(dict(stall=stall, betw=betw, inter=inter, uni=uni, rows=None))
n = len(rows)
print(f"\n{n} consecutive pass pairs (C={round(frac * 512)}, committed MTP depth 3, greedy)")
CEIL = 318.0
tot_stall = np.mean([r["stall"] for r in rows]) * 1e3
tot_inter = np.mean([r["inter"] for r in rows]) * 1e3
grp = {"gdn": [], "qsa": []}
for L in range(NL - 1):
    k = kinds[L + 1]
    t = np.mean([r["betw"][L] for r in rows]) * 1e3
    by = (moe_b[L] + att_b[L + 1] + np.mean([r["uni"][L] for r in rows]) * EXPERT_MB * 1e6
          + (2 * gdn_state if k == "gdn" else 0))
    grp[k].append((t, by))
tot = tot_stall + tot_inter + sum(t for g in grp.values() for t, _ in g)
print(f"pass wall (gate to gate) {tot:.1f} ms: loader stalls {tot_stall:.1f} ms ({tot_stall / tot:.0%}), inter-pass block "
      f"{tot_inter:.1f} ms ({tot_inter / tot:.0%})")
print("component                         count  ms/each  ms/pass  MB/each   GB/s   % of 318")
for k, g in grp.items():
    if not g:
        continue
    t = np.mean([x for x, _ in g]); b = np.mean([y for _, y in g])
    print(f"MoE(L) + {k.upper()} attention(L+1)        {len(g):3d}   {t:6.3f}   {t * len(g):6.1f}   {b / 1e6:6.1f}   "
          f"{b / 1e6 / t:6.0f}   {b / 1e6 / t / CEIL:5.0%}")
print(f"inter-pass block (head, sampling, drafts)  1   {tot_inter:6.2f}")
print(f"loader stalls (48 gates)                  48   {tot_stall / NL:6.3f}   {tot_stall:6.1f}")
