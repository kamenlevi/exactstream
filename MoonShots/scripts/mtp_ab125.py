"""MTP acceptance A/B on the 125B, one load, draft head and MTP-history policy varied in place.
Draft heads:
  A  the head as BigMoE loads it (bare sidecar: 4-bit g32 experts, 4-bit g64 attention, 8-bit router/indexer)
  R  control: the same sidecar re-attached by attach() below (must equal A token for token and draft for draft)
  B  the official bf16 head (Qwen/Qwen3.8-Flash-Next, fetched by fetch_mtp_bf16.py, mapped as mtp_bf16_check.py)
  D  control for B: the sidecar dequantized to bf16 in dense modules (A's values, B's kernels)
MTP-history policy (generate_mtpk's mtp_history_policy): "cycle" is the Python API default that BigMoE's scripts get
(the head never sees the prompt), "committed" is MTPLX's product default (server, CLI): the head's own KV cache is
primed with the prompt and extended with every committed token.
Prompts and samplers as mtp_accept125.py (code, math, prose; greedy depth 3 and 1, temperature 1.0 depth 3; 128 tokens).
Greedy outputs must be identical across all runs (the verify keeps outputs exact for any draft head and history).
usage: mtp_ab125.py OFFICIAL.safetensors [FRAC] [RUNS]   RUNS like "A:cycle,A:committed,R:committed,B:committed,D:committed"
(GPU, BigMoE-size memory: exclusive window)"""
import gc, hashlib, json, os, sys, time
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for _k, _v in (("MTPLX_FAMILY_CAPTURE_COMMIT", "1"), ("MTPLX_DROP_EVENTS", "1"), ("MTPLX_LAZY_MTP_HISTORY_APPEND", "1"),
               ("MTPLX_DEFER_REPAIR_EVAL", "1")):
    os.environ.setdefault(_k, _v)                 # as BigMoE's mtp_diag.py (MTPLX server defaults)
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mtp_bf16_check import official_to_mtplx
from bigmoe.loader import load_streamed
from bigmoe import fuse
OFF = sys.argv[1]
frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
RUNS = [r.split(":") for r in (sys.argv[3] if len(sys.argv) > 3 else
                                "A:cycle,A:committed,R:committed,B:committed,D:committed,B:cycle").split(",")]
MD = os.path.expanduser("~/BigMoE/models/bare")
C = round(frac * 512)
rt, st = load_streamed(MD, C)
import mtplx.models.qwen4_exp as Q
from mtplx.sampling import SamplerConfig
from mtplx import generation as G
from bigmoe.gated import ext
TM = rt.model.language_model
ARGS = TM.args


def stats(v):
    v = v.astype(mx.float32)
    return f"mean {v.mean().item():+.4f} min {v.min().item():+.4f} max {v.max().item():+.4f}"


print("runtime MTP norms as loaded (pre_fc norms should read raw + 1):", flush=True)
for n, p in tree_flatten(TM.mtp.parameters()):
    if p.ndim == 1:
        print(f"  {n:58s} {stats(p)}", flush=True)


def stripped_of(d):
    """{mtp.*} -> attach_mtp's stripped dict (prefix removed, +1 on the raw pre_fc norms, as attach_mtp does)."""
    s = {k[4:]: v for k, v in d.items() if k.startswith("mtp.")}
    for n in ("pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight"):
        s[n] = (s[n].astype(mx.float32) + 1.0).astype(s[n].dtype)
    return s


def module_inputs():
    mtp = Q.Qwen4ExpMTP(ARGS)
    return {p[:-7]: int(a.shape[-1]) for p, a in tree_flatten(mtp.parameters()) if p.endswith(".weight") and a.ndim >= 2}


def attach(s):
    """attach_mtp's build, from a dict: quantize the modules the dict ships packed, load strict, publish."""
    mtp = Q.Qwen4ExpMTP(ARGS)
    qmap = {}
    for mod, in_f in module_inputs().items():
        w, sc = s.get(f"{mod}.weight"), s.get(f"{mod}.scales")
        if w is None or sc is None or w.dtype != mx.uint32:
            continue
        qmap[mod] = {"bits": int(w.shape[-1]) * 32 // in_f, "group_size": in_f // int(sc.shape[-1])}
    if qmap:
        nn.quantize(mtp, group_size=64, bits=8, class_predicate=lambda p, m: qmap.get(p) or False)
    mtp.load_weights(list(s.items()), strict=True)
    mtp.eval()
    mx.eval(mtp.parameters())
    TM.mtp = mtp
    return len(qmap)


def dequantized(s):
    """the dict with every packed module dequantized to bf16 (bits/group inferred as attach_mtp does)."""
    out = dict(s)
    for mod, in_f in module_inputs().items():
        w, sc = s.get(f"{mod}.weight"), s.get(f"{mod}.scales")
        if w is None or sc is None or w.dtype != mx.uint32:
            continue
        bits, gs = int(w.shape[-1]) * 32 // in_f, in_f // int(sc.shape[-1])
        out[f"{mod}.weight"] = mx.dequantize(w, sc, s[f"{mod}.biases"], group_size=gs, bits=bits).astype(mx.bfloat16)
        del out[f"{mod}.scales"], out[f"{mod}.biases"]
    return out


g8 = json.loads(open(os.path.expanduser("~/MoEPrefetch/data/gsm8k/test.jsonl")).readline())["question"]
PROMPTS = {
    "code": open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read(),
    "math": "Solve the problem step by step, briefly. End with a line of the form 'Answer: <number>'.\n\n" + g8,
    "prose": "Explain how a refrigerator works to a curious 10 year old, in about 150 words.",
}
IDS = {}
for name, text in PROMPTS.items():
    ids = rt.tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    IDS[name] = list(ids if isinstance(ids[0], int) else ids["input_ids"])
SAMPLERS = {"greedy": SamplerConfig(temperature=0.0, top_p=1.0, top_k=0), "t1.0": SamplerConfig(temperature=1.0, top_p=1.0, top_k=0)}
CONFIGS = [("greedy", 3), ("greedy", 1), ("t1.0", 3)]
rows = []


def measure(head, policy):
    for name, ids in IDS.items():
        for sname, depth in CONFIGS:
            stamps = []

            def cb(toks):
                if not stamps:
                    ext.reset_stats()               # decode only: the prefill is done by the first callback
                stamps.append((time.perf_counter(), len(toks) if isinstance(toks, list) else 1))
            fuse.reset_generation_state(rt.model)
            t0 = time.perf_counter()
            res = G.generate_mtpk(rt, ids, max_tokens=128, sampler=SAMPLERS[sname], speculative_depth=depth, seed=0,
                                  mtp_history_policy=policy, token_callback=cb)
            dt = time.perf_counter() - t0
            x = ext.stats()
            toks, S = list(res.tokens), res.stats
            acc, dr = list(S.accepted_by_depth), list(S.drafted_by_depth)
            rounds = dr[0] if dr else 0
            cond = [round(acc[0] / dr[0], 3) if dr[0] else None] + [round(acc[i] / acc[i - 1], 3) if acc[i - 1] else None
                                                                    for i in range(1, len(acc))]
            n_dec = max(1, sum(k for _, k in stamps) - stamps[0][1]) if stamps else 1
            dec = stamps[-1][0] - stamps[0][0] if len(stamps) > 1 else float("nan")
            row = dict(head=head, policy=policy, prompt=name, sampler=sname, depth=depth, tokens=len(toks),
                       verify_calls=S.verify_calls, mtp_rounds=rounds, accepted_by_depth=acc, drafted_by_depth=dr,
                       conditional_accept=cond, overall_accept=round(sum(acc) / max(1, sum(dr)), 3),
                       mtp_tokens_per_round=round((sum(acc) + rounds) / max(1, rounds), 2),
                       tokens_per_verify=round(len(toks) / max(1, S.verify_calls), 2),
                       decode_tok_s=round(n_dec / dec, 2) if dec == dec and dec > 0 else None,
                       stalled_per_tok=round(x["stalled"] / n_dec, 2), misses_per_tok=round(x["misses"] / n_dec, 2),
                       io_ms_per_tok=round(1e3 * x.get("io_s", 0.0) / n_dec, 2),
                       sha=hashlib.sha1(str(toks).encode()).hexdigest()[:10], seconds=round(dt, 1))
            rows.append(row)
            print(json.dumps(row), flush=True)


cur = "A"
for head, policy in RUNS:
    t0 = time.perf_counter()
    if head != cur:
        if head in ("R", "D"):
            bare = stripped_of(mx.load(os.path.join(MD, "mtp.safetensors")))
            nq = attach(dequantized(bare) if head == "D" else bare)
            del bare
        elif head == "B":
            off = stripped_of(official_to_mtplx(mx.load(OFF)))
            nq = attach(off)
            del off
        else:
            raise SystemExit(f"head {head}: A is only the head as loaded (run it first)")
        cur = head
        gc.collect(); mx.clear_cache()
        print(f"[{head}] attached ({nq} quantized modules) in {time.perf_counter() - t0:.0f}s, "
              f"active {mx.get_active_memory() / 1e9:.1f} GB", flush=True)
    try:
        measure(head, policy)
    except Exception:
        import traceback
        traceback.print_exc()
        print(f"[{head}:{policy}] FAILED, continuing", flush=True)
    json.dump(rows, open(os.path.expanduser("~/MoonShots/data/mtp_ab125.json"), "w"), indent=1)
st.gate.check()
print(f"\nC={C} slots/layer ({C * 2.77 * 48 / 1e3:.0f} GB experts). Pooled over the 3 prompts:")
print("head policy     config    cond. accept by position   overall  MTP tok/round  tok/verify  decode tok/s  greedy sha")
for head, policy in RUNS:
    for sname, depth in CONFIGS:
        rs = [r for r in rows if r["head"] == head and r["policy"] == policy and r["sampler"] == sname and r["depth"] == depth]
        if not rs:
            continue
        acc = [sum(r["accepted_by_depth"][i] for r in rs) for i in range(depth)]
        dr = [sum(r["drafted_by_depth"][i] for r in rs) for i in range(depth)]
        cond = [acc[0] / max(1, dr[0])] + [acc[i] / acc[i - 1] if acc[i - 1] else 0 for i in range(1, depth)]
        tpr = (sum(acc) + dr[0]) / max(1, dr[0])
        tpv = sum(r["tokens"] for r in rs) / sum(r["verify_calls"] for r in rs)
        tps = sum(r["decode_tok_s"] or 0 for r in rs) / len(rs)
        sha = "/".join(r["sha"][:4] for r in rs) if sname == "greedy" else ""
        print(f"{head:4s} {policy:10s} {sname:6s} d{depth}  " + " ".join(f"{c:.3f}" for c in cond).ljust(27)
              + f"{sum(acc) / max(1, sum(dr)):.3f}    {tpr:.2f}           {tpv:.2f}        {tps:5.1f}        {sha}", flush=True)
