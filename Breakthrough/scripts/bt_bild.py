"""Breakthrough D1c (2026-10-21): confidence-gated coverage (BiLD-style fallback) on the fast path, eager decoding.
Each step runs frugal (wait for ranks 0-1, drop missing ranks 2-9 without renormalising: the night probes' best cheap
rule); if the step's next-token entropy exceeds tau, the step is rolled back (MTPLX snapshot_untrimmable_cache_lazy +
rollback_after_verify, verified_tokens=1) and recomputed with every top-10 expert waited through the native gate.
Engine: ~/Breakthrough/fork125bt (BigMoE ffc859b + selective waiting + protect-what-fits). Spin gate (daytime only:
a SIGSTOP mid-pass invalidates it; the night probes use tf_frugal_var.py's CPU-synchronous residency instead).
usage: bt_bild.py FRAC TAU TASK      TASK = speed (PROMPT env code|prose|edit, BT_N tokens, 128) | gsmN (OFFSET env)
   TAU < 0 = plain frugal (never re-run); TAU >= 99 = exact (every step exact).
Prints tok/s, re-run share, SSD loads per token (ext misses), peak memory; GSM8K accuracy and answers equal to BigMoE's
exact eager run (MoEPrefetch/data/w125_gsm_eager.json, problems 0-149). Checkpoints in data/bild/."""
import hashlib, json, os, re, sys, time
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for _k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(_k, "0")
os.environ["BIGMOE_SPARE"] = "0"
os.environ["BIGMOE_PF_NOPUBLISH"] = "1"
for _k, _v in (("BIGMOE_COMPILE_HC", "1"), ("BIGMOE_HC_VREAD", "1"), ("BIGMOE_ASYNC_EVERY", "1")):
    os.environ.setdefault(_k, _v)
sys.path.insert(0, os.path.expanduser("~/Breakthrough/fork125bt"))
sys.path.insert(1, os.path.expanduser("~/MoEPrefetch/scripts"))
import numpy as np
import mlx.core as mx
from bigmoe.loader import load_streamed
from bigmoe.gated import ext
from fused_route import topk_probs

OUT = os.path.expanduser("~/Breakthrough/data/bild"); os.makedirs(OUT, exist_ok=True)
MD = os.path.expanduser("~/BigMoE/models/bare")
frac, TAU, TASK = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
cfg = json.load(open(os.path.join(MD, "config.json"))); cfg = cfg.get("text_config", cfg)
E, K = cfg["num_experts"], cfg["num_experts_per_tok"]
C = int(os.environ.get("BT_C", "0")) or max(16, round(frac * E))
t0 = time.perf_counter()
rt, st = load_streamed(MD, C, mtp=False)
print(f"bt_bild: C={C} of {E} ({100 * C / E:.0f} %), tau {TAU}, loaded in {time.perf_counter() - t0:.0f}s", flush=True)
from mtplx.models import qwen4_exp
from mtplx.attention_math import attention_gate
from mtplx.cache_state import rollback_after_verify, snapshot_untrimmable_cache_lazy
orig_call = qwen4_exp.SparseMoeBlock.__call__
MODE = {"exact": TAU >= 99}


def block(self, x):
    sw = self.switch_mlp
    bm = getattr(sw, "_bm", None)
    if bm is None or x.shape[1] != 1 or bm["full"]:
        return orig_call(self, x)
    if bm["L"] == bm["first"]:
        bm["gate"].new_pass()
    ae = bm.get("async_every", 0)
    if ae and bm["L"] % ae == ae - 1:
        mx.async_eval(x)
    ids, pk = topk_probs(self.gate(x)[0], K)                            # [1, K] top-10, descending
    nw = K if MODE["exact"] else 2
    slots = bm["gate"].route_and_gate(bm["L"], ids[0], nw=nw, nk=K).reshape(1, K).astype(mx.int32)
    res = slots >= 0
    w = pk * res / pk.sum(axis=-1, keepdims=True)                      # no renormalisation over the computed ones
    safe = mx.where(res, slots, slots[:, :1]).astype(mx.uint32)
    base = type(sw).__mro__[1]
    y = (base.__call__(sw, x, safe[None]) * w[None][..., None].astype(x.dtype)).sum(axis=-2)
    return y + attention_gate(self.shared_expert(x), self.shared_expert_gate(x))


qwen4_exp.SparseMoeBlock.__call__ = block
model, tok = rt.model, rt.tokenizer
EOS = set()
for _a in ("eos_token_ids", "eos_token_id"):
    _v = getattr(tok, _a, None)
    if _v is not None:
        EOS |= set(_v) if isinstance(_v, (list, set, tuple)) else {_v}
for _t in ("<|im_end|>", "<|endoftext|>"):
    try:
        _i = tok.convert_tokens_to_ids(_t)
        if isinstance(_i, int) and _i >= 0: EOS.add(_i)
    except Exception:
        pass


def chat_ids(text):
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    return list(ids if isinstance(ids[0], int) else ids["input_ids"])


def generate(prompt_ids, n, stop_eos=False):
    """greedy BiLD decode -> tokens, stats"""
    from bigmoe.fuse import reset_generation_state
    reset_generation_state(model)
    cache = model.make_cache()
    ids = mx.array([list(prompt_ids)[:-1]])
    for i in range(0, ids.shape[1], 2048):
        lg = model(ids[:, i:i + 2048], cache=cache)
        mx.eval(lg[0] if isinstance(lg, tuple) else lg)
    y, out = int(prompt_ids[-1]), []
    s = dict(steps=0, reruns=0, t=0.0)
    ext.reset_stats()
    m0, t0 = ext.stats()["misses"], time.perf_counter()
    while len(out) < n:
        snap = snapshot_untrimmable_cache_lazy(cache) if 0 <= TAU < 99 else None
        lg = model(mx.array([[y]]), cache=cache)
        lg = (lg[0] if isinstance(lg, tuple) else lg)[0, -1].astype(mx.float32)
        lp = lg - mx.logsumexp(lg)
        ent, nxt = -(mx.exp(lp) * lp).sum(), mx.argmax(lg)
        mx.eval(ent, nxt)
        if snap is not None and float(ent) > TAU:
            rollback_after_verify(cache, snap, verified_tokens=1)
            MODE["exact"] = True
            try:
                lg = model(mx.array([[y]]), cache=cache)
                nxt = mx.argmax((lg[0] if isinstance(lg, tuple) else lg)[0, -1])
                mx.eval(nxt)
            finally:
                MODE["exact"] = False
            s["reruns"] += 1
        s["steps"] += 1
        y = int(nxt.item())
        if stop_eos and y in EOS:
            break
        out.append(y)
    st.gate.check()
    s["t"] = time.perf_counter() - t0
    s["loads"] = ext.stats()["misses"] - m0
    return out, s


sha = lambda t: hashlib.sha1(str(list(t)).encode()).hexdigest()[:10]
if TASK == "speed":
    P = os.environ.get("PROMPT", "code")
    text = {"code": lambda: open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read(),
            "prose": lambda: ("Explain, in plain prose for a curious 14-year-old, how a CPU cache works and why it makes "
                              "programs faster. Use one everyday analogy and keep it to about 300 words."),
            "edit": lambda: ("Here is a file:\n\n```python\n" + open(os.path.expanduser("~/SpecDecode/scripts/inventory_sample.py")).read()
                             + "```\n\nRewrite the whole file with total_value renamed to total everywhere. Output only the full file in one code block.")}[P]()
    ids = chat_ids(text)
    N = int(os.environ.get("BT_N", "128"))
    generate(ids, 16)                                                   # warm-up
    out, s = generate(ids, N)
    print(f"[{P}] bild C={C} tau {TAU}: {s['steps'] / s['t']:.1f} tok/s, re-ran {s['reruns'] / s['steps']:.1%} of steps, "
          f"{s['loads'] / s['steps']:.1f} expert loads/token ({s['loads'] / s['steps'] * 2.765:.0f} MB), peak "
          f"{mx.get_peak_memory() / 1e9:.1f} GB, sha {sha(out)} | {tok.decode(out)[:70]!r}", flush=True)
elif TASK.startswith("gsm"):
    N = int(TASK[3:]); OFF = int(os.environ.get("OFFSET", "0"))
    G8 = [json.loads(l) for l in open(os.path.expanduser("~/MoEPrefetch/data/gsm8k/test.jsonl"))][OFF:OFF + N]
    INSTR = "Solve the problem step by step, briefly. End with a line of the form 'Answer: <number>'.\n\n"
    tag = os.environ.get("BT_TAG", f"r{round(100 * C / E):02d}_t{TAU:g}")
    ck = os.path.join(OUT, f"gsm_{tag}.json")
    state = json.load(open(ck)) if os.path.exists(ck) else {}
    ans, ok, st_l = (state.get(k, {}) for k in ("answers", "ok", "stats"))
    for j, p in enumerate(G8):
        key = str(OFF + j)
        if key in ok:
            continue
        out, s = generate(chat_ids(INSTR + p["question"]), 600, stop_eos=True)
        text = tok.decode(out)
        m = re.findall(r"Answer:\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)", text) or re.findall(r"(-?[\d,]*\.?\d+)", text)
        gold = p["answer"].split("####")[-1].strip().replace(",", "")
        a = m[-1].replace(",", "").rstrip(".") if m else None
        try:
            good = int(a is not None and abs(float(a) - float(gold)) < 1e-6)
        except ValueError:
            good = 0
        ans[key], ok[key], st_l[key] = a, good, s
        json.dump({"answers": ans, "ok": ok, "stats": st_l, "C": C, "tau": TAU}, open(ck, "w"))
    keys = [str(OFF + j) for j in range(len(G8))]
    ref = json.load(open(os.path.expanduser("~/MoEPrefetch/data/w125_gsm_eager.json")))["answers"]
    same = sum(1 for k in keys if int(k) < len(ref) and ref[int(k)] == ans.get(k))
    steps = sum(st_l[k]["steps"] for k in keys); rr = sum(st_l[k]["reruns"] for k in keys)
    ld = sum(st_l[k]["loads"] for k in keys); tt = sum(st_l[k]["t"] for k in keys)
    print(f"125B C={C} bild tau {TAU} [{tag}]: GSM8K {sum(ok[k] for k in keys)}/{len(keys)} (offset {OFF}), answers equal "
          f"to exact eager {same}/{sum(1 for k in keys if int(k) < len(ref))}, re-ran {rr / max(1, steps):.1%} of steps, "
          f"{ld / max(1, steps):.1f} loads/token, {steps / max(tt, 1e-9):.1f} tok/s decode; per-problem "
          f"{''.join(str(ok[k]) for k in keys)}", flush=True)
