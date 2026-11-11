"""Breakthrough D1 quality policies on the fast path (2026-10-21): wait-by-gate-mass routing with the wait decision made
on the GPU and carried to the loader as a flag bit (bit 20) on each routing-row id (~/Breakthrough/fork125bf: fork125bt
+ flag mode: post kernel masks the flag, the loader waits for exactly the flagged ids, ensure_wait_list). No CPU sync per
layer, works for multi-row verify (each row waits for its own prefix).
Rule per row: rank r of the top-10 is waited if the gate mass of ranks < r is below theta (so ranks up to the one that
crosses theta); missing non-waited ranks take the first resident ranks 10-19 at their own weight; no renormalisation
(weights = p / sum of the top-10), as tf_frugal_var.py's mNN_nr probes.
  BT_THETA=0.85 (default) | BT_ADA=lo,hi,tau (eager only: theta = lo if the previous step's entropy <= tau else hi)
usage: bt_mass.py FRAC TASK [CONFIG]   TASK = speed (PROMPT env code|prose|edit, BT_N) | gsmN (OFFSET env)
  CONFIG = ar (eager, default) | k:md:lookup (spec125.generate, fixed theta)
Prints tok/s, SSD reads per token (ext misses), peak; GSM8K accuracy + answers equal to BigMoE's exact eager run."""
import hashlib, json, os, re, sys, time
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for _k, _v in (("MTPLX_FAMILY_CAPTURE_COMMIT", "1"), ("MTPLX_DROP_EVENTS", "1"), ("MTPLX_LAZY_MTP_HISTORY_APPEND", "1"),
               ("MTPLX_DEFER_REPAIR_EVAL", "1")):
    os.environ.setdefault(_k, _v)
for _k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(_k, "0")
PF = os.environ.get("BT_PF") == "1"        # look-ahead prefetch of predicted experts into spare slots (part of C)
os.environ["BIGMOE_SPARE"] = os.environ.get("BT_SPARE", "12") if PF else "0"
if PF:
    os.environ.setdefault("BIGMOE_PREFETCH", "12")
    os.environ.setdefault("BIGMOE_PF_JIT", "1")
    os.environ.setdefault("BIGMOE_PF_QD", "4")
os.environ["BIGMOE_PF_NOPUBLISH"] = "1"
for _k, _v in (("BIGMOE_MAXK", "80"), ("BIGMOE_COMPILE_HC", "1"), ("BIGMOE_HC_VREAD", "1"), ("BIGMOE_ASYNC_EVERY", "1"),
               ("BIGMOE_FRSPEC", "1"), ("BIGMOE_MTP_STREAM", "1"), ("BIGMOE_MTP_C", "64")):
    os.environ.setdefault(_k, _v)
if (sys.argv[3] if len(sys.argv) > 3 else "ar") == "ar":
    os.environ["BIGMOE_MTP_STREAM"] = "0"   # no MTP head in eager runs: a streamed MTP gate layer would be a phantom
sys.path.insert(0, os.path.expanduser("~/Breakthrough/fork125bf"))
sys.path.insert(1, os.path.expanduser("~/MoEPrefetch/scripts"))
sys.path.insert(2, os.path.expanduser("~/SpecDecode"))
import numpy as np
import mlx.core as mx
from bigmoe.loader import load_streamed
from bigmoe.gated import ext, MAXK, PRED_LINES
from fused_route import topk_probs

OUT = os.path.expanduser("~/Breakthrough/data/mass"); os.makedirs(OUT, exist_ok=True)
MD = os.path.expanduser("~/BigMoE/models/bare")
frac, TASK = float(sys.argv[1]), sys.argv[2]
CONF = sys.argv[3] if len(sys.argv) > 3 else "ar"
cfg = json.load(open(os.path.join(MD, "config.json"))); cfg = cfg.get("text_config", cfg)
E, K = cfg["num_experts"], cfg["num_experts_per_tok"]
C = int(os.environ.get("BT_C", "0")) or max(16, round(frac * E))
THETA = float(os.environ.get("BT_THETA", "0.85"))
ADA = [float(v) for v in os.environ["BT_ADA"].split(",")] if os.environ.get("BT_ADA") else None
assert ADA is None or CONF == "ar", "adaptive theta is eager-only"
t0 = time.perf_counter()
rt, st = load_streamed(MD, C, mtp=CONF != "ar")
print(f"bt_mass: C={C} of {E} ({100 * C / E:.0f} %), theta {THETA if ADA is None else 'ada %s' % ADA}, {CONF}, "
      f"loaded in {time.perf_counter() - t0:.0f}s", flush=True)
from mtplx.models import qwen4_exp
from mtplx.attention_math import attention_gate
orig_call = qwen4_exp.SparseMoeBlock.__call__
PREV_ENT = {"v": mx.array(99.0)}         # previous step's next-token entropy (lazy, on the GPU)


def block(self, x):
    sw = self.switch_mlp
    bm = getattr(sw, "_bm", None)
    S = x.shape[1]
    kc = min(20, MAXK // max(1, S))
    if bm is None or bm["full"] or S > 8 or kc < K:
        return orig_call(self, x)
    if bm["L"] == bm["first"]:
        bm["gate"].new_pass()
    ae = bm.get("async_every", 0)
    if ae and bm["L"] % ae == ae - 1:
        mx.async_eval(x)
    ids, pk = topk_probs(self.gate(x)[0], kc)                           # [S, kc]
    p10 = pk[:, :K]; tot = p10.sum(axis=-1, keepdims=True)
    before = (mx.cumsum(p10, axis=-1) - p10) / tot                       # mass of the ranks before r
    th = mx.array(THETA) if ADA is None else mx.where(PREV_ENT["v"] <= ADA[2], ADA[0], ADA[1])
    wait = before < th                                                   # rank 0 always waited
    if S > 1 and S * K > C - 4:   # IDEAS D1d: a pass may not wait for more distinct experts than the pool holds
        M = max(0, C - 4 - S)                                            # every row's rank 0, then M more by priority
        prio = mx.where(wait, p10 / tot * (0.8 ** mx.arange(S, dtype=mx.float32))[:, None], -1.0)
        prio = mx.concatenate([mx.full((S, 1), -1.0), prio[:, 1:]], axis=-1)   # rank 0 is handled separately
        kth = mx.sort(prio.reshape(-1))[-M] if M > 0 else mx.array(9.0)
        wait = mx.concatenate([mx.ones((S, 1), mx.bool_), (prio[:, 1:] >= kth) & (prio[:, 1:] >= 0)], axis=-1)
    wait = wait.astype(mx.int32)
    flags = mx.concatenate([wait << 20, mx.zeros((S, kc - K), mx.int32)], axis=-1) if kc > K else wait << 20
    idf = ids.astype(mx.int32) | flags
    flat = idf.T.reshape(-1) if S > 1 else idf.reshape(-1)               # rank-major
    preds, counts = None, (0, 0)
    nx = bm.get("next") if PF else None
    if nx is not None:   # BT_PF: look-ahead predictions for layers L+1.. (stream.py's unranked path) -> spare-slot prefetch
        rows = x.reshape(-1, x.shape[-1])
        parts, cnt, left = [], [], 30 + 31 * PRED_LINES
        for router, ratio, B in nx:
            b = min(B, left // rows.shape[0])
            if b <= 0:
                break
            lg = router(rows * ratio if ratio is not None else rows)
            parts.append(mx.argpartition(-lg, kth=b - 1, axis=-1)[:, :b].reshape(-1)); cnt.append(b * rows.shape[0])
            left -= cnt[-1]
        if parts:
            preds = mx.concatenate(parts) if len(parts) > 1 else parts[0]
            counts = tuple(cnt + [0] * (3 - len(cnt))) + (0,)
    slots = bm["gate"].route_and_gate(bm["L"], flat, preds=preds, counts=counts, nw=255, nk=K * S)
    slots = (slots.reshape(kc, S).T if S > 1 else slots.reshape(1, kc)).astype(mx.int32)
    res = slots >= 0
    sel = res & (mx.cumsum(res.astype(mx.int32), axis=-1) <= K)         # first K resident in rank order
    w = pk * sel / tot                                                   # no renormalisation
    safe = mx.where(sel, slots, slots[:, :1]).astype(mx.uint32)
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


class _Stop(Exception):
    pass


def generate(prompt_ids, n, stop_eos=False):
    """-> tokens, stats (steps, seconds of decode, SSD reads)"""
    from bigmoe.fuse import reset_generation_state
    reset_generation_state(model)
    out, s = [], {}
    if CONF == "ar":
        cache = model.make_cache()
        ids = mx.array([list(prompt_ids)[:-1]])
        for i in range(0, ids.shape[1], 2048):
            lg = model(ids[:, i:i + 2048], cache=cache)
            mx.eval(lg[0] if isinstance(lg, tuple) else lg)
        y = int(prompt_ids[-1]); PREV_ENT["v"] = mx.array(99.0)
        ext.reset_stats(); m0, t0 = ext.stats()["misses"], time.perf_counter()
        while len(out) < n:
            lg = model(mx.array([[y]]), cache=cache)
            lg = (lg[0] if isinstance(lg, tuple) else lg)[0, -1].astype(mx.float32)
            nxt = mx.argmax(lg)
            if ADA is not None:
                lp = lg - mx.logsumexp(lg); PREV_ENT["v"] = -(mx.exp(lp) * lp).sum()
            y = int(nxt.item())
            if stop_eos and y in EOS:
                break
            out.append(y)
    else:
        import spec125
        k, md, lk = (int(v) for v in CONF.split(":")[:3])
        first = {}                                  # clock and read counter start at the first generated token
        m0, t0 = None, None                         # (spec125.generate runs the prompt prefill itself)

        def cb(t):
            if not first:
                first.update(t=time.perf_counter(), m=ext.stats()["misses"])
            if stop_eos and int(t) in EOS:
                raise _Stop
            out.append(int(t))
        try:
            spec125.generate(model, prompt_ids, n, k_mtp=k, max_draft=md, lookup=lk, stats=s, on_token=cb)
        except _Stop:
            pass
    st.gate.check()
    if CONF != "ar":                                # decode only: from the first token on (one token of the count fewer)
        t0, m0 = first.get("t", time.perf_counter()), first.get("m", ext.stats()["misses"])
        s.update(t=time.perf_counter() - t0, reads=ext.stats()["misses"] - m0, n=max(1, len(out) - 1))
        return out, s
    s.update(t=time.perf_counter() - t0, reads=ext.stats()["misses"] - m0, n=len(out))
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
    print(f"[{P}] mass C={C} theta {THETA if ADA is None else ADA} {CONF}: {s['n'] / s['t']:.1f} tok/s (incl. the first step), "
          f"{s['reads'] / max(1, s['n']):.1f} SSD reads/token ({s['reads'] / max(1, s['n']) * 2.765:.0f} MB), "
          f"{s.get('tok_per_pass', float('nan')):.2f} tok/pass, peak {mx.get_peak_memory() / 1e9:.1f} GB, sha {sha(out)} | "
          f"{tok.decode(out)[:70]!r}", flush=True)
elif TASK.startswith("gsm"):
    N = int(TASK[3:]); OFF = int(os.environ.get("OFFSET", "0"))
    G8 = [json.loads(l) for l in open(os.path.expanduser("~/MoEPrefetch/data/gsm8k/test.jsonl"))][OFF:OFF + N]
    INSTR = "Solve the problem step by step, briefly. End with a line of the form 'Answer: <number>'.\n\n"
    tag = os.environ.get("BT_TAG", f"r{round(100 * C / E):02d}_{'ada' if ADA else 't%g' % THETA}_{CONF.replace(':', '')}")
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
        ans[key], ok[key], st_l[key] = a, good, {k2: s[k2] for k2 in ("t", "reads", "n")}
        json.dump({"answers": ans, "ok": ok, "stats": st_l, "C": C, "theta": THETA, "ada": ADA, "conf": CONF}, open(ck, "w"))
    keys = [str(OFF + j) for j in range(len(G8))]
    ref = json.load(open(os.path.expanduser("~/MoEPrefetch/data/w125_gsm_eager.json")))["answers"]
    same = sum(1 for k in keys if int(k) < len(ref) and ref[int(k)] == ans.get(k))
    nt = sum(st_l[k]["n"] for k in keys); tt = sum(st_l[k]["t"] for k in keys); rd = sum(st_l[k]["reads"] for k in keys)
    print(f"125B C={C} mass {THETA if ADA is None else ADA} {CONF} [{tag}]: GSM8K {sum(ok[k] for k in keys)}/{len(keys)} "
          f"(offset {OFF}), answers equal to exact eager {same}/{sum(1 for k in keys if int(k) < len(ref))}, "
          f"{nt / max(tt, 1e-9):.1f} tok/s decode, {rd / max(1, nt):.1f} reads/token; per-problem "
          f"{''.join(str(ok[k]) for k in keys)}", flush=True)
