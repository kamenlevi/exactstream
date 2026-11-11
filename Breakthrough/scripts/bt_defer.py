"""Breakthrough I1 speed prototype (2026-10-21): half-layer deferred experts on the fast path. fork125bd = fork125bf +
BIGMOE_DEFER=M: in flag mode the loader reads the flagged (waited) experts, releases the GPU with the waited + resident
slots (phase A, late ones -1), then reads the late ones and publishes up to M (position, slot) pairs (phase B).
Semantics = NewIdeas' tf_defer125.py dh_wK: top-10 by score, ranks < K waited, resident ranks combined now, every other
top-10 rank read and its gate-weighted output (on layer L's MLP input) written into the hyper-connection streams with
layer L's inject weights after layer L+1's attention, before its MLP read (phase B's spin is ordered after that
attention); weights p / sum(top-10 p) (no renormalisation); the last layer waits for all. Eager 1-row calls only.
  BT_K=4 waited ranks | BT_M=6 phase-B pairs (more late ids than M: the first ones are waited) | BT_SAME=1: add the late
  experts at the end of layer L instead (within-layer overlap only, for comparison) | BT_K=10: exact (no late)
  BT_THETA=0.5: only ranks before 50 % of the top-10 mass are used at all (tf_defer125's _m50)
  BT_FRUGAL=1: the qualifying frugal (w2f): top-20 candidates, ranks 0-1 waited, the first 10 resident in rank order,
  renormalised; BT_LATE1=1 waits for rank 0 only and adds a missing rank 1 half a layer late (same set and weights).
The GPU marks waited ids with bit 20 and late-eligible ids with bit 21 (fork125bd's loader reads both).
usage: bt_defer.py FRAC TASK   TASK = speed (PROMPT env code|prose, BT_N) | gsmN (OFFSET env) | check (BT_K=10 vs 4)
Prints tok/s, SSD reads per token (waited / late / overflow), peak memory."""
import hashlib, json, os, re, sys, time
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for _k, _v in (("MTPLX_FAMILY_CAPTURE_COMMIT", "1"), ("MTPLX_DROP_EVENTS", "1"), ("MTPLX_LAZY_MTP_HISTORY_APPEND", "1"),
               ("MTPLX_DEFER_REPAIR_EVAL", "1")):
    os.environ.setdefault(_k, _v)
for _k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(_k, "0")
KW = int(os.environ.get("BT_K", "4"))
M = int(os.environ.get("BT_M", "6"))
SAME = os.environ.get("BT_SAME") == "1"
THETA = float(os.environ.get("BT_THETA", "0"))
FRUGAL = os.environ.get("BT_FRUGAL") == "1"
LATE1 = FRUGAL and os.environ.get("BT_LATE1") == "1"
os.environ["BIGMOE_SPARE"] = "0"
os.environ["BIGMOE_PF_NOPUBLISH"] = "1"
os.environ["BIGMOE_SPLIT"] = str(M)        # allocates the phase-B region buffer (flag-mode rows never take the split path)
os.environ["BIGMOE_DEFER"] = str(M)
for _k, _v in (("BIGMOE_MAXK", "80"), ("BIGMOE_COMPILE_HC", "1"), ("BIGMOE_HC_VREAD", "1"), ("BIGMOE_ASYNC_EVERY", "1"),
               ("BIGMOE_FRSPEC", "1"), ("BIGMOE_MTP_STREAM", "0")):
    os.environ.setdefault(_k, _v)
sys.path.insert(0, os.path.expanduser("~/Breakthrough/fork125bd"))
sys.path.insert(1, os.path.expanduser("~/MoEPrefetch/scripts"))
import numpy as np
import mlx.core as mx
from bigmoe.loader import load_streamed
from bigmoe.gated import ext
from fused_route import topk_probs

OUT = os.path.expanduser("~/Breakthrough/data/defer"); os.makedirs(OUT, exist_ok=True)
MD = os.path.expanduser("~/BigMoE/models/bare")
frac, TASK = float(sys.argv[1]), sys.argv[2]
cfg = json.load(open(os.path.join(MD, "config.json"))); cfg = cfg.get("text_config", cfg)
E, K, NL = cfg["num_experts"], cfg["num_experts_per_tok"], cfg["num_hidden_layers"]
C = int(os.environ.get("BT_C", "0")) or max(16, round(frac * E))
t0 = time.perf_counter()
rt, st = load_streamed(MD, C, mtp=False)
print(f"bt_defer: C={C} of {E} ({100 * C / E:.0f} %), " + (f"frugal, rank 1 {'late' if LATE1 else 'waited'}" if FRUGAL else f"wait ranks < {KW}" + (f", mass < {THETA}" if THETA else "")) + f", {M} phase-B pairs, "
      f"{'same layer' if SAME else 'half a layer late'}, loaded in {time.perf_counter() - t0:.0f}s", flush=True)
from mtplx.models import qwen4_exp
from mtplx.models.qwen4_exp import _hyper_residual_write
from mtplx.attention_math import attention_gate
orig_call = qwen4_exp.SparseMoeBlock.__call__
STASH, PENDH = {}, {}
KC = 20 if FRUGAL else K
R = mx.arange(KC)
if FRUGAL:                                   # rank 0 waited; rank 1 waited, or late-eligible (bit 21) with BT_LATE1
    FLAGS = mx.where(R == 0, 1 << 20, mx.where(R == 1, (1 << 21) if LATE1 else (1 << 20), 0)).astype(mx.int32)[None]
else:                                        # ranks < K waited, the rest of the top-10 late-eligible
    FLAGS = mx.where(R < KW, 1 << 20, 1 << 21).astype(mx.int32)[None]
WALL = mx.where(R < (2 if FRUGAL else K), 1 << 20, 0).astype(mx.int32)[None]   # last layer: nothing late


def late_fn(xL, wv, dep, seq, n, sw, L):
    """phase B of layer L as a function of a later array (`after`): the spin is ordered after it"""
    base = type(sw).__mro__[1]

    def f(after):
        o = mx.isnan(after.reshape(-1)[:1]).astype(mx.int32) * 0
        depB = mx.concatenate([dep[:1] + o, mx.array([n, 0] * M, dtype=mx.int32)])
        lb = ext.spin_wait(depB, L, seq, 2 * M, 1)
        pos, slb = lb[0::2], lb[1::2]
        xf = xL.reshape(-1, xL.shape[-1])
        xpad = mx.concatenate([xf, mx.zeros((1, xf.shape[-1]), dtype=xf.dtype)])
        yB = base.__call__(sw, xpad[pos // n][None], slb[:, None][None].astype(mx.uint32))[0, :, 0, :]   # [M, D]
        wpad = mx.concatenate([wv.reshape(-1), mx.zeros((1,), wv.dtype)])                               # pads: weight 0
        return (yB * wpad[pos][:, None].astype(yB.dtype)).sum(axis=0).reshape(1, 1, -1)
    return f


def block(self, x):
    sw = self.switch_mlp
    bm = getattr(sw, "_bm", None)
    if bm is None or bm["full"] or x.shape[1] != 1:
        return orig_call(self, x)
    L = bm["L"]
    if L == bm["first"]:
        bm["gate"].new_pass()
    ae = bm.get("async_every", 0)
    if ae and L % ae == ae - 1:
        mx.async_eval(x)
    ids, pk = topk_probs(self.gate(x)[0], KC)                          # [1, KC] by score
    last = L == NL - 1
    fl = WALL if last else FLAGS
    use = None
    if THETA > 0 and not FRUGAL:                                       # mass rule: ranks past theta are not used at all
        p10 = pk[:, :K]
        use = ((mx.cumsum(p10, axis=-1) - p10) / p10.sum(axis=-1, keepdims=True)) < THETA
        fl = mx.where(use | (R < KW), fl, 0)
    flat = (ids.astype(mx.int32) | fl).reshape(-1)
    slots = bm["gate"].route_and_gate(L, flat, nw=255, nk=K).reshape(1, KC).astype(mx.int32)
    dep, seq, n = bm["gate"]._last
    res = slots >= 0
    if FRUGAL:                                                         # rank 0, rank 1 (now or late), then the first
        r1 = (R == 1)[None]                                            # resident ranks 2-19 up to 10 in all
        pool = res & ~r1
        sel = (pool & (mx.cumsum(pool.astype(mx.int32), axis=-1) <= K - 1)) | r1
        w = pk * sel
        w = w / w.sum(axis=-1, keepdims=True)                          # renormalised over the 10 (as select_resident)
    else:
        w = pk / pk.sum(axis=-1, keepdims=True)                        # no renormalisation beyond the top-10 sum
        if use is not None:
            w = w * use
    now = res                                                          # late ids are -1 here: phase B adds them
    safe = mx.where(res, slots, slots[:, :1]).astype(mx.uint32)        # rank 0 is always waited -> resident
    base = type(sw).__mro__[1]
    y = (base.__call__(sw, x, safe[None]) * (w * now)[None][..., None].astype(x.dtype)).sum(axis=-2)
    if not last and (LATE1 or (not FRUGAL and KW < K)):
        STASH[L] = late_fn(x, w, dep, seq, n, sw, L)
    return y + attention_gate(self.shared_expert(x), self.shared_expert_gate(x))


def layer_call(self, hidden, *, input_ids, ssm_mask, cache):          # MTPLX DecoderLayer.__call__ + the late writes
    if "ple" in self:
        hidden = hidden + self.ple(hidden, input_ids, cache)
    mixed, hyper, inject = self.attn_hyper_connection(hidden)
    if self.is_linear:
        block_out = self.linear_attn(mixed, ssm_mask, cache)
    else:
        block_out = self.self_attn(mixed, cache)
    L = self._bt_L
    for f, inj in PENDH.pop(L, []):                                   # layer L-1's late experts, before this MLP read
        hyper = _hyper_residual_write(hyper, f(block_out), inj)
    mixed, hyper, inject = self.mlp_hyper_connection(hyper, pending=(block_out, inject))
    block_out = self.mlp(mixed)
    hidden = _hyper_residual_write(hyper, block_out, inject)
    f = STASH.pop(L, None)
    if f is not None:
        if SAME:
            hidden = _hyper_residual_write(hidden, f(block_out), inject)
        else:
            PENDH.setdefault(L + 1, []).append((f, inject))
    return hidden


TM = rt.model.language_model.model if hasattr(rt.model, "language_model") else rt.model.model
for i, l in enumerate(TM.layers):
    l._bt_L = i
qwen4_exp.SparseMoeBlock.__call__ = block
qwen4_exp.DecoderLayer.__call__ = layer_call
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
    """-> tokens, stats (seconds of decode, SSD reads, waited misses, late reads, overflow)"""
    from bigmoe.fuse import reset_generation_state
    reset_generation_state(model)
    STASH.clear(); PENDH.clear()
    out = []
    cache = model.make_cache()
    ids = mx.array([list(prompt_ids)[:-1]])
    for i in range(0, ids.shape[1], 2048):
        lg = model(ids[:, i:i + 2048], cache=cache)
        mx.eval(lg[0] if isinstance(lg, tuple) else lg)
    y = int(prompt_ids[-1])
    ext.reset_stats(); ext.wt_reset(); ext.defer_reset()
    m0, t0 = ext.stats()["misses"], time.perf_counter()
    while len(out) < n:
        lg = model(mx.array([[y]]), cache=cache)
        y = int(mx.argmax((lg[0] if isinstance(lg, tuple) else lg)[0, -1]).item())
        if stop_eos and y in EOS:
            break
        out.append(y)
    t = time.perf_counter() - t0
    st.gate.check()
    wm = ext.wt_stats()[0]; rows, late, over = ext.defer_stats()
    return out, dict(t=t, n=len(out), reads=ext.stats()["misses"] - m0, waited=wm, late=late, over=over, rows=rows)


sha = lambda t: hashlib.sha1(str(list(t)).encode()).hexdigest()[:10]
PROMPTS = {"code": lambda: open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read(),
           "prose": lambda: ("Explain, in plain prose for a curious 14-year-old, how a CPU cache works and why it makes "
                             "programs faster. Use one everyday analogy and keep it to about 300 words.")}
if TASK == "speed":
    P = os.environ.get("PROMPT", "code")
    ids = chat_ids(PROMPTS[P]())
    N = int(os.environ.get("BT_N", "128"))
    generate(ids, 16)                                                   # warm-up
    out, s = generate(ids, N)
    nt = max(1, s["n"])
    print(f"[{P}] defer C={C} K={KW} M={M} {'same' if SAME else 'dh'}: {s['n'] / s['t']:.1f} tok/s, SSD reads/token "
          f"{s['reads'] / nt:.1f} (waited misses {s['waited'] / nt:.1f}, late {s['late'] / nt:.1f}, overflow-waited "
          f"{s['over'] / nt:.2f}), peak {mx.get_peak_memory() / 1e9:.1f} GB, sha {sha(out)} | {tok.decode(out)[:70]!r}", flush=True)
elif TASK.startswith("gsm"):
    N = int(TASK[3:]); OFF = int(os.environ.get("OFFSET", "0"))
    G8 = [json.loads(l) for l in open(os.path.expanduser("~/MoEPrefetch/data/gsm8k/test.jsonl"))][OFF:OFF + N]
    INSTR = "Solve the problem step by step, briefly. End with a line of the form 'Answer: <number>'.\n\n"
    tag = os.environ.get("BT_TAG", f"r{round(100 * C / E):02d}_k{KW}{'s' if SAME else 'h'}")
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
        json.dump({"answers": ans, "ok": ok, "stats": st_l, "C": C, "K": KW, "M": M, "same": SAME}, open(ck, "w"))
    keys = [str(OFF + j) for j in range(len(G8))]
    ref = json.load(open(os.path.expanduser("~/MoEPrefetch/data/w125_gsm_eager.json")))["answers"]
    same_n = sum(1 for k in keys if int(k) < len(ref) and ref[int(k)] == ans.get(k))
    nt = sum(st_l[k]["n"] for k in keys); tt = sum(st_l[k]["t"] for k in keys); rd = sum(st_l[k]["reads"] for k in keys)
    print(f"125B C={C} defer K={KW} [{tag}]: GSM8K {sum(ok[k] for k in keys)}/{len(keys)} (offset {OFF}), answers equal "
          f"to exact eager {same_n}/{sum(1 for k in keys if int(k) < len(ref))}, {nt / max(tt, 1e-9):.1f} tok/s decode, "
          f"{rd / max(1, nt):.1f} reads/token; per-problem {''.join(str(ok[k]) for k in keys)}", flush=True)
elif TASK == "check":   # K=10 must reproduce exact streaming; K<10 must stay coherent (text printed)
    ids = chat_ids(PROMPTS["code"]())
    out, s = generate(ids, 48)
    print(f"check K={KW} {'same' if SAME else 'dh'}: sha {sha(out)} reads {s['reads']} waited {s['waited']} late {s['late']} "
          f"over {s['over']} rows {s['rows']} | {tok.decode(out)[:160]!r}", flush=True)
