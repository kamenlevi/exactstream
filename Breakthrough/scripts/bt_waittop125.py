"""Breakthrough copy of MoEPrefetch scripts/waittop125.py (2026-10-19): identical logic; reads MoEPrefetch's fork125
and GSM8K data read-only, writes checkpoints only to ~/Breakthrough/data/expA. Original docstring:
Selective waiting on Qwen3.8-Flash-Next 125B (BigMoE's model, MTPLX runtime), forked gate in fork125/
(upstream BigMoE 1fde1b8 + fork/apply_patch.py). Decode only (MTP off: BigMoE measured no MTP gain here).
  waittop125.py FRAC MODE TASK
    MODE  exact  : BigMoE's unmodified streamed block (reference)
          w10    : our block, wait for all top-10 (should reproduce exact)
          w2     : wait for missing top-2 of 10, substitute ranks 3-10 from resident ranks 11-20
          wl     : layer-aware: early layers (0-11 of 48) wait only for top-1, later layers top-2
                   (the gate cannot express "wait for none" without a C++ change; top-1 is the conservative stand-in)
    TASK  speed  : BigMoE's code prompt, 128 tokens -> tok/s, stalls, sha
          trace  : (MODE exact) log each decode token's router top-20 per layer on code/chat/math prompts
                   -> data/traces125/<name>.npz  (125B routing traces for residency policy work)
          gsmN   : first N GSM8K problems (env OFFSET), greedy, 600-token cap -> accuracy + per-problem string
Run with BigMoE's venv. Model ~/BigMoE/models/bare is read-only. Summary only."""
import hashlib, json, os, re, sys, time
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
os.environ.setdefault("BIGMOE_SPARE", "8")
os.environ["BIGMOE_PF_NOPUBLISH"] = "1"
ROOT = os.path.expanduser("~/MoEPrefetch")          # read-only: fork125, data/gsm8k
OUT = os.path.expanduser("~/Breakthrough/data/expA")  # all writes go here
UPSTREAM = os.environ.get("W125_UPSTREAM") == "1"     # exact mode via BigMoE's current package (its new flags)
import numpy as np
import mlx.core as mx
if UPSTREAM:
    sys.path.insert(0, os.path.expanduser("~/BigMoE"))
    from bigmoe.loader import load_streamed
    from bigmoe.gated import ext
else:
    sys.path.insert(0, os.path.join(ROOT, "fork125"))
    from bigmoe_wt.loader import load_streamed
    from bigmoe_wt.gated import ext
sys.path.insert(1, os.path.join(ROOT, "scripts"))
from fused_route import topk_probs, select_resident

MD = os.path.expanduser("~/BigMoE/models/bare")
frac, MODE, TASK = float(sys.argv[1]), sys.argv[2], sys.argv[3]
cfg = json.load(open(os.path.join(MD, "config.json"))); cfg = cfg.get("text_config", cfg)
E, K = cfg["num_experts"], cfg["num_experts_per_tok"]
C = max(16, round(frac * E))
if MODE == "exact":
    os.environ["BIGMOE_SPARE"] = "0"
rt, st = load_streamed(MD, C, mtp=False)

if MODE != "exact":
    from mtplx.models import qwen4_exp
    from mtplx.attention_math import attention_gate
    orig_call = qwen4_exp.SparseMoeBlock.__call__
    KC = 20
    NK = int(os.environ.get("BT_NK", K))   # Breakthrough: ranks nw..NK-1 are LRU-touched / loaded in the background;
                                           # BT_NK=2 with w2 = frugal mode: missing ranks >= 2 are never read from the SSD

    def block(self, x):
        sw = self.switch_mlp
        bm = getattr(sw, "_bm", None)
        if bm is None or x.shape[1] != 1 or bm["full"]:
            return orig_call(self, x)
        if bm["L"] == bm["first"]:
            bm["gate"].new_pass()
        ids, pk = topk_probs(self.gate(x)[0], KC)        # fused: softmax over 512 + bitonic top-20
        if MODE == "w10":
            sl = bm["gate"].route_and_gate(bm["L"], ids[0, :K]).reshape(1, 1, K)
            w = pk[None, :, :K]
            if self.norm_topk_prob:
                w = w / mx.maximum(w.sum(axis=-1, keepdims=True), 1e-6)
        else:
            nw = 1 if (MODE == "wl" and bm["L"] < 12) else 2
            slots = bm["gate"].route_and_gate(bm["L"], ids[0], nw=nw, nk=NK).reshape(1, KC).astype(mx.int32)
            sl, w = select_resident(slots, pk, topk=K)     # fused: first K resident, renormalised
            sl, w = sl[None], w[None]
        base = type(sw).__mro__[1]
        y = (base.__call__(sw, x, sl.astype(mx.uint32)) * w[..., None].astype(x.dtype)).sum(axis=-2)
        return y + attention_gate(self.shared_expert(x), self.shared_expert_gate(x))

    qwen4_exp.SparseMoeBlock.__call__ = block

TRACE = {}
if TASK == "trace":
    from mtplx.models import qwen4_exp as _q
    _orig_tr = _q.SparseMoeBlock.__call__
    def _traced(self, x):
        bm = getattr(self.switch_mlp, "_bm", None)
        if bm is not None and x.shape[1] == 1:
            g = self.gate(x)[0, 0].astype(mx.float32)
            top = mx.argsort(-g, axis=-1)[:20]
            TRACE.setdefault(bm["L"], []).append(top.astype(mx.int16))
            TRACE.setdefault(f"G{bm['L']}", []).append(mx.softmax(g)[top].astype(mx.float16))   # Breakthrough: router probs
            for d, (router, ratio, _B) in enumerate(bm.get("next") or [], 1):   # BigMoE's look-ahead predictor:
                xin = x[0, 0] * ratio if ratio is not None else x[0, 0]           # router_{L+d}(MoE input of L), top-32
                TRACE.setdefault(f"P{d}_{bm['L']}", []).append(mx.argsort(-router(xin), axis=-1)[:32].astype(mx.int16))
        return _orig_tr(self, x)
    _q.SparseMoeBlock.__call__ = _traced

from mtplx.sampling import SamplerConfig
from mtplx import generation as G
greedy = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)


def chat_ids(text):
    ids = rt.tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    return list(ids if isinstance(ids[0], int) else ids["input_ids"])


import threading
_last = [None]                       # Breakthrough watchdog: time of the last generated token (None = not started)


def _watchdog(limit=float(os.environ.get("BT_WATCHDOG_S", "180"))):
    while True:
        time.sleep(10)
        if _last[0] is not None and time.time() - _last[0] > limit:
            print(f"HANG watchdog: no token for {time.time() - _last[0]:.0f}s, exiting", flush=True)
            os._exit(3)


threading.Thread(target=_watchdog, daemon=True).start()


def run(ids, n):
    stamps = []
    _last[0] = time.time()           # a new generation: prefill + first tokens must arrive within the limit
    def cb(toks):
        _last[0] = time.time()
        if not stamps:
            ext.reset_stats()
            if hasattr(ext, 'wt_reset'): ext.wt_reset()
        stamps.append((time.perf_counter(), len(toks) if isinstance(toks, list) else 1))
    res = G.generate_ar(rt, ids, max_tokens=n, sampler=greedy, token_callback=cb)
    toks = getattr(res, "tokens", None) or getattr(res, "token_ids", None) or (res.get("tokens") if isinstance(res, dict) else None)
    if toks is None and isinstance(res, tuple): toks = res[0]
    toks = list(toks or [])
    n_dec = max(1, sum(k for _, k in stamps) - stamps[0][1]) if stamps else 1
    dec = stamps[-1][0] - stamps[0][0] if len(stamps) > 1 else float("nan")
    return toks, n_dec / dec if dec == dec and dec > 0 else float("nan"), n_dec


if TASK == "trace":
    os.makedirs(os.path.join(OUT, "traces125"), exist_ok=True)
    G8 = [json.loads(l) for l in open(os.path.join(ROOT, "data", "gsm8k", "test.jsonl"))][:3]
    prompts = {"code": open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read(),
               "chat": "Explain how a refrigerator works to a curious 10 year old.",
               "bulgarian": "Разкажи ми накратко историята на Рилския манастир.",
               "math": "\n\n".join(p["question"] for p in G8)}
    NT = int(os.environ.get("TRACE_TOKENS", "200"))
    for name, text in prompts.items():
        TRACE.clear()
        run(chat_ids(text), NT)
        arr = {(k if isinstance(k, str) else f"L{k}"): np.array(mx.stack(v)) for k, v in TRACE.items()}
        np.savez_compressed(os.path.join(OUT, "traces125", f"{name}{'_' + str(NT) if NT != 200 else ''}{os.environ.get('TRACE_SUFFIX', '')}.npz"), **arr)
        print(f"125B trace {name}: {len(next(iter(TRACE.values())))} decode tokens x {sum(isinstance(k, int) for k in TRACE)} streamed layers, {sum(isinstance(k, str) for k in TRACE)} look-ahead lists", flush=True)
elif TASK == "speed":
    toks, tps, n_dec = run(chat_ids(open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read()), 128)
    st.gate.check()
    s = ext.stats(); wm, rl, pr = ext.wt_stats() if hasattr(ext, 'wt_stats') else (0, 0, 0)
    print(f"125B C={C} {MODE}: decode {tps:.1f} tok/s, stalled layers/tok {s['stalled']/n_dec:.1f}, waited misses/tok {wm/n_dec:.1f}, "
          f"bg loads/tok {rl/n_dec:.1f}, sha {hashlib.sha1(str(toks).encode()).hexdigest()[:10]}, peak {mx.get_peak_memory()/1e9:.1f} GB")
else:
    N = int(TASK[3:])
    OFF = int(os.environ.get("OFFSET", "0"))
    probs = [json.loads(l) for l in open(os.path.join(ROOT, "data", "gsm8k", "test.jsonl"))][OFF:OFF + N]
    INSTR = "Solve the problem step by step, briefly. End with a line of the form 'Answer: <number>'.\n\n"
    tag = os.environ.get("W125_TAG", MODE)
    ck = os.path.join(OUT, f"w125_gsm_{tag}.json")
    state = json.load(open(ck)) if os.path.exists(ck) else {"offset": OFF, "answers": [], "ok": []}
    ok, answers, t0 = state["ok"], state["answers"], time.perf_counter()
    budget = int(os.environ.get("W125_CHUNK", "0")) or len(probs)    # problems per invocation (resumable)
    done_now = 0
    for p in probs[len(answers):]:
        if done_now >= budget:
            break
        done_now += 1
        toks, _, _ = run(chat_ids(INSTR + p["question"]), 600)
        text = rt.tokenizer.decode(toks)
        m = re.findall(r"Answer:\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)", text) or re.findall(r"(-?[\d,]*\.?\d+)", text)
        gold = p["answer"].split("####")[-1].strip().replace(",", "")
        answers.append(m[-1].replace(",", "").rstrip(".") if m else None)
        try:
            ok.append(int(bool(m) and abs(float(m[-1].replace(",", "").rstrip(".")) - float(gold)) < 1e-6))
        except ValueError:
            ok.append(0)
        json.dump({"offset": OFF, "answers": answers, "ok": ok}, open(ck, "w"))   # checkpoint every problem
    st.gate.check()
    print(f"125B C={C} {MODE} [{tag}]: GSM8K {sum(ok)}/{len(ok)} of {len(probs)} target ({time.perf_counter()-t0:.0f}s) per-problem {''.join(map(str, ok))}")
