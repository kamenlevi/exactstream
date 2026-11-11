"""Breakthrough D1 x speculation (2026-10-20): frugal wait-top-2 routing on the streamed 125B with MTP / prompt-lookup
speculative decoding (SpecDecode's spec125.generate, read-only import).

Engine: ~/Breakthrough/fork125bt = MoEPrefetch's fork125 (BigMoE ffc859b + selective waiting) with one change in the
native loader: ensure_wait protects only what fits in the pool (waited ids, then resident candidates in list order), so
S verify rows x KC candidate ids can exceed C at 5-10 % residency without "no free slot" aborts. Eviction is unchanged
whenever all candidates fit (every eager frugal run so far).

Frugal block (every streamed MoE call with S <= 8 rows; longer calls = prefill run the exact streamed path):
  router top-KC per row (KC = min(20, MAXK // S)); the routing row is rank-major (all rows' rank 0, then all rank 1,
  ...), so the loader waits only for the first 2*S ids (each row's ranks 0-1), LRU-touches resident ranks 2-9 and never
  reads a missing one (BIGMOE_SPARE=0); per row the K = 10 experts used are the first 10 resident in rank order,
  gates renormalised over them. S = 1 is exactly the eager frugal setup of data/expA (GSM8K-150 146 at 10 %).
usage: bt_frugal_spec.py FRAC TASK [CONFIGS]
  TASK  smoke : 24 tokens eager + 24 tokens with the first config (mechanics only)
        speed : PROMPT env (code | prose | edit), BT_N tokens (128): eager (ar2, pipelined) then each config
        gsmN  : GSM8K problems OFFSET..OFFSET+N with the first config ('ar2' = eager), 600-token cap, EOS stop;
                checkpoint per problem in data/expS/bt_gsm_<tag>.json; compared with BigMoE's exact eager answers
  CONFIGS: comma list of k_mtp:max_draft:lookup (spec125.generate), default 3:3:1
Env: BT_TAG (checkpoint tag), BT_SMAX (8), BIGMOE_MTP_C (64). Run with BigMoE's venv, PYTHONDONTWRITEBYTECODE=1."""
import hashlib, json, os, re, sys, time

os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for _k, _v in (("MTPLX_FAMILY_CAPTURE_COMMIT", "1"), ("MTPLX_DROP_EVENTS", "1"), ("MTPLX_LAZY_MTP_HISTORY_APPEND", "1"),
               ("MTPLX_DEFER_REPAIR_EVAL", "1")):
    os.environ.setdefault(_k, _v)
for _k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(_k, "0")
PF = os.environ.get("BT_PF") == "1"        # frugal + look-ahead prefetch of predicted experts into spare slots
os.environ["BIGMOE_SPARE"] = os.environ.get("BT_SPARE", "12") if PF else "0"   # spare slots are part of C (C_main = C - spare)
if PF:
    os.environ.setdefault("BIGMOE_PREFETCH", "12")
    os.environ.setdefault("BIGMOE_PF_JIT", "1")
    os.environ.setdefault("BIGMOE_PF_QD", "4")
os.environ["BIGMOE_PF_NOPUBLISH"] = "1"
for _k, _v in (("BIGMOE_MAXK", "80"), ("BIGMOE_COMPILE_HC", "1"), ("BIGMOE_HC_VREAD", "1"), ("BIGMOE_ASYNC_EVERY", "1"),
               ("BIGMOE_FRSPEC", "1"), ("BIGMOE_MTP_STREAM", "1"), ("BIGMOE_MTP_C", "64")):
    os.environ.setdefault(_k, _v)
ENG = os.path.expanduser("~/Breakthrough/fork125bt")
sys.path.insert(0, ENG)                                   # `bigmoe` -> fork125bt/bigmoe_wt (symlink)
sys.path.insert(1, os.path.expanduser("~/MoEPrefetch/scripts"))
sys.path.insert(2, os.path.expanduser("~/SpecDecode"))
import mlx.core as mx
from bigmoe.loader import load_streamed
from bigmoe.gated import ext, MAXK, PRED_LINES
from bigmoe.stream import _CORR                         # None unless BIGMOE_PF_CORR names MoEPrefetch's factors
from bigmoe import decode as D
from fused_route import topk_probs, select_resident
import spec125

OUT = os.path.expanduser("~/Breakthrough/data/expS")
os.makedirs(OUT, exist_ok=True)
MD = os.path.expanduser("~/BigMoE/models/bare")
frac, TASK = float(sys.argv[1]), sys.argv[2]
CONFIGS = (sys.argv[3] if len(sys.argv) > 3 else "3:3:1").split(",")
cfg = json.load(open(os.path.join(MD, "config.json"))); cfg = cfg.get("text_config", cfg)
E, K = cfg["num_experts"], cfg["num_experts_per_tok"]
C = int(os.environ.get("BT_C", "0")) or max(16, round(frac * E))   # BT_C: exact slot count (plane-cache emulation)
SMAX = int(os.environ.get("BT_SMAX", "8"))
NW = 2
t0 = time.perf_counter()
rt, st = load_streamed(MD, C, mtp=True)
print(f"bt_frugal_spec: C={C} of {E} ({100 * C / E:.0f} %), loaded in {time.perf_counter() - t0:.0f}s, "
      f"active {mx.get_active_memory() / 1e9:.1f} GB, MAXK {MAXK}, SMAX {SMAX}", flush=True)

from mtplx.models import qwen4_exp
from mtplx.attention_math import attention_gate
orig_call = qwen4_exp.SparseMoeBlock.__call__
CALLS = {"frugal": 0, "rows": 0, "exact": 0}
PLAN = os.environ.get("BT_PLAN")   # e.g. 4433332222: bits by position in the used list (MoonShots planes125, one-row
if PLAN:                           # calls only: eager decode; verify rows stay 4-bit). Emulates a plane cache whose
    sys.path.insert(0, os.path.expanduser("~/MoonShots/patches/planes125"))   # MSB tier holds C experts (plane_sim.py)
    import planes125
    planes125.install(0, rt.model, plan=PLAN)
    print(f"bt_frugal_spec: plan {PLAN} on one-row calls (planes125)", flush=True)


def block(self, x):
    sw = self.switch_mlp
    bm = getattr(sw, "_bm", None)
    S = x.shape[1]
    kc = min(20, MAXK // max(1, S))
    if bm is None or bm["full"] or S > SMAX or kc < K:
        CALLS["exact"] += 1
        return orig_call(self, x)
    CALLS["frugal"] += 1; CALLS["rows"] += S
    if bm["L"] == bm["first"]:
        bm["gate"].new_pass()
    ae = bm.get("async_every", 0)
    if ae and bm["L"] % ae == ae - 1:   # as the streamed SwitchGLU: hand the graph so far to the GPU
        mx.async_eval(x)
    ids, pk = topk_probs(self.gate(x)[0], kc)                     # [S, kc] router top-kc, softmax probs (descending)
    flat = ids.T.reshape(-1) if S > 1 else ids.reshape(-1)        # rank-major: every row's rank 0 first, then rank 1 ...
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
            if _CORR is not None and router is nx[0][0] and bm["L"] < len(_CORR):   # MoEPrefetch's corrected d=1
                _c = _CORR[bm["L"]]                                               # (BIGMOE_PF_CORR, as stream.py)
                lg = lg.astype(mx.float32) + ((rows.astype(mx.float32) - _c[2]) @ _c[0]) @ _c[1] + _c[3]
            parts.append(mx.argpartition(-lg, kth=b - 1, axis=-1)[:, :b].reshape(-1)); cnt.append(b * rows.shape[0])
            left -= cnt[-1]
        if parts:
            preds = mx.concatenate(parts) if len(parts) > 1 else parts[0]
            counts = tuple(cnt + [0] * (3 - len(cnt))) + (0,)
    slots = bm["gate"].route_and_gate(bm["L"], flat, preds=preds, counts=counts, nw=NW * S, nk=K * S)
    slots = (slots.reshape(kc, S).T if S > 1 else slots.reshape(1, kc)).astype(mx.int32)
    sl, w = select_resident(slots, pk, topk=K)                    # first K resident per row, renormalised
    base = type(sw).__mro__[1]
    y = (base.__call__(sw, x, sl[None].astype(mx.uint32)) * w[None][..., None].astype(x.dtype)).sum(axis=-2)
    return y + attention_gate(self.shared_expert(x), self.shared_expert_gate(x))


qwen4_exp.SparseMoeBlock.__call__ = block
tok = rt.tokenizer


def chat_ids(text):
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    return list(ids if isinstance(ids[0], int) else ids["input_ids"])


EOS = set()
for _a in ("eos_token_ids", "eos_token_id"):
    _v = getattr(tok, _a, None)
    if _v is not None:
        EOS |= set(_v) if isinstance(_v, (list, set, tuple)) else {_v}
for _t in ("<|im_end|>", "<|endoftext|>"):
    try:
        _i = tok.convert_tokens_to_ids(_t)
        if isinstance(_i, int) and _i >= 0:
            EOS.add(_i)
    except Exception:
        pass
EOS = {int(e) for e in EOS if e is not None}


class _Stop(Exception):
    pass


def gen(ids, n, conf, stop_eos=False):
    """-> tokens, tok/s, stats dict (passes, tok/pass, waited misses / token, ...)"""
    out, stamps = [], []

    def cb(t):
        if stamps == []:
            ext.reset_stats()
            if hasattr(ext, "wt_reset"): ext.wt_reset()
        stamps.append(time.perf_counter())
        if stop_eos and int(t) in EOS:
            raise _Stop
        out.append(int(t))
    s = {}
    try:
        if conf == "ar2":
            D.greedy(rt.model, ids, n, on_token=cb)
        else:
            k, md, lk = (int(v) for v in conf.split(":")[:3])
            spec125.generate(rt.model, ids, n, k_mtp=k, max_draft=md, lookup=lk, stats=s, on_token=cb)
    except _Stop:
        pass
    st.gate.check()
    nd = max(1, len(stamps) - 1)
    tps = nd / (stamps[-1] - stamps[0]) if len(stamps) > 1 else float("nan")
    wm = ext.wt_stats()[0] if hasattr(ext, "wt_stats") else 0
    s.update(waited_per_tok=wm / nd, n=len(out))
    return out, tps, s


sha = lambda t: hashlib.sha1(str(list(t)).encode()).hexdigest()[:10]
PROMPTS = {"code": lambda: open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read(),
           "prose": lambda: ("Explain, in plain prose for a curious 14-year-old, how a CPU cache works and why it makes "
                             "programs faster. Use one everyday analogy and keep it to about 300 words."),
           "edit": lambda: ("Here is a file:\n\n```python\n" + open(os.path.expanduser("~/SpecDecode/scripts/inventory_sample.py")).read()
                            + "```\n\nRewrite the whole file with total_value renamed to total everywhere. Output only the full file in one code block.")}

if TASK in ("smoke", "speed"):
    P = os.environ.get("PROMPT", "code")
    ids = chat_ids(PROMPTS[P]())
    N = 24 if TASK == "smoke" else int(os.environ.get("BT_N", "128"))
    ref, tps, s = gen(ids, N, "ar2")
    print(f"[{P}] ar2 C={C}: {tps:6.1f} tok/s, waited misses/tok {s['waited_per_tok']:.1f}, sha {sha(ref)} | "
          f"{tok.decode(ref)[:70]!r}", flush=True)
    if TASK == "speed":
        gen(ids, 16, CONFIGS[0])                                  # warm-up (first spec run after load is slow)
    for conf in CONFIGS:
        CALLS.update(frugal=0, rows=0, exact=0)
        out, tps, s = gen(ids, N, conf)
        div = next((i for i, (a, b) in enumerate(zip(out, ref)) if a != b), None)
        print(f"[{P}] spec {conf} C={C}: {tps:6.1f} tok/s, {s.get('tok_per_pass', float('nan')):.2f} tok/pass, "
              f"lookup {s.get('lookup_used', 0)}/{s.get('passes', 0)}, waited misses/tok {s['waited_per_tok']:.1f}, "
              f"draft {s.get('t_draft', 0):.2f}s verify {s.get('t_verify', 0):.2f}s, refused {s.get('commit_refused', 0)}, "
              f"frugal calls {CALLS['frugal']} ({CALLS['rows'] / max(1, CALLS['frugal']):.2f} rows) exact {CALLS['exact']}, "
              f"sha {sha(out)} {'= ar2' if out == ref else 'differs from ar2 at ' + str(div)} | {tok.decode(out)[:70]!r}", flush=True)
    print(f"peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
elif TASK.startswith("gsm"):
    N = int(TASK[3:])
    OFF = int(os.environ.get("OFFSET", "0"))
    G8 = [json.loads(l) for l in open(os.path.expanduser("~/MoEPrefetch/data/gsm8k/test.jsonl"))]
    probs = G8[OFF:OFF + N]
    INSTR = "Solve the problem step by step, briefly. End with a line of the form 'Answer: <number>'.\n\n"
    conf = CONFIGS[0]
    tag = os.environ.get("BT_TAG", f"r{round(frac * 100):02d}_{conf.replace(':', '')}")
    ck = os.path.join(OUT, f"bt_gsm_{tag}.json")
    state = json.load(open(ck)) if os.path.exists(ck) else {}
    ans, ok, tps_l, n_l = (state.get(k, {}) for k in ("answers", "ok", "tps", "ntok"))
    t0 = time.perf_counter()
    for j, p in enumerate(probs):
        key = str(OFF + j)
        if key in ok:
            continue
        toks, tps, s = gen(chat_ids(INSTR + p["question"]), 600, conf, stop_eos=True)
        text = tok.decode(toks)
        m = re.findall(r"Answer:\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)", text) or re.findall(r"(-?[\d,]*\.?\d+)", text)
        gold = p["answer"].split("####")[-1].strip().replace(",", "")
        a = m[-1].replace(",", "").rstrip(".") if m else None
        try:
            good = int(a is not None and abs(float(a) - float(gold)) < 1e-6)
        except ValueError:
            good = 0
        ans[key], ok[key], tps_l[key], n_l[key] = a, good, round(tps, 2), len(toks)
        json.dump({"answers": ans, "ok": ok, "tps": tps_l, "ntok": n_l, "C": C, "conf": conf}, open(ck, "w"))
    keys = [str(OFF + j) for j in range(len(probs))]
    ref = json.load(open(os.path.expanduser("~/MoEPrefetch/data/w125_gsm_eager.json")))   # BigMoE exact eager, 0-149
    ra = ref.get("answers", [])
    same = sum(1 for k in keys if int(k) < len(ra) and ra[int(k)] == ans.get(k))
    nref = sum(1 for k in keys if int(k) < len(ra))
    tot_tok = sum(n_l[k] for k in keys)
    mean_tps = tot_tok / max(1e-9, sum(n_l[k] / tps_l[k] for k in keys if tps_l[k] == tps_l[k] and tps_l[k] > 0))
    print(f"125B C={C} frugal spec {conf} [{tag}]: GSM8K {sum(ok[k] for k in keys)}/{len(keys)} (offset {OFF}), "
          f"answers equal to exact eager {same}/{nref}, decode {mean_tps:.1f} tok/s (token-weighted, {tot_tok} tokens), "
          f"{time.perf_counter() - t0:.0f}s, peak {mx.get_peak_memory() / 1e9:.1f} GB; per-problem "
          f"{''.join(str(ok[k]) for k in keys)}", flush=True)
