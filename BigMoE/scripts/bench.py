"""Real decode run: Qwen3.8-Flash-Next 4-bit, routed experts streamed from SSD (exact).
usage: bench.py C_FRACTION MODE[,MODE..] [N_TOKENS] [PROMPT_FILE]
  MODE = ar | mtp1 | mtp2 | mtp3
Prints one summary line per mode; tokens + stats -> data/runs/<tag>.json"""
import hashlib, json, os, sys, time
# The gate runs Python per layer call; a compiled (traced) trunk forward would run it only once.
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")   # MTPLX would pre-read up to ~18 GiB of the n-gram table into page cache
for _k, _v in (("MTPLX_FAMILY_CAPTURE_COMMIT", "1"), ("MTPLX_DROP_EVENTS", "1"), ("MTPLX_LAZY_MTP_HISTORY_APPEND", "1"), ("MTPLX_DEFER_REPAIR_EVAL", "1")):
    os.environ.setdefault(_k, _v)   # MTPLX server defaults: commit rejected MTP rounds without a 2nd trunk forward
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import mlx.core as mx
from bigmoe.loader import load_streamed

MD = os.path.expanduser(os.environ.get("BIGMOE_MD", "~/BigMoE/models/bare"))   # BIGMOE_MD: dry runs on other models
frac = float(sys.argv[1]); modes = sys.argv[2].split(",")
from bigmoe.gated import ext
N = int(sys.argv[3]) if len(sys.argv) > 3 else 128
pf = sys.argv[4] if len(sys.argv) > 4 else os.path.expanduser("~/BigMoE/data/prompt_code.txt")
_tc = json.load(open(os.path.join(MD, "config.json"))); _tc = _tc.get("text_config", _tc)
C = max(16, round(frac * _tc["num_experts"]))
t0 = time.perf_counter()
rt, st = load_streamed(MD, C, mtp=not os.environ.get("BIGMOE_NOMTP"))
t_load = time.perf_counter() - t0
from mtplx.sampling import SamplerConfig
from mtplx import generation as G
greedy = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)
msgs = [{"role": "user", "content": open(pf).read()}]
ids = rt.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
ids = list(ids if isinstance(ids[0], int) else ids["input_ids"])
mem = lambda: (mx.get_active_memory() / 1e9, mx.get_peak_memory() / 1e9)
print(f"loaded C={C}/512 ({frac:.0%}) in {t_load:.0f}s; pools {st.resident_gb():.1f} GB; active {mem()[0]:.1f} GB; prompt {len(ids)} tok", flush=True)
os.makedirs(os.path.expanduser("~/BigMoE/data/runs"), exist_ok=True)
for mode in modes:
    stamps = []
    def cb(toks):
        if not stamps: ext.reset_stats()         # decode-only stats (prefill excluded)
        stamps.append((time.perf_counter(), len(toks) if isinstance(toks, list) else 1))
    ext.reset_stats(); t0 = time.perf_counter()
    from bigmoe import stream as _S; _S.SYNC.update(calls=0, rows=0)
    if mode == "ar2":                            # pipelined greedy loop (bigmoe/decode.py)
        from bigmoe import decode as _D
        res = {"tokens": _D.greedy(rt.model, ids, N, on_token=lambda t: cb([t]))}
    elif mode == "ar":
        res = G.generate_ar(rt, ids, max_tokens=N, sampler=greedy, token_callback=cb)
    else:
        res = G.generate_mtpk(rt, ids, max_tokens=N, sampler=greedy, speculative_depth=int(mode[3:]), token_callback=cb, mtp_history_policy="committed")
    t_all = time.perf_counter() - t0
    toks = getattr(res, "tokens", None) or getattr(res, "token_ids", None) or (res.get("tokens") if isinstance(res, dict) else None)
    if toks is None and isinstance(res, tuple): toks = res[0]
    toks = list(toks or [])
    n_cb = sum(n for _, n in stamps)
    t_first = stamps[0][0] - t0 if stamps else float("nan")
    dec = (stamps[-1][0] - stamps[0][0]) if len(stamps) > 1 else float("nan")
    n_dec = max(1, n_cb - stamps[0][1]) if stamps else 1
    st.gate.check()                              # raises if any spin gate timed out
    print(f"   sync-path calls during run: {_S.SYNC['calls']} ({_S.SYNC['rows']} rows); gated calls {ext.stats()['gated']}", flush=True)
    s = ext.stats(); per = max(1, n_dec)
    tag = f"C{C}_{mode}_N{N}"
    out = dict(C=C, mode=mode, N=N, prompt_tokens=len(ids), tokens=toks, t_first=t_first, decode_s=dec, decode_tokens=n_dec,
               stats=s, fast_hits=st.gate.fast_hits(),
               active_gb=mem()[0], peak_gb=mem()[1], text=rt.tokenizer.decode(toks)[:400])
    json.dump(out, open(os.path.expanduser(f"~/BigMoE/data/runs/{tag}.json"), "w"), indent=1)
    h = hashlib.sha1(str(toks).encode()).hexdigest()[:10]
    print(f"{mode:5s} C={C}: decode {n_dec/dec if dec==dec and dec>0 else float('nan'):6.1f} tok/s ({n_dec} tok), prefill+first {t_first:6.1f}s | "
          f"per gen token: misses {s['misses']/per:5.1f} stalled layers {s['stalled']/per:5.1f} io {s['io_s']/per*1e3:5.1f} ms "
          f"slow calls {s['slow_calls']/per:4.1f} x {s['slow_s']/max(1, s['slow_calls'])*1e6:.0f} us, fast hits {st.gate.fast_hits()/per:4.1f} | "
          f"peak {mem()[1]:.1f} GB | tokens {len(toks)} sha {h} | {out['text'][:60]!r}", flush=True)
