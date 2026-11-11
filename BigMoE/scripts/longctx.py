"""Long-context decode on the 125B (MoonShots request): prompts of ~4K/16K/32K tokens made of real repo text
(FINDINGS/LEDGER/source of ~/BigMoE, ~/SpecDecode, ~/MoEPrefetch, ~/MoonShots), then plain greedy (bigmoe.decode)
and spec125 k=3 (lookup off), N tokens each. Prints prefill time, tok/s, misses and stalled layers per token, peak.
Usage: longctx.py [frac=0.50] [N=128] [lengths=4096,16384,32768]   (reference env)"""
import os, sys, time, glob, hashlib
for k, v in (("MTPLX_NGRAM_PREWARM", "off"), ("MTPLX_COMPILED_VERIFY", "0"), ("MTPLX_QWEN4EXP_COMPILE", "0"),
             ("MTPLX_COMPILE_AR_FORWARD", "0"), ("MTPLX_COMPILED_TARGET_PREFIX", "0"), ("MTPLX_FAMILY_CAPTURE_COMMIT", "1")):
    os.environ.setdefault(k, v)
sys.path.insert(0, os.path.expanduser("~/BigMoE")); sys.path.insert(0, os.path.expanduser("~/SpecDecode"))
import mlx.core as mx
from bigmoe.loader import load_streamed
from bigmoe.gated import ext
from bigmoe.decode import greedy
import spec125
frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.50
N = int(sys.argv[2]) if len(sys.argv) > 2 else 128
lens = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "4096,16384,32768").split(",")]
rt, st = load_streamed(os.path.expanduser("~/BigMoE/models/bare"), round(frac * 512))
tok = rt.tokenizer
files = []
for r in ("BigMoE", "SpecDecode", "MoEPrefetch", "MoonShots"):
    d = os.path.expanduser("~/" + r)
    files += sorted(glob.glob(d + "/*.md")) + sorted(glob.glob(d + "/**/*.py", recursive=True))
files = [f for f in files if "/.venv" not in f and "/site-packages/" not in f]
corpus = ""
for f in files:
    try:
        corpus += f"\n\n===== {f} =====\n" + open(f, errors="ignore").read()
    except OSError:
        pass
body = tok.encode(corpus)
print(f"corpus {len(body)} tokens from {len(files)} files", flush=True)
sha = lambda t: hashlib.sha1(str(list(t)).encode()).hexdigest()[:10]
for T in lens:
    text = tok.decode(body[:T - 64])
    msgs = [{"role": "user", "content": text + "\n\nSummarize what the code above does, file by file."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
    ids = ids if isinstance(ids[0], int) else ids["input_ids"]
    for mode in ("plain", "spec3"):
        mx.reset_peak_memory()
        stamps = []
        cb = lambda t: (stamps.append(time.perf_counter()), ext.reset_stats() if len(stamps) == 1 else None)
        t0 = time.perf_counter()
        if mode == "plain":
            out = greedy(rt.model, ids, N, on_token=cb)
        else:
            out, sst = spec125.generate(rt.model, ids, N, k_mtp=3, max_draft=5, lookup=False, on_token=cb)
        s = ext.stats()
        n = len(stamps) - 1
        dec = (stamps[-1] - stamps[0]) if n > 0 else float("nan")
        extra = f" tok/pass {sst['tok_per_pass']:.2f}" if mode == "spec3" else ""
        print(f"T={len(ids)} {mode}: prefill+first {stamps[0] - t0:.1f}s, decode {n / dec:.1f} tok/s, per token: misses "
              f"{s['misses'] / n:.1f} stalled {s['stalled'] / n:.1f}{extra}, peak {mx.get_peak_memory() / 1e9:.1f} GB, "
              f"sha {sha(out)}", flush=True)
st.gate.check()
