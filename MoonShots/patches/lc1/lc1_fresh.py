"""LC1 discriminator: is the 32K plain-vs-spec3 mismatch a kernel (batch invariance) or in-process state?
longctx.py runs plain then spec3 in ONE process. On the streamed 35B, repeated MTP-spec generations in one process are not
reproducible (MoonShots data/host35_exact.txt: runs 1-3 sha 70904c380e with 18/17/18 passes, run 4 844b0186b9; every
fresh process gives 70904c380e). So first compare each mode in its own fresh process on the SAME frozen prompt.
usage (BigMoE's reference env + the LC1 flags; quiet lock; ~2 x (load + 32K prefill + N tokens)):
  lc1_fresh.py plain [T=32768] [N=128]      # fresh process, plain greedy only
  lc1_fresh.py spec3 [T=32768] [N=128]      # fresh process, spec125 k=3 only
  lc1_fresh.py both  [T=32768] [N=128]      # longctx.py's in-process order (plain, then spec3), for the control
  lc1_fresh.py cmp   [T=32768]              # first differing token between the saved runs
The prompt ids are built once with longctx.py's recipe and frozen in IDS (the corpus changes whenever a repo changes)."""
import os, sys, time, glob, hashlib, json
for k, v in (("MTPLX_NGRAM_PREWARM", "off"), ("MTPLX_COMPILED_VERIFY", "0"), ("MTPLX_QWEN4EXP_COMPILE", "0"),
             ("MTPLX_COMPILE_AR_FORWARD", "0"), ("MTPLX_COMPILED_TARGET_PREFIX", "0"), ("MTPLX_FAMILY_CAPTURE_COMMIT", "1")):
    os.environ.setdefault(k, v)                       # as scripts/longctx.py
mode = sys.argv[1]; T = int(sys.argv[2]) if len(sys.argv) > 2 else 32768; N = int(sys.argv[3]) if len(sys.argv) > 3 else 128
OUT = os.environ.get("LC1_OUT", os.path.expanduser("~/MoonShots/data/lc1"))
os.makedirs(OUT, exist_ok=True)
IDS = f"{OUT}/ids_{T}.json"
if mode == "cmp":
    runs = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/tok_{T}_*.json"))}
    names = sorted(runs)
    ref = runs.get(f"tok_{T}_plain_fresh") or runs[names[0]]
    for n in names:
        t = runs[n]["tokens"]; r = ref["tokens"]
        d = next((i for i, (a, b) in enumerate(zip(t, r)) if a != b), None)
        print(f"{n:28s} sha {runs[n]['sha']}  passes {runs[n].get('passes', '-')}  first diff vs plain_fresh: {d}")
    sys.exit()
sys.path.insert(0, os.path.expanduser("~/BigMoE")); sys.path.insert(0, os.path.expanduser("~/SpecDecode"))
import mlx.core as mx
from bigmoe.loader import load_streamed
from bigmoe.gated import ext
from bigmoe.decode import greedy
import spec125
rt, st = load_streamed(os.path.expanduser("~/BigMoE/models/bare"), round(float(os.environ.get("LC1_FRAC", "0.50")) * 512))
tok = rt.tokenizer
if not os.path.exists(IDS):
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
    text = tok.decode(body[:T - 64])
    ids = tok.apply_chat_template([{"role": "user", "content": text + "\n\nSummarize what the code above does, file by file."}],
                                  add_generation_prompt=True, enable_thinking=False)
    ids = list(ids if isinstance(ids[0], int) else ids["input_ids"])
    json.dump(ids, open(IDS, "w"))
ids = json.load(open(IDS))
sha = lambda t: hashlib.sha1(str(list(t)).encode()).hexdigest()[:10]
tag = "fresh" if mode != "both" else "inproc"
for m in (("plain", "spec3") if mode == "both" else (mode,)):
    t0 = time.perf_counter()
    if m == "plain":
        out, extra = greedy(rt.model, ids, N), {}
    else:
        out, sst = spec125.generate(rt.model, ids, N, k_mtp=3, max_draft=5, lookup=False)
        extra = {"passes": round(N / sst["tok_per_pass"])}
    out = [int(t) for t in out]
    json.dump(dict(tokens=out, sha=sha(out), env={k: v for k, v in os.environ.items() if k.startswith(("BIGMOE_", "MTPLX_"))}, **extra),
              open(f"{OUT}/tok_{T}_{m}_{tag}.json", "w"))
    print(f"T={len(ids)} {m} ({tag}): sha {sha(out)} in {time.perf_counter() - t0:.0f}s", flush=True)
