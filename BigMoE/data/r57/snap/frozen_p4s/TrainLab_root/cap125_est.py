"""TrainLab: capture for a learned top-2 pre-router ESTIMATOR on the 125B (NewIdeas 19:40; Speculating Experts style).
Teacher-forced prefill with BigMoE's upstream streamed package at C = 25 % (MoEPrefetch's cap125.py recipe, read-only
use of ~/BigMoE). Feasibility first: five representative target layers L in {8, 16, 24, 32, 39}; per document saves
x_{L-1} (the MoE input of layer L-1 = BigMoE's d=1 tap, fp16), the true router logits of L (fp16), and the true top-10
ids of L-1 and of L (routing history features). ~1 GB for ~40K tokens -> data/cap125est/<doc>.npz. Fit with
fit125_est.py (CPU). Corpus disjoint from MoEPrefetch's validation docs where possible; held-out split by document.
  cap125_est.py [max_tokens_per_doc=1536]      (BigMoE's venv; GPU, ~24 GB)"""
import glob, json, os, subprocess, sys, sysconfig, time
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
os.environ["BIGMOE_PREFETCH"] = "1"                          # bm["next"] = router_{L+1}: the training-free tap
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import numpy as np
import mlx.core as mx
from bigmoe.loader import load_streamed

TL = os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p4s/TrainLab_root")
OUT = os.path.join(TL, "data", "cap125est"); os.makedirs(OUT, exist_ok=True)
NT = int(sys.argv[1]) if len(sys.argv) > 1 else 1536
TARGETS = (8, 16, 24, 32, 39)
NEED = set(TARGETS) | {L - 1 for L in TARGETS}
MD = os.path.expanduser("~/BigMoE/models/bare")
cfg = json.load(open(os.path.join(MD, "config.json"))); cfg = cfg.get("text_config", cfg)
t0 = time.time()
rt, st = load_streamed(MD, max(16, round(0.25 * cfg["num_experts"])), mtp=False)
print(f"loaded in {time.time() - t0:.0f}s", flush=True)

from mtplx.models import qwen4_exp as _q
CAP = {}
_orig = _q.SparseMoeBlock.__call__


def _cap(self, x):
    bm = getattr(self.switch_mlp, "_bm", None)
    if bm is not None and x.shape[1] > 1 and bm["L"] in NEED:             # prefill chunks of the layers we need
        g = self.gate(x)[0]
        top = mx.argpartition(-g.astype(mx.float32), 10, axis=-1)[:, :10]
        nx = bm.get("next")
        tap = nx[0][0](x[0]).astype(mx.float16) if (nx and bm["L"] + 1 in TARGETS) else None
        CAP.setdefault(bm["L"], []).append((x[0].astype(mx.float16), g.astype(mx.float16), top.astype(mx.int16), tap))
    return _orig(self, x)


_q.SparseMoeBlock.__call__ = _cap
from mtplx.sampling import SamplerConfig
from mtplx import generation as G
greedy = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)

# corpus: TrainLab's mixed training sources (code, chat transcripts, multilingual wiki, notes, GSM8K outside 200-399)
docs = {}
std = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py")))
for i, f in enumerate(std[5::13][:6]): docs[f"py{i}"] = open(f, errors="ignore").read()
for i, f in enumerate(sorted(glob.glob(os.path.join(TL, "data/multi/*.txt")))): docs[f"wiki{i}"] = open(f).read()[20000:40000]
G8 = [json.loads(l) for l in open(os.path.expanduser("~/MoEPrefetch/data/gsm8k/test.jsonl"))]
docs["gsm_c"] = "\n\n".join(p["question"] + "\n" + p["answer"] for p in G8[600:640])
docs["gsm_d"] = "\n\n".join(p["question"] + "\n" + p["answer"] for p in G8[700:740])
sys.path.insert(0, TL)
import build_data as B
for i, d in enumerate(B.chat_docs()):
    if i >= 8: break
    docs[f"chat{i}"] = d
for i, f in enumerate(sorted(glob.glob(os.path.expanduser("~/Breakthrough/research/*.md")))[:4]): docs[f"note{i}"] = open(f).read()
done = 0
for name, text in docs.items():
    if os.path.exists(os.path.join(OUT, f"{name}.npz")) or len(text) < 800:
        continue
    ids = rt.tokenizer.encode(text)[:NT]
    if isinstance(ids, dict): ids = ids["input_ids"]
    CAP.clear()
    t1 = time.time()
    G.generate_ar(rt, list(ids), max_tokens=1, sampler=greedy)
    d = {}
    for L in TARGETS:
        d[f"x{L - 1}"] = np.array(mx.concatenate([c[0] for c in CAP[L - 1]]))      # [T, D] MoE input of L-1
        d[f"g{L}"] = np.array(mx.concatenate([c[1] for c in CAP[L]]))              # [T, E] true logits of L
        d[f"top{L}"] = np.array(mx.concatenate([c[2] for c in CAP[L]]))            # [T, 10] true top-10 of L
        d[f"top{L - 1}"] = np.array(mx.concatenate([c[2] for c in CAP[L - 1]]))    # [T, 10] true top-10 of L-1
        d[f"tap{L}"] = np.array(mx.concatenate([c[3] for c in CAP[L - 1]]))       # [T, E] router_L(x_{L-1})
    np.savez(os.path.join(OUT, f"{name}.npz"), **d)
    done += d[f"g{TARGETS[0]}"].shape[0]
    print(f"{name}: {d[f'g{TARGETS[0]}'].shape[0]} tokens ({time.time() - t1:.0f}s)", flush=True)
print(f"TOTAL {done} tokens, {time.time() - t0:.0f}s")
