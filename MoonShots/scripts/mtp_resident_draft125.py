"""Round 32o: resident-only routing for the MTP drafter (24 GB mode, MTP_STREAM=1). The MTP head's experts stream through
gate layer 48 and its 3 draft steps stall on misses (~1.4-1.8 ms each, ~4.5-5.1 ms/pass: data/mtp_draft_stalls125.txt).
Drafts only affect acceptance (verification stays exact), so the MTP router may pick among RESIDENT experts only: mask
the router's probabilities with the layer-48 residency snapshot (ext.lookup, CPU table; the MTP pool only evicts on its
own misses, which then no longer happen), top-k among them, renormalize as the block does. Arms (one process each):
  A  normal drafts                B  resident-only drafts
Same greedy prompts; reports tokens, verify passes (tokens/pass = acceptance), wall tok/s, and checks that B's output
tokens == A's (they must: verification is exact). usage: [REF ENV] mtp_resident_draft125.py [FRAC] [MAXTOK]
(GPU, model memory; 24 GB mode needs BIGMOE_MTP_STREAM=1 BIGMOE_RECORDS=3 in the environment)"""
import hashlib, json, os, subprocess, sys, time
ARM = os.environ.get("MRD_ARM")
FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 192
if ARM is None:
    res = {}
    for arm in ("A", "B"):
        out = subprocess.run([sys.executable, __file__, str(FRAC), str(MAXTOK)], env=dict(os.environ, MRD_ARM=arm),
                             capture_output=True, text=True)
        lines = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
        res[arm] = json.loads(lines[-1][7:]) if lines else {"error": out.stderr[-800:]}
        print(arm, json.dumps(res[arm]), flush=True)
    if all("prompts" in r for r in res.values()):
        for p in res["A"]["prompts"]:
            a, b = res["A"]["prompts"][p], res["B"]["prompts"][p]
            print(f"{p}: tokens/pass A {a['tpp']:.2f} B {b['tpp']:.2f} | tok/s A {a['tps']:.1f} B {b['tps']:.1f} ({b['tps'] / a['tps'] - 1:+.1%}) | "
                  f"outputs identical: {a['sha'] == b['sha']}", flush=True)
    sys.exit(0)
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import numpy as np
import mlx.core as mx
from bigmoe.loader import load_streamed
from bigmoe.gated import ext
import mtplx.models.qwen4_exp as Q
from mtplx.sampling import SamplerConfig
from mtplx import generation as Gen
MD = os.path.expanduser("~/BigMoE/models/bare")
rt, st = load_streamed(MD, round(FRAC * 512))
TM = rt.model.language_model
mtp = getattr(TM, "mtp", None) or getattr(rt.model, "mtp", None)
MTP_BLOCK = mtp.layers[0].mlp
MTP_L = getattr(MTP_BLOCK.switch_mlp, "_bm", {}).get("L")
print("MTP gate layer:", MTP_L, flush=True)
if ARM == "B" and MTP_L is not None:
    _orig = type(MTP_BLOCK).__call__
    ALL = list(range(int(MTP_BLOCK.gate.weight.shape[0])))

    def call(self, x):
        if self is not MTP_BLOCK:
            return _orig(self, x)
        slots = np.array(ext.lookup(MTP_L, ALL))            # CPU residency snapshot of the MTP pool
        res = mx.array(slots >= 0)
        if int(res.sum().item()) < self.top_k:               # pool not warm yet: normal routing
            return _orig(self, x)
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        gates = mx.where(res, gates, mx.zeros_like(gates))
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        y = (self.switch_mlp(x, inds) * scores[..., None]).sum(axis=-2)
        return y + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)

    type(MTP_BLOCK).__call__ = call
probs = [json.loads(l) for l in open(os.path.expanduser("~/MoEPrefetch/data/gsm8k/test.jsonl"))]
PROMPTS = {"code": open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read(),
           "math": "Solve the problem step by step, briefly. End with a line of the form 'Answer: <number>'.\n\n" + probs[0]["question"],
           "chat": "Explain to a curious teenager why the sky is blue, in a few short paragraphs."}
smp = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)
out = {}
for name, text in PROMPTS.items():
    ids = rt.tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    ids = list(ids if isinstance(ids[0], int) else ids["input_ids"])
    first = []
    cb = lambda toks: first.append(time.perf_counter()) if not first else None
    t0 = time.perf_counter()
    r = Gen.generate_mtpk(rt, ids, max_tokens=MAXTOK, sampler=smp, speculative_depth=3, seed=0,
                          mtp_history_policy="committed", token_callback=cb)
    t1 = time.perf_counter()
    toks = list(r.tokens)
    dec = t1 - (first[0] if first else t0)
    out[name] = dict(tokens=len(toks), passes=int(r.stats.verify_calls), tpp=len(toks) / max(1, r.stats.verify_calls),
                     tps=len(toks) / max(1e-9, dec), sha=hashlib.sha256(str(toks).encode()).hexdigest()[:12])
    print(name, out[name], flush=True)
print("RESULT " + json.dumps({"prompts": out}), flush=True)
