"""GSM8K screen for nested bit-plane experts (NOT bit-exact; Kamen OK'd testing non-exact ideas 07:4x). Qwen3.6-35B-A3B 4-bit.
Arm A: plain 4-bit (stock mlx-lm MoE call). Arm B: VARIANT, e.g. mix4-msb2m = each token's top-4 of 8 gate ranks on the
exact 4-bit experts (MSB + LSB planes), the other ranks on the MSB plane only (2.5 bits/weight, per-group mean-LSB offset).
Same prompt, greedy CoT, max 900 tokens and answer extraction as ~/MoEPrefetch/scripts/gsm8k.py (data read from there).
Prints both accuracies, per-problem strings, how many final answers and whole outputs match arm A, and the teacher-forced
ppl of both arms on the expert_planes35 text.   usage: gsm8k_planes35.py VARIANT N   (GPU, ~30 GB: exclusive window)"""
import json, math, os, re, sys, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

VARIANT, N = sys.argv[1], int(sys.argv[2])
MODEL = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit")
model, tok = load(MODEL)
tm = model.language_model.model
OFF = int(os.environ.get("OFFSET", "0"))                         # REUSE_A=<json>: take arm A from an earlier run; SKIP_C=1
probs = [json.loads(l) for l in open(os.path.expanduser("~/MoEPrefetch/data/gsm8k/test.jsonl"))][OFF:OFF + N]
INSTR = "Solve the problem step by step, briefly. End with a line of the form 'Answer: <number>'.\n\n"
MAXT = 900
eos = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])


def extract(text):
    m = re.findall(r"Answer:\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    if not m:
        m = re.findall(r"(-?[\d,]*\.?\d+)", text)
    return m[-1].replace(",", "").rstrip(".") if m else None


def gold(a):
    return a.split("####")[-1].strip().replace(",", "")


def correct(pred, p):
    try:
        return int(pred is not None and abs(float(pred) - float(gold(p["answer"]))) < 1e-6)
    except ValueError:
        return 0


text = ""
for f in ("~/SpecDecode/FINDINGS.md", "~/BigMoE/FINDINGS.md", "~/MoEPrefetch/FINDINGS.md"):
    text += open(os.path.expanduser(f)).read() + "\n\n"
tids = mx.array(tok.encode(text)[:2049])[None]


def ppl():
    out = model(tids[:, :-1])
    lg = (out[0] if isinstance(out, tuple) else out)[0].astype(mx.float32)
    nll = -mx.take_along_axis(lg - mx.logsumexp(lg, axis=-1, keepdims=True), tids[0, 1:][:, None], axis=-1)[:, 0].mean()
    return math.exp(nll.item())


def arm(name):
    res, t0, ntok = [], time.perf_counter(), 0
    for p in probs:
        ids = tok.apply_chat_template([{"role": "user", "content": INSTR + p["question"]}], add_generation_prompt=True,
                                      enable_thinking=False)
        out = []
        for t, _ in generate_step(mx.array(ids), model, max_tokens=MAXT, sampler=make_sampler(temp=0.0)):
            t = int(t)
            if t in eos:
                break
            out.append(t)
        ntok += len(out)
        txt = tok.decode(out)
        pred = extract(txt)
        res.append({"pred": pred, "ok": correct(pred, p), "text": txt})
    dt = time.perf_counter() - t0
    acc = sum(r["ok"] for r in res)
    print(f"{name:12s}: GSM8K {acc}/{len(res)} = {100 * acc / len(res):.1f}%  ({dt:.0f}s, {ntok / dt:.1f} tok/s)  "
          f"per-problem {''.join(str(r['ok']) for r in res)}", flush=True)
    return res


# ---- the nested bit-plane variant (same construction as expert_planes35.py)
P3 = ("gate_proj", "up_proj", "down_proj")
sws = [l.mlp.switch_mlp for l in tm.layers]
orig = [{p: (getattr(sw, p).weight, getattr(sw, p).scales, getattr(sw, p).biases, getattr(sw, p).bits, getattr(sw, p).group_size)
         for p in P3} for sw in sws]
assert all(v[3] == 4 and v[4] == 64 for o in orig for v in o.values())
SH4 = mx.arange(0, 32, 4).astype(mx.uint32)
SH2 = mx.arange(0, 32, 2).astype(mx.uint32)


def unpack4(w):
    return ((w[..., None] >> SH4) & 15).reshape(*w.shape[:-1], -1)


def pack2(q):
    return (q.reshape(*q.shape[:-1], -1, 16) << SH2).sum(-1).astype(mx.uint32)


def msb_term(w, s, b, off):
    q = unpack4(w)
    if off is None:
        o = (q & 3).reshape(*s.shape, 64).astype(mx.float32).mean(-1)
        bias = (b.astype(mx.float32) + s.astype(mx.float32) * o).astype(s.dtype)
    else:
        bias = (b.astype(mx.float32) + off * s.astype(mx.float32)).astype(s.dtype)
    return (pack2(q >> 2), 4 * s, bias, 2, 64)


def glu(sw, x, idx, terms):
    x, sidx, inv = _gather_sort(mx.expand_dims(x, (-2, -3)), idx)

    def proj(p, h):
        y = None
        for (w, s, b, bits, gs) in terms[p]:
            o = mx.gather_qmm(h, w, s, b, rhs_indices=sidx, transpose=True, group_size=gs, bits=bits, sorted_indices=True)
            y = o if y is None else y + o
        return y

    h = sw.activation(proj("up_proj", x), proj("gate_proj", x))
    return _scatter_unsort(proj("down_proj", h), inv, idx.shape).squeeze(-2)


CUR = {}


def moe_call(self, x):
    gates = mx.softmax(self.gate(x), axis=-1, precise=True)
    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if self.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    order = mx.argsort(-scores, axis=-1)
    inds, scores = mx.take_along_axis(inds, order, axis=-1), mx.take_along_axis(scores, order, axis=-1)
    R = CUR["R"]
    y = glu(self.switch_mlp, x, inds[..., :R], CUR["hi"][self._li]) if R > 0 else None
    if R < k:
        lo = glu(self.switch_mlp, x, inds[..., R:], CUR["lo"][self._li])
        y = lo if y is None else mx.concatenate([y, lo], axis=-2)
    y = (y * scores[..., None]).sum(axis=-2)
    return y + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)


m = re.fullmatch(r"mix(\d)-(msb2m?)|(msb2m?)", VARIANT)
assert m, VARIANT
R, kind = (int(m.group(1)), m.group(2)) if m.group(1) else (0, m.group(3))

ppl_a = ppl()
print(f"arm A plain 4-bit: teacher-forced ppl {ppl_a:.4f}", flush=True)
if os.environ.get("REUSE_A"):
    A = json.load(open(os.environ["REUSE_A"]))["A"]
    print(f"4-bit (reused from {os.environ['REUSE_A']}): GSM8K {sum(r['ok'] for r in A)}/{len(A)}  per-problem {''.join(str(r['ok']) for r in A)}", flush=True)
else:
    A = arm("4-bit")
W4 = [{p: [v] for p, v in o.items()} for o in orig]
LO = []
for o in orig:
    d = {p: [msb_term(w, s, b, 1.5 if kind == "msb2" else None)] for p, (w, s, b, _, _) in o.items()}
    mx.eval([t[:3] for v in d.values() for t in v])
    LO.append(d)
for li, l in enumerate(tm.layers):
    object.__setattr__(l.mlp, "_li", li)
type(tm.layers[0].mlp).__call__ = moe_call
CUR.update(R=R, hi=W4, lo=LO)
ppl_b = ppl()
print(f"arm B {VARIANT}: teacher-forced ppl {ppl_b:.4f} ({100 * (ppl_b / ppl_a - 1):+.2f} %)", flush=True)
B = arm(VARIANT)


def agree(X, name):
    print(f"{name} vs arm A: final answers {sum(a['pred'] == x['pred'] for a, x in zip(A, X))}/{N}, whole outputs "
          f"{sum(a['text'] == x['text'] for a, x in zip(A, X))}/{N}; both correct {sum(a['ok'] and x['ok'] for a, x in zip(A, X))}, "
          f"A-only {sum(a['ok'] and not x['ok'] for a, x in zip(A, X))}, {name}-only {sum(x['ok'] and not a['ok'] for a, x in zip(A, X))}",
          flush=True)


agree(B, "B")
# arm C (noise floor): the same 4-bit weights through this script's MoE path; only summation order and row tiling differ
C = None
if not os.environ.get("SKIP_C"):
    CUR.update(R=8, hi=W4, lo=None)
    C = arm("path4")
    agree(C, "C")
json.dump({"variant": VARIANT, "A": A, "B": B, "C": C, "ppl": [ppl_a, ppl_b]},
          open(os.path.expanduser(f"~/MoonShots/data/gsm8k_planes35_{VARIANT}{'_off%d' % OFF if OFF else ''}.json"), "w"))
