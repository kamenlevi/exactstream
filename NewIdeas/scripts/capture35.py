"""Capture routed-MoE inputs on the Qwen3.6-35B-A3B proxy for CPU studies of what an expert computes (NewIdeas I2/I3).
Exact forward (experts requantized to 4-bit g64 as in defer35.py), independent teacher-forced segments of NI_CSEG
tokens from mixed sources; for layers NI_LAYERS keeps every row's MoE input x (post-attention RMSNorm output, f16),
its top-8 expert ids and normalized scores, and the block's routed output (f16).
-> ~/NewIdeas/data/cap35/L<l>.npz with X [N, 2048] f16, I [N, 8] int16, S [N, 8] f32, Y [N, 2048] f16, seg [N] int8
"""
import glob, json, os, sys, time
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import qwen3_next

sys.path.insert(0, os.path.dirname(__file__))
from defer35 import MODEL, to4bit

OUT = os.path.expanduser("~/NewIdeas/data/cap35"); os.makedirs(OUT, exist_ok=True)
CSEG = int(os.environ.get("NI_CSEG", "2048"))
LAYERS = [int(v) for v in os.environ.get("NI_LAYERS", "3,10,17,24,31,38").split(",")]


def texts():
    md = [os.path.expanduser(p) for p in ("~/BigMoE/FINDINGS.md", "~/MoEPrefetch/FINDINGS.md", "~/SpecDecode/FINDINGS.md",
                                          "~/Breakthrough/SCOREBOARD.md", "~/MoonShots/IDEAS.md")]
    gsm = [json.loads(l) for l in open(os.path.expanduser("~/MoEPrefetch/data/gsm8k/test.jsonl"))]
    py = sorted(glob.glob(os.path.expanduser("~/MoonShots/scripts/*.py"))) + sorted(
        glob.glob(os.path.expanduser("~/Breakthrough/scripts/*.py")))
    return [("prose", "\n\n".join(open(f).read() for f in md)),
            ("gsm", "\n\n".join("Question: " + p["question"] + "\nAnswer: " + p["answer"] for p in gsm[400:900])),
            ("code", "\n\n".join(open(f).read() for f in py))]


def main():
    nseg = int(os.environ.get("NI_NSEG_PER_SRC", "4"))
    t0 = time.time()
    model, tok = load(MODEL, lazy=True)
    to4bit(model)
    layers = model.language_model.model.layers
    rows = {L: {"X": [], "I": [], "S": [], "Y": [], "seg": []} for L in LAYERS}
    cur = {"seg": 0}
    orig = qwen3_next.Qwen3NextSparseMoeBlock.__call__

    def hook(self, x):
        L = self._ni_layer
        if L in rows:
            gates = mx.softmax(self.gate(x), axis=-1, precise=True)
            inds = mx.argpartition(gates, kth=-self.top_k, axis=-1)[..., -self.top_k:]
            sc = mx.take_along_axis(gates, inds, axis=-1)
            if self.norm_topk_prob:
                sc = sc / sc.sum(axis=-1, keepdims=True)
            y = (self.switch_mlp(x, inds) * sc[..., None]).sum(axis=-2)
            mx.eval(inds, sc, y)
            r = rows[L]
            r["X"].append(np.array(x[0].astype(mx.float16))); r["I"].append(np.array(inds[0]).astype(np.int16))
            r["S"].append(np.array(sc[0].astype(mx.float32))); r["Y"].append(np.array(y[0].astype(mx.float16)))
            r["seg"].append(np.full(x.shape[1], cur["seg"], np.int8))
        return orig(self, x)
    for i, l in enumerate(layers):
        l.mlp._ni_layer = i
    qwen3_next.Qwen3NextSparseMoeBlock.__call__ = hook
    k = 0
    for name, text in texts():
        ids = tok.encode(text)
        for j in range(nseg):
            seg = ids[j * CSEG:(j + 1) * CSEG]
            if len(seg) < CSEG:
                break
            cur["seg"] = k
            out = model(mx.array(seg)[None])
            mx.eval(out)
            print(f"{name} seg {j}: {len(seg)} tokens ({time.time() - t0:.0f}s)", flush=True)
            k += 1
    for L, r in rows.items():
        np.savez(os.path.join(OUT, f"L{L}.npz"), **{a: np.concatenate(v) for a, v in r.items()})
    print(f"capture: {k} segments x {CSEG} tokens, layers {LAYERS}, {time.time() - t0:.0f}s, "
          f"peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
