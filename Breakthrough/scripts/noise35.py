"""D2 first experiment: noise-injection sweep on the Qwen3.6-35B-A3B proxy (Breakthrough, 2026-10-19).
How much weight error (NMSE) can each tensor class take before it costs more than today's Q4 (MLX affine 4-bit g64)?
Rationale (research/bits_limits.md): HIGGS' linearity theorem (equal weighted error -> equal quality), and an ideal
trellis code at R bits has NMSE ~1.1 x 2^-2R: 3.5 / 3.25 / 3.0 / 2.75 / 2.5 bits -> 0.0086 / 0.012 / 0.017 / 0.024 / 0.034.
Base: the 8-bit unsloth 35B (near-bf16), which is the reference. For ONE tensor class at a time, every matching quantized
module is
  q4  : requantized to affine 4-bit g64 (the anchor: what Q4 costs in that class), or
  none: nothing modified (the floor: run-to-run differences against the separate reference pass)
  nX  : given white Gaussian noise with per-row variance X * mean(w_row^2) on its dequantized weights, then requantized to
        8-bit g64 (which adds ~1e-5 NMSE).
Measured on 3 teacher-forced segments (technical prose, GSM8K solutions, Python code; BT_SEG tokens each): mean and P99
KLD(ref || variant) over the reference's top-64 plus the tail mass, top-1 agreement with the reference, mean NLL delta,
and the class's measured NMSE. The model is reloaded for every arm.
usage: noise35.py ref                  -> data/noise35/ref.npz
       noise35.py CLASS ARM [ARM ...]  -> one line per arm appended to data/noise35/results.txt
CLASS: experts_gu, experts_down, experts, shared, gdn, attn, lm_head, all (every quantized linear but routers/embeddings)
"""
import glob, json, os, sys, time
import numpy as np
import mlx.core as mx
from mlx_lm import load

MODEL = os.environ.get("BT_MODEL") or glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MLX-8bit/snapshots/*/"))[0]
OUT = os.path.expanduser(os.environ.get("BT_OUT", "~/Breakthrough/data/noise35"))
os.makedirs(OUT, exist_ok=True)
SEG = int(os.environ.get("BT_SEG", "1365"))
TOPK = 64
CLASSES = {
    "experts_gu": lambda n: n.endswith("switch_mlp.gate_proj") or n.endswith("switch_mlp.up_proj"),
    "experts_down": lambda n: n.endswith("switch_mlp.down_proj"),
    "experts": lambda n: ".switch_mlp." in n,
    "shared": lambda n: ".shared_expert." in n,
    "gdn": lambda n: ".linear_attn." in n,
    "attn": lambda n: ".self_attn." in n,
    "lm_head": lambda n: n.endswith("lm_head"),
    "all": lambda n: not (n.endswith("mlp.gate") or n.endswith("shared_expert_gate") or "embed" in n),
}


def segments(tok):
    srcs = [[os.path.expanduser("~/BigMoE/FINDINGS.md")], None,
            sorted(glob.glob(os.path.expanduser("~/MoonShots/scripts/*.py")))]
    gsm = [json.loads(l) for l in open(os.path.expanduser("~/MoEPrefetch/data/gsm8k/test.jsonl"))][200:400]
    out = []
    for k, files in enumerate(srcs):
        text = ("\n\n".join("Question: " + p["question"] + "\nAnswer: " + p["answer"] for p in gsm) if files is None
                else "\n\n".join(open(f).read() for f in files[:40]))
        ids = tok.encode(text)[:SEG + 1]
        assert len(ids) == SEG + 1, (k, len(ids))
        out.append(mx.array(ids)[None])
    return out


def logprobs(model, seg):
    out = model(seg[:, :-1])
    lg = (out[0] if isinstance(out, tuple) else out)[0].astype(mx.float32)
    return lg - mx.logsumexp(lg, axis=-1, keepdims=True)


def ref_pass():
    model, tok = load(MODEL)
    d = {}
    for i, s in enumerate(segments(tok)):
        lp = logprobs(model, s)
        top = mx.argpartition(-lp, TOPK - 1, axis=-1)[:, :TOPK]
        tlp = mx.take_along_axis(lp, top, axis=-1)
        nll = -mx.take_along_axis(lp, s[0, 1:, None], axis=-1)[:, 0]
        a1 = mx.argmax(lp, axis=-1)
        mx.eval(top, tlp, nll, a1)
        d[f"top{i}"], d[f"tlp{i}"], d[f"nll{i}"], d[f"a1{i}"] = (np.array(x) for x in (top, tlp, nll, a1))
    np.savez(os.path.join(OUT, "ref.npz"), **d)
    print(f"ref: {len(d) // 4} segments x {SEG} tokens, mean NLL {np.mean([d[f'nll{i}'].mean() for i in range(3)]):.4f}, "
          f"peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)


def modify(model, cls, arm, seed):
    if arm == "none":                      # null arm: reload only, nothing modified (measures the floor vs the ref pass)
        return 0, 0.0
    match = CLASSES[cls]
    mx.random.seed(seed)
    se = sw = 0.0
    n = 0
    for name, m in model.named_modules():
        if "language_model" not in name or not hasattr(m, "scales") or not hasattr(m, "bits") or not match(name):
            continue
        gs, bits, dt = m.group_size, m.bits, m.scales.dtype
        w = mx.dequantize(m.weight, m.scales, m.biases, group_size=gs, bits=bits).astype(mx.float32)
        if arm == "q4":
            ngs, nb, wn = 64, 4, w
        elif arm.startswith("t"):            # QTIP-style trellis fake-quant at k bits (scripts/trellis.py, L=12, CPU Viterbi)
            from trellis import quantize as tq
            ngs, nb = gs, bits
            wn = tq(w, int(arm[1:]), L=int(os.environ.get("BT_TL", "12")))
        else:
            ngs, nb = gs, bits
            wn = w + mx.random.normal(w.shape) * mx.sqrt(float(arm[1:]) * mx.mean(w * w, axis=-1, keepdims=True))
        wq, s, b = mx.quantize(wn.astype(dt), group_size=ngs, bits=nb)
        wd = mx.dequantize(wq, s, b, group_size=ngs, bits=nb).astype(mx.float32)
        e, t = mx.sum((wd - w) ** 2), mx.sum(w * w)
        mx.eval(wq, s, b, e, t)
        se += e.item()
        sw += t.item()
        n += 1
        m.weight, m.scales, m.biases, m.group_size, m.bits = wq, s, b, ngs, nb
        del w, wn, wd
        mx.clear_cache()
    return n, se / max(sw, 1e-30)


def compare(model, segs, ref):
    kl, agree, dnll, seg_kl = [], [], [], []
    for i, s in enumerate(segs):
        lp = logprobs(model, s)
        top, p_lp = mx.array(ref[f"top{i}"]), mx.array(ref[f"tlp{i}"])
        q_lp = mx.take_along_axis(lp, top, axis=-1)
        p, q = mx.exp(p_lp), mx.exp(q_lp)
        pt, qt = mx.maximum(1 - p.sum(-1), 1e-12), mx.maximum(1 - q.sum(-1), 1e-12)
        k = (p * (p_lp - q_lp)).sum(-1) + pt * (mx.log(pt) - mx.log(qt))
        nll = -mx.take_along_axis(lp, s[0, 1:, None], axis=-1)[:, 0]
        a1 = mx.argmax(lp, axis=-1)
        mx.eval(k, nll, a1)
        kl.append(np.array(k))
        seg_kl.append(float(kl[-1].mean()))
        agree.append(np.array(a1) == ref[f"a1{i}"])
        dnll.append(np.array(nll) - ref[f"nll{i}"])
    kl, agree, dnll = (np.concatenate(x) for x in (kl, agree, dnll))
    return kl.mean(), np.percentile(kl, 99), agree.mean(), dnll.mean(), seg_kl


def main():
    if sys.argv[1] == "ref":
        return ref_pass()
    cls, arms = sys.argv[1], sys.argv[2:]
    ref = np.load(os.path.join(OUT, "ref.npz"))
    for j, arm in enumerate(arms):
        t0 = time.time()
        model, tok = load(MODEL)
        segs = segments(tok)
        n, nmse = modify(model, cls, arm, seed=1000 + j)
        kl, kl99, agree, dnll, seg_kl = compare(model, segs, ref)
        line = (f"{cls:12s} {arm:8s} modules {n:4d}  NMSE {nmse:.5f}  KLD {kl:.5f} (P99 {kl99:.4f}; prose/gsm/code "
                f"{seg_kl[0]:.4f}/{seg_kl[1]:.4f}/{seg_kl[2]:.4f})  top-1 agree {agree:.4f}  dNLL {dnll:+.4f}  "
                f"{time.time() - t0:.0f}s peak {mx.get_peak_memory() / 1e9:.1f} GB")
        print(line, flush=True)
        open(os.path.join(OUT, "results.txt"), "a").write(line + "\n")
        del model, segs
        mx.clear_cache()


if __name__ == "__main__":
    main()
