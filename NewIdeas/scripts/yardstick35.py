"""35B analogue of the Eval suite's ar4 yardstick (NewIdeas, 2026-10-21). The 125B gate allows a policy KLD <= 1.1x the KLD
between the reference and ANOTHER 4-bit build (AutoRound int4), per corpus part. Here: reference = defer35.py's model
(8-bit checkpoint, experts requantized to 4-bit g64; data/defer35/ref_1365.npz); yardstick = LM Studio's separately
converted 4-bit 35B (every weight 4-bit). Per-segment KLD (prose / gsm / code) and top-1 agreement on the same three
teacher-forced segments -> data/defer35/yardstick.txt; a policy passes the 35B gate if its per-segment KLD <= 1.1x these.
"""
import os, sys, time
import numpy as np
import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import defer35 as D

Y = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit")


def main():
    t0 = time.time()
    model, tok = load(Y)
    segs = D.segments(tok)
    ref = np.load(os.path.join(D.OUT, f"ref_{D.SEG}.npz"))
    out = []
    for i, s in enumerate(segs):
        o = model(s[:, :-1])
        lg = (o[0] if isinstance(o, tuple) else o)[0].astype(mx.float32)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        top, p_lp = mx.array(ref[f"top{i}"]), mx.array(ref[f"tlp{i}"])
        q_lp = mx.take_along_axis(lp, top, axis=-1)
        p, q = mx.exp(p_lp), mx.exp(q_lp)
        pt, qt = mx.maximum(1 - p.sum(-1), 1e-12), mx.maximum(1 - q.sum(-1), 1e-12)
        k = (p * (p_lp - q_lp)).sum(-1) + pt * (mx.log(pt) - mx.log(qt))
        a1 = mx.argmax(lp, axis=-1)
        mx.eval(k, a1)
        k = np.array(k)
        out.append((float(k.mean()), float(np.percentile(k, 99.9)), float((np.array(a1) == ref[f"a1{i}"]).mean())))
    line = (f"yardstick (LM Studio 4-bit 35B vs reference), per segment prose/gsm/code: KLD "
            + "/".join(f"{o[0]:.5f}" for o in out) + "  P99.9 " + "/".join(f"{o[1]:.4f}" for o in out)
            + "  top-1 " + "/".join(f"{o[2]:.4f}" for o in out) + f"  ({time.time() - t0:.0f}s, peak {mx.get_peak_memory() / 1e9:.1f} GB)")
    print(line, flush=True)
    open(os.path.join(D.OUT, "yardstick.txt"), "a").write(line + "\n")


if __name__ == "__main__":
    main()
