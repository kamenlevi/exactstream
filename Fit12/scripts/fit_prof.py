"""Fit12 GP1 (2026-10-24): time of the GDN / attention mixers and the PLE layers inside a real B12LL decode (the
orchestrator's ask: how much of the ~43 ms compute per pass at 12 GB is GDN; Leap's 35B profile puts its biggest kernel gap
there). F12_PROF=1 patches qwen4_exp's GatedDeltaNet, Attention, PLELayer and DecoderLayer __call__ (class level, this
process only) with a sync before and after each call: build = Python building the lazy graph, gpu = its eval minus a null
round trip measured at the same point, split into encode (mx.async_eval returning: Metal encoding on this thread) and wait. The syncs break ASYNC_EVERY's overlap, so a profile arm's tok/s is not comparable;
its shas must still equal U1 (the wrapper only moves eval points). Trunk modules (model.layers) are keyed apart from the rest
(the MTP head)."""
import os, time

ON = os.environ.get("F12_PROF") == "1"
T = {}                      # (name, S, trunk) -> [calls, build_s, gpu_s, wall_s, encode_s, wait_s]
NULL = [0, 0.0]             # null round trips: count, seconds
TRUNK = set()
NLAY = [48]


def _wrap(cls, name):
    import mlx.core as mx
    orig = cls.__call__

    def call(self, *a, **kw):
        x = a[0]
        mx.eval(x)
        tn = time.perf_counter(); mx.eval(mx.zeros((1,)) + 1); t0 = time.perf_counter()
        out = orig(self, *a, **kw)
        t1 = time.perf_counter(); mx.async_eval(out); t1b = time.perf_counter(); mx.eval(out); t2 = time.perf_counter()
        NULL[0] += 1; NULL[1] += t0 - tn
        r = T.setdefault((name, x.shape[1], id(self) in TRUNK), [0, 0.0, 0.0, 0.0, 0.0, 0.0])
        r[0] += 1; r[1] += t1 - t0; r[2] += t2 - t1; r[3] += t2 - t0; r[4] += t1b - t1; r[5] += t2 - t1b
        return out
    cls.__call__ = call


def apply(Q, model):
    if not ON:
        return False
    NLAY[0] = len(model.language_model.model.layers)
    for l in model.language_model.model.layers:
        TRUNK.add(id(l))
        for n in ("linear_attn", "self_attn", "ple"):
            if n in l:
                TRUNK.add(id(l[n]))
    for cls, name in ((Q.GatedDeltaNet, "gdn"), (Q.Attention, "attn"), (Q.PLELayer, "ple"), (Q.DecoderLayer, "layer")):
        _wrap(cls, name)
    return True


def reset():
    T.clear(); NULL[0] = 0; NULL[1] = 0.0


def report(tag, P):
    """Per trunk pass, for each trunk width S < 16 seen (passes = that width's layer calls / layer count)."""
    if not ON:
        return
    ss = sorted({S for (n, S, tr) in T if n == "layer" and tr and S < 16})
    rt = 1e3 * NULL[1] / max(1, NULL[0])
    parts = []
    for S in ss:
        lay = T.get(("layer", S, True))
        npass = lay[0] / NLAY[0] if lay else 0
        if not npass:
            continue
        seg = [f"S={S}: {npass:.0f} passes"]
        for n in ("layer", "gdn", "attn", "ple"):
            r = T.get((n, S, True))
            if r:
                g = max(0.0, r[2] - rt / 1e3 * r[0])
                seg.append(f"{n} x{r[0] / npass:.0f}: wall {1e3 * r[3] / npass:.1f}, build {1e3 * r[1] / npass:.1f}, "
                           f"gpu {1e3 * g / npass:.1f} (encode {1e3 * r[4] / npass:.1f} + wait {1e3 * r[5] / npass:.1f}) ms")
        parts.append(" | ".join(seg))
    other = {f"{n}/S{S}": f"{r[0]}x {1e3 * r[3] / r[0]:.2f} ms" for (n, S, tr), r in T.items() if not tr}
    print(f"[{tag}] {P} prof (per pass, null round trip {rt:.3f} ms subtracted from gpu): " + " || ".join(parts)
          + f" || non-trunk calls (mean wall): {other}", flush=True)
