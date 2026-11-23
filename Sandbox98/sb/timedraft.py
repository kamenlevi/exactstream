"""Drafter step timing at 1 / 2 / 3 layers (results/screens/DEPTH_LINES.md, addition 1): decode-style, one row, a
~400-pair context in the cache (depth 1 over the true pairs), then 30 sequential drafted steps (each: the stacked layers
over the cache + its own chain entries, the target's 4-bit lm_head, argmax, synced like real drafting). Median of 5 runs
after a warm-up; the lm_head + argmax time is measured alone too. -> results/timedraft.json
usage: timedraft.py [--ctx 400] [--layers 1,2,3]"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from sb import core as C

KEYS = ("input_layernorm", "post_attention_layernorm", "q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm",
        "gate_proj", "up_proj", "down_proj")


def build(nl):
    D = C.load_drafter("s30")
    if nl > 1:
        P = dict(tree_flatten(D.parameters()))
        extra = []
        for _ in range(nl - 1):
            L = C.Layer()
            L.load_weights([(k, mx.array(v)) for k, v in P.items() if k.split(".")[0] in KEYS])
            extra.append(L)
        D.xl = extra
        object.__setattr__(D, "_stack", [D] + list(D.xl))
    mx.eval(D.parameters())
    return D


def one_draft(D, W, ids, S, steps=30):
    a0 = S - 2
    h, K1, V1 = C.depth1(D, W[:, :S], mx.array(ids[:S]), a0)
    tok = mx.argmax(C.lm_logits(D.norm(h))[0, 0])
    mx.eval(h, K1, V1, tok)
    buf = list(ids[:S]) + [int(tok.item())]
    anc = mx.array([a0])
    ck, cv, dts = [], [], []
    for j in range(2, steps + 2):
        ida = mx.array(np.asarray(buf, np.int32))
        t = time.perf_counter()
        h = C.deeper(D, h, ida, anc, j, K1, V1, ck, cv)
        tok = mx.argmax(C.lm_logits(D.norm(h))[0, 0])
        mx.eval(tok, h, *ck[-1:], *cv[-1:])
        dts.append(time.perf_counter() - t)
        buf.append(int(tok.item()))
    return dts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=400); ap.add_argument("--layers", default="1,2,3")
    a = ap.parse_args()
    C.limit_memory(5); C.load_target(); C.drop_trunk()
    f = next(f for f in C.files("mc") if len(np.load(f)["ids"]) > a.ctx + 40)
    z = np.load(f)
    ids = np.asarray(z["ids"], np.int32); W = C.load_W(z)
    hn = mx.random.normal((1, 1, 2048))
    for _ in range(5):
        mx.eval(mx.argmax(C.lm_logits(hn)[0, 0]))
    lm = []
    for _ in range(50):
        t = time.perf_counter(); mx.eval(mx.argmax(C.lm_logits(hn)[0, 0])); lm.append(time.perf_counter() - t)
    out = {"ctx": a.ctx, "lm_head_ms": 1e3 * float(np.median(lm)), "layers": {}}
    for nl in [int(x) for x in a.layers.split(",")]:
        D = build(nl)
        one_draft(D, W, ids, a.ctx)                      # warm-up
        runs = [one_draft(D, W, ids, a.ctx) for _ in range(5)]
        per_tok = float(np.median([np.median(r) for r in runs])) * 1e3
        per_draft = float(np.median([np.sum(r) for r in runs])) * 1e3
        out["layers"][nl] = {"ms_per_token": per_tok, "ms_per_30_draft": per_draft, "ms_layers_per_token": per_tok - out["lm_head_ms"]}
        print(f"[td] {nl} layer(s): {per_tok:.2f} ms per drafted token ({per_tok - out['lm_head_ms']:.2f} layers + "
              f"{out['lm_head_ms']:.2f} lm_head+argmax), {per_draft:.1f} ms per 30-deep draft", flush=True)
    json.dump(out, open(os.path.join(C.RES, "timedraft.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
