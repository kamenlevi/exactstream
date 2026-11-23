"""BREAK98c PREFIX EYES cache (sb/kvx.py; results/LEVERS_A.md section B98C): the frozen 2B trunk's own keys / values at the
full-attention layers 15 and 19, as its KVCache holds them (k-normed, RoPE'd; v as projected), for every position of each
file, from ONE capture-shaped forward per file (batch 1, whole length). Only layers 15 and 19 get KVCache objects; every
other layer runs with no cache, as the capture did (the masks stay 'causal' / none), so the forward is the capture's.
Files: the training files of pass_slices(pool2, 96, 1115)[:400] (the 400 slices of WALK-MIX's setup), MBPP-50, HE-30.
  -> data/kvx/<set dir>/<key>.safetensors  K, V [2, KV, S, hd] bf16 (S = the file's length)
  -> data/kvx/init.safetensors             n{L} / q{L} / qn{L}: the trunk layer's input_layernorm, q_proj (dequantized,
                                           with its gate half) and q_norm, fp32: the cross-attention's init
  -> data/kvx/manifest.json                rows, bytes, the W check
Check (the first --check files): the forward's final post-norm state equals the stored W (core.load_W), bitwise or the
max |diff| is printed. Resumable (files already written are skipped); writes are atomic.
usage: kvxgen.py [--sets train,mbpp,he] [--check 3]"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx
from sb import core as C
from sb.kvx import LAYERS, kv_path, init_path


def file_list(sets):
    out = []
    for st in sets:
        if st == "train":
            from sb.train import split
            tr, _ = split("pool2")
            seen = []
            for f, _, _ in C.pass_slices(tr, 96, 1115)[:400]:
                if f not in seen:
                    seen.append(f)
            out += seen
        else:
            out += C.files(st)[:{"mbpp": 50, "he": 30}[st]]
    return out


def capture(t, ids):
    """one capture-shaped forward -> (K [len(LAYERS), KV, S, hd], V, post [S, d]) as the trunk computes them"""
    from mlx_lm.models.cache import KVCache
    cache = [None] * len(t.layers)
    for L in LAYERS:
        assert not t.layers[L].is_linear, f"layer {L} is not a full-attention layer"
        cache[L] = KVCache()
    post = t(mx.array(np.asarray(ids, np.int32))[None], cache=cache)
    K = mx.stack([cache[L].state[0][0] for L in LAYERS]); V = mx.stack([cache[L].state[1][0] for L in LAYERS])
    return K, V, post[0]


def write_init(t):
    out = {}
    for L in LAYERS:
        a = t.layers[L].self_attn
        qp = a.q_proj
        if hasattr(qp, "scales"):
            w = mx.dequantize(qp.weight, qp.scales, qp.biases, group_size=qp.group_size, bits=qp.bits)
        else:
            w = qp.weight
        out[f"q{L}"] = w.astype(mx.float32)
        out[f"qn{L}"] = a.q_norm.weight.astype(mx.float32)
        out[f"n{L}"] = t.layers[L].input_layernorm.weight.astype(mx.float32)
    p = init_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    mx.save_safetensors(p + ".tmp.safetensors", out); os.replace(p + ".tmp.safetensors", p)
    return {k: list(v.shape) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="train,mbpp,he"); ap.add_argument("--check", type=int, default=3)
    ap.add_argument("--mem", type=float, default=4.5)
    a = ap.parse_args()
    C.limit_memory(a.mem)
    t0 = time.time()
    C.load_target()
    t = C.tm()
    shapes = write_init(t)
    fs = file_list(a.sets.split(","))
    print(f"[kv] {len(fs)} files ({a.sets}); layers {LAYERS}; init {shapes}; ready in {time.time() - t0:.0f}s", flush=True)
    rows = nbytes = done = 0; checks = []
    for i, f in enumerate(fs):
        C.wait_pause()
        p = kv_path(f)
        z = np.load(f)
        ids = np.asarray(z["ids"], np.int32)
        if os.path.exists(p) and i >= a.check:
            rows += len(ids); nbytes += os.path.getsize(p); continue
        K, V, post = capture(t, ids)
        mx.eval(K, V, post)
        if i < a.check:
            W = C.load_W(z)[0]
            d = float(mx.abs(post.astype(mx.float32) - W.astype(mx.float32)).max())
            same = bool(mx.array_equal(post, W)) if post.shape == W.shape else False
            checks.append({"file": os.path.basename(f), "S": len(ids), "post_shape": list(post.shape), "W_shape": list(W.shape),
                           "bitwise": same, "max_abs_diff": d})
            print(f"[kv] W check {os.path.basename(f)}: post {list(post.shape)} vs stored W {list(W.shape)}: "
                  f"{'BITWISE' if same else f'max |diff| {d:.3e}'}", flush=True)
        assert K.shape[2] == len(ids), (K.shape, len(ids))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        mx.save_safetensors(p + ".tmp.safetensors", {"K": K, "V": V}); os.replace(p + ".tmp.safetensors", p)
        rows += len(ids); nbytes += os.path.getsize(p); done += 1
        del K, V, post
        mx.clear_cache()
        if (i + 1) % 50 == 0:
            print(f"[kv] {i + 1}/{len(fs)} files, {rows} rows, {nbytes / 1e9:.2f} GB [{time.time() - t0:.0f}s]", flush=True)
    man = {"layers": LAYERS, "files": len(fs), "written": done, "rows": rows, "bytes": nbytes, "checks": checks,
           "wall_s": time.time() - t0, "mlx_peak_gb": mx.get_peak_memory() / 1e9}
    json.dump(man, open(os.path.join(C.DATA, "kvx", "manifest.json"), "w"), indent=1)
    print(f"[kv] DONE {len(fs)} files ({done} written), {rows} rows, {nbytes / 1e9:.2f} GB in {time.time() - t0:.0f}s; "
          f"W check {'BITWISE' if checks and all(c['bitwise'] for c in checks) else checks}; peak {man['mlx_peak_gb']:.2f} GB",
          flush=True)


if __name__ == "__main__":
    main()
