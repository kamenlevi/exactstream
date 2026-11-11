"""Fit12 TRL view reader for the decision map (map screen 7; Wed 4 Nov 2026). Read only; for Breakthrough's harness hook.
The TRL store (/Users/kamen/Fit12/scripts/fit_trl_store.py) keeps, per (layer, expert, projection, k), the DECODED trellis
weight W = Q R^T as an MLX 8-bit affine view (group 64, bf16 scales / biases):
    /Users/kamen/Fit12/data/trl/L{L}/e{EEE}_{gate|up|down}_k{k}.safetensors   keys w (uint32 [out, in / 4]), s, b ([out, in / 64])
API as dm_t3 (~/FableGod/decision_map/dm_t3.py: read(L, ids) + _proj):
    W = read(L, uniq, form)          -> {"gate_proj": {"w", "s", "b"}, "up_proj": ..., "down_proj": ...}, experts stacked in
                                        the order of `uniq` (so dm_t3's idx into uniq works unchanged)
    y = proj(h, W["up_proj"], idx)   -> mx.gather_qmm(h, w, s, b, rhs_indices=idx, transpose=True, group_size=64, bits=8)
    has(L, e, form) / coverage(L, ids, form): which experts exist (the encode may be partial; the harness decides the
                                        fallback, e.g. records3 at 3 bits, and should count it).
FORMS: mix = gate / up k 2 + down k 3 (2.40 b); trl2 = k 2 everywhere (2.06 b); trl3 = k 3 everywhere (3.07 b).
Store dir: F12_TRL_STORE (default above). Self-test: python trl_view.py test [L] (CPU, seconds)."""
import os, sys
import mlx.core as mx

STORE = os.environ.get("F12_TRL_STORE", "/Users/kamen/Fit12/data/trl")
KS = {"mix": {"gate": 2, "up": 2, "down": 3}, "trl2": {"gate": 2, "up": 2, "down": 2},
      "trl3": {"gate": 3, "up": 3, "down": 3}}
NAMES = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}


def _f(L, e, p, k):
    return os.path.join(STORE, f"L{L}", f"e{int(e):03d}_{p}_k{k}.safetensors")


def has(L, e, form="mix"):
    return all(os.path.exists(_f(L, e, p, k)) for p, k in KS[form].items())


def coverage(L, ids, form="mix"):
    ids = [int(e) for e in ids]
    return sum(has(L, e, form) for e in ids), len(ids)


def read(L, ids, form="mix"):
    out = {}
    for p, k in KS[form].items():
        ws = [mx.load(_f(L, e, p, k)) for e in ids]
        out[NAMES[p]] = {key: mx.stack([w[key] for w in ws]) for key in ("w", "s", "b")}
    return out


def proj(h, W, idx):
    return mx.gather_qmm(h, W["w"], W["s"], W["b"], rhs_indices=idx, transpose=True, group_size=64, bits=8)


def test(L=18):
    """layout check: gather_qmm on the view == x @ dequantize(view)^T for the first two stored experts. CPU, in float32
    (MLX's CPU qmm with bf16 activations is only bf16-accurate, ~5 % here; the GPU kernels the harness uses accumulate in
    fp32, as dm_t3's bits=3 path)."""
    mx.set_default_device(mx.cpu)
    ids = [e for e in range(512) if has(L, e)][:2]
    assert len(ids) == 2, f"need 2 encoded experts at L{L}"
    W = read(L, ids)
    idx = mx.array([0, 1, 1, 0], dtype=mx.uint32)
    for name, din in (("gate_proj", 2560), ("up_proj", 2560), ("down_proj", 640)):
        V = {"w": W[name]["w"], "s": W[name]["s"].astype(mx.float32), "b": W[name]["b"].astype(mx.float32)}
        h = mx.random.normal((4, 1, din), key=mx.random.key(0)) * 0.05
        y = proj(h, V, idx)[:, 0, :]
        ref = mx.stack([h[r, 0] @ mx.dequantize(V["w"][int(idx[r].item())], V["s"][int(idx[r].item())],
                                                 V["b"][int(idx[r].item())], group_size=64, bits=8).T for r in range(4)])
        rel = (mx.abs(y - ref).max() / mx.abs(ref).max()).item()
        print(f"[trl_view] L{L} experts {ids} {name}: out {tuple(y.shape)}, max rel diff vs dense {rel:.2e}", flush=True)
        assert rel < 1e-4, rel
    print("[trl_view] test OK", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test(int(sys.argv[2]) if len(sys.argv) > 2 else 18)
