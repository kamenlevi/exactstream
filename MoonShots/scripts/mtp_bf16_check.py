"""Validate the official bf16 MTP tensors (fetch_mtp_bf16.py) against BigMoE's bare sidecar (mtp.safetensors) before
swapping them in: map the official names/layout to MTPLX's (experts.gate_up_proj [E, 2*inter, hidden] split on axis 1 into
switch_mlp.gate_proj / up_proj, experts.down_proj -> switch_mlp.down_proj, the converter's +1 on hc/q/k/qk-layernorm norms;
pre_fc norms stay raw, attach_mtp adds +1), then for every sidecar tensor:
  bf16 in the sidecar  -> must be bit-identical to the mapped official tensor;
  packed in the sidecar -> re-quantize the official tensor with the sidecar's own bits/group (inferred as attach_mtp does)
                           and report identical words, and the error of the sidecar's dequantized values vs bf16.
CPU only.   usage: mtp_bf16_check.py OFFICIAL.safetensors"""
import os, sys
import mlx.core as mx
SIDE = os.path.expanduser("~/BigMoE/models/bare/mtp.safetensors")
SHIFT = (".q_norm.weight", ".k_norm.weight", ".q_layernorm.weight", ".k_layernorm.weight", ".hc_norm.weight")


def official_to_mtplx(off):
    """official {mtp.*: bf16} -> {mtp.*: bf16} in the sidecar's names and layout (pre_fc norms raw)."""
    out = {}
    for k, v in off.items():
        if k.endswith("mlp.experts.gate_up_proj"):
            p = k[: -len("experts.gate_up_proj")]
            assert v.shape[1] == 2 * 640 and v.shape[2] == 2560, v.shape
            g, u = mx.split(v, 2, axis=1)
            out[p + "switch_mlp.gate_proj.weight"], out[p + "switch_mlp.up_proj.weight"] = g, u
        elif k.endswith("mlp.experts.down_proj"):
            assert v.shape[1] == 2560 and v.shape[2] == 640, v.shape
            out[k[: -len("experts.down_proj")] + "switch_mlp.down_proj.weight"] = v
        elif v.ndim == 1 and k.endswith(SHIFT):
            out[k] = (v.astype(mx.float32) + 1.0).astype(v.dtype)
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    mx.set_default_device(mx.cpu)      # only for the standalone check: importers keep the GPU
    off = official_to_mtplx(mx.load(sys.argv[1]))
    side = mx.load(SIDE)
    mods = sorted({k.rsplit(".", 1)[0] for k in side})
    worst = 0.0
    for m in mods:
        w = side.get(m + ".weight")
        ref = off.get(m + ".weight")
        if ref is None:
            print(f"MISSING in official: {m}.weight"); continue
        if w.dtype != mx.uint32:
            same = bool(mx.array_equal(w, ref).item())
            print(f"{m:62s} bf16 {tuple(w.shape)}  bit-identical: {same}")
            if not same:
                d = mx.abs(w.astype(mx.float32) - ref.astype(mx.float32))
                print(f"    max |diff| {d.max().item():.4g}, mean {d.mean().item():.4g}")
            continue
        s, b = side[m + ".scales"], side[m + ".biases"]
        in_f = ref.shape[-1]
        bits = w.shape[-1] * 32 // in_f
        gs = in_f // s.shape[-1]
        # big expert tensors: compare a slice of experts to keep the CPU check short
        sl = (slice(0, 8),) if ref.ndim == 3 else (slice(None),)
        r = ref[sl]
        qw, qs, qb = mx.quantize(r, group_size=gs, bits=bits)
        eq_w = mx.mean((qw == w[sl]).astype(mx.float32)).item()
        eq_s = mx.mean((qs == s[sl]).astype(mx.float32)).item()
        deq = mx.dequantize(w[sl], s[sl], b[sl], group_size=gs, bits=bits).astype(mx.float32)
        rf = r.astype(mx.float32)
        rel = (mx.sqrt(mx.mean((deq - rf) ** 2)) / mx.sqrt(mx.mean(rf ** 2))).item()
        own = mx.dequantize(qw, qs, qb, group_size=gs, bits=bits).astype(mx.float32)
        rel_own = (mx.sqrt(mx.mean((own - rf) ** 2)) / mx.sqrt(mx.mean(rf ** 2))).item()
        worst = max(worst, rel)
        print(f"{m:62s} {bits}-bit g{gs} {tuple(r.shape)}  words equal {eq_w:.4f} scales equal {eq_s:.4f}  "
              f"rel.err sidecar {rel:.4f} (own RTN {rel_own:.4f})")
    print(f"worst sidecar rel.err vs official bf16: {worst:.4f}  (a layout or mapping error would be ~1.4)")
