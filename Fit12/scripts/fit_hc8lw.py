"""Fit12 (2026-10-23): layer-wise hc8 install. v3f/hc8.install builds every 8-bit mixer lazily and evaluates them all at the
end, so loading holds all bf16 mixers (1.27 GB) and their 8-bit copies at once: B12LL's lifetime peak footprint (11.35 GB) is
this load transient, while it runs at 10.4-10.7 GB (Fit12 M1). This version quantizes, evaluates and releases one module at
a time. Same modules, same mx.quantize call, so the weights are bitwise those of hc8.install.
apply(v3f_dir): pre-import v3f's hc8 (bt_b12 later does `import hc8` and gets this module object) and swap its install()."""
import sys


def apply(v3f):
    if v3f not in sys.path:
        sys.path.insert(0, v3f)
    import hc8
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    def install(model, bits=8, group_size=64, ple=True):
        import mtplx.models.qwen4_exp as Q
        TM = getattr(model, "language_model", model)
        swapped, before, after = 0, 0, 0
        for name, mod in TM.named_modules():
            if name.startswith("mtp"):
                continue
            targets = []
            if isinstance(mod, Q.GatedResidual):
                targets = ["input_mix_weight_down", "input_mix_weight_up"]
            elif ple and isinstance(mod, Q.PLELayer):
                targets = ["key_proj", "value_proj"]
            for attr in targets:
                lin = getattr(mod, attr)
                if not isinstance(lin, nn.Linear) or lin.weight.shape[1] % group_size:
                    continue
                q = nn.QuantizedLinear.from_linear(lin, group_size, bits)
                mx.eval(q.parameters())                     # materialize now, then drop the bf16 original
                before += lin.weight.nbytes
                after += sum(a.nbytes for _, a in tree_flatten(q.parameters()))
                setattr(mod, attr, q)
                del lin
                swapped += 1
            if targets:
                mx.clear_cache()
        mx.eval(TM.parameters())
        return swapped, before / 1e6, after / 1e6

    hc8.install = install
    return hc8
