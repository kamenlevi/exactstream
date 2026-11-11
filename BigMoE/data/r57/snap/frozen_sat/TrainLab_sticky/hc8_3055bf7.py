"""Round 28 (orchestrator 20:45, test only; adoption is Kamen's call): the 125B's hyper-connection mixers and PLE key/value
projections in 8-bit instead of bf16. MTPLX's quantization recipe keeps input_mix_weight_down / up, block_inject_weight
and ple.key_proj / value_proj in full precision; they are 26 of each layer's ~65 MB of resident weights (1.25 GB of the
3.2 GB a 4-row verify pass reads, round 26c). This swaps input_mix_weight_down [320, 10240] and input_mix_weight_up
[10240, 320] of every trunk GatedResidual (2 per layer + the final mixer; the MTP head's are left alone) and the PLE
layer's key_proj [10240, 2560] / value_proj [2560, 2560] for MLX affine 8-bit, group 64 (nn.QuantizedLinear.from_linear).
block_inject_weight [4, 10240] stays bf16 (40 KB). MTPLX's fused HC read kernels require bf16 mixers, so the patched
reads take MTPLX's eager path; don't combine with BigMoE's HC fusion switches (BIGMOE_COMPILE_HC / HC_VREAD / HC_PACK).
install(model) -> (modules swapped, MB before, MB after)."""
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
            before += lin.weight.nbytes
            after += sum(a.nbytes for _, a in tree_flatten(q.parameters()))
            setattr(mod, attr, q)
            swapped += 1
    mx.eval(TM.parameters())
    return swapped, before / 1e6, after / 1e6
