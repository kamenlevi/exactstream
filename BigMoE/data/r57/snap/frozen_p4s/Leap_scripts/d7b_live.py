"""D7b arm E for the live loop (Leap, Mon 2026-11-02 21:2x; for BigMoE's r57_screen; R-85b in ~/Leap/TABLE_350.md 'D7b'):
the L1-20 drafter's 5 layers in 4-bit (RTN, group 64, MLX's nn.quantize when the drafter is loaded; fc, the candidate
selector and the head unchanged), so its 16-row projections run on REF2's qmm16c while ref2 is applied. Drafts change
(R-85b's quiet replay: propose 6.30 / 6.25 ms vs 10.39 / 10.16 bf16, both with R-84's qmm16c head; kept code +0.00 %,
math -3.39 %), the target's tokens don't (REF2-T verifies every draft), so the run's check is every token == REF2-T.
No new files: ~/Leap/data/l120_ft.safetensors is quantized as it loads.
Use, after draft.load_weights(l120_ft) and before draft.bind(model) and any draft_fused install:
    import d7b_live; d7b_live.apply(draft)
then install draft_fused with forms=False (R-76's conv kernel). R-78's forms are bf16 spellings: _forms_on asserts
nn.Linear (a QuantizedLinear is not one), and quantizing after the forms fails (form C moves the weight to _wt). So
D7b replaces K-83's drafter forms; R-76's conv kernel and R-77's chain_fused stack with it."""
import mlx.core as mx
import mlx.nn as nn


def apply(draft, group_size=64, bits=4):
    """quantize the drafter's layer projections in place (R-85b arm E); returns the drafter"""
    nn.quantize(draft, group_size=group_size, bits=bits,
                class_predicate=lambda p, m: isinstance(m, nn.Linear) and p.startswith("layers"))
    mx.eval(draft.parameters())
    return draft
