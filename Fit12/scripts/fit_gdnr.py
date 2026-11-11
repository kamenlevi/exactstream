"""Fit12 GR1 (2026-10-24): Leap's exact GDN recurrence (recurrence_r4 in ~/Leap/scripts/gdn35_exact.py: mlx_lm 0.31.3's
gated_delta kernel expressions in the same order, 4 value rows per thread, chunks of 16 tokens staged in threadgroup memory;
bitwise == mlx_lm per Leap, -24 % at T = 16 on the 35B) in place of mlx_lm's gated_delta_kernel on unmasked scalar-g calls
(B12LL's verify rows, the deferred commit repair, prefill). g and beta stay mlx_lm's own ops, so MLX 0.32.2 vs 0.32.4's
sigmoid difference (Leap's recurrence_gates) does not enter. Leap's file is loaded by path, read-only (no sys.path change;
PYTHONDONTWRITEBYTECODE=1 in the batch scripts). F12_GDNR=1; apply() after load_b12. CALLS = [r4 calls, fallback calls]."""
import importlib.util, os

ON = os.environ.get("F12_GDNR") == "1"
CALLS = [0, 0]


def apply():
    if not ON:
        return False
    spec = importlib.util.spec_from_file_location("leap_gdn35_exact", os.path.expanduser("~/Leap/scripts/gdn35_exact.py"))
    LG = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(LG)
    import mlx_lm.models.gated_delta as GD
    orig = GD.gated_delta_kernel

    def kern(q, k, v, g, beta, state, mask=None):
        if mask is None and g.ndim == 3 and v.shape[-1] % 16 == 0:
            CALLS[0] += 1
            return LG.recurrence_r4(q, k, v, g, beta, state)
        CALLS[1] += 1
        return orig(q, k, v, g, beta, state, mask)
    GD.gated_delta_kernel = kern
    return True
