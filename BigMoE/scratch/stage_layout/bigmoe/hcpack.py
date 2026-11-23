"""Row 15a (MoonShots): lossless exponent-offset pack of the bf16 hyper-connection mixer weights (78 % of the
bytes), read through a clone of MLX's own GemvWide with the lane count MLX itself picks for the reference's row
count (max(2, M) under dup-row) -> bit-identical to the compiled dup-row reference (MoonShots test_packed_hc.py).
Pack and GemvWide clone are imported read-only from ~/MoonShots/patches/hc_exp_pack."""
import os, sys
import mlx.core as mx
import mlx.nn as nn
_P = os.path.expanduser("~/MoonShots/patches/hc_exp_pack")
if _P not in sys.path:
    sys.path.insert(0, _P)
from hc_pack import pack_bf16          # noqa: E402
from gemv_probe import gemv_wide       # noqa: E402

NAMES = ("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight")
_LANES = {}


def lane_table(out_dims, in_dims, Ms=range(2, 16)):
    key = (out_dims, in_dims)
    if key not in _LANES:
        W = (mx.random.normal((out_dims, in_dims)) * 0.02).astype(mx.bfloat16)
        tab = {}
        for M in Ms:
            x = mx.random.normal((M, in_dims)).astype(mx.bfloat16)
            ref = x @ W.T
            for kl in (32, 16, 8, 4):
                if mx.array_equal(gemv_wide(W, x, kl), ref).item():
                    tab[M] = kl
                    break
        _LANES[key] = tab
    return _LANES[key]


class PackedLinear(nn.Module):
    def __init__(self, lin):
        super().__init__()
        self.out_dims, self.in_dims = lin.weight.shape
        self._pack = pack_bf16(lin.weight)
        self._lanes = lane_table(self.out_dims, self.in_dims)
        missing = [M for M in range(2, 16) if M not in self._lanes]
        assert not missing, f"MLX does not use GemvWide for {self.out_dims}x{self.in_dims} at M={missing}"

    def __call__(self, x):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        y = gemv_wide(None, x2, self._lanes[max(2, x2.shape[0])], packed=(self._pack, self.out_dims))
        return y.reshape(*shp[:-1], self.out_dims)


def pack_hyper_connections(model):
    n = 0
    for layer in model.language_model.model.layers:
        for hc_name in ("attn_hyper_connection", "mlp_hyper_connection"):
            hc = getattr(layer, hc_name, None)
            if hc is None:
                continue
            for name in NAMES:
                lin = getattr(hc, name, None)
                if isinstance(lin, nn.Linear) and lin.weight.dtype == mx.bfloat16:
                    setattr(hc, name, PackedLinear(lin)); n += 1
        mx.eval(layer.parameters())
    return n


class PinnedLinear(nn.Module):
    """BIGMOE_HC_PIN=1: the bf16 HC mixer with MLX's GemvWide at a FIXED lane count (32, what MLX itself picks for
    M=2..5) for every row count up to 15. MLX switches the down/up mixers to 16 lanes at M=6 (and irregularly beyond),
    which is the reference's S=6 band break; pinned, S=6.. reproduce the S<=5 arithmetic. Prefill-width calls (M>15)
    keep MLX's own matmul, as the reference does."""
    LANES = 32

    def __init__(self, lin):
        super().__init__()
        self.weight = lin.weight
        self.out_dims, self.in_dims = lin.weight.shape

    def __call__(self, x):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        if x2.shape[0] > 15:
            return x @ self.weight.T
        y = gemv_wide(self.weight, x2, self.LANES)
        return y.reshape(*shp[:-1], self.out_dims)


def pin_hyper_connections(model):
    n = 0
    for layer in model.language_model.model.layers:
        for hc_name in ("attn_hyper_connection", "mlp_hyper_connection"):
            hc = getattr(layer, hc_name, None)
            if hc is None:
                continue
            for name in NAMES:
                lin = getattr(hc, name, None)
                if isinstance(lin, nn.Linear) and lin.weight.dtype == mx.bfloat16 and "bias" not in lin:
                    setattr(hc, name, PinnedLinear(lin)); n += 1
    return n
