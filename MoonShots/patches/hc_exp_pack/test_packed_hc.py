"""Row 15a, bitwise test against BigMoE's reference: MTPLX GatedResidual, mx.compile'd as in bigmoe.fuse,
with BIGMOE dup-row active, on real layer weights of ~/BigMoE/models/bare. The packed variant swaps the three mixer
nn.Linear modules for PackedLinear (exponent-offset pack + MLX GemvWide clone with the k_lanes MLX itself uses).
Compares every output bitwise at S = 1, 3, 6, with and without a pending residual write.
Usage: test_packed_hc.py [layer]   (lane counts are probed against MLX itself; small GPU job)"""
import json, os, struct, sys
from types import SimpleNamespace
import numpy as np
import mlx.core as mx
import mlx.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
from hc_pack import pack_bf16, packed_bytes          # noqa: E402
from gemv_probe import gemv_wide                     # noqa: E402
from bigmoe.fuse import duprow                       # noqa: E402
from mtplx.models.qwen4_exp import GatedResidual     # noqa: E402

KL = dict(input_mix_weight_down=None, input_mix_weight_up=None, block_inject_weight=None)   # lanes probed per shape
LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 0
D = os.path.expanduser("~/BigMoE/models/bare")
cfg = json.load(open(f"{D}/config.json")); cfg = cfg.get("text_config", cfg)
args = SimpleNamespace(hc_count=cfg["hc_count"], hidden_size=cfg["hidden_size"], hc_lowrank=cfg["hc_lowrank"],
                       rms_norm_eps=cfg["rms_norm_eps"])


_LANES = {}


def lane_table(out_dims, in_dims, Ms=range(2, 16)):
    """k_lanes MLX's own x @ W.T uses for each row count M (found by bitwise match against the GemvWide clone).
    Depends only on the shape, so it is probed once per shape with random weights."""
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
    """Drop-in for a bias-free bf16 nn.Linear under BigMoE's dup-row reference: y = x @ W.T read from the
    exponent-offset pack with MLX's own GemvWide arithmetic and the lane count MLX picks for the reference's row
    count (max(2, M) under dup-row). Per-row results of GemvWide don't depend on M, so no row duplication is needed."""

    def __init__(self, lin, kl=None):
        super().__init__()
        self.out_dims, self.in_dims = lin.weight.shape
        self._pack = pack_bf16(lin.weight)
        self._lanes = lane_table(self.out_dims, self.in_dims)
        missing = [M for M in range(2, 16) if M not in self._lanes]
        assert not missing, f"MLX does not use GemvWide for {self.out_dims}x{self.in_dims} at M={missing}"

    def __call__(self, x):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        kl = self._lanes[max(2, x2.shape[0])]
        y = gemv_wide(None, x2, kl, packed=(self._pack, self.out_dims))
        return y.reshape(*shp[:-1], self.out_dims)


def load_layer(part):
    idx = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
    pre = f"language_model.model.layers.{LAYER}.{part}."
    out = {}
    for nm in [n for n in idx if n.startswith(pre)]:
        f = f"{D}/{idx[nm]}"
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            m = json.loads(fh.read(n))[nm]
        a, b = m["data_offsets"]
        u = np.array(np.memmap(f, np.uint16, "r", offset=8 + n + a, shape=((b - a) // 2,))).reshape(m["shape"])
        out[nm[len(pre):]] = mx.array(u).view(mx.bfloat16)
    return out


def build(part, packed):
    hc = GatedResidual(args, use_combine=True)
    hc.load_weights(list(load_layer(part).items()), strict=False)
    if packed:
        for name, kl in KL.items():
            setattr(hc, name, PackedLinear(getattr(hc, name), kl))
    mx.eval(hc.parameters())
    base = GatedResidual
    f0 = mx.compile(lambda h, _hc=hc: base.__call__(_hc, h))
    f1 = mx.compile(lambda h, bo, inj, _hc=hc: base.__call__(_hc, h, (bo, inj)))
    return hc, f0, f1


duprow()                                               # the reference runs with dup-row


def main():
    ok = True
    for part in ("attn_hyper_connection", "mlp_hyper_connection"):
        ref_hc, r0, r1 = build(part, False)
        pk_hc, p0, p1 = build(part, True)
        w_bytes = sum(getattr(ref_hc, n).weight.nbytes for n in KL)
        p_bytes = sum(packed_bytes(getattr(pk_hc, n)._pack) for n in KL)
        print(f"{part}: mixer weights {w_bytes/1e6:.2f} MB bf16 -> {p_bytes/1e6:.2f} MB packed ({p_bytes/w_bytes:.0%})")
        for S in (1, 3, 6):
            h = (mx.random.normal((1, S, args.hc_count * args.hidden_size)) * 0.5).astype(mx.bfloat16)
            bo = (mx.random.normal((1, S, args.hidden_size)) * 0.5).astype(mx.bfloat16)
            inj = (mx.random.uniform(shape=(1, S, args.hc_count)) * 2).astype(mx.bfloat16)
            for tag, a, b in (("read", r0(h), p0(h)), ("write+read", r1(h, bo, inj), p1(h, bo, inj))):
                a = a if isinstance(a, (tuple, list)) else (a,)
                b = b if isinstance(b, (tuple, list)) else (b,)
                eq = all(mx.array_equal(x, y).item() for x, y in zip(a, b))
                ok &= eq
                print(f"  S={S} {tag:10s}: packed == compiled reference (dup-row), bitwise: {eq}")
    print("ALL BIT-IDENTICAL" if ok else "MISMATCH")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
