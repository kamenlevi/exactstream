"""gdn_exact.py vs the eager chain of MTPLX's GatedDeltaNet (verbatim ops: concatenate, nn.Conv1d(groups=C), nn.silu, split,
_l2norm, inv_scale * q, new state = last 3 stream rows), bitwise, S = 1..8, random bf16 data at several scales.
usage: test_gdn_exact.py   (GPU, seconds)"""
import os, sys
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p4s/Leap_third_party/ms_gdn"))   # read-only use
import mlx.core as mx
import mlx.nn as nn
from gdn_exact import gdn_prework_exact

C, KW, KD, VD = 10240, 4, 2048, 6144
conv = nn.Conv1d(C, C, kernel_size=KW, groups=C, bias=False)


def eager(qkv, conv_state):
    """MTPLX GatedDeltaNet.__call__'s stock branch (B = 1), the lines between the in_proj and gated_delta_update."""
    S = qkv.shape[1]
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    new_state = mx.contiguous(conv_input[:, -(KW - 1):, :])
    conv_out = nn.silu(conv(conv_input))
    q, k, v = [t.reshape(1, S, h, d) for t, h, d in zip(mx.split(conv_out, [KD, 2 * KD], -1), [16, 16, 48], [128, 128, 128])]
    inv_scale = k.shape[-1] ** -0.5

    def _l2norm(x):
        xf = x.astype(mx.float32)
        return (xf * mx.rsqrt((xf * xf).sum(-1, keepdims=True) + 1e-6)).astype(x.dtype)

    q = inv_scale * _l2norm(q)
    k = _l2norm(k)
    return q, k, v, new_state


if __name__ == "__main__":
    scale_bf16 = mx.array(1.0, mx.bfloat16) * (128 ** -0.5)       # how MLX turns the Python scalar into a bf16 operand
    ok = True
    for seed, (sx, sw) in enumerate(((1.0, 0.3), (3.0, 0.5), (0.2, 1.0))):
        mx.random.seed(seed)
        conv.weight = (mx.random.normal(conv.weight.shape) * sw).astype(mx.bfloat16)
        for S in range(1, 17):
            qkv = (mx.random.normal((1, S, C)) * sx).astype(mx.bfloat16)
            st = (mx.random.normal((1, KW - 1, C)) * sx).astype(mx.bfloat16)
            rq, rk, rv, rs = eager(qkv, st)
            fq, fk, fv, fs = gdn_prework_exact(qkv.reshape(S, C), st.reshape(KW - 1, C), conv.weight.reshape(C, KW), scale_bf16)
            mism = [int((a.reshape(-1).view(mx.uint16) != b.reshape(-1).view(mx.uint16)).sum().item())
                    for a, b in ((rq, fq), (rk, fk), (rv, fv), (rs, fs))]
            ok &= sum(mism) == 0
            print(f"scale x {sx} w {sw} S={S}: mismatches q/k/v/state {mism}", flush=True)
    print("PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)
