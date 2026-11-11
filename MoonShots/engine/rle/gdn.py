"""Roofline engine: the attention half of a GDN layer (HC read -> in_proj -> conv/l2norm -> recurrence + gated norm ->
out_proj -> HC write folded into the MLP-side HC read). Weights come from MTPLX's loaded DecoderLayer (read-only use)."""
import mlx.core as mx
from . import kernels as KN

_IN = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")


def qw(m):
    """(weight, scales, biases) of an affine-quantized Linear"""
    return m.weight, m.scales, m.biases


def pack_hc(hc):
    """GatedResidual -> dict(Wd [324 (or 320), 10240] bf16, Wu [10240, 320], nw [10240])"""
    Wd = hc.input_mix_weight_down.weight
    if "block_inject_weight" in hc:
        Wd = mx.concatenate([Wd, hc.block_inject_weight.weight], axis=0)
    return dict(Wd=Wd, Wu=hc.input_mix_weight_up.weight, nw=hc.hc_norm.weight)


def pack_gdn(layer):
    la = layer.linear_attn
    if "in_proj_fused" in la:
        w_in = qw(la.in_proj_fused)
    else:
        parts = [qw(getattr(la, n)) for n in _IN]
        w_in = tuple(mx.concatenate([p[i] for p in parts], axis=0) for i in range(3))
    assert w_in[0].shape[0] == 16480, w_in[0].shape
    return dict(attn_hc=pack_hc(layer.attn_hyper_connection), mlp_hc=pack_hc(layer.mlp_hyper_connection), w_in=w_in,
                cw=la.conv1d.weight.reshape(10240, 4), A_log=la.A_log, dt_bias=la.dt_bias, nwg=la.norm.weight,
                w_out=qw(la.out_proj))


def qmm(x, w):
    return mx.quantized_matmul(x, *w, transpose=True, group_size=64, bits=4)


def attn_half(P, H, cst, state, M):
    """H [M, 10240] bf16 residual, cst [3, 10240] bf16 conv tail, state [48*16384] fp32. Returns (H1 = H after the
    attention write, mixed_m [M, 2560] = the MLP-side HC read, y_m [M, 324] fp32 (MLP inject at 320:324), cst2, state2)."""
    y1, mixed = KN.hc_read(H, P["attn_hc"]["Wd"], P["attn_hc"]["Wu"], P["attn_hc"]["nw"], M)
    gin, cst2 = KN.gdn_prework(qmm(mixed, P["w_in"]), P["cw"], cst, M)
    yr, state2 = KN.gdn_recurrence(gin, state, P["A_log"], P["dt_bias"], P["nwg"], M)
    H1 = KN.hc_write(H, y1, qmm(yr, P["w_out"]), M)
    y_m, mixed_m = KN.hc_read(H1, P["mlp_hc"]["Wd"], P["mlp_hc"]["Wu"], P["mlp_hc"]["nw"], M)
    return H1, mixed_m, y_m, cst2, state2


def compile_graph():
    """lazy graphs of every attention-half kernel at M = 1 and 4 with zero weights (for compile_check.py)"""
    z = lambda *s, dt=mx.bfloat16: mx.zeros(s, dt)
    q = lambda n, k: (z(n, k // 8, dt=mx.uint32), z(n, k // 64), z(n, k // 64))
    hc = lambda inj: dict(Wd=z(324 if inj else 320, 10240), Wu=z(10240, 320), nw=z(10240))
    P = dict(attn_hc=hc(True), mlp_hc=hc(True), w_in=q(16480, 2560), cw=z(10240, 4), A_log=z(48), dt_bias=z(48),
             nwg=z(128), w_out=q(2560, 6144))
    for M in (1, 4):
        attn_half(P, z(M, 10240), z(3, 10240), z(48 * 16384, dt=mx.float32), M)
