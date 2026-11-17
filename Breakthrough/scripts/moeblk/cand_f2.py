"""cand_f2.py: bench arm 'f2' = F1 + the FRONT fold: router logits (MLX qmm) -> router_fused -> front (t1cf + slots +
the upgrade grouping + rowsum64(x) in one threadgroup) -> the routed + shared kernels -> combine_phase -> gated add"""
import moeblk as MB
import msh
import front as FR

mx = MB.mx


def block(lay, x):
    import bigmoe.tier as BT
    c = lay.c
    M, K = x.shape[0], c["top_k"]
    logits = lay.qmm(x, c["router"]["tensor"], c["router"]["bits"])
    ids, scores = MB.RF.router_fused(logits, K)
    f = FR.front(lay, ids, scores, x)
    y, _, _ = msh.experts_and_shared(lay, x, f["pm"], f["pl"], f["idn"], xs=f["xs"], grp=f["grp"])
    TG = BT._tier_glu_mod()
    yr = y[:M * K]
    yc = TG.combine_phase(yr, yr, mx.ones((M, K), mx.int32), f["sc"].reshape(-1, K))
    sg = lay.qmm(x, c["shared"]["gate_tensor"], c["shared"]["gate_bits"])
    return msh._gated_add(yc, y[M * K:], sg)
