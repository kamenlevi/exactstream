"""PK v2: the HC read as two persistent phases (pk_ops.pk_hc_a / pk_hc_b, 20 TGs x 1024, the grid barrier = two
dispatches here) vs the live GatedResidual modules of r57's Q8H32 head, bit for bit: the attention HC (no pending,
inject), the MLP HC (pending write, inject), the mixer (no inject) (MoonShots, Wed 18 Nov 2026).
usage: ~/BigMoE/.venv/bin/python -I -B pk_hcbit.py"""
import os
import sys

import numpy as np

H = os.path.expanduser
sys.path.insert(0, H("~/MoonShots/week98/draft"))
sys.path.insert(0, H("~/BigMoE/scripts"))
import mlx.core as mx  # noqa: E402
import pk_ops  # noqa: E402
import r57_draft as R  # noqa: E402
import dd_ceiling as CE  # noqa: E402
import dd_ttt as DD  # noqa: E402

NTG, NT = 20, 1024
SRC_A = """
  const uint tg = threadgroup_position_in_grid.x, lid = thread_position_in_threadgroup.x, nt = threads_per_threadgroup.x;
  const uint sg = simdgroup_index_in_threadgroup, nsg = nt / 32u, lane = thread_index_in_simdgroup;
  const uint ntg = threadgroups_per_grid.x;
  threadgroup float part[84];
  threadgroup float inv[4];
  pk_hc_a<PENDING, COMBINE>(hyper, bout, pinj, nw, eps[0], wd, sd, bd, wi, nrm + tg * 10240u, hw, mix1, ipart, part, inv,
                            tg, ntg, ntg - 1u, lid, nt, sg, nsg, lane);
"""
SRC_B = """
  const uint tg = threadgroup_position_in_grid.x, sg = simdgroup_index_in_threadgroup;
  const uint nsg = threads_per_threadgroup.x / 32u, lane = thread_index_in_simdgroup, ntg = threadgroups_per_grid.x;
  pk_hc_b<COMBINE>(mix1, wu, su, bu, nrm + tg * 10240u, mixed, ipart, inj, tg, ntg, sg, nsg, lane);
"""
HDR = pk_ops.HEADER + pk_ops.HC
KA = mx.fast.metal_kernel(name="pk_hc_a", input_names=["hyper", "bout", "pinj", "nw", "eps", "wd", "sd", "bd", "wi"],
                          output_names=["nrm", "hw", "mix1", "ipart"], source=SRC_A, header=HDR)
KB = mx.fast.metal_kernel(name="pk_hc_b", input_names=["mix1", "wu", "su", "bu", "nrm", "ipart"],
                          output_names=["mixed", "inj"], source=SRC_B, header=HDR)


def pk_read(gr, hyper, pending=None):
    combine = "block_inject_weight" in gr
    bout = pending[0].reshape(-1) if pending is not None else mx.zeros((2560,), mx.bfloat16)
    pinj = pending[1].reshape(-1) if pending is not None else mx.zeros((4,), mx.bfloat16)
    pinj = mx.concatenate([pinj, mx.zeros((60,), mx.bfloat16)])     # >= 8 elements: device memory, as in the PK
    wi = gr.block_inject_weight.weight.reshape(-1) if combine else mx.zeros((64,), mx.bfloat16)
    dn, up = gr.input_mix_weight_down, gr.input_mix_weight_up
    t = [("PENDING", pending is not None), ("COMBINE", combine)]
    nrm, hw, mix1, ipart = KA(inputs=[hyper.reshape(-1), bout, pinj, gr.hc_norm.weight, mx.array([gr.hc_norm.eps], mx.float32),
                                      dn.weight, dn.scales, dn.biases, wi], template=t,
                              grid=(NTG * NT, 1, 1), threadgroup=(NT, 1, 1),
                              output_shapes=[(NTG * 10240,), (10240,), (320,), (32,)],
                              output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16, mx.float32])
    mixed, inj = KB(inputs=[mix1, up.weight, up.scales, up.biases, nrm, ipart], template=[("COMBINE", combine)],
                    grid=(NTG * NT, 1, 1), threadgroup=(NT, 1, 1), output_shapes=[(2560,), (4,)],
                    output_dtypes=[mx.bfloat16, mx.bfloat16])
    return mixed, inj, hw


def eq(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return bool(mx.array_equal(a, b).item()), int((a != b).sum().item())


def main():
    rt, lm, head = DD.M.load(0.03)
    R.install(lm, "q8h32")
    L0 = head.layers[0]
    tot = ok = 0
    for fi, s in ((0, 182), (2, 400), (5, 300)):
        z = np.load(CE.HEP[fi])
        ids = np.asarray(z["ids"], np.int32)
        W = mx.array(z["wd"]).astype(mx.bfloat16)[None]
        x = head._mtp_prepare_inputs_impl(W[:, s:s + 1], lm.model.embed_tokens(mx.array([[int(ids[s + 1])]])))
        m_r, h_r, i_r = L0.attn_hyper_connection(x)
        m, i, _ = pk_read(L0.attn_hyper_connection, x)
        for nm, (a, b) in (("attn mixed", (m, m_r)), ("attn inject", (i, i_r))):
            e, nd = eq(a, b); tot += 1; ok += int(e)
            print(f"  w{fi} s{s} {nm}: {'BITWISE' if e else f'{nd} differ'}", flush=True)
        bo = (mx.random.normal((1, 1, 2560)) * 0.5).astype(mx.bfloat16)
        m2_r, h2_r, i2_r = L0.mlp_hyper_connection(h_r, pending=(bo, i_r))
        m2, i2, hw2 = pk_read(L0.mlp_hyper_connection, h_r, pending=(bo, i_r))
        for nm, (a, b) in (("mlp mixed", (m2, m2_r)), ("mlp inject", (i2, i2_r)), ("mlp written", (hw2, h2_r))):
            e, nd = eq(a, b); tot += 1; ok += int(e)
            print(f"  w{fi} s{s} {nm}: {'BITWISE' if e else f'{nd} differ'}", flush=True)
        xm_r = head.hyper_connection_mixer(h2_r)
        xm, _, _ = pk_read(head.hyper_connection_mixer, h2_r)
        e, nd = eq(xm, xm_r); tot += 1; ok += int(e)
        print(f"  w{fi} s{s} mixer mixed: {'BITWISE' if e else f'{nd} differ'}", flush=True)
    print(f"[pk_hcbit] bitwise {ok} / {tot}", flush=True)


if __name__ == "__main__":
    main()
