"""PK v2: the tree attention phase (pk_ops.pk_attn_head: q / k norms, RoPE at POS, sdpa_vector one-pass over the cache
prefix + drafted rows (ancestor mask) + self, attention_gate) vs forward_static's stock ops on r57's live Q8H32 head,
bit for bit; t = 0 (no mask) and t > 0 (random ancestor masks) (MoonShots, Wed 18 Nov 2026).
usage: ~/BigMoE/.venv/bin/python -I -B pk_attnbit.py"""
import os
import sys

import numpy as np

H = os.path.expanduser
sys.path.insert(0, H("~/MoonShots/week98/draft"))
sys.path.insert(0, H("~/MoonShots/week98/bet7"))
sys.path.insert(0, H("~/BigMoE/scripts"))
import mlx.core as mx  # noqa: E402
import pk_ops  # noqa: E402
import ra_sf as RA  # noqa: E402
import r57_draft as R  # noqa: E402
import dd_ceiling as CE  # noqa: E402
import dd_ttt as DD  # noqa: E402
from fl_exact import RMS_HEADER  # noqa: E402

SRC = """
  const uint lid = thread_position_in_threadgroup.x, simd_gid = simdgroup_index_in_threadgroup;
  const uint simd_lid = thread_index_in_simdgroup, nsg = threads_per_threadgroup.x / 32u;
  threadgroup float part[33];
  threadgroup bfloat16_t qr[256], kr[256], qn[256], kn[256];
  threadgroup float cs[64], sn[64], outputs[32 * 32], max_scores[32], sum_exp_scores[32];
  for (uint h = threadgroup_position_in_grid.x; h < 24u; h += threadgroups_per_grid.x)
    pk_attn_head(h, qf, kf, vf, kc, vc, uint(meta[0]), kd, vd, uint(meta[1]), uint(meta[2]), uint(meta[3]), uint(meta[4]),
                 qnw, knw, inv_freq, eps[0], gated, krot, part, qr, kr, cs, sn, qn, kn, outputs, max_scores,
                 sum_exp_scores, lid, simd_gid, simd_lid, nsg);
"""
KER = mx.fast.metal_kernel(name="pk_attnbit", input_names=["qf", "kf", "vf", "kc", "vc", "kd", "vd", "meta", "qnw", "knw",
                                                           "inv_freq", "eps"],
                           output_names=["gated", "krot"], source=SRC,
                           header=pk_ops.HEADER + RMS_HEADER + pk_ops.HC + pk_ops.ATTN)


def main():
    import mtplx.models.qwen4_exp as Q
    from mtplx.attention_math import attention_gate
    rt, lm, head = DD.M.load(0.03)
    R.install(lm, "q8h32")
    L0 = head.layers[0]
    att = L0.self_attn
    rng = np.random.default_rng(3)
    tot = ok = 0
    for fi, s in ((0, 182), (2, 400), (5, 600)):
        z = np.load(CE.HEP[fi])
        ids = np.asarray(z["ids"], np.int32)
        W = mx.array(z["wd"]).astype(mx.bfloat16)[None]
        R.begin_window(lm, ids[:int(z['gen'])])
        mtp = RA.Head(lm)
        mtp.append(W[:, :s + 1], ids[1:s + 2])
        mx.eval(mtp.logits, mtp.h)
        qc = mtp.cache[0]
        Lc = int(qc.offset)
        x = head._mtp_prepare_inputs_impl(mtp.h, lm.model.embed_tokens(mx.array([[int(ids[s + 2])]])))
        mixed, _, _ = L0.attn_hyper_connection(x)
        qf, kf, vf = att.q_proj(mixed), att.k_proj(mixed), att.v_proj(mixed)
        mx.eval(qf, kf, vf)
        for t, d0 in ((0, 0), (5, 2), (14, 6), (9, 9)):
            pos = mx.array([Lc + d0], mx.int32)
            q, gate = mx.split(qf.reshape(1, 1, 24, -1), 2, axis=-1)
            gate = gate.reshape(1, 1, -1)
            k, v = kf.reshape(1, 1, 2, -1), vf.reshape(1, 1, 2, -1)
            q, k = att.q_norm(q), att.k_norm(k)
            q, k = RA._rope_qk(att, q, k, pos)
            q, k, v = q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)
            Kc, Vc = qc.kv.keys[..., :Lc, :], qc.kv.values[..., :Lc, :]
            k, v = k.astype(Kc.dtype), v.astype(Vc.dtype)
            if t:
                Kd = (mx.random.normal((1, 2, t, 256)) * 1.5).astype(mx.bfloat16)
                Vd = mx.random.normal((1, 2, t, 256)).astype(mx.bfloat16)
                bits = rng.integers(0, 2, t)
                anc = int(sum(int(b) << i for i, b in enumerate(bits)))
                mask = mx.concatenate([mx.ones((Lc,), mx.bool_), mx.array(bits.astype(bool)), mx.ones((1,), mx.bool_)]).reshape(1, 1, 1, -1)
                Ka, Va = mx.concatenate([Kc, Kd, k], axis=2), mx.concatenate([Vc, Vd, v], axis=2)
            else:
                Kd = Vd = mx.zeros((1, 2, 1, 256), mx.bfloat16)
                anc, mask = 0, None
                Ka, Va = mx.concatenate([Kc, k], axis=2), mx.concatenate([Vc, v], axis=2)
            out = Q._verify_sdpa(q, Ka, Va, scale=att.scale, mask=mask)
            ref = attention_gate(out.transpose(0, 2, 1, 3).reshape(1, 1, -1), gate).reshape(-1)
            kb, vb = qc.kv.keys, qc.kv.values
            cap = int(kb.shape[2])
            meta = mx.array([cap * 256, Lc, t, anc, Lc + d0], mx.int32)
            g, kr = KER(inputs=[qf.reshape(-1), kf.reshape(-1), vf.reshape(-1), kb.reshape(-1), vb.reshape(-1),
                                Kd.transpose(0, 2, 1, 3).reshape(-1), Vd.transpose(0, 2, 1, 3).reshape(-1), meta,
                                att.q_norm.weight, att.k_norm.weight, att._inv_freq, mx.array([att.q_norm.eps], mx.float32)],
                        grid=(20 * 1024, 1, 1), threadgroup=(1024, 1, 1), output_shapes=[(6144,), (512,)],
                        output_dtypes=[mx.bfloat16, mx.bfloat16])
            e1 = bool(mx.array_equal(g, ref).item())
            e2 = bool(mx.array_equal(kr, k.reshape(-1)).item())
            tot += 2; ok += int(e1) + int(e2)
            print(f"  w{fi} s{s} Lc {Lc} t {t} pos +{d0}: gated {'BITWISE' if e1 else str(int((g != ref).sum().item())) + ' differ'}"
                  f" | krot {'BITWISE' if e2 else 'differs'}", flush=True)
    print(f"[pk_attnbit] bitwise {ok} / {tot}", flush=True)


if __name__ == "__main__":
    main()
