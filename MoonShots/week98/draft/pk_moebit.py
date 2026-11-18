"""PK v2: the fuse step and the MoE block as persistent phases (one dispatch per phase here) vs r57's live Q8H32 head,
bit for bit (MoonShots, Wed 18 Nov 2026):
  fuse: emb row (4-bit dequant) | rms norms | fc_embedding (8-bit qmv_fast) | fc_hidden x4 (qmv_wide) | fh + en
  MoE:  P9 router | shared gate / up + swiglu | the fp32 gate's dot partials; P10 route (TG 0) | shared down;
        P11 experts gate / up + swiglu (4-bit g32, R 2); P12 experts down (R 4); P13 combine + sigmoid(seg) * shared
usage: ~/BigMoE/.venv/bin/python -I -B pk_moebit.py"""
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

HDR = pk_ops.HEADER + RMS_HEADER + pk_ops.HC + pk_ops.MOE
COMMON = """
  const uint tg = threadgroup_position_in_grid.x, lid = thread_position_in_threadgroup.x, nt = threads_per_threadgroup.x;
  const uint sg = simdgroup_index_in_threadgroup, nsg = nt / 32u, lane = thread_index_in_simdgroup;
  const uint ntg = threadgroups_per_grid.x, gsg = tg * nsg + sg, nall = ntg * nsg;
  float tv_ = 0.0f; int ti_ = 0;
"""
FUSE1 = COMMON + """
  threadgroup float part[33];
  device bfloat16_t* mine = priv + tg * 16384u;
  pk_fuse_prep(ew, es, eb, uint(tok[0]), wid, wne, wnh, eps[0], eps[1], mine, mine + 2560u, mine + 5120u, part,
               lid, nt, sg, nsg, lane);
  for (uint u = gsg; u < 2560u + 640u; u += nall) {
    if (u < 2560u) qmv_pf<bfloat16_t, 64, 8, 1, 2, true, false>(few, fes, feb, mine + 2560u, en, 2560, int(u), lane, tv_, ti_);
    else { uint v = u - 2560u; qmv_wide_impl<bfloat16_t, 64, 8, 4, 8>(fhw, fhs, fhb, mine + 5120u, fh, 2560, 2560, 4,
                                                                       uint3(0, v >> 1, 0), v & 1u, lane); }
  }
  if (tg == 0u) for (uint i = lid; i < 2560u; i += nt) embo[i] = mine[i];
"""
FUSE2 = """
  uint i = thread_position_in_grid.x;
  if (i < 10240u) x[i] = fh[i] + en[i % 2560u];
"""
MOE_SRC = COMMON + """
  device bfloat16_t* S_ = (device bfloat16_t*)scr;
  device float* SF_ = (device float*)scr;
  device uint* SU_ = (device uint*)scr;
  device bfloat16_t* rl = S_ + 0u;  device bfloat16_t* sh = S_ + 512u;  device bfloat16_t* wts = S_ + 1152u;
  device bfloat16_t* segv = S_ + 1168u;  device bfloat16_t* sout = S_ + 1184u;  device bfloat16_t* eh = S_ + 3744u;
  device bfloat16_t* yout = S_ + 10144u;  device bfloat16_t* bo = S_ + 35744u;
  device float* spart = SF_ + 20000u;  device uint* inds = SU_ + 20064u;
  threadgroup float local_max[32], local_norm[32];
  threadgroup bfloat16_t P[512], tvv[10];
  threadgroup uint ti[10], sgk[40];
  if (PH == 9) {
    if (tg == ntg - 1u && sg < 16u) pk_seg_partial(x, segw, spart, sg, lane);
    const uint skip = 16u, wid = (tg == ntg - 1u) ? (sg >= 16u ? tg * nsg + sg - 16u : 0xFFFFFFFFu) : gsg;
    for (uint u = wid; u < 512u + 640u; u += nall - skip) {
      if (u < 512u) qmv_pf<bfloat16_t, 64, 8, 1, 2, true, false>(rw, rs, rb, x, rl, 2560, int(u), lane, tv_, ti_);
      else {
        uint j = u - 512u;
        bfloat16_t g[1], uu[1];
        qmv_rows<bfloat16_t, 64, 8, 1, true>(gw, gs_, gb, x, 2560, int(j), lane, g);
        qmv_rows<bfloat16_t, 64, 8, 1, true>(uw, us, ub, x, 2560, int(j), lane, uu);
        if (lane == 0u) sh[j] = fl_silu_jit(g[0]) * uu[0];
      }
    }
  } else if (PH == 10) {
    if (tg == 0u) {
      pk_route(rl, inds, wts, local_max, local_norm, P, ti, tvv, sgk, lid, lane, sg);
      if (sg == 0u) { bfloat16_t s = pk_seg_value(spart, lane); if (lane == 0u) segv[0] = s; }
    } else {
      for (uint u = (tg - 1u) * nsg + sg; u < 2560u; u += (ntg - 1u) * nsg)
        qmv_pf<bfloat16_t, 64, 8, 1, 1, false, false>(dw, ds, db, sh, sout, 640, int(u), lane, tv_, ti_);
    }
  } else if (PH == 11) {
    for (uint u = gsg; u < 10u * 320u; u += nall) {
      uint e = u / 320u, j = (u % 320u) * 2u;
      size_t ex = inds[e];
      bfloat16_t g[2], uu[2];
      qmv_rows<bfloat16_t, 32, 4, 2, true>(egw + ex * (640 * 320), egs + ex * (640 * 80), egb + ex * (640 * 80), x, 2560, int(j), lane, g);
      qmv_rows<bfloat16_t, 32, 4, 2, true>(euw + ex * (640 * 320), eus + ex * (640 * 80), eub + ex * (640 * 80), x, 2560, int(j), lane, uu);
      if (lane == 0u) { eh[e * 640u + j] = fl_silu_jit(g[0]) * uu[0]; eh[e * 640u + j + 1u] = fl_silu_jit(g[1]) * uu[1]; }
    }
  } else if (PH == 12) {
    for (uint u = gsg; u < 10u * 640u; u += nall) {
      uint e = u / 640u, r0 = (u % 640u) * 4u;
      size_t ex = inds[e];
      qmv_pf<bfloat16_t, 32, 4, 4, 1, false, false>(edw + ex * (2560 * 80), eds + ex * (2560 * 20), edb + ex * (2560 * 20),
                                                    eh + e * 640u, yout + e * 2560u, 640, int(r0), lane, tv_, ti_);
    }
  } else if (PH == 13) {
    uint c = thread_position_in_grid.x;
    if (c < 2560u) bo[c] = pk_moe_out(yout, wts, segv[0], sout, c);
  }
"""
ARGS = ["x", "rw", "rs", "rb", "gw", "gs_", "gb", "uw", "us", "ub", "dw", "ds", "db", "segw", "egw", "egs", "egb", "euw",
        "eus", "eub", "edw", "eds", "edb"]
OUTS = ["rl", "sh", "spart", "inds", "wts", "segv", "sout", "eh", "yout", "bo"]
OSH = [(512,), (640,), (64,), (16,), (16,), (16,), (2560,), (6400,), (25600,), (2560,)]
ODT = [mx.bfloat16, mx.bfloat16, mx.float32, mx.uint32, mx.bfloat16, mx.bfloat16, mx.bfloat16, mx.bfloat16, mx.bfloat16,
       mx.bfloat16]


def main():
    rt, lm, head = DD.M.load(0.03)
    R.install(lm, "q8h32")
    L0 = head.layers[0]
    att, moe = L0.self_attn, L0.mlp
    kf1 = mx.fast.metal_kernel(name="pk_fuse1", input_names=["ew", "es", "eb", "tok", "wid", "wne", "wnh", "eps", "few", "fes",
                                                              "feb", "fhw", "fhs", "fhb"],
                               output_names=["priv", "en", "fh", "embo"], source=FUSE1, header=HDR)
    kf2 = mx.fast.metal_kernel(name="pk_fuse2", input_names=["fh", "en"], output_names=["x"], source=FUSE2, header=HDR)
    et = lm.model.embed_tokens
    sw = moe.switch_mlp
    se = moe.shared_expert
    tot = ok = 0
    for fi, s in ((0, 182), (2, 400), (5, 300)):
        z = np.load(CE.HEP[fi])
        ids = np.asarray(z["ids"], np.int32)
        W = mx.array(z["wd"]).astype(mx.bfloat16)[None]
        R.begin_window(lm, ids[:int(z["gen"])])
        mtp = RA.Head(lm)
        mtp.append(W[:, :s + 1], ids[1:s + 2])
        mx.eval(mtp.logits, mtp.h)
        tok = int(ids[s + 2])
        emb_r = et(mx.array([[tok]]))
        x_r = head._mtp_prepare_inputs_impl(mtp.h, emb_r)
        fc_e, fc_h = head.fc_embedding, head.fc_hidden
        priv, en, fh, embo = kf1(inputs=[et.weight.view(mx.uint8).reshape(-1), et.scales.reshape(-1), et.biases.reshape(-1),
                                         mx.array([tok], mx.int32), mtp.h.reshape(-1), head.pre_fc_norm_embedding.weight,
                                         head.pre_fc_norm_hidden.weight,
                                         mx.array([head.pre_fc_norm_embedding.eps, head.pre_fc_norm_hidden.eps], mx.float32),
                                         fc_e.weight, fc_e.scales, fc_e.biases, fc_h.weight, fc_h.scales, fc_h.biases],
                                 grid=(20 * 1024, 1, 1), threadgroup=(1024, 1, 1),
                                 output_shapes=[(20 * 16384,), (2560,), (10240,), (2560,)], output_dtypes=[mx.bfloat16] * 4)
        x = kf2(inputs=[fh, en], grid=(10240, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(10240,)],
                output_dtypes=[mx.bfloat16])[0]
        for nm, a, b in (("emb row", embo, emb_r), ("fuse x", x, x_r)):
            e = bool(mx.array_equal(a.reshape(-1), b.reshape(-1)).item()); tot += 1; ok += int(e)
            print(f"  w{fi} s{s} {nm}: {'BITWISE' if e else str(int((a.reshape(-1) != b.reshape(-1)).sum().item())) + ' differ'}", flush=True)
        # the MoE on a real MLP-HC output (the stock path up to it)
        mixed, hyper, inject = L0.attn_hyper_connection(x_r)
        bo1 = att.o_proj(mx.zeros((1, 1, 6144), mx.bfloat16) + (mx.random.normal((1, 1, 6144)) * 0.3).astype(mx.bfloat16))
        mixed2, _, _ = L0.mlp_hyper_connection(hyper, pending=(bo1, inject))
        ref = L0.mlp(mixed2).reshape(-1)
        ins = [mixed2.reshape(-1), moe.gate.weight, moe.gate.scales, moe.gate.biases, se.gate_proj.weight, se.gate_proj.scales,
               se.gate_proj.biases, se.up_proj.weight, se.up_proj.scales, se.up_proj.biases, se.down_proj.weight,
               se.down_proj.scales, se.down_proj.biases, moe.shared_expert_gate.weight.reshape(-1),
               sw.gate_proj.weight, sw.gate_proj.scales, sw.gate_proj.biases, sw.up_proj.weight, sw.up_proj.scales,
               sw.up_proj.biases, sw.down_proj.weight, sw.down_proj.scales, sw.down_proj.biases]
        scr = mx.zeros((20200,), mx.float32)
        km = mx.fast.metal_kernel(name="pk_moe", input_names=ARGS + ["scr"], output_names=["dummy"], source=MOE_SRC,
                                  header=HDR)
        for ph in (9, 10, 11, 12, 13):
            d = km(inputs=ins + [scr], template=[("PH", ph)], grid=(20 * 1024, 1, 1), threadgroup=(1024, 1, 1),
                   output_shapes=[(1,)], output_dtypes=[mx.float32])[0]
            mx.eval(d)
        mx.synchronize()
        S_ = scr.view(mx.bfloat16)
        st = {"bo": S_[35744:35744 + 2560], "inds": scr.view(mx.uint32)[20064:20064 + 10]}
        e = bool(mx.array_equal(st["bo"], ref).item()); tot += 1; ok += int(e)
        print(f"  w{fi} s{s} MoE block: {'BITWISE' if e else str(int((st['bo'] != ref).sum().item())) + ' differ'}"
              f" (experts {sorted(np.array(st['inds'][:10]).tolist())})", flush=True)
    print(f"[pk_moebit] bitwise {ok} / {tot}", flush=True)


if __name__ == "__main__":
    main()
