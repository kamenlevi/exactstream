"""3c check (tierglu.py) on real experts: MSB/LSB pools of 64 experts of one layer (make_records3's converter, in memory,
from records/), against BigMoE's 3b path (bigmoe/tier.py _proj: rebuild4 + gather_qmm bits=4 / bits=3, SwiGLU):
  1. the activation (TgSigmoid/TgMultiply in bfloat16) == mlx_lm's compiled swiglu on all 65536 gate values, bitwise;
  2. plan 4433333333, rows M = 1, 2, 4, 8: h and per-rank y == 3b, bitwise (else max |d|);
  3. plan 4433332222 (2-bit tail = MSB-3 codes with the low bit cleared, bias b2 = bf16(b + s mean(q & 3))): y == the
     same tiers through gather_qmm(bits=3) on masked codes, bitwise; and vs planes125's masked-4-bit tiers (allclose).
usage: test_tierglu.py [layer]   (GPU: ~0.6 GB, seconds)"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.join(HERE, "..", "planes125"))
import mlx.core as mx
import make_records3 as M
from tierglu import tier_glu as _tier_glu, PROJ, _HDR
LD = int(os.environ.get("TG_LD", "0"))          # kernel A load variant (0 = byte loads, 1 = ushort loads)
tier_glu = lambda *a, **k: _tier_glu(*a, ld=LD, **k)
from mlx_lm.models.switch_layers import SwiGLU

L = int(sys.argv[1]) if len(sys.argv) > 1 else 0
NE = 64
MD = os.path.expanduser("~/BigMoE/models/bare/records")
lay = M.layout()
R3 = lay["recsz"]
if os.path.exists(os.path.join(MD, f"L{L:02d}.bin")):          # 4-bit records still there: convert in memory
    si = json.load(open(os.path.join(MD, "index.json")))["layers"][str(L)]
    rs, po, ps = si["recsz"], si["piece_off"], si["piece_size"]
    fd = M.open_nocache(os.path.join(MD, si["file"]), os.O_RDONLY)
    src = M.aligned(NE * rs)
    M.pread_into(fd, src, 0)
    os.close(fd)
    r3 = M.convert_chunk(src, NE, si, lay)
else:                                                          # records/ converted away: records3 + the verified rebuild
    fd = M.open_nocache(os.path.expanduser(f"~/BigMoE/models/bare/records3/L{L:02d}.bin"), os.O_RDONLY)
    r3 = M.aligned(NE * R3)
    M.pread_into(fd, r3, 0)
    os.close(fd)
    po, ps, o = [], [], 0
    for of, K in M.SHAPES:
        for n in (of * K // 2, of * K // 64 * 2, of * K // 64 * 2):
            po.append(o); ps.append(n); o += n
    rs = o
    src = np.zeros(NE * rs, np.uint8)
    for i in range(3):
        mo, lo = lay["msb_piece_off"][3 * i:3 * i + 3], lay["lsb_piece_off"][2 * i:2 * i + 2]
        ms, ls = lay["msb_piece_size"][3 * i:3 * i + 3], lay["lsb_piece_size"][2 * i:2 * i + 2]
        w4 = M.merge_codes(M.view(r3, NE, R3, mo[0], ms[0], np.uint8, inner=3), M.view(r3, NE, R3, lo[0], ls[0], np.uint8))
        s = M.f32_to_bf16(M.bf16_to_f32(M.view(r3, NE, R3, mo[1], ms[1], np.uint16)) * np.float32(0.5))
        b = M.view(r3, NE, R3, lo[1], ls[1], np.uint16)
        for k, a in enumerate((w4, s, b)):
            M.view(src, NE, rs, po[3 * i + k], ps[3 * i + k], a.dtype)[...] = a


def piece(buf, rec, off, size, dt, shape):
    a = np.stack([np.frombuffer(buf, dt, size // np.dtype(dt).itemsize, e * rec + off) for e in range(NE)])
    return mx.array(a.reshape(NE, *shape))


orig, msb, lsb, b2 = {}, {}, {}, {}
for i, (p, (out, K)) in enumerate(zip(PROJ, M.SHAPES)):
    orig[p] = (piece(src, rs, po[3 * i], ps[3 * i], np.uint32, (out, K // 8)),
               piece(src, rs, po[3 * i + 1], ps[3 * i + 1], np.uint16, (out, K // 64)).view(mx.bfloat16),
               piece(src, rs, po[3 * i + 2], ps[3 * i + 2], np.uint16, (out, K // 64)).view(mx.bfloat16))
    mo, ms, lo, ls = (lay["msb_piece_off"][3 * i:3 * i + 3], lay["msb_piece_size"][3 * i:3 * i + 3],
                      lay["lsb_piece_off"][2 * i:2 * i + 2], lay["lsb_piece_size"][2 * i:2 * i + 2])
    msb[p] = {"w3": piece(r3, R3, mo[0], ms[0], np.uint32, (out, K * 3 // 32)),
              "s3": piece(r3, R3, mo[1], ms[1], np.uint16, (out, K // 64)).view(mx.bfloat16),
              "b3": piece(r3, R3, mo[2], ms[2], np.uint16, (out, K // 64)).view(mx.bfloat16)}
    lsb[p] = {"l1": piece(r3, R3, lo[0], ls[0], np.uint32, (out, K // 32)),
              "b": piece(r3, R3, lo[1], ls[1], np.uint16, (out, K // 64)).view(mx.bfloat16)}
    # 2-bit tier bias, as planes125._tier_proj(n=2): bf16(fp32(b) + fp32(s) * mean(q & 3))
    w4, s4, bias4 = orig[p]
    q = (w4[..., None] >> mx.arange(0, 32, 4).astype(mx.uint32)) & 15
    off = (q & 3).reshape(*s4.shape, 64).sum(-1).astype(mx.float32) / 64
    b2[p] = (bias4.astype(mx.float32) + s4.astype(mx.float32) * off).astype(mx.bfloat16)
mx.eval(orig, msb, lsb, b2)
ok = True

# 1. activation, exhaustive over the gate's bf16 values
act_k = mx.fast.metal_kernel(name="tg_act_test", input_names=["g", "u"], output_names=["h"], header=_HDR, source="""
    const uint i = thread_position_in_grid.x;
    h[i] = TgMultiply()(TgMultiply()(g[i], TgSigmoid()(g[i])), u[i]);""")
g = mx.arange(65536, dtype=mx.uint32).astype(mx.uint16).view(mx.bfloat16)
for seed in range(3):
    u = (mx.random.normal((65536,), key=mx.random.key(seed)) * 2.0).astype(mx.bfloat16)
    (h1,) = act_k(inputs=[g, u], grid=(65536, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(65536,)],
                  output_dtypes=[mx.bfloat16])
    h2 = SwiGLU()(u, g)                                    # SwitchGLU: activation(x_up, x_gate) = silu(gate) * up
    same = mx.array_equal(h1.view(mx.uint16), h2.view(mx.uint16)).item()
    nan_same = mx.array_equal(mx.isnan(h1), mx.isnan(h2)).item()
    finite = ~(mx.isnan(h1) | mx.isnan(h2))
    neq = int(((h1.view(mx.uint16) != h2.view(mx.uint16)) & finite).sum().item())
    ok &= (neq == 0) and nan_same
    print(f"1. activation (seed {seed}): bitwise equal {same}; non-NaN mismatches {neq}; NaN pattern equal {nan_same}", flush=True)

import bigmoe.tier as T
# per u32 word of a row's 3-bit stream: clear each code's low bit (stream bits k with k % 3 == 0)
MASK2W = mx.array([0xB6DB6DB6, 0x6DB6DB6D, 0xDB6DB6DB], mx.uint32)
pools = (msb, lsb)


def ref_3b(x, sm, sl, plan, b2pool=None):
    """BigMoE's tier_call body (bigmoe/tier.py) on these pools; 2-bit segments: gather_qmm(bits=3) on masked codes + b2."""
    xe = mx.expand_dims(x[None], (-2, -3))                 # [1, M, 1, 1, K]
    sm, sl = sm[None], (sl[None] if sl is not None else None)
    outs, hs = [], []
    for a, b, n in T.segments(plan):
        if n == 2:
            def pj(h, p):                                  # 3-bit words: bit k of the stream, field = k // 3: mask per word
                w = msb[p]["w3"][sm[..., a:b]]
                shp = w.shape
                wm = (w.reshape(*shp[:-1], -1, 3) & MASK2W).reshape(shp)
                idx = mx.arange(sm[..., a:b].size, dtype=mx.uint32).reshape(sm[..., a:b].shape)
                return mx.gather_qmm(h, wm, msb[p]["s3"][sm[..., a:b]], b2pool[p][sm[..., a:b]], rhs_indices=idx,
                                     transpose=True, group_size=64, bits=3)
            up, gt = pj(xe, "up_proj"), pj(xe, "gate_proj")
            h = SwiGLU()(up, gt)
            hs.append(h); outs.append(pj(h, "down_proj").squeeze(-2))
            continue
        sl_ab = (sm[..., a:b], sl[..., a:b]) if n == 4 else sm[..., a:b]
        up = T._proj(xe, pools, "up_proj", sl_ab, n)
        gt = T._proj(xe, pools, "gate_proj", sl_ab, n)
        h = SwiGLU()(up, gt)
        hs.append(h)
        outs.append(T._proj(h, pools, "down_proj", sl_ab, n).squeeze(-2))
    return mx.concatenate(outs, axis=-2)[0], mx.concatenate(hs, axis=-3)[0]


def get_h(x, sm, sl, plan, b2pool=None):
    """tierglu's kernel A output (h) for the same call, via a second call that returns it."""
    import tierglu as TG
    Mr, K = x.shape
    m4, m2, r4 = TG.plan_masks(plan)
    bb = b2pool if b2pool is not None else {p: msb[p]["b3"] for p in PROJ}
    g_, u_ = msb["gate_proj"], msb["up_proj"]
    N = g_["s3"].shape[1]
    (h,) = TG._ka()(inputs=[x, g_["w3"], u_["w3"], g_["s3"], u_["s3"], g_["b3"], u_["b3"], bb["gate_proj"], bb["up_proj"],
                            lsb["gate_proj"]["l1"], lsb["up_proj"]["l1"], lsb["gate_proj"]["b"], lsb["up_proj"]["b"],
                            sm.astype(mx.int32).reshape(-1), sl.astype(mx.int32).reshape(-1)],
                    template=[("T", x.dtype), ("NR_C", len(plan)), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)),
                              ("K_C", K), ("N_C", N), ("RA_C", 4), ("LD_C", LD)],
                    grid=(N // 4 * 64, Mr * len(plan), 1), threadgroup=(64, 1, 1),
                    output_shapes=[(Mr * len(plan), N)], output_dtypes=[x.dtype])
    return h.reshape(Mr, len(plan), N)


rng = np.random.default_rng(7)
for plan, b2p in (("4433333333", None), ("4433332222", b2)):
    for Mr in (1, 2, 4, 8):
        x = mx.array((rng.standard_normal((Mr, 2560)) * 0.6).astype(np.float32)).astype(mx.bfloat16)
        sm = mx.array(np.stack([rng.choice(NE, 10, replace=False) for _ in range(Mr)]).astype(np.int32))
        sl = mx.array(np.stack([rng.choice(NE, 2, replace=False) for _ in range(Mr)]).astype(np.int32))
        y = tier_glu(x, msb, lsb, sm, sl, plan=plan, b2=b2p)
        yr, hr = ref_3b(x, sm, sl, plan, b2p)
        h = get_h(x, sm, sl, plan, b2p)
        hr = hr.reshape(Mr, 10, -1)
        mx.eval(y, yr, h, hr)
        heq = mx.array_equal(h.view(mx.uint16), hr.view(mx.uint16)).item()
        per_rank = [mx.array_equal(y[:, k].view(mx.uint16), yr[:, k].view(mx.uint16)).item() for k in range(10)]
        hd = mx.abs(h.astype(mx.float32) - hr.astype(mx.float32)).max().item()
        yd = mx.abs(y.astype(mx.float32) - yr.astype(mx.float32)).max().item()
        ymax = mx.abs(yr.astype(mx.float32)).max().item()
        ok &= heq and all(per_rank)
        print(f"{'2' if plan == '4433333333' else '3'}. plan {plan} M={Mr}: h bitwise {heq} (max |d| {hd:.3g}); "
              f"y per rank bitwise {''.join('1' if e else '0' for e in per_rank)} (max |d| {yd:.3g}, |y| max {ymax:.3g})", flush=True)

# 3b. the 2-bit tier vs planes125's evaluated values (masked 4-bit codes; same weights, other kernel arithmetic)
import planes125 as P125
x = mx.array((rng.standard_normal((1, 2560)) * 0.6).astype(np.float32)).astype(mx.bfloat16)
sm = mx.array(rng.choice(NE, 10, replace=False)[None].astype(np.int32))
xe = mx.expand_dims(x[None], (-2, -3))
for p in ("gate_proj", "up_proj"):
    class Lin:
        weight, scales, biases, group_size = orig[p][0], orig[p][1], orig[p][2], 64
    sl3 = sm[0, 6:]
    r2 = P125._tier_proj(Lin, xe, sl3, mx.arange(4, dtype=mx.uint32).reshape(1, 1, 4), 2)
    yr, _ = ref_3b(x, sm, mx.array([[0, 1]], mx.int32), "4433332222", b2)
    # compare the gate/up projection of ranks 6-9 through the masked-3-bit reference path
    w = msb[p]["w3"][sl3]
    wm = (w.reshape(*w.shape[:-1], -1, 3) & MASK2W).reshape(w.shape)
    y3 = mx.gather_qmm(xe, wm, msb[p]["s3"][sl3], b2[p][sl3], rhs_indices=mx.arange(4, dtype=mx.uint32).reshape(1, 1, 4),
                       transpose=True, group_size=64, bits=3)
    d = mx.abs(y3.astype(mx.float32) - r2.astype(mx.float32)).max().item()
    rel = d / mx.abs(r2.astype(mx.float32)).max().item()
    ok &= rel < 2e-2
    print(f"3b. {p} 2-bit tier: masked-3-bit gather_qmm vs planes125 _tier_proj(n=2): max |d| {d:.3g} (rel {rel:.2g})", flush=True)

# 4. the fused combine (tier_glu(..., scores=)) vs the block's (y * scores[..., None]).sum(-2) on tier_glu's per-rank y,
# for each candidate MLX reduction order LY (informational: the per-rank path above is the equivalence that must hold)
for Mr in (1, 2, 4, 8):
    x = mx.array((rng.standard_normal((Mr, 2560)) * 0.6).astype(np.float32)).astype(mx.bfloat16)
    sm = mx.array(np.stack([rng.choice(NE, 10, replace=False) for _ in range(Mr)]).astype(np.int32))
    sl = mx.array(np.stack([rng.choice(NE, 2, replace=False) for _ in range(Mr)]).astype(np.int32))
    w = mx.softmax(mx.array(rng.standard_normal((Mr, 10)).astype(np.float32)), axis=-1).astype(mx.bfloat16)
    ref = (tier_glu(x, msb, lsb, sm, sl) * w[..., None]).sum(axis=-2)
    res = {}
    for ly in (1, 2, 4, 8):
        yc = tier_glu(x, msb, lsb, sm, sl, scores=w, ly=ly)
        res[ly] = int((yc.view(mx.uint16) != ref.view(mx.uint16)).sum().item())
    print(f"4. fused combine M={Mr}: mismatches vs the block's combine by LY {res}", flush=True)
print("PASS" if ok else "FAIL", flush=True)
sys.exit(0 if ok else 1)
