"""tierglu4 (records4 planes) vs MLX references per tier, bitwise, on 64 real experts of layer 0 (records4 from
scratch/records4/L00.bin, records3 from BigMoE's records3/L00.bin): tier 4 == gather_qmm(bits=4) on the original 4-bit
expert; tier 3 == gather_qmm(bits=3) on records3's MSB part (v3); tier 2 == gather_qmm(bits=2) on the P2 plane; SwiGLU
and down per tier. Plans 4433333333 (v3), 4433332222, 4422222222, 4222222222; M = 1, 2, 4, 8.
usage: test_tierglu4.py   (GPU, seconds)"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mlx.core as mx
from mlx_lm.models.switch_layers import SwiGLU
import make_records3 as M
import make_records4 as M4
from tierglu4 import tier_glu4, tier_glu4_phase
from tierglu import combine_phase

NE = 64
PROJ = ("gate_proj", "up_proj", "down_proj")
lay3, lay4 = M.layout(), M4.layout4()
R3, R4 = lay3["recsz"], lay4["recsz"]
f3 = M.open_nocache(os.path.expanduser("~/BigMoE/models/bare/records3/L00.bin"), os.O_RDONLY)
f4 = M.open_nocache(os.path.expanduser("~/MoonShots/scratch/records4/L00.bin"), os.O_RDONLY)
b3, b4 = M.aligned(NE * R3), M.aligned(NE * R4)
M.pread_into(f3, b3, 0); M.pread_into(f4, b4, 0)
arr = lambda buf, R, off, size, dt, shape: mx.array(np.ascontiguousarray(M.view(buf, NE, R, off, size, dt)).reshape(NE, *shape))
p2, s3p, s4p, ref3, ref4 = {}, {}, {}, {}, {}
for i, (p, (out, K)) in enumerate(zip(PROJ, M.SHAPES)):
    po, ps = lay4["piece_off"], lay4["piece_size"]
    i2, i3, i4 = 3 * i, 9 + 2 * i, 15 + 2 * i
    p2[p] = {"w2": arr(b4, R4, po[i2], ps[i2], np.uint32, (out, K // 16)),
             "s2": arr(b4, R4, po[i2 + 1], ps[i2 + 1], np.uint16, (out, K // 64)).view(mx.bfloat16),
             "b2": arr(b4, R4, po[i2 + 2], ps[i2 + 2], np.uint16, (out, K // 64)).view(mx.bfloat16)}
    s3p[p] = {"c": arr(b4, R4, po[i3], ps[i3], np.uint32, (out, K // 32)),
              "b3": arr(b4, R4, po[i3 + 1], ps[i3 + 1], np.uint16, (out, K // 64)).view(mx.bfloat16)}
    s4p[p] = {"l": arr(b4, R4, po[i4], ps[i4], np.uint32, (out, K // 32)),
              "b": arr(b4, R4, po[i4 + 1], ps[i4 + 1], np.uint16, (out, K // 64)).view(mx.bfloat16)}
    mo, ms, lo, ls = lay3["msb_piece_off"], lay3["msb_piece_size"], lay3["lsb_piece_off"], lay3["lsb_piece_size"]
    ref3[p] = (arr(b3, R3, mo[3 * i], ms[3 * i], np.uint32, (out, K * 3 // 32)),
               arr(b3, R3, mo[3 * i + 1], ms[3 * i + 1], np.uint16, (out, K // 64)).view(mx.bfloat16),
               arr(b3, R3, mo[3 * i + 2], ms[3 * i + 2], np.uint16, (out, K // 64)).view(mx.bfloat16))
    q = M.merge_codes(M.view(b3, NE, R3, mo[3 * i], ms[3 * i], np.uint8, inner=3), M.view(b3, NE, R3, lo[2 * i], ls[2 * i], np.uint8))
    s = M.f32_to_bf16(M.bf16_to_f32(M.view(b3, NE, R3, mo[3 * i + 1], ms[3 * i + 1], np.uint16)) * np.float32(0.5))
    ref4[p] = (mx.array(q.reshape(NE, out, K // 8)), mx.array(s.reshape(NE, out, K // 64)).view(mx.bfloat16),
               arr(b3, R3, lo[2 * i + 1], ls[2 * i + 1], np.uint16, (out, K // 64)).view(mx.bfloat16))
mx.eval(p2, s3p, s4p, ref3, ref4)
act = SwiGLU()


def reference(x, sm, plan):
    xe = mx.expand_dims(x, (-2, -3))
    outs = []
    for r, c in enumerate(plan):
        ids = sm[:, r:r + 1]
        idx = mx.arange(ids.size, dtype=mx.uint32).reshape(ids.shape)
        if c == "4":
            pj = lambda h, p: mx.gather_qmm(h, *(a[ids] for a in ref4[p]), rhs_indices=idx, transpose=True, group_size=64, bits=4)
        elif c == "3":
            pj = lambda h, p: mx.gather_qmm(h, *(a[ids] for a in ref3[p]), rhs_indices=idx, transpose=True, group_size=64, bits=3)
        else:
            pj = lambda h, p: mx.gather_qmm(h, p2[p]["w2"][ids], p2[p]["s2"][ids], p2[p]["b2"][ids], rhs_indices=idx,
                                            transpose=True, group_size=64, bits=2)
        h = act(pj(xe, "up_proj"), pj(xe, "gate_proj"))
        outs.append(pj(h, "down_proj").squeeze(-2))
    return mx.concatenate(outs, axis=-2)


rng = np.random.default_rng(5)
ok = True
for plan in ("4433333333", "4433332222", "4422222222", "4222222222"):
    for Mr in (1, 2, 4, 8):
        x = mx.array((rng.standard_normal((Mr, 2560)) * 0.6).astype(np.float32)).astype(mx.bfloat16)
        sm = mx.array(np.stack([rng.choice(NE, 10, replace=False) for _ in range(Mr)]).astype(np.int32))
        t = np.array([int(c) for c in plan])
        cs = mx.where(mx.array(t >= 3)[None, :], sm, -1)
        lsl = mx.where(mx.array(t == 4)[None, :], sm, -1)
        y = tier_glu4(x, p2, s3p, s4p, sm, cs, lsl, plan=plan)
        yr = reference(x, sm, plan)
        per = [mx.array_equal(y[:, k].view(mx.uint16), yr[:, k].view(mx.uint16)).item() for k in range(10)]
        d = mx.abs(y.astype(mx.float32) - yr.astype(mx.float32)).max().item()
        sc = mx.softmax(mx.array(rng.standard_normal((Mr, 10)).astype(np.float32)), axis=-1).astype(mx.bfloat16)
        yc = tier_glu4(x, p2, s3p, s4p, sm, cs, lsl, plan=plan, scores=sc)
        comb = mx.array_equal(yc.view(mx.uint16), (y * sc[..., None]).sum(axis=-2).view(mx.uint16)).item()
        ph = mx.array(rng.integers(1, 3, (Mr, 10)).astype(np.int32))
        y1 = tier_glu4_phase(x, p2, s3p, s4p, sm, cs, lsl, ph, 1, plan=plan)
        y2 = tier_glu4_phase(x, p2, s3p, s4p, sm, cs, lsl, ph, 2, plan=plan)
        oo = mx.array_equal(combine_phase(y1, y2, ph, sc).view(mx.uint16), yc.view(mx.uint16)).item()
        ok &= all(per) and comb and oo
        print(f"plan {plan} M={Mr}: per rank bitwise {''.join('1' if e else '0' for e in per)} (max |d| {d:.3g}); "
              f"fused combine == MLX (y * scores).sum(-2) bitwise: {comb}; out-of-order phases == fused: {oo}", flush=True)
print("PASS" if ok else "FAIL", flush=True)
sys.exit(0 if ok else 1)
