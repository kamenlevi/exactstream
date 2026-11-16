"""WEEK98 MODULE 2 (MoonShots, Mon 16 Nov 2026): what binds ooo1cq's kernel A (and B)? Ablations on the captured calls
of exp32's real blocks (m2_common.py's setup), each a copy of Leap's kernel source with ONE part removed (own pipeline
names; diagnosis only, the outputs are not used):
  base     the kernel as built by ooo1cq (copied source; must equal the reference kernel's output bitwise)
  nocode   the 6 code words per lane and group come from registers (no code bytes from memory; decode + MMA kept)
  nox      the activation rows (x for A, h for B) come from registers (no activation loads; the MMA kept)
  nodec    no spread3 / spread8 decode (the raw 3-byte chunks go to threadgroup memory)
  nomma    no MMA (op.run removed; the activation loads die with it)
  stream   the same weight reads (codes, LSB words, the 11-bit scale / bias pieces) XOR-folded: no decode, no
           activations, no threadgroup memory, no MMA: the access pattern's own streaming time
  empty    every threadgroup returns at once (nseg forced to 0)
Per variant: GPU us per call (libmtlprof, one command buffer each, median of --reps), per block, then means.
usage: ~/BigMoE/.venv/bin/python -I -B m2_abl.py [--blocks 8] [--reps 7]   (MEM ~3 GB; indicative unless quiet)"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--blocks", type=int, default=8)
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--out", default=os.path.expanduser("~/MoonShots/week98/m2/m2_abl"))
A_ = ap.parse_args()
os.environ["M2_BLOCKS"] = str(A_.blocks)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m2_common import *                                  # noqa: E402,F403
from m2_common import L_, say                            # noqa: E402
import tierglu_nax6 as N6                                # noqa: E402
import tierglu_nax6n as NN                               # noqa: E402
import p11 as P11                                        # noqa: E402
import ooo1c as OC                                       # noqa: E402

SRC_A = OQ.a_one_call_q(NN._A_NP)
SRC_B = OQ.b_drop(NN._B_NP)
HDR_A = N6._HDR4 + P11.P11_METAL + OC.SYS
HDR_B = N6._HDR4 + P11.P11_METAL


def rep1(s, a, b):
    assert s.count(a) == 1, (a[:70], s.count(a))
    return s.replace(a, b)


LOADS = "const uint w0 = c[0], w1 = c[1], w2 = c[2], w3 = c[3], w4 = c[4], w5 = c[5];"
FAKE = ("const uint w0 = (uint)lane * 2654435761u + (uint)g * 40503u + e, w1 = w0 ^ 0x55555555u, w2 = w0 + 0x1234567u, "
        "w3 = w0 * 3u, w4 = w0 ^ 0xAAAAAAAAu, w5 = w0 + 0x7654321u;")
XLOADS = [("const vec<T, 4> a00 = *(const device vec<T, 4>*)(ar0 + k0 + fn0);",
           "const vec<T, 4> a00 = vec<T, 4>(static_cast<T>(1e-3f * (float)(k0 + fn0)));"),
          ("const vec<T, 4> a10 = *(const device vec<T, 4>*)(ar1 + k0 + fn0);",
           "const vec<T, 4> a10 = vec<T, 4>(static_cast<T>(2e-3f * (float)(k0 + fn0)));"),
          ("const vec<T, 4> a01 = *(const device vec<T, 4>*)(ar0 + k0 + 16 + fn0);",
           "const vec<T, 4> a01 = vec<T, 4>(static_cast<T>(3e-3f * (float)(k0 + fn0)));"),
          ("const vec<T, 4> a11 = *(const device vec<T, 4>*)(ar1 + k0 + 16 + fn0);",
           "const vec<T, 4> a11 = vec<T, 4>(static_cast<T>(4e-3f * (float)(k0 + fn0)));")]
DEC4 = "for (int q = 0; q < 8; ++q) nb[q] = spread8(ch[q], ((q < 4 ? l0 : l1) >> (8 * (q & 3))) & 0xFFu);"
DEC3 = "            } else {\n                for (int q = 0; q < 8; ++q) nb[q] = spread3(ch[q]);"
M1 = "            uint ch[8] = {"
M2 = "        simdgroup_barrier(mem_flags::mem_threadgroup);"
M3 = "\n    threadgroup_barrier(mem_flags::mem_threadgroup);"
GPL = "    for (int gp = kc * GPC; gp < (kc + 1) * GPC; ++gp) {"


def variant(src, kind, out_name):
    if kind == "base":
        return src
    if kind == "nocode":
        return rep1(src, LOADS, FAKE)
    if kind == "nox":
        for a, b in XLOADS:
            src = rep1(src, a, b)
        return src
    if kind == "nodec":
        src = rep1(src, DEC4, "for (int q = 0; q < 8; ++q) nb[q] = ch[q] ^ (q < 4 ? l0 : l1);")
        return rep1(src, DEC3, "            } else {\n                for (int q = 0; q < 8; ++q) nb[q] = ch[q];")
    if kind == "nomma":
        return rep1(src, "op.run(ca, tbt, cg);", "")
    if kind == "stream":
        i1, i2 = src.index(M1), src.index(M2)
        i3 = src.index(M3, i2)                           # the epilogue's first barrier (after the gp loop)
        assert i1 < i2 < i3 and src.count(M1) == 1 and src.count(M2) == 1
        body = ("            uint lx = 0u;\n"
                "            if (tier == 4) lx = lrow[64 * g] ^ lrow[64 * g + 1];\n"
                "            xacc ^= w0 ^ w1 ^ w2 ^ w3 ^ w4 ^ w5 ^ lx ^ as_type<uint>((float)srow[32 * g]) "
                "^ as_type<uint>((float)brow[32 * g]);\n"
                "        }\n    }\n")
        s = src[:i1] + body + f"    if (xacc == 0x9E3779B9u) {out_name}[0] = static_cast<T>(1.0f);\n"
        return rep1(s, GPL, "    uint xacc = 0u;\n" + GPL)
    raise ValueError(kind)


KINDS = ["base", "nocode", "nox", "nodec", "nomma", "stream", "empty"]
_KV = {}


def ka_v(kind):
    if kind not in _KV:
        _KV[kind] = mx.fast.metal_kernel(
            name=f"m2abl_a_{kind}",
            input_names=["x", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                         "msb_slot", "lsb_slot", "pidx", "seg", "nseg", "region", "offs", "status"],
            output_names=["h", "rs_m", "rs_l"], source=variant(SRC_A, "base" if kind == "empty" else kind, "h"),
            header=HDR_A)
    return _KV[kind]


def kb_v(kind):
    key = ("b", kind)
    if key not in _KV:
        _KV[key] = mx.fast.metal_kernel(
            name=f"m2abl_b_{kind}",
            input_names=["h", "hs", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
            output_names=["y"], source=variant(SRC_B, "base" if kind == "empty" else kind, "y"), header=HDR_B)
    return _KV[key]


def run_a(st, kind, grp, xs):
    pidx, seg, nsg_, nseg = grp
    if kind == "empty":
        nsg_ = mx.zeros((1,), mx.int32)
    g, u = st.msb_n["gate_proj"], st.msb_n["up_proj"]
    L = st.lsb_n
    return ka_v(kind)(
        inputs=[st.x, xs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], g["b3"], u["b3"],
                L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], st.ms, st.ls,
                pidx, seg, nsg_, st.region, st.offs, st.status],
        template=st.tmpl + [("K_C", st.Kd), ("N_C", st.N), ("KS_C", st.ks)],
        grid=(st.N // 32 * 64 * st.ks, nseg, 1), threadgroup=(64 * st.ks, 1, 1),
        output_shapes=[(st.M * st.R, st.N), (st.M * st.R,), (max(st.M * st.r4, 1),)],
        output_dtypes=[st.x.dtype, mx.int32, mx.int32])


def run_b(st, kind, grp, h, hs, rs_m, rs_l):
    pidx, seg, nsg_, nseg = grp
    if kind == "empty":
        nsg_ = mx.zeros((1,), mx.int32)
    d = st.msb_n["down_proj"]
    L = st.lsb_n
    (y,) = kb_v(kind)(
        inputs=[h, hs, d["w3"], d["s3"], d["b3"], d["b3"], L["down_proj"]["l1"], L["down_proj"]["b"], rs_m, rs_l,
                pidx, seg, nsg_],
        template=st.tmpl + [("N_C", st.N), ("D_C", st.D), ("NSG_C", st.nsg_b), ("KS_C", st.ks_b)],
        grid=(st.D // (32 * st.nsg_b) * 32 * st.nsg_b * st.ks_b, nseg, 1), threadgroup=(32 * st.nsg_b * st.ks_b, 1, 1),
        output_shapes=[(st.M * st.R, st.D)], output_dtypes=[st.x.dtype])
    return y


for blk in B32[:2]:
    mx.eval(call(blk)); mx.synchronize()
ROWS = []
t_run = time.time()
for bi in range(NB):
    CAP.clear()
    yref = call(B32[bi])
    mx.eval(yref)
    st = Staged(*CAP["oq"], CAP["comb"])
    grp0, _, _, _ = gpu(st.grp)
    xs0, _, _, _ = gpu(st.xs)
    (h0, rsm0, rsl0), _, _, _ = gpu(lambda: st.ka(grp0, xs0))
    hs0, _, _, _ = gpu(lambda: st.hs(h0))
    y0, _, _, _ = gpu(lambda: st.kb(grp0, h0, hs0, rsm0, rsl0))
    ha = run_a(st, "base", grp0, xs0)
    yb = run_b(st, "base", grp0, h0, hs0, rsm0, rsl0)
    mx.eval(ha, yb)
    eqa = bool(mx.array_equal(ha[0], h0).item())        # rows of dropped / unused pairs are unwritten in both: compare
    eqb = bool(mx.array_equal(yb, y0).item())           # only if every pair row is written (km path: no -2 bases)
    sb = seg_bytes(st, grp0)
    R_ = {f"A_{k}": [] for k in KINDS}
    R_.update({f"B_{k}": [] for k in KINDS})
    for rep in range(A_.reps):
        for k in KINDS:
            R_[f"A_{k}"].append(gpu(lambda: run_a(st, k, grp0, xs0))[1])
            R_[f"B_{k}"].append(gpu(lambda: run_b(st, k, grp0, h0, hs0, rsm0, rsl0))[1])
    med = {k: float(np.median(v)) for k, v in R_.items()}
    ROWS.append(dict(block=bi, eq_a=eqa, eq_b=eqb, **sb, **{f"us_{k}": round(v, 2) for k, v in med.items()}))
    say(f"  block {bi:2d}: base == ooo1cq A {eqa}, B {eqb}; {sb['nseg']} segments; A MB {sb['a_mb']:.1f}, B MB "
        f"{sb['b_mb']:.1f} | A: " + ", ".join(f"{k} {med['A_' + k]:.1f}" for k in KINDS)
        + " | B: " + ", ".join(f"{k} {med['B_' + k]:.1f}" for k in KINDS))

col = lambda k: np.array([r_[k] for r_ in ROWS], float)
say(f"\nmeans over {NB} blocks (GPU us per kernel; A {np.mean(col('a_mb')):.1f} MB, B {np.mean(col('b_mb')):.1f} MB of "
    f"segment bytes; {np.mean(col('nseg')):.1f} segments):")
for p_, mb in (("A", np.mean(col("a_mb"))), ("B", np.mean(col("b_mb")))):
    say(f"  {p_}: " + ", ".join(f"{k} {np.mean(col(f'us_{p_}_{k}')):.1f}" for k in KINDS)
        + f" | bytes at 290 GB/s {mb / 290 * 1e3:.1f}; stream rate {mb / np.mean(col(f'us_{p_}_stream')) * 1e3:.1f} GB/s, "
        f"base rate {mb / np.mean(col(f'us_{p_}_base')) * 1e3:.1f} GB/s")
say(f"all base == ooo1cq: A {all(r_['eq_a'] for r_ in ROWS)}, B {all(r_['eq_b'] for r_ in ROWS)}; peak "
    f"{mx.get_peak_memory() / 1e9:.2f} GB; {time.time() - t_run:.0f} s")
os.makedirs(os.path.dirname(A_.out), exist_ok=True)
open(A_.out + ".txt", "a").write("\n".join(L_) + "\n")
json.dump(dict(rows=ROWS), open(A_.out + ".json", "w"), indent=1)
