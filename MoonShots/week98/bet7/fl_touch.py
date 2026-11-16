"""WEEK98 BET 7 (a'): prefetch without a persistent kernel (NewIdeas 14:4x; MoonShots, Mon 16 Nov 2026).
fused2 plus a data-independent TOUCH kernel per step that reads the next step's weights (one 4-byte load per 64 bytes)
while the current step runs, so the next step starts on warm cache lines. Ordering, with no barrier before a touch:
the touch takes the overlapped step's INPUT as an unused anchor input (it is no longer in the encoder's set of fresh
outputs once that step has been dispatched), and mx.depends ties the touch into the next step's input, so the next
step's barrier waits for both. Outputs are untouched: every arm must equal fused2 bitwise.
Modes (same 9 touch sets per level, same bytes):
  pre    each set touched while the step BEFORE its consumer runs (prefetch). Sets: attention HC (during S1G), S4 (during
         the HC read), o_proj (during attention), MLP HC (during o_proj), router + shared expert (during the MLP HC
         read), the 10 routed experts' down weights (during S10, from S9b's ids), mixer HC (during S11), head (during
         the mixer HC read), the next level's S1G (during S15P). The other routed weights can't be known in time.
  late   each set touched while its OWN consumer runs (the control: the same extra reads, no prefetch benefit)
  cap    pre with each touch capped at --cap MB (a prefix of each array: the first wave's rows)
Timing as fl_time.py / fl_overlap.py (T0 400, full head, F = 1, L = 21, async per level, interleaved, 2 rounds
dropped); self-check: every arm's tokens equal stock's in every block. --smoke: one block per mode, bit-checked against
fused2 (hh, top-K ids, log-probs, tokens at every level).
usage: ~/BigMoE/.venv/bin/python -I -B fl_touch.py [--smoke] [--rounds 30] [--cap 4] [--out <json>]   (MEM ~4 GB)"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                          # noqa: E402
import mlx.core as mx                                       # noqa: E402
from mlx.utils import tree_flatten                          # noqa: E402

import fl_attn                                              # noqa: E402
import fl_fused                                             # noqa: E402
import fl_hc                                                # noqa: E402
import fl_level                                             # noqa: E402
import fl_moe                                               # noqa: E402
import fl_stock as FS                                       # noqa: E402

STRIDE = 16                      # words: one 4-byte load per 64-byte line
MAGIC = "0x9E3779B9u"            # the loads feed a store that practically never fires, so they can't be dropped
_TK = {}


def _touch_kernel(m):
    if m not in _TK:
        src = ["uint tid = thread_position_in_grid.x;", "uint nt = threads_per_grid.x;", "uint acc = 0u;"]
        for j in range(m):
            src.append(f"{{ const device uint* p = reinterpret_cast<const device uint*>(a{j}); "
                       f"for (uint i = tid * {STRIDE}; i < N{j}; i += nt * {STRIDE}) acc ^= p[i]; }}")
        src.append(f"if (acc == {MAGIC}) out[0] = acc;")
        _TK[m] = mx.fast.metal_kernel(name=f"fl_touch{m}", input_names=[f"a{j}" for j in range(m)] + ["anchor"],
                                      output_names=["out"], source="\n".join(src))
    return _TK[m]


def _grid(words):
    lines = sum(w // STRIDE + 1 for w in words)
    return min(max(256, (lines + 255) // 256 * 256), 256 * 1024)


def touch(arrs, cap_mb, anchor):
    """read every 64th byte of each array's prefix (all of it, or cap_mb MB in total across the arrays)"""
    words = [int(a.nbytes) // 4 for a in arrs]
    f = 1.0 if cap_mb is None else min(1.0, cap_mb * 1e6 / (4 * sum(words)))
    ns = [min(w, max(STRIDE, int(w * f))) for w in words]
    return _touch_kernel(len(arrs))(inputs=list(arrs) + [anchor], template=[(f"N{j}", n) for j, n in enumerate(ns)],
                                    grid=(_grid(ns), 1, 1), threadgroup=(256, 1, 1), output_shapes=[(1,)],
                                    output_dtypes=[mx.uint32])[0]


def touch_experts(down, inds, cap_mb, anchor):
    """read every 64th byte of the 10 selected experts' down weights, scales and biases (per-expert prefix if capped)"""
    w, s, b = down["weight"], down["scales"], down["biases"]
    E = int(w.shape[0])
    per = [int(a.nbytes) // 4 // E for a in (w, s, b)]
    f = 1.0 if cap_mb is None else min(1.0, cap_mb * 1e6 / (10 * 4 * sum(per)))
    n = [min(p, max(STRIDE, int(p * f))) for p in per]
    name = f"fl_touchx_{n[0]}_{n[1]}_{n[2]}"
    if name not in _TK:
        body = []
        for v, p, nn in zip(("w", "s", "b"), per, n):
            body.append(f"  {{ const device uint* q = reinterpret_cast<const device uint*>({v}) + (ulong)ex * {p}; "
                        f"for (uint i = tid * {STRIDE}; i < {nn}; i += nt * {STRIDE}) acc ^= q[i]; }}")
        src = ("uint tid = thread_position_in_grid.x;\nuint nt = threads_per_grid.x;\nuint acc = 0u;\n"
               "for (int e = 0; e < 10; e++) {\n  uint ex = inds[e];\n" + "\n".join(body) + "\n}\n"
               f"if (acc == {MAGIC}) out[0] = acc;")
        _TK[name] = mx.fast.metal_kernel(name=name, input_names=["inds", "w", "s", "b", "anchor"],
                                         output_names=["out"], source=src)
    return _TK[name](inputs=[inds, w, s, b, anchor], grid=(_grid([10 * x for x in n]), 1, 1),
                     threadgroup=(256, 1, 1), output_shapes=[(1,)], output_dtypes=[mx.uint32])[0]


def dep(x, t):
    return mx.depends([x], [t])[0]


def moe3(blk, x, cap, pre, late_anchor_touch=None):
    """fl_moe.moe with two hooks: pre -> the selected experts' down weights touched during S10 (anchor: S10's input
    inds; tied into S11's input eh); late -> the router + shared-expert set touched during S9 (late_anchor_touch(x),
    tied into S9b's input). Returns (out, inds, eh) so the caller can anchor the next touches."""
    x = x.reshape(-1)
    g, se, sw, seg = blk.gate, blk.shared_expert, blk.switch_mlp, blk.shared_expert_gate
    BF = mx.bfloat16
    s9 = fl_moe._k("fl_moe_s9", ["x", "rw", "rs", "rb", "gw", "gsc", "gb", "uw", "usc", "ub", "segw", "segs", "segb"],
                   ["rl", "gout", "uout", "sh", "segl"], fl_moe._S9)
    rl, _, _, sh, segl = s9(
        inputs=[x, g["weight"], g["scales"], g["biases"], se.gate_proj["weight"], se.gate_proj["scales"],
                se.gate_proj["biases"], se.up_proj["weight"], se.up_proj["scales"], se.up_proj["biases"],
                seg["weight"], seg["scales"], seg["biases"]],
        grid=(145 * 32, 2, 1), threadgroup=(32, 2, 1),
        output_shapes=[(512,), (640,), (640,), (640,), (1,)], output_dtypes=[BF] * 5)
    if late_anchor_touch is not None:
        rl = dep(rl, late_anchor_touch(x))
    s9b = fl_moe._k("fl_moe_s9b", ["rl"], ["inds", "wts", "probs_top"], fl_moe._S9B)
    inds, wts, ptop = s9b(inputs=[rl], grid=(128, 1, 1), threadgroup=(128, 1, 1),
                          output_shapes=[(10,), (10,), (10,)], output_dtypes=[mx.uint32, BF, BF])
    s10 = fl_moe._k("fl_moe_s10", ["x", "inds", "egw", "egs", "egb", "euw", "eus", "eub", "sh", "sdw", "sds", "sdb"],
                    ["gout", "uout", "eh", "sout"], fl_moe._S10)
    _, _, eh, sout = s10(
        inputs=[x, inds, sw.gate_proj["weight"], sw.gate_proj["scales"], sw.gate_proj["biases"],
                sw.up_proj["weight"], sw.up_proj["scales"], sw.up_proj["biases"], sh, se.down_proj["weight"],
                se.down_proj["scales"], se.down_proj["biases"]],
        grid=((800 + 320) * 32, 2, 1), threadgroup=(32, 2, 1),
        output_shapes=[(6400,), (6400,), (6400,), (2560,)], output_dtypes=[BF] * 4)
    if pre:
        eh = dep(eh, touch_experts(sw.down_proj, inds, cap, inds))
    s11 = fl_moe._k("fl_moe_s11", ["inds", "wts", "eh", "edw", "eds", "edb", "segl", "sout"], ["yout", "out"],
                    fl_moe._S11)
    _, out = s11(
        inputs=[inds, wts, eh, sw.down_proj["weight"], sw.down_proj["scales"], sw.down_proj["biases"], segl, sout],
        grid=(320 * 32, 2, 1), threadgroup=(32, 2, 1),
        output_shapes=[(25600,), (2560,)], output_dtypes=[BF, BF])
    return out, inds, eh


def _arrs(*mods):
    """a module's parameters, minus arrays under 64 elements (negligible bytes; MLX would pass them as constant)"""
    out = []
    for m in mods:
        out += [v for _, v in tree_flatten(m.parameters()) if v.size >= 64]
    return out


class Fused3(fl_fused.Fused2):
    """fused2 + touches; mode 'pre' (prefetch), 'late' (control), cap_mb = None (whole sets) or MB per touch"""

    def __init__(self, st, mode="pre", cap_mb=None):
        super().__init__(st)
        self.mode, self.cap = mode, cap_mb
        mtp, L0, attn = st.mtp, self.L0, self.attn
        mlp = L0.mlp
        self.S = {"S1G": [mtp.fc_embedding.weight, mtp.fc_hidden.weight],
                  "attnHC": _arrs(L0.attn_hyper_connection),
                  "S4": _arrs(attn.q_proj, attn.k_proj, attn.v_proj, attn.indexer.index_qk_proj),
                  "S6": _arrs(attn.o_proj),
                  "mlpHC": _arrs(L0.mlp_hyper_connection),
                  "S9": _arrs(mlp.gate, mlp.shared_expert, mlp.shared_expert_gate),
                  "mixHC": _arrs(mtp.hyper_connection_mixer)}
        self.mb = {k: round(sum(int(a.nbytes) for a in v) / 1e6, 2) for k, v in self.S.items()}

    def head_set(self):
        sh = self.st.sh
        return [sh.w, sh.s, sh.b]

    def t(self, name, anchor):
        arrs = self.head_set() if name == "S14" else self.S[name]
        return touch(arrs, self.cap, anchor)

    def level(self, h, tok, sc, F=1, K=8, prep=None):
        assert F == 1, "the prototype covers F = 1"
        st, mtp, L0, attn = self.st, self.st.mtp, self.L0, self.attn
        cache = st.cache[0]
        T = int(cache.offset)
        idx = attn.indexer
        assert (T + 1) // idx.ratio <= idx.block_topk, "beyond the indexer's budget: the stock level's job"
        pre = self.mode == "pre"
        hn, en_in = prep if prep is not None else fl_level.s0_prep(mtp, st.emb, h, tok)
        hpre = fl_level.s1g(mtp, hn, en_in)                                                  # S1
        hpre = dep(hpre, self.t("attnHC" if pre else "S1G", hn))                             # during S1G
        mixed1, inj1, _, _ = fl_hc.hc_read(L0.attn_hyper_connection, hpre)                   # S2, S3
        mixed1 = dep(mixed1, self.t("S4" if pre else "attnHC", hpre))                        # during the HC read
        qf, kf, vf, iqk = fl_level.s4_proj(attn, mixed1)                                     # S4
        if not pre:
            qf = dep(qf, self.t("S4", mixed1))                                               # during S4
        kb, vb = cache.kv.keys, cache.kv.values
        if kb is None or T > kb.shape[2]:
            raise RuntimeError("the cache backing must hold the T rows S5 reads")
        gated, krot = fl_attn.fused_attn(attn, qf, kf, vf, kb[0], vb[0], T)                  # S5
        if pre:
            gated = dep(gated, self.t("S6", qf))                                             # during attention
        bo1 = fl_level.s6_oproj(attn, gated)                                                 # S6
        bo1 = dep(bo1, self.t("mlpHC" if pre else "S6", gated))                              # during o_proj
        mixed2, inj2, hw2, _ = fl_hc.hc_read(L0.mlp_hyper_connection, hpre, pending=(bo1, inj1))   # S7, S8
        mixed2 = dep(mixed2, self.t("S9", bo1) if pre else self.t("mlpHC", bo1))             # during the HC read
        late9 = None if pre else (lambda a: self.t("S9", a))
        bo2, inds, eh = moe3(L0.mlp, mixed2, self.cap, pre, late9)                           # S9-S11
        if pre:
            bo2 = dep(bo2, self.t("mixHC", eh))                                              # during S11
        else:
            bo2 = dep(bo2, touch_experts(L0.mlp.switch_mlp.down_proj, inds, self.cap, eh))   # during S11
        x, _, hh, _ = fl_hc.hc_read(mtp.hyper_connection_mixer, hw2, pending=(bo2, inj2))    # S12, S13
        x = dep(x, self.t("S14" if pre else "mixHC", bo2))                                   # during the HC read
        lg = fl_level.s14_head(st.sh, x)                                                     # S14
        if not pre:
            lg = dep(lg, self.t("S14", x))                                                   # during S14
        kidx, klp, cand, sel, nsc, ntok, hn2, en2 = fl_level.s15p_tail(lg, sc, st.sh.ids, hh, mtp, st.emb, K)  # S15
        tnext = self.t("S1G", lg) if pre else None                                           # during S15P
        cache.indexer_budget = idx.block_topk * idx.ratio
        _, kidxr = mx.split(iqk.reshape(1, 1, 640), [idx.n_heads * idx.head_dim], axis=-1)
        cache.write_raw(kidxr.reshape(1, 1, idx.head_dim))
        idx._extend_pooled(cache, T + 1)
        k4 = krot.reshape(1, 2, 1, 256)
        v4 = vf.reshape(1, 1, 2, 256).transpose(0, 2, 1, 3)
        from mtplx.models.qwen4_exp import _after_index_block_writes
        k4 = _after_index_block_writes(k4, cache)
        cache.kv.update_and_fetch(k4, v4)
        hh3 = hh.reshape(1, 1, -1)
        return dict(h=hh3, tok=ntok.reshape(1, 1), sc=nsc, hh=hh3, kidx=kidx.reshape(1, K), klp=klp.reshape(1, K),
                    cand=cand, sel=sel, par=mx.zeros((1,), mx.uint32), prep=(hn2, en2), touch=tnext)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--L", type=int, default=21)
    ap.add_argument("--T0", type=int, default=400)
    ap.add_argument("--cap", type=float, default=4.0)
    ap.add_argument("--out", default=os.path.expanduser("~/MoonShots/week98/bet7/fl_touch.json"))
    A = ap.parse_args()
    BW, MB = 292.7, 177.6
    t00 = time.time()
    st = FS.Stock(nh=32768, T0=A.T0)
    K = 8
    fu2 = fl_fused.Fused2(st)
    impl = {"fused2": fu2, "pre": Fused3(st, "pre"), "cap": Fused3(st, "pre", A.cap), "late": Fused3(st, "late")}
    print(f"built at {time.time() - t00:.0f} s; touch sets (MB): {impl['pre'].mb}, head "
          f"{sum(int(a.nbytes) for a in impl['pre'].head_set()) / 1e6:.2f}", flush=True)

    def run(arm, tok0, keep_all=False):
        h, tok, sc = st.h0, mx.array([[tok0]], dtype=mx.int32), mx.zeros((1,), mx.float32)
        prep = None
        toks, outs = [], []
        t0 = time.perf_counter()
        for _ in range(A.L):
            if arm == "stock":
                o = st.level(h, tok, sc, 1, K)
                h, tok, sc = o["h"], o["tok"], o["sc"]
                mx.async_eval(h, tok, sc, o["kidx"], o["klp"])
            else:
                o = impl[arm].level(h, tok, sc, 1, K, prep=prep)
                h, tok, sc, prep = o["h"], o["tok"], o["sc"], o["prep"]
                extra = [o["touch"]] if o.get("touch") is not None else []
                mx.async_eval(h, tok, sc, o["kidx"], o["klp"], prep[0], prep[1], *extra)
            toks.append(tok)
            if keep_all:
                outs.append((o["hh"], o["kidx"], o["klp"], o["tok"]))
        mx.eval(toks, h, sc)
        t1 = time.perf_counter()
        st.cache[0].trim(A.L)
        if keep_all:
            mx.eval(outs)
        return (t1 - t0) / A.L, [int(t.reshape(-1)[0].item()) for t in toks], outs

    if A.smoke:
        _, _, ref = run("fused2", 1000, True)
        ok = {}
        for arm in ("pre", "cap", "late"):
            _, _, o = run(arm, 1000, True)
            ok[arm] = all(bool(mx.array_equal(a, b).item()) for x, y in zip(o, ref) for a, b in zip(x, y))
        print(f"smoke: {A.L} levels, hh / top-K ids / log-probs / tokens bit-equal to fused2: {ok}", flush=True)
        return
    ARMS = ["stock", "fused2", "pre", "cap", "late"]
    W = {a: [] for a in ARMS}
    same = True
    for r in range(A.rounds + 2):
        tk = {}
        for a in (ARMS if r % 2 == 0 else ARMS[::-1]):
            w, tk[a], _ = run(a, 1000 + r)
            if r >= 2:
                W[a].append(1e3 * w)
        same &= all(tk[a] == tk["stock"] for a in ARMS)
    res = {"rounds": A.rounds, "L": A.L, "T0": A.T0, "cap_mb": A.cap, "tokens_equal": same, "sets_mb": impl["pre"].mb,
           "arms": {}}
    print(f"tokens equal stock's in every block, every arm: {same}")
    print("ms per level, median [p10-p90] | f = ms - 177.6 MB / 292.7")
    for a in ARMS:
        x = np.array(W[a])
        v = res["arms"][a] = dict(ms=round(float(np.median(x)), 4), p10=round(float(np.percentile(x, 10)), 4),
                                  p90=round(float(np.percentile(x, 90)), 4), f=round(float(np.median(x)) - MB / BW, 4))
        print(f"  {a:7s} {v['ms']:.4f} [{v['p10']:.4f}-{v['p90']:.4f}] | f {v['f']:.4f}")
    m = {a: res["arms"][a]["ms"] for a in ARMS}
    best = min(m["pre"], m["cap"])
    verdict = "PASS" if best <= 0.76 else ("KILL" if best >= 0.85 else "BETWEEN")
    res["verdict"] = verdict
    print(f"best prefetch arm {best:.4f} ms (PASS <= 0.76, KILL >= 0.85): {verdict}; pre - late "
          f"{m['pre'] - m['late']:+.4f} ms, late - fused2 {m['late'] - m['fused2']:+.4f} ms")
    json.dump(res, open(A.out, "w"), indent=1)
    print(f"wrote {A.out} ({time.time() - t00:.0f} s)")


if __name__ == "__main__":
    main()
