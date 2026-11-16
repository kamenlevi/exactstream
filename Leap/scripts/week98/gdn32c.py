"""gdn32c.py (Leap, Mon 16 Nov 2026 13:0x; CHECK25): the gated delta rule over a 17-32-row verify window in CHUNKED form, as
one dispatch per layer (one threadgroup per v-head), in place of mlx_lm's sequential step kernel (one thread row per dv,
the T steps in series: ~117 us per layer at T 32, ~4.2 ms per 32-row check). FAST: same recurrence, sums reassociated.
The recurrence (mlx_lm gated_delta_step; S [Dv, Dk] per head, fp32):
  S'_t = g_t S_{t-1};  delta_t = beta_t (v_t - S'_t k_t);  S_t = S'_t + delta_t k_t^T;  y_t = S_t q_t
Chunked (C = T rows, Lambda_t = sum_{i <= t} log g_i, Gam_ts = exp(Lambda_t - Lambda_s); all fp32, log space, so no decay
product underflows):
  Delta = (I + diag(beta) Lstrict)^-1 diag(beta) (V - diag(gamma) K S_0^T),   Lstrict_ts = Gam_ts (k_t . k_s), s < t
  Y     = diag(gamma) Q S_0^T + Mlow Delta,                                   Mlow_ts = Gam_ts (q_t . k_s), s <= t
  S_C   = gamma_C S_0 + Delta^T diag(gamma_C / gamma) K
Threads: 4 per dv row (each a 32-wide quarter of Dk: its S_0 quarter in registers), 512 per head; K, Q (bf16, 16 KB) and
the two C x C matrices (fp32, 8 KB) in threadgroup memory; the triangular solve per dv column in registers (each of the 4 threads redundantly).
  chunk(q, k, v, g, beta, state) -> (y [B, T, Hv, Dv], state' [B, Hv, Dv, Dk])  (gated_delta_kernel's signature, no mask)
  check()  vs mlx_lm's kernel on random inputs at T 17 / 24 / 32 (relative error of y and of the state)
usage: [BigMoE venv] gdn32c.py check"""
from functools import lru_cache
import os
import sys

import mlx.core as mx

_SRC = r"""
    constexpr int C = T_C, DK = Dk, DV = Dv, QW = DK / 4;
    const uint n = threadgroup_position_in_grid.x;               // b * Hv + hv
    const uint b_idx = n / Hv, hv_idx = n % Hv, hk_idx = hv_idx / (Hv / Hk);
    const uint tid = thread_index_in_threadgroup;               // 512 threads: dv = tid / 4, quarter = tid % 4
    const uint dv = tid >> 2, qt = tid & 3;
    threadgroup InT Ks[C][DK];                                   // bf16 (as given): 2 x 8 KB at C 32
    threadgroup InT Qs[C][DK];
    threadgroup float Lm[C][C];                                  // strictly lower: beta_t Gam_ts (k_t . k_s)
    threadgroup float Mm[C][C];                                  // lower incl. diagonal: Gam_ts (q_t . k_s)
    threadgroup float lam[C];
    threadgroup float bet[C];
    // load K, Q (this k-head), the log decays and betas
    for (uint o = tid; o < (uint)(C * DK); o += 512) {
        const uint t = o / DK, c = o % DK;
        const size_t qk = ((size_t)(b_idx * T_C + t) * Hk + hk_idx) * DK + c;
        Ks[t][c] = k[qk];
        Qs[t][c] = q[qk];
    }
    if (tid == 0) {
        float acc = 0.0f;
        for (int t = 0; t < C; ++t) {
            const float gv = static_cast<float>(g[(b_idx * T_C + t) * Hv + hv_idx]);
            acc += gv > 0.0f ? log(gv) : -1e30f;
            lam[t] = acc;
            bet[t] = static_cast<float>(beta[(b_idx * T_C + t) * Hv + hv_idx]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // the C x C matrices: entry (t, s), 2 per thread
    for (uint o = tid; o < (uint)(C * C); o += 512) {
        const int t = o / C, s = o % C;
        float kk = 0.0f, qk = 0.0f;
        if (s <= t) {
            for (int c = 0; c < DK; ++c) {
                const float ks = static_cast<float>(Ks[s][c]);
                kk += static_cast<float>(Ks[t][c]) * ks;
                qk += static_cast<float>(Qs[t][c]) * ks;
            }
        }
        const float gm = s <= t ? exp(lam[t] - lam[s]) : 0.0f;
        Lm[t][s] = s < t ? bet[t] * gm * kk : 0.0f;
        Mm[t][s] = s <= t ? gm * qk : 0.0f;
    }
    // this thread's quarter of S_0's row dv
    const device StT* s_in = state_in + ((size_t)n * DV + dv) * DK + qt * QW;
    float s0[QW];
    for (int i = 0; i < QW; ++i) s0[i] = static_cast<float>(s_in[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // R_t = beta_t (v_t - gamma_t (S_0 k_t)[dv]); the 4 quarters' partial dots summed across the 4 lanes
    float dl[C];
    for (int t = 0; t < C; ++t) {
        float p = 0.0f;
        for (int i = 0; i < QW; ++i) p += s0[i] * static_cast<float>(Ks[t][qt * QW + i]);
        p += simd_shuffle_xor(p, 1);
        p += simd_shuffle_xor(p, 2);
        const float vv = static_cast<float>(v[((size_t)(b_idx * T_C + t) * Hv + hv_idx) * DV + dv]);
        dl[t] = bet[t] * (vv - exp(lam[t]) * p);
    }
    // Delta = (I + Lm)^-1 R (forward substitution; the same on the 4 lanes of a dv row)
    for (int t = 1; t < C; ++t) {
        float a = dl[t];
        for (int s = 0; s < t; ++s) a -= Lm[t][s] * dl[s];
        dl[t] = a;
    }
    // Y_t = gamma_t (S_0 q_t)[dv] + sum_{s <= t} Mm_ts Delta_s
    for (int t = 0; t < C; ++t) {
        float p = 0.0f;
        for (int i = 0; i < QW; ++i) p += s0[i] * static_cast<float>(Qs[t][qt * QW + i]);
        p += simd_shuffle_xor(p, 1);
        p += simd_shuffle_xor(p, 2);
        float yv = exp(lam[t]) * p;
        for (int s = 0; s <= t; ++s) yv += Mm[t][s] * dl[s];
        if (qt == 0) y[((size_t)(b_idx * T_C + t) * Hv + hv_idx) * DV + dv] = static_cast<InT>(yv);
    }
    // S_C = gamma_C S_0 + sum_t Delta_t (gamma_C / gamma_t) k_t (this quarter)
    const float gC = exp(lam[C - 1]);
    float w[C];
    for (int t = 0; t < C; ++t) w[t] = dl[t] * exp(lam[C - 1] - lam[t]);
    device StT* s_out = state_out + ((size_t)n * DV + dv) * DK + qt * QW;
    for (int i = 0; i < QW; ++i) {
        float a = gC * s0[i];
        for (int t = 0; t < C; ++t) a += w[t] * static_cast<float>(Ks[t][qt * QW + i]);
        s_out[i] = static_cast<StT>(a);
    }
"""


@lru_cache(None)
def _k():
    return mx.fast.metal_kernel(name="leap_gdn32c", input_names=["q", "k", "v", "g", "beta", "state_in"],
                                output_names=["y", "state_out"], source=_SRC)


def chunk(q, k, v, g, beta, state):
    """mlx_lm gated_delta_kernel's signature (scalar gating g [B, T, Hv], no mask), T <= 32, Dk = Dv = 128"""
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    assert 1 <= T <= 32 and Dk == 128 and Dv == 128 and g.ndim == 3 and Hv % Hk == 0, (k.shape, v.shape, g.shape)
    tmem = 2 * 2 * T * Dk + 4 * (2 * T * T + 2 * T)
    assert tmem <= 32768, tmem
    return _k()(inputs=[q, k, v, g, beta, state],
                template=[("InT", q.dtype), ("StT", state.dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv), ("T_C", T)],
                grid=(512 * B * Hv, 1, 1), threadgroup=(512, 1, 1),
                output_shapes=[(B, T, Hv, Dv), state.shape], output_dtypes=[q.dtype, state.dtype])


def check(seeds=3):
    """vs mlx_lm's sequential kernel: max |dy| / max |y| and max |dS| / max |S| at T 17 / 24 / 32, typical decays and a
    strong-decay head mix (g down to ~0.02, where gamma underflows fp32 within 32 steps)"""
    import numpy as np
    from mlx_lm.models import gated_delta as GD
    rng = np.random.default_rng(1134)
    lines, worst = [], 0.0
    for sd in range(seeds):
        for strong in (False, True):
            for T in (17, 24, 32):
                B, Hk, Hv, D = 1, 16, 48, 128
                f = lambda *s_: mx.array(rng.standard_normal(s_).astype(np.float32))
                l2 = lambda x: x * mx.rsqrt((x * x).sum(-1, keepdims=True) + 1e-6)
                q = (l2(f(B, T, Hk, D)) * (D ** -0.5)).astype(mx.bfloat16)
                k = l2(f(B, T, Hk, D)).astype(mx.bfloat16)
                v = (f(B, T, Hv, D) * 0.5).astype(mx.bfloat16)
                lo = 0.02 if strong else 0.6
                g = mx.array(rng.uniform(lo, 0.999, (B, T, Hv)).astype(np.float32))
                beta = mx.sigmoid(f(B, T, Hv)).astype(mx.bfloat16)
                st = f(B, Hv, D, D) * 0.1
                y0, s0 = GD.gated_delta_kernel(q, k, v, g, beta, st)
                y1, s1 = chunk(q, k, v, g, beta, st)
                mx.eval(y0, s0, y1, s1)
                ey = float(mx.max(mx.abs(y1.astype(mx.float32) - y0.astype(mx.float32))) / mx.max(mx.abs(y0.astype(mx.float32))))
                es = float(mx.max(mx.abs(s1 - s0)) / mx.max(mx.abs(s0)))
                fin = bool(mx.all(mx.isfinite(y1)).item() and mx.all(mx.isfinite(s1)).item())
                worst = max(worst, ey, es) if fin else float("inf")
                lines.append(f"  seed {sd} {'strong' if strong else 'typical'} decay T {T}: y rel err {ey:.2e}, state rel err "
                             f"{es:.2e}, finite {fin}")
                print(lines[-1], flush=True)
    hdr = f"GDN32C check vs mlx_lm's sequential kernel: worst relative error {worst:.2e} (bf16 y output rounding ~4e-3)"
    print(hdr, flush=True)
    return worst, [hdr] + lines


def bench(rounds=7, nl=36, T=32):
    """ms per check for its 36 GDN recurrences at T rows (independent calls, distinct states): mlx_lm's kernel vs chunk"""
    import time
    import numpy as np
    from mlx_lm.models import gated_delta as GD
    B, Hk, Hv, D = 1, 16, 48, 128
    kk = mx.random.split(mx.random.key(3), 6)
    l2 = lambda x: x * mx.rsqrt((x * x).sum(-1, keepdims=True) + 1e-6)
    q = (l2(mx.random.normal((B, T, Hk, D), key=kk[0])) * D ** -0.5).astype(mx.bfloat16)
    k = l2(mx.random.normal((B, T, Hk, D), key=kk[1])).astype(mx.bfloat16)
    v = mx.random.normal((B, T, Hv, D), key=kk[2]).astype(mx.bfloat16)
    g = mx.random.uniform(0.6, 0.999, (B, T, Hv), key=kk[3])
    beta = mx.sigmoid(mx.random.normal((B, T, Hv), key=kk[4])).astype(mx.bfloat16)
    sts = [mx.random.normal((B, Hv, D, D), key=mx.random.key(100 + i)) * 0.1 for i in range(nl)]
    mx.eval(q, k, v, g, beta, sts)
    forms = {"mlx_lm step": lambda st: GD.gated_delta_kernel(q, k, v, g, beta, st), "chunk": lambda st: chunk(q, k, v, g, beta, st)}
    T_ = {f: [] for f in forms}
    for r in range(rounds):
        for f in (list(forms)[r % 2:] + list(forms)[:r % 2]):
            mx.synchronize()
            t0 = time.perf_counter()
            mx.eval([forms[f](st) for st in sts])
            mx.synchronize()
            T_[f].append(time.perf_counter() - t0)
    out = {f: 1e3 * float(np.median(v[1:])) for f, v in T_.items()}
    line = f"GDN32C bench (T {T}, {nl} layers per check, independent calls): " + ", ".join(
        f"{f} {v:.3f} ms ({1e3 * v / nl:.1f} us/layer)" for f, v in out.items())
    print(line, flush=True)
    return line


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        line = bench()
        open(os.path.expanduser("~/Leap/data/week98/check25/gdn32c_bench.txt"), "a").write(line + "  [" +
             (sys.argv[2] if len(sys.argv) > 2 else "") + "]\n")
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        w, lines = check()
        open(os.path.expanduser("~/Leap/data/week98/check25/gdn32c_check.txt"), "w").write("\n".join(lines) + "\n")
        sys.exit(0 if w < 2e-2 else 1)
