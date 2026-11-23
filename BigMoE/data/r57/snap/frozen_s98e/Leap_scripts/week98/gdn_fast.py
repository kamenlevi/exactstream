"""gdn_fast.py (Leap, Mon 16 Nov 2026 14:0x; CHECK25: the GDN core, ~3.9 ms of a 32-row check in mlx_lm's sequential
gated delta kernel, 36 layers x ~107 us): the same recurrence, bit for bit, with a different launch shape.
mlx_lm's gated_delta_kernel (gated_delta_step: scalar g per head, no mask) runs one simdgroup per (head, value column), each
lane holding Dk / 32 state entries: 32 x Dv x B*Hv threads = 196,608 at Dv 128, Hv 48 (~5 waves of the GPU), and every
serial step waits on its own device loads. Here:
  (1) C value columns per simdgroup (C x Dk / 32 state entries per lane): threads / C, the q / k loads of a step shared;
  (2) the inputs of U steps are loaded ahead of the serial chain, in their own types (InT q / k / v, g's and beta's own).
Per column the expressions are mlx_lm's, in its order and with its operand types: per lane the same Dk / 32 entries
(s_idx = NP * lane + i), state * g then kv_mem += state * k, simd_sum, delta = (v - kv_mem) * beta, state + k * delta,
out += state * q, simd_sum, y = InT(out); the state written back as StT.
  fast(q, k, v, g, beta, state, mask=None, C=4, U=4) -> (y, state_out), as mlx_lm's gated_delta_kernel; it falls back to
    that kernel for a mask, a vectorized g (ndim 4), or shapes it does not cover (Dk % 32, Dv % (C * S))
  check()  bitwise vs mlx_lm's kernel: T 1..32, C 1/2/4/8, U 1/2/4/8, real-scale inputs (l2-normed q / k, g in (0, 1))
  bench()  us per call, alternating with mlx_lm's kernel (ABAB; rough unless the Mac is quiet)
usage: [BigMoE venv] gdn_fast.py [check|bench]"""
import os
import sys
import time

import numpy as np
import mlx.core as mx

_K = {}
MODE = ["chain"]


def _src():
    return """
    uint n = thread_position_in_grid.z;
    uint b_idx = n / Hv;
    uint hv_idx = n % Hv;
    uint hk_idx = hv_idx / (Hv / Hk);
    constexpr int NP = Dk / 32;
    auto q_ = q + b_idx * T_C * Hk * Dk + hk_idx * Dk;
    auto k_ = k + b_idx * T_C * Hk * Dk + hk_idx * Dk;
    auto v_ = v + b_idx * T_C * Hv * Dv + hv_idx * Dv;
    auto g_ = g + b_idx * T_C * Hv;
    auto beta_ = beta + b_idx * T_C * Hv;
    auto y_ = y + b_idx * T_C * Hv * Dv + hv_idx * Dv;
    uint dk_idx = thread_position_in_threadgroup.x;
    uint col0 = thread_position_in_grid.y * CC;
    float st[CC][NP];
    for (int c = 0; c < CC; ++c) {
      auto i_state = state_in + (n * Dv + col0 + c) * Dk;
      for (int i = 0; i < NP; ++i) {
        auto s_idx = NP * dk_idx + i;
        st[c][i] = static_cast<float>(i_state[s_idx]);
      }
    }
    for (int t0 = 0; t0 < T_C; t0 += U) {
      InT kk[U][NP];
      InT qq[U][NP];
      InT vv[U][CC];
      GT gg[U];
      BT bb[U];
      for (int u = 0; u < U; ++u) {
        int t = t0 + u;
        if (t < T_C) {
          for (int i = 0; i < NP; ++i) {
            auto s_idx = NP * dk_idx + i;
            kk[u][i] = k_[t * Hk * Dk + s_idx];
            qq[u][i] = q_[t * Hk * Dk + s_idx];
          }
          for (int c = 0; c < CC; ++c) {
            vv[u][c] = v_[t * Hv * Dv + col0 + c];
          }
          gg[u] = g_[t * Hv + hv_idx];
          bb[u] = beta_[t * Hv + hv_idx];
        }
      }
      for (int u = 0; u < U; ++u) {
        int t = t0 + u;
        if (t < T_C) {
          for (int c = 0; c < CC; ++c) {
            float kv_mem = 0.0f;
            for (int i = 0; i < NP; ++i) {
              st[c][i] = st[c][i] * gg[u];
              kv_mem += st[c][i] * kk[u][i];
            }
            kv_mem = simd_sum(kv_mem);
            auto delta = (vv[u][c] - kv_mem) * bb[u];
            float out = 0.0f;
            for (int i = 0; i < NP; ++i) {
              st[c][i] = st[c][i] + kk[u][i] * delta;
              out += st[c][i] * qq[u][i];
            }
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {
              y_[t * Hv * Dv + col0 + c] = static_cast<InT>(out);
            }
          }
        }
      }
    }
    for (int c = 0; c < CC; ++c) {
      auto o_state = state_out + (n * Dv + col0 + c) * Dk;
      for (int i = 0; i < NP; ++i) {
        auto s_idx = NP * dk_idx + i;
        o_state[s_idx] = static_cast<StT>(st[c][i]);
      }
    }
"""


def _kernel():
    if "k" not in _K:
        _K["k"] = mx.fast.metal_kernel(name="gdn_fast_step", input_names=["q", "k", "v", "g", "beta", "state_in"],
                                       output_names=["y", "state_out"], source=_src())
    return _K["k"]


def _ref():
    import mlx_lm.models.gated_delta as GD
    return GD


def fast(q, k, v, g, beta, state, mask=None, C=4, U=4, S=4):
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if mask is not None or g.ndim == 4 or Dk % 32 or Dv % (C * S) or Hv % Hk:
        return _ref().gated_delta_kernel(q, k, v, g, beta, state, mask)
    return _kernel()(
        inputs=[q, k, v, g, beta, state],
        template=[("InT", q.dtype), ("GT", g.dtype), ("BT", beta.dtype), ("StT", state.dtype), ("Dk", Dk), ("Dv", Dv),
                  ("Hk", Hk), ("Hv", Hv), ("T_C", T), ("CC", C), ("U", U)],
        grid=(32, Dv // C, B * Hv), threadgroup=(32, S, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape], output_dtypes=[q.dtype, state.dtype])


def inputs(T, seed, B=1, Hk=16, Hv=48, Dk=128, Dv=128, dt=mx.bfloat16):
    """real-scale inputs: q / k l2-normed per head (q scaled by Dk^-0.5, as the prework does), v ~ N(0, 1), g = the decay
    in (0, 1) (fp32, as compute_g returns), beta = sigmoid (InT), a state from 64 earlier steps' scale"""
    mx.random.seed(seed)
    q = mx.random.normal((B, T, Hk, Dk))
    k = mx.random.normal((B, T, Hk, Dk))
    q = (q / mx.linalg.norm(q, axis=-1, keepdims=True) * Dk ** -0.5).astype(dt)
    k = (k / mx.linalg.norm(k, axis=-1, keepdims=True)).astype(dt)
    v = mx.random.normal((B, T, Hv, Dv)).astype(dt)
    a = mx.random.normal((B, T, Hv)) * 2.0
    g = mx.exp(-mx.exp(mx.random.normal((Hv,)) * 0.5) * mx.log1p(mx.exp(a))).astype(mx.float32)
    beta = mx.sigmoid(mx.random.normal((B, T, Hv))).astype(dt)
    state = (mx.random.normal((B, Hv, Dv, Dk)) * 0.3).astype(mx.float32)
    mx.eval(q, k, v, g, beta, state)
    return q, k, v, g, beta, state


def _same(a, b):
    a8 = a.view(mx.uint16) if a.dtype in (mx.bfloat16, mx.float16) else a.view(mx.uint32)
    b8 = b.view(mx.uint16) if b.dtype in (mx.bfloat16, mx.float16) else b.view(mx.uint32)
    return bool(mx.array_equal(a8, b8).item())


def check(Ts=(1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 24, 31, 32), CU=((1, 1), (2, 2), (4, 4), (4, 1), (4, 8), (8, 4), (2, 8))):
    GD = _ref()
    bad, n = [], 0
    for T in Ts:
        for seed in (0, 1):
            x = inputs(T, 1000 * T + seed)
            y0, s0 = GD.gated_delta_kernel(*x)
            mx.eval(y0, s0)
            for C, U in CU:
                y1, s1 = fast(*x, C=C, U=U)
                mx.eval(y1, s1)
                n += 1
                if not (_same(y0, y1) and _same(s0, s1)):
                    d = float(mx.max(mx.abs(y0.astype(mx.float32) - y1.astype(mx.float32))).item())
                    ds = float(mx.max(mx.abs(s0 - s1)).item())
                    bad.append((T, seed, C, U, d, ds))
    print(f"gdn_fast check: {n - len(bad)} / {n} bitwise (y and state) vs mlx_lm's gated_delta_kernel"
          + ("" if not bad else f"; NOT bitwise: {bad[:6]}"))
    return not bad


def bench(T=32, reps=40, CU=((1, 1), (2, 4), (4, 4), (8, 4), (4, 8), (8, 8))):
    GD = _ref()
    x = inputs(T, 7)

    def tm(f, k=reps):
        """'ind': k independent calls in one graph (throughput); 'chain': each call takes the previous call's state
        (latency, as in the engine: one GDN core per layer, dependent on the layer before)"""
        mx.synchronize()
        t0 = time.perf_counter()
        if MODE[0] == "ind":
            ys = [f(x[5]) for _ in range(k)]
            mx.eval(ys)
        else:
            s_ = x[5]
            for _ in range(k):
                s_ = f(s_)[1]
            mx.eval(s_)
        mx.synchronize()
        return 1e6 * (time.perf_counter() - t0) / k

    q, k, v, g, beta = x[:5]
    ref = lambda s_: GD.gated_delta_kernel(q, k, v, g, beta, s_)
    var = {(C, U): (lambda s_, C=C, U=U: fast(q, k, v, g, beta, s_, C=C, U=U)) for C, U in CU}
    res = {}
    for mode in ("chain", "ind"):
        MODE[0] = mode
        for f in [ref] + list(var.values()):
            tm(f, 5)
        out = {}
        for rep in range(3):                              # ABAB rounds: the reference, then each variant
            out.setdefault("ref", []).append(tm(ref))
            for cu, f in var.items():
                out.setdefault(cu, []).append(tm(f))
        r0 = float(np.median(out["ref"]))
        print(f"gdn_fast bench T {T}, {mode} ({reps} calls per round, median of 3 rounds): mlx_lm {r0:.1f} us per call")
        for C, U in CU:
            r = float(np.median(out[(C, U)]))
            print(f"  C {C} U {U}: {r:.1f} us ({r0 / r:.2f}x); 36 layers: {36 * (r0 - r) / 1e3:+.2f} ms per check")
        res[mode] = {str(kk): float(np.median(vv)) for kk, vv in out.items()}
    return res


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "check"
    if a == "check":
        sys.exit(0 if check() else 1)
    bench()
