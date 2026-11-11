"""GDN-STEP-R (Leap, Sat 2026-11-07; PLAN 220 1a): mlx_lm's gated_delta_step with R dv rows per simdgroup.
Why: at 16 rows mlx_lm's kernel takes 58.6 us per layer (16-row chain, v1' + wired, gap_p220b) for 6.7 MB of state and
inputs (~28 us at the in_proj streams' ~240 GB/s); at 2 rows 27.6. Fri's GDN-STEP3 ablations (no simd_sum, constant
k / q, prefetch) all kept the reference grid (32, Dv, B * Hv) / (32, 4, 1) = 1,536 threadgroups and found no per-row
culprit. Suspect left: the grid runs in several waves and every wave walks all T rows. Here a simdgroup carries R
consecutive dv rows: grid (32, Dv / R, B * Hv), the same threadgroup, 1,536 / R threadgroups. Each dv row's arithmetic
is mlx_lm's loop body verbatim (the same 4 state values per lane, the same order, the same simd_sum), so the outputs
should be bitwise == mlx_lm's (check() proves it on the engine's shapes). k / q / g / beta loads are shared by the R
rows. kernel(q, k, v, g, beta, state, mask=None, R=4) -> (y, state') as mlx_lm's gated_delta_kernel (scalar g only).
usage: python gdn_step_r.py [check|bench] -> ~/Leap/data/gdn_step_r.txt"""
import os
import sys
import time

import mlx.core as mx

OUT = os.path.expanduser("~/Leap/data/gdn_step_r.txt")
PRE_ONLY = os.environ.get("GS_PRE_ONLY") == "1"
TG_ONLY = os.environ.get("GS_TG_ONLY") == "1"
_K = {}


def _src(has_mask):
    mask_source = "mask[b_idx * T + t]" if has_mask else "true"
    return f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        const int dv0 = int(thread_position_in_grid.y) * R;

        float state[R][n_per_t];
        for (int r = 0; r < R; ++r) {{
          auto i_state = state_in + (n * Dv + dv0 + r) * Dk;
          for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            state[r][i] = static_cast<float>(i_state[s_idx]);
          }}
        }}

        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {{
          if ({mask_source}) {{
            for (int r = 0; r < R; ++r) {{
              const int dv_idx = dv0 + r;
              float kv_mem = 0.0f;
              for (int i = 0; i < n_per_t; ++i) {{
                auto s_idx = n_per_t * dk_idx + i;
                state[r][i] = state[r][i] * g_[hv_idx];
                kv_mem += state[r][i] * k_[s_idx];
              }}
              kv_mem = simd_sum(kv_mem);

              auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

              float out = 0.0f;
              for (int i = 0; i < n_per_t; ++i) {{
                auto s_idx = n_per_t * dk_idx + i;
                state[r][i] = state[r][i] + k_[s_idx] * delta;
                out += state[r][i] * q_[s_idx];
              }}
              out = simd_sum(out);
              if (thread_index_in_simdgroup == 0) {{
                y[dv_idx] = static_cast<InT>(out);
              }}
            }}
          }} else {{
            for (int r = 0; r < R; ++r) {{
              if (thread_index_in_simdgroup == 0) {{
                y[dv0 + r] = static_cast<InT>(0);
              }}
            }}
          }}
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }}
        for (int r = 0; r < R; ++r) {{
          auto o_state = state_out + (n * Dv + dv0 + r) * Dk;
          for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            o_state[s_idx] = static_cast<StT>(state[r][i]);
          }}
        }}
    """


def _src_pre(has_mask):
    """variant PRE: every row's k / q (this lane's 4 values), v (the R dv rows), g, beta (and the mask) loaded into
    registers BEFORE the time loop (TT = T as a template), then mlx_lm's loop body verbatim on the registers. The values
    and the arithmetic are the same, so the bits are too; tests whether the per-row cost is a load round trip."""
    mask_load = "mk[t] = mask[b_idx * TT + t];" if has_mask else "mk[t] = true;"
    return f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;
        auto dk_idx = thread_position_in_threadgroup.x;
        const int dv0 = int(thread_position_in_grid.y) * R;

        float kr[TT][n_per_t], qr[TT][n_per_t], vr[TT][R], gr[TT], br[TT];
        bool mk[TT];
        for (int t = 0; t < TT; ++t) {{
          auto q_ = q + ((b_idx * TT + t) * Hk + hk_idx) * Dk;
          auto k_ = k + ((b_idx * TT + t) * Hk + hk_idx) * Dk;
          auto v_ = v + ((b_idx * TT + t) * Hv + hv_idx) * Dv;
          for (int i = 0; i < n_per_t; ++i) {{
            kr[t][i] = static_cast<float>(k_[n_per_t * dk_idx + i]);
            qr[t][i] = static_cast<float>(q_[n_per_t * dk_idx + i]);
          }}
          for (int r = 0; r < R; ++r) vr[t][r] = static_cast<float>(v_[dv0 + r]);
          gr[t] = static_cast<float>(g[(b_idx * TT + t) * Hv + hv_idx]);
          br[t] = static_cast<float>(beta[(b_idx * TT + t) * Hv + hv_idx]);
          {mask_load}
        }}

        float state[R][n_per_t];
        for (int r = 0; r < R; ++r) {{
          auto i_state = state_in + (n * Dv + dv0 + r) * Dk;
          for (int i = 0; i < n_per_t; ++i) state[r][i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
        }}
        y += b_idx * TT * Hv * Dv + hv_idx * Dv;
        for (int t = 0; t < TT; ++t) {{
          if (mk[t]) {{
            for (int r = 0; r < R; ++r) {{
              const int dv_idx = dv0 + r;
              float kv_mem = 0.0f;
              for (int i = 0; i < n_per_t; ++i) {{
                state[r][i] = state[r][i] * gr[t];
                kv_mem += state[r][i] * kr[t][i];
              }}
              kv_mem = simd_sum(kv_mem);
              auto delta = (vr[t][r] - kv_mem) * br[t];
              float out = 0.0f;
              for (int i = 0; i < n_per_t; ++i) {{
                state[r][i] = state[r][i] + kr[t][i] * delta;
                out += state[r][i] * qr[t][i];
              }}
              out = simd_sum(out);
              if (thread_index_in_simdgroup == 0) y[dv_idx] = static_cast<InT>(out);
            }}
          }} else {{
            for (int r = 0; r < R; ++r) if (thread_index_in_simdgroup == 0) y[dv0 + r] = static_cast<InT>(0);
          }}
          y += Hv * Dv;
        }}
        for (int r = 0; r < R; ++r) {{
          auto o_state = state_out + (n * Dv + dv0 + r) * Dk;
          for (int i = 0; i < n_per_t; ++i) o_state[n_per_t * dk_idx + i] = static_cast<StT>(state[r][i]);
        }}
    """


def _src_tg():
    """variant TG: the threadgroup first copies every row's k / q (the head's 128 values, bf16 as stored), its 4 R dv
    rows' v, g and beta into threadgroup memory, one barrier, then mlx_lm's loop body verbatim reading them there.
    Same values, same arithmetic: bitwise. Tests the per-row global-load round trip without register spills."""
    return """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;
        constexpr int NDV = 4 * R;                       // dv rows per threadgroup
        auto dk_idx = thread_position_in_threadgroup.x;
        const uint tid = thread_index_in_threadgroup;    // 0 .. 127
        const int dvt = int(threadgroup_position_in_grid.y) * NDV;   // the threadgroup's first dv row
        const int dv0 = int(thread_position_in_grid.y) * R;

        threadgroup InT ks[TT][Dk], qs[TT][Dk], vs[TT][NDV];
        threadgroup float gs[TT];
        threadgroup InT bs[TT];
        for (uint e = tid; e < (uint)(TT * Dk); e += 128) {
          const int t = int(e) / Dk, c = int(e) % Dk;
          ks[t][c] = k[((b_idx * TT + t) * Hk + hk_idx) * Dk + c];
          qs[t][c] = q[((b_idx * TT + t) * Hk + hk_idx) * Dk + c];
        }
        for (uint e = tid; e < (uint)(TT * NDV); e += 128) {
          const int t = int(e) / NDV, c = int(e) % NDV;
          vs[t][c] = v[((b_idx * TT + t) * Hv + hv_idx) * Dv + dvt + c];
        }
        for (uint e = tid; e < (uint)TT; e += 128) {
          gs[e] = g[(b_idx * TT + e) * Hv + hv_idx];
          bs[e] = beta[(b_idx * TT + e) * Hv + hv_idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float state[R][n_per_t];
        for (int r = 0; r < R; ++r) {
          auto i_state = state_in + (n * Dv + dv0 + r) * Dk;
          for (int i = 0; i < n_per_t; ++i) state[r][i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
        }
        y += b_idx * TT * Hv * Dv + hv_idx * Dv;
        for (int t = 0; t < TT; ++t) {
          for (int r = 0; r < R; ++r) {
            const int dv_idx = dv0 + r;
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[r][i] = state[r][i] * gs[t];
              kv_mem += state[r][i] * ks[t][s_idx];
            }
            kv_mem = simd_sum(kv_mem);
            auto delta = (vs[t][dv_idx - dvt] - kv_mem) * bs[t];
            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[r][i] = state[r][i] + ks[t][s_idx] * delta;
              out += state[r][i] * qs[t][s_idx];
            }
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) y[dv_idx] = static_cast<InT>(out);
          }
          y += Hv * Dv;
        }
        for (int r = 0; r < R; ++r) {
          auto o_state = state_out + (n * Dv + dv0 + r) * Dk;
          for (int i = 0; i < n_per_t; ++i) o_state[n_per_t * dk_idx + i] = static_cast<StT>(state[r][i]);
        }
    """


def kernel_tgr(q, k, v, g, beta, state, mask=None, R=2):
    """variant TG with T a runtime input (rows <= TMAX 32 in threadgroup memory): one compiled kernel for every T, the
    same values and arithmetic as TG (so bitwise == mlx_lm). ref2's REF2_GDN_TG path; T > 32 or a mask -> None"""
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if mask is not None or T > 32 or g.ndim != 3 or Dv % (4 * R):
        return None
    if "tgr" not in _K:
        src = (_src_tg().replace("for (uint e = tid; e < (uint)(TT * Dk); e += 128)", "for (uint e = tid; e < (uint)(T * Dk); e += 128)")
               .replace("for (uint e = tid; e < (uint)(TT * NDV); e += 128)", "for (uint e = tid; e < (uint)(T * NDV); e += 128)")
               .replace("for (uint e = tid; e < (uint)TT; e += 128)", "for (uint e = tid; e < (uint)T; e += 128)")
               .replace("for (int t = 0; t < TT; ++t)", "for (int t = 0; t < T; ++t)")
               .replace("(b_idx * TT + t)", "(b_idx * T + t)").replace("(b_idx * TT + e)", "(b_idx * T + e)")
               .replace("y += b_idx * TT * Hv * Dv", "y += b_idx * T * Hv * Dv")
               .replace("threadgroup InT ks[TT][Dk], qs[TT][Dk], vs[TT][NDV];", "threadgroup InT ks[TMAX][Dk], qs[TMAX][Dk], vs[TMAX][NDV];")
               .replace("threadgroup float gs[TT];", "threadgroup float gs[TMAX];").replace("threadgroup InT bs[TT];", "threadgroup InT bs[TMAX];"))
        assert "TT" not in src.replace("TMAX", ""), "kernel_tgr: a TT left"
        _K["tgr"] = mx.fast.metal_kernel(name="leap_gated_delta_step_tgr", input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
                                         output_names=["y", "state_out"], source=src)
    return _K["tgr"](inputs=[q, k, v, g, beta, state, T], template=[("InT", q.dtype), ("StT", state.dtype), ("Dk", Dk), ("Dv", Dv),
                                                                     ("Hk", Hk), ("Hv", Hv), ("R", R), ("TMAX", 32)],
                     grid=(32, Dv // R, B * Hv), threadgroup=(32, 4, 1),
                     output_shapes=[(B, T, Hv, Dv), state.shape], output_dtypes=[q.dtype, state.dtype])


def kernel_tg(q, k, v, g, beta, state, mask=None, R=1):
    """variant TG (see _src_tg); no row mask (the verify / commit calls pass none)"""
    assert mask is None
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if "tg" not in _K:
        _K["tg"] = mx.fast.metal_kernel(name="leap_gated_delta_step_tg", input_names=["q", "k", "v", "g", "beta", "state_in"],
                                        output_names=["y", "state_out"], source=_src_tg())
    return _K["tg"](inputs=[q, k, v, g, beta, state], template=[("InT", q.dtype), ("StT", state.dtype), ("Dk", Dk), ("Dv", Dv),
                                                                  ("Hk", Hk), ("Hv", Hv), ("R", R), ("TT", T)],
                    grid=(32, Dv // R, B * Hv), threadgroup=(32, 4, 1),
                    output_shapes=[(B, T, Hv, Dv), state.shape], output_dtypes=[q.dtype, state.dtype])


def kernel_pre(q, k, v, g, beta, state, mask=None, R=1):
    """variant PRE (see _src_pre)"""
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    key = ("pre", mask is not None)
    if key not in _K:
        ins = ["q", "k", "v", "g", "beta", "state_in"] + (["mask"] if mask is not None else [])
        _K[key] = mx.fast.metal_kernel(name=f"leap_gated_delta_step_pre{'_mask' if mask is not None else ''}",
                                       input_names=ins, output_names=["y", "state_out"], source=_src_pre(mask is not None))
    ins = [q, k, v, g, beta, state] + ([mask] if mask is not None else [])
    return _K[key](inputs=ins, template=[("InT", q.dtype), ("StT", state.dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk),
                                         ("Hv", Hv), ("R", R), ("TT", T)],
                   grid=(32, Dv // R, B * Hv), threadgroup=(32, 4, 1),
                   output_shapes=[(B, T, Hv, Dv), state.shape], output_dtypes=[q.dtype, state.dtype])


def _kern(has_mask):
    if has_mask not in _K:
        ins = ["q", "k", "v", "g", "beta", "state_in", "T"] + (["mask"] if has_mask else [])
        _K[has_mask] = mx.fast.metal_kernel(name=f"leap_gated_delta_step_r{'_mask' if has_mask else ''}",
                                            input_names=ins, output_names=["y", "state_out"], source=_src(has_mask))
    return _K[has_mask]


def kernel(q, k, v, g, beta, state, mask=None, R=4):
    """mlx_lm's gated_delta_kernel (scalar g) with R dv rows per simdgroup"""
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    assert g.ndim == 3 and Dv % (4 * R) == 0, (g.shape, Dv, R)
    ins = [q, k, v, g, beta, state, T] + ([mask] if mask is not None else [])
    return _kern(mask is not None)(
        inputs=ins,
        template=[("InT", q.dtype), ("StT", state.dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv), ("R", R)],
        grid=(32, Dv // R, B * Hv), threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape], output_dtypes=[q.dtype, state.dtype])


def _inputs(rng, T):
    import numpy as np
    k = rng.standard_normal((1, T, 16, 128)).astype(np.float32); k /= np.linalg.norm(k, axis=-1, keepdims=True)
    q = rng.standard_normal((1, T, 16, 128)).astype(np.float32); q /= np.linalg.norm(q, axis=-1, keepdims=True)
    q *= 128 ** -0.5
    v = rng.standard_normal((1, T, 48, 128)).astype(np.float32)
    g = rng.uniform(0.5, 1.0, (1, T, 48)).astype(np.float32)
    b = rng.uniform(0.0, 1.0, (1, T, 48)).astype(np.float32)
    st = (rng.standard_normal((1, 48, 128, 128)) * 0.3).astype(np.float32)
    a = [mx.array(q).astype(mx.bfloat16), mx.array(k).astype(mx.bfloat16), mx.array(v).astype(mx.bfloat16),
         mx.array(g), mx.array(b).astype(mx.bfloat16), mx.array(st)]
    mx.eval(a)
    return a


def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")


def check():
    """bitwise vs mlx_lm's gated_delta_kernel: T 1 / 2 / 5 / 8 / 16 / 32, R 1 / 2 / 4 / 8, with and without a row mask"""
    import numpy as np
    from mlx_lm.models.gated_delta import gated_delta_kernel
    rng = np.random.default_rng(11)
    bad, n = 0, 0
    for T in (1, 2, 5, 8, 16, 32):
        q, k, v, g, b, st = _inputs(rng, T)
        for masked in (False, True):
            m = mx.array(rng.random((1, T)) < 0.8) if masked else None
            y0, s0 = gated_delta_kernel(q, k, v, g, b, st, m)
            for R in ((1, 2, 4, 8) if not PRE_ONLY else ()) + (tuple(("pre", r_) for r_ in (1, 2)) if not TG_ONLY else ()) \
                    + (tuple(("tg", r_) for r_ in (1, 2, 4)) + (("tgr", 2),) if not masked else ()):
                if isinstance(R, tuple):
                    y1, s1 = {"tg": kernel_tg, "tgr": kernel_tgr, "pre": kernel_pre}[R[0]](q, k, v, g, b, st, m, R=R[1])
                else:
                    y1, s1 = kernel(q, k, v, g, b, st, m, R=R)
                mx.eval(y0, s0, y1, s1)
                ok = bool(mx.array_equal(y0, y1)) and bool(mx.array_equal(s0, s1))
                n += 1
                bad += not ok
                if not ok:
                    say(f"  DIFF T {T} R {R} mask {masked}: y max |d| {float(mx.abs(y0.astype(mx.float32) - y1.astype(mx.float32)).max()):.3g}, "
                        f"state max |d| {float(mx.abs(s0 - s1).max()):.3g}")
    say(f"GDN-STEP-R check: {n - bad} of {n} cases bitwise == mlx_lm's kernel (y and state)")
    return bad


def bench(reps=5, n=36):
    """dependent chains of n calls (each call's state = the last call's output, as the 36 layers would not be, but
    the cold state is re-read each call: 36 distinct states, rotated), per call us, T 2 and 16, mlx_lm vs R 2 / 4 / 8"""
    import numpy as np
    from mlx_lm.models.gated_delta import gated_delta_kernel
    rng = np.random.default_rng(5)
    for T in (2, 16):
        q, k, v, g, b, _ = _inputs(rng, T)
        sts = [mx.array((rng.standard_normal((1, 48, 128, 128)) * 0.3).astype(np.float32)) for _ in range(n)]
        mx.eval(sts)
        res = {}
        for name, fn in [("mlx_lm", lambda s: gated_delta_kernel(q, k, v, g, b, s)),
                         ("R2", lambda s: kernel(q, k, v, g, b, s, R=2)), ("R4", lambda s: kernel(q, k, v, g, b, s, R=4)),
                         ("R8", lambda s: kernel(q, k, v, g, b, s, R=8)), ("P1", None), ("P2", None),
                         ("T1", None), ("T2", None), ("T4", None), ("TR2", None), ("mlx_lm2", None)]:
            if TG_ONLY and name[0] in "RP":
                continue
            ts = []
            for rep in range(reps + 1):
                mx.synchronize()
                t0 = time.perf_counter()
                y = None
                for i in range(n):                       # a dependent chain: each call's v depends on the last y
                    vi = v if y is None else v + 0 * y[..., :1].astype(v.dtype)
                    if name.startswith("mlx_lm"):
                        y, _s = gated_delta_kernel(q, k, vi, g, b, sts[i])
                    elif name.startswith("TR"):
                        y, _s = kernel_tgr(q, k, vi, g, b, sts[i], R=int(name[2:]))
                    elif name.startswith("T"):
                        y, _s = kernel_tg(q, k, vi, g, b, sts[i], R=int(name[1:]))
                    elif name.startswith("P"):
                        y, _s = kernel_pre(q, k, vi, g, b, sts[i], R=int(name[1:]))
                    else:
                        y, _s = kernel(q, k, vi, g, b, sts[i], R=int(name[1:]))
                mx.eval(y)
                if rep:
                    ts.append(1e6 * (time.perf_counter() - t0) / n)
            res[name] = sorted(ts)[len(ts) // 2]
        say(f"GDN-STEP-R bench T {T}: per call us (median of {reps} chains of {n}, incl. the ~3 us link): " +
            ", ".join(f"{k_} {v_:.1f}" for k_, v_ in res.items()))


if __name__ == "__main__":
    say(f"\nGDN-STEP-R {time.strftime('%a %H:%M:%S')} ({os.environ.get('GS_COND', 'share lock')}; MLX {mx.__version__})")
    what = sys.argv[1:] or ["check", "bench"]
    if "check" in what:
        check()
    if "bench" in what:
        bench()


def check_tg_all(R=2, seeds=(1, 2, 3)):
    """TG (R) vs mlx_lm's kernel for EVERY T in 1 .. 16 (each T is its own compiled kernel), engine shapes, no mask"""
    import numpy as np
    from mlx_lm.models.gated_delta import gated_delta_kernel
    bad = n = 0
    for sd in seeds:
        rng = np.random.default_rng(100 + sd)
        for T in range(1, 17):
            q, k, v, g, b, st = _inputs(rng, T)
            y0, s0 = gated_delta_kernel(q, k, v, g, b, st)
            y1, s1 = kernel_tg(q, k, v, g, b, st, R=R)
            mx.eval(y0, s0, y1, s1)
            n += 1
            bad += not (bool(mx.array_equal(y0, y1)) and bool(mx.array_equal(s0, s1)))
    say(f"GDN TG {R} vs mlx_lm, every T 1-16 x {len(seeds)} seeds: {n - bad} of {n} bitwise")
    return bad


def precompile_tg(R=2):
    """ref2's REF2_GDN_TG: build TG R's kernels for T 1-16 at the engine's shapes now (each T is its own pipeline)"""
    for T in range(1, 17):
        q = mx.zeros((1, T, 16, 128), mx.bfloat16)
        v = mx.zeros((1, T, 48, 128), mx.bfloat16)
        y, s = kernel_tg(q, q, v, mx.ones((1, T, 48), mx.float32), mx.zeros((1, T, 48), mx.bfloat16),
                         mx.zeros((1, 48, 128, 128), mx.float32), R=R)
        mx.eval(y, s)
