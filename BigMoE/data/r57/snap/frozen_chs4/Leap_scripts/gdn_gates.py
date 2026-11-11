"""R-67 GATES (Leap, Sun 2026-11-01; REF2c candidate): mlx_lm's gated_delta_update with the gate math inside the recurrence
kernel. Stock: beta = sigmoid(b) and g = compute_g(A_log, a, dt_bias) (exp(-exp(A_log) * softplus(a + dt_bias)), bf16
intermediates) run as their own dispatches before the kernel; here each (t, head) computes them in fp32 from the bf16
inputs inside the kernel (one dispatch instead of three). The recurrence itself is mlx_lm's gated_delta_step (scalar
gates, no mask) unchanged, so per step the arithmetic does not depend on T: a 16-row call and 16 one-row calls give the
same bits (INV, r67_gates_test.py). The gate values differ from the stock bf16 chain in the last bits (fp32 instead of bf16
rounding), so this is a reference change (REF2c), not exact-safe.
gated_delta_update_g(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True) -> (y, state), stock's signature;
falls back to stock for masks or vectorized gates."""
from functools import lru_cache
import mlx.core as mx
from mlx_lm.models import gated_delta as GD

_SRC = r"""
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
    auto dv_idx = thread_position_in_grid.y;
    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;
    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = static_cast<float>(i_state[s_idx]);
    }
    // gates in fp32 from the bf16 inputs: g = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b)
    const float ea = metal::exp(static_cast<float>(A_log[hv_idx]));
    const float dtb = static_cast<float>(dt_bias[hv_idx]);
    auto a_ = a + b_idx * T * Hv;
    auto bb_ = bg + b_idx * T * Hv;
    for (int t = 0; t < T; ++t) {
      const float xa = static_cast<float>(a_[hv_idx]) + dtb;
      const float sp = xa > 20.0f ? xa : metal::log(1.0f + metal::exp(xa));
      const float g = metal::exp(-ea * sp);
      const float beta = 1.0f / (1.0f + metal::exp(-static_cast<float>(bb_[hv_idx])));
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] * g;
        kv_mem += state[i] * k_[s_idx];
      }
      kv_mem = simd_sum(kv_mem);
      auto delta = (v_[dv_idx] - kv_mem) * beta;
      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] + k_[s_idx] * delta;
        out += state[i] * q_[s_idx];
      }
      out = simd_sum(out);
      if (thread_index_in_simdgroup == 0) {
        y[dv_idx] = static_cast<InT>(out);
      }
      q_ += Hk * Dk;
      k_ += Hk * Dk;
      v_ += Hv * Dv;
      y += Hv * Dv;
      a_ += Hv;
      bb_ += Hv;
    }
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      o_state[s_idx] = static_cast<StT>(state[i]);
    }
"""


@lru_cache(None)
def _kernel():
    return mx.fast.metal_kernel(name="gated_delta_step_gates", input_names=["q", "k", "v", "a", "bg", "A_log", "dt_bias",
                                                                             "state_in", "T"],
                                output_names=["y", "state_out"], source=_SRC)


def gated_delta_update_g(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if (mask is not None or not use_kernel or mx.default_device() != mx.gpu or a.ndim != 3 or b.ndim != 3
            or Dk % 32 != 0):
        return GD.gated_delta_update(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel)
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    return tuple(_kernel()(
        inputs=[q, k, v, a, b, A_log, dt_bias, state, T],
        template=[("InT", q.dtype), ("StT", state.dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
        grid=(32, Dv, B * Hv), threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape], output_dtypes=[q.dtype, state.dtype]))
