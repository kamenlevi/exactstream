"""Drop-in for MTPLX's fused GDN conv+silu+l2norm kernels with the exact version (gdn_exact.py): after install(), the
branches that MTPLX_FUSED_GDN_CONVNORM=1 (decode rows, S=1) and MTPLX_FUSED_CONVNORM_VERIFY=1 (verify rows, S=2..6) route
to are bitwise equal to the eager chain (tensor level), so the reference's shas stay unchanged at any context length,
while the ~10 dependent kernels per GDN layer between the in_proj and gated_delta_update become one dispatch.
Usage (in the process, before the first forward):  import gdn_exact_patch; gdn_exact_patch.install()
and set MTPLX_FUSED_GDN_CONVNORM=1 MTPLX_FUSED_CONVNORM_VERIFY=1 in the environment."""
import mlx.core as mx
from gdn_exact import gdn_prework_exact, gated_norm_exact

_SCALE = None


def _scale():
    global _SCALE
    if _SCALE is None:
        _SCALE = mx.array(1.0, mx.bfloat16) * (128 ** -0.5)   # MLX's bf16 operand for the Python scalar inv_scale
    return _SCALE


def fused_gdn_conv_norm_rows(qkv_rows, conv_state, conv_w):
    """Same contract as mtplx.kernels.gdn_conv_norm.fused_gdn_conv_norm_rows; exact."""
    return gdn_prework_exact(qkv_rows.reshape(qkv_rows.shape[0], -1), conv_state.reshape(3, -1), conv_w.reshape(10240, 4), _scale())


def fused_gdn_conv_norm(qkv_row, conv_state, conv_w):
    """Same contract as mtplx.kernels.gdn_conv_norm.fused_gdn_conv_norm (1-D q, k, v); exact."""
    q, k, v, ns = gdn_prework_exact(qkv_row.reshape(1, -1), conv_state.reshape(3, -1), conv_w.reshape(10240, 4), _scale())
    return q.reshape(-1), k.reshape(-1), v.reshape(-1), ns


def install(norm=True):
    """norm=True also routes SigmoidRMSNormGated (decode/verify rows, < 32 rows, all-bf16 operands: the tested case) to
    the exact one-dispatch kernel (rms_norm + sigmoid gate: 4 dependent kernels -> 1); anything else keeps MTPLX's code."""
    import mtplx.kernels.gdn_conv_norm as G
    G.fused_gdn_conv_norm_rows = fused_gdn_conv_norm_rows
    G.fused_gdn_conv_norm = fused_gdn_conv_norm
    if norm:
        import mtplx.models.qwen4_exp as Q
        orig = Q.SigmoidRMSNormGated.__call__

        def call(self, hidden_states, gate=None):
            rows = hidden_states.size // max(1, hidden_states.shape[-1])
            if (gate is not None and rows < 32 * hidden_states.shape[-2] and hidden_states.shape[-1] == 128
                    and hidden_states.dtype == mx.bfloat16 and gate.dtype == mx.bfloat16
                    and self.weight.dtype == mx.bfloat16 and hidden_states.shape == gate.shape):
                return gated_norm_exact(hidden_states, gate, self.weight, self.eps)
            return orig(self, hidden_states, gate)

        Q.SigmoidRMSNormGated.__call__ = call
    return G
