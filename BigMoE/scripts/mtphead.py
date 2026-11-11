"""R-60 MTPDROP (NewIdeas 10-31): overlay a trained MTP head's dense parts on a loaded stack (ported from Breakthrough's
_bt_b12_r60.py mtp_head_overlay). Lives in scripts/ (not bigmoe/) while EvalSuite's runners pin bigmoe/*.py; harnesses call
overlay(rt, path) right after load_streamed when BIGMOE_MTP_HEAD=<safetensors> is set.
MTP Head's D-22 head (~/TrainLab/ckpt/d22_b3.safetensors, fp32 by mtp125.Trainable's names): fc_hidden, fc_embedding,
attention q/k/v/o, router, shared expert + its gate (69.1M weights) become bf16 Linear modules (bits=8: 8-bit g64; stock
4-bit storage would erase most of the fine-tune). The MoE block and its (streamed) routed experts stay. Drafts only:
verification, and so the output tokens, are unchanged."""
import os


def overlay(rt, path, bits=0):
    import mlx.core as mx
    import mlx.nn as nn
    head = rt.model.language_model.mtp
    layer = head.layers[0]
    att, moe = layer.self_attn, layer.mlp
    W = mx.load(path)

    def lin(name):
        w = W[name + ".weight"]
        l = nn.Linear(w.shape[1], w.shape[0], bias=False)
        l.weight = w.astype(mx.bfloat16)
        mx.eval(l.weight)
        return nn.QuantizedLinear.from_linear(l, group_size=64, bits=bits) if bits else l

    class _SharedMLP(nn.Module):
        def __init__(self, g, u, d):
            super().__init__()
            self.gate_proj, self.up_proj, self.down_proj = g, u, d

        def __call__(self, x):
            return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))
    head.fc_hidden, head.fc_embedding = lin("fc_hidden"), lin("fc_embedding")
    att.q_proj, att.k_proj, att.v_proj, att.o_proj = lin("q_proj"), lin("k_proj"), lin("v_proj"), lin("o_proj")
    if getattr(att, "qkv_fused", None) is not None:
        att.qkv_fused = None                                  # the fused q/k/v would keep the stock weights
    moe.gate = lin("router")
    moe.shared_expert = _SharedMLP(lin("se_gate"), lin("se_up"), lin("se_down"))
    if "se_gatew.weight" in W:
        moe.shared_expert_gate = lin("se_gatew")
    object.__setattr__(head, "_mtp_prepare_inputs_impl", head._prepare_inputs_eager)   # no compiled graph on old weights
    del W
    mx.eval(head.parameters())
    mx.clear_cache()
    comp = "eager"
    if os.environ.get("BIGMOE_MTP_HEAD_COMPILE", "1") == "1" and hasattr(head, "install_compiled_prepare"):
        try:                                             # R-60b: MTPLX's exact compiled prepare graph, rebuilt on the new
            head.install_compiled_prepare(); comp = "compiled"   # weights (it checks parity with the eager graph itself)
        except Exception as e:
            object.__setattr__(head, "_mtp_prepare_inputs_impl", head._prepare_inputs_eager); comp = f"eager ({e})"
    print(f"[bigmoe] MTP head overlay {os.path.basename(path)} ({bits or 16}-bit dense parts, prepare {comp})", flush=True)


def from_env(rt):
    p = os.environ.get("BIGMOE_MTP_HEAD")
    if p:
        overlay(rt, os.path.expanduser(p), int(os.environ.get("BIGMOE_MTP_HEAD_BITS", "0") or 0))
    return p
