"""P-CAP (MTP Head, Wed 11-04 18:0x; pre-registered in REPORT 'P-CAP'): grow DFlash2 from 5 to 10 layers by block
expansion (LLaMA Pro, Wu et al. 2024): after each layer, a copy of it whose output projections (self_attn.o_proj,
mlp.down_proj) are zero. The copy's residual branches then add exact zeros (the dynamic convs in DFlash2's finish() are
linear in their input), so the expanded drafter computes exactly the original function before training.
Self-test (python sticky/pcap_expand.py): incoai's logits before vs after expansion on random context rows, max |diff|."""
import os
import mlx.core as mx


def expand_interleave(draft):
    """in place: layers [L0, L0', L1, L1', ...]; returns the indices of the new layers"""
    cfg = draft.config
    old, types = list(draft.layers), tuple(cfg.layer_types)
    new, idx = [], []
    for i, L in enumerate(old):
        new.append(L)
        c = type(L)(cfg, i)                              # same layer type as L (sliding window, causal flag)
        c.update(L.parameters())                         # MLX arrays are immutable: sharing them is a copy
        c.self_attn.o_proj.weight = mx.zeros_like(L.self_attn.o_proj.weight)
        c.mlp.down_proj.weight = mx.zeros_like(L.mlp.down_proj.weight)
        idx.append(len(new))
        new.append(c)
    draft.layers = new
    cfg.layer_types = tuple(t for t in types for _ in (0, 1))
    cfg.num_hidden_layers = len(new)
    return idx


if __name__ == "__main__":
    import sys
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p1a/Leap_third_party"))
    import mlx.nn as nn
    import dflash_model_mlx as DF

    class Target(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.QuantizedEmbedding(248320, 2560, group_size=64, bits=4)
            self.lm_head = nn.QuantizedLinear(2560, 248320, bias=False, group_size=64, bits=4)

    w = mx.load(os.path.expanduser("~/BigMoE/models/bare/model-00016-of-00017.safetensors"))
    tgt = Target()
    tgt.load_weights([(f"embed_tokens.{s}", w[f"language_model.model.embed_tokens.{s}"]) for s in ("weight", "scales", "biases")]
                     + [(f"lm_head.{s}", w[f"language_model.lm_head.{s}"]) for s in ("weight", "scales", "biases")])
    del w
    d = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
    d.bind(tgt)
    MASK = int(d.config.mask_token_id)
    mx.random.seed(0)
    ctx = (mx.random.normal((1, 300, 12800)) * 2.0).astype(mx.bfloat16)
    blk = mx.array([[1234] + [MASK] * 15])

    def logits(m):
        cache = m.make_cache()
        return m.compute_logits(m.hidden_states(blk, ctx, cache, logits_start=1)).astype(mx.float32)

    a = logits(d)
    mx.eval(a)
    n_old = len(d.layers)
    newi = expand_interleave(d)
    b = logits(d)
    mx.eval(b)
    diff = float(mx.abs(a - b).max())
    print(f"P-CAP expand self-test: {n_old} -> {len(d.layers)} layers (new at {newi}), layer types "
          f"{d.config.layer_types}; max |logit diff| before vs after {diff:.3e} ({'PASS' if diff == 0.0 else 'FAIL'}); "
          f"argmax agree {float((mx.argmax(a, -1) == mx.argmax(b, -1)).astype(mx.float32).mean()):.4f}")
