"""BigMoE's D2 family (Wed 11 Nov 2026; ~/BigMoE/docs/PREREG_D2_2026-11-11.md), imported lazily by
dm_ablations.ablation() for names d<k>_* (Breakthrough's hook, its OK 18:4x). Registers into dm_ablations.ABLATIONS:

  d<b>_trunk, b in 2 / 3 / 4: the D2 trunk set at b bits, group 64 (32 where the input width is not a multiple of 64).
    Each module is dequantized, then requantized: conversion only, from the weights the store has.
    - The D2 set, with exact per-type counts asserted (Breakthrough's census of the model config):
      - GDN in_proj_a / in_proj_b / in_proj_qkv / in_proj_z / out_proj: 180 (4-bit in the store);
      - attention q / k / v / o_proj: 48 (4-bit);
      - indexer.index_qk_proj: 12 (8-bit);
      - HC input_mix_weight_down / up: 194 (bf16);
      - HC block_inject_weight: 96 (bf16);
      - PLE key / value_proj: 2 (bf16).
    - Unchanged, per D2's spec ("router + shared 0.17 GB"): the router (mlp.gate, 8-bit), the shared expert + its gate,
      embed_tokens, lm_head and the MTP head.
    - The originals stay referenced and a TEARDOWN puts them back, so a later name in the same ticket sees the store's trunk.
  d0_resid: the routed experts restricted BY EXPERT ID to a fixed resident set.
    - The set, per layer L: the C_m[L] most popular experts by ~/FableGod/fablec/masks/pop_cap45.npy, where C_m is the
      45 GB default's base pool (~/MoonShots/data/laysz45_256.json; mean 295 of 512).
    - Drop semantics: a non-resident expert in a row's own top-10 contributes 0, with no renorm. That is a drafter that
      skips its misses. It applies on every row and every MoE layer, on the parent's ops path (dm_common's < 32-row path,
      used for every chunk here), with the id mask on the scores.
  d0_ops: the same ops path with every expert resident: the noise floor of that path against the stock combine.
Stack with '+': d2_trunk+d0_resid is D2."""
import json, os

import numpy as np
import mlx.core as mx
import mlx.nn as nn

import dm_common as C
from dm_ablations import ABLATIONS

D2_SET = {"in_proj_a": "gdn", "in_proj_b": "gdn", "in_proj_qkv": "gdn", "in_proj_z": "gdn", "out_proj": "gdn",
          "q_proj": "att", "k_proj": "att", "v_proj": "att", "o_proj": "att", "index_qk_proj": "idx",
          "input_mix_weight_down": "mix", "input_mix_weight_up": "mix", "block_inject_weight": "inject",
          "key_proj": "ple", "value_proj": "ple"}
WANT = {"gdn": 180, "att": 48, "idx": 12, "mix": 194, "inject": 96, "ple": 2}


def _model():
    rt = C.RT["rt"]
    m = rt.model if hasattr(rt, "model") else rt
    return getattr(m, "language_model", m)


def _requant(lin, bits):
    """a Linear / QuantizedLinear -> QuantizedLinear at `bits` (dequantize first): conversion only."""
    if isinstance(lin, nn.QuantizedLinear):
        w = mx.dequantize(lin.weight, lin.scales, lin.get("biases"), lin.group_size, lin.bits)
    else:
        w = lin.weight
    out_f, in_f = int(w.shape[0]), int(w.shape[1])
    gs = 64 if in_f % 64 == 0 else 32
    has_b = "bias" in lin
    tmp = nn.Linear(in_f, out_f, bias=has_b)
    tmp.weight = w.astype(mx.bfloat16)
    if has_b:
        tmp.bias = lin.bias
    q = nn.QuantizedLinear.from_linear(tmp, gs, bits)
    mx.eval(q.parameters())
    return q, gs


def trunk(bits):
    def setup():
        TM = _model()
        undo, cnt, g32 = [], {k: 0 for k in WANT}, 0
        for name, mod in TM.named_modules():
            if name.startswith("mtp") or ".mtp" in name:
                continue
            for attr, kind in D2_SET.items():
                lin = mod.get(attr) if isinstance(mod, nn.Module) and attr in mod else None
                if not isinstance(lin, (nn.Linear, nn.QuantizedLinear)):
                    continue
                if kind == "gdn" and attr == "out_proj" and not isinstance(mod, C.Q.GatedDeltaNet):
                    continue                                 # out_proj of GDN only (attention's is o_proj)
                q, gs = _requant(lin, bits)
                g32 += gs == 32
                setattr(mod, attr, q); undo.append((mod, attr, lin)); cnt[kind] += 1
        def restore():
            for mod, attr, lin in undo:
                setattr(mod, attr, lin)
            mx.clear_cache()
        C.TEARDOWN.append(restore)
        assert cnt == WANT, f"d{bits}_trunk swapped {cnt}, want {WANT}"
        print(f"[dm] d{bits}_trunk: {sum(cnt.values())} linears at {bits} bits ({cnt}; group 32 on {g32})", flush=True)
    return setup


_RES = {}


def _resident():
    if not _RES:
        pop = np.load(os.path.expanduser("~/FableGod/fablec/masks/pop_cap45.npy"))       # [48, 512]
        cm = json.load(open(os.path.expanduser("~/MoonShots/data/laysz45_256.json")))["C_m"]
        assert pop.shape == (48, 512) and len(cm) == 48, (pop.shape, len(cm))
        for L in range(48):
            keep = np.zeros(512, np.float32)
            keep[np.argsort(-pop[L], kind="stable")[:int(cm[L])]] = 1.0                  # ties: the lower expert id first
            _RES[L] = mx.array(keep)
        _RES["sizes"] = [int(c) for c in cm]
    return _RES


def moe_ops(mask):
    """the SparseMoeBlock call on the parent's ops path for every chunk; mask True: the fixed resident set by expert id."""
    def setup():
        Q = C.Q
        R = _resident() if mask else None
        prev = Q.SparseMoeBlock.__call__
        def call(self, x):
            li = C.LAYERS.get(id(self), (None, None))[0]
            gates = mx.softmax(self.gate(x), axis=-1, precise=True)
            k = self.top_k
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            scores = mx.take_along_axis(gates, inds, axis=-1)
            if self.norm_topk_prob:
                scores = scores / scores.sum(axis=-1, keepdims=True)
            if R is not None and li is not None:
                scores = scores * mx.take(R[li], inds).astype(scores.dtype)   # drop semantics, no renorm
            y = self.switch_mlp(x, inds)
            y = (y * scores[..., None]).sum(axis=-2)
            shared = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
            return y + shared
        Q.SparseMoeBlock.__call__ = call
        def restore():
            Q.SparseMoeBlock.__call__ = prev
        C.TEARDOWN.append(restore)
        if R is not None:
            print(f"[dm] d0_resid: fixed resident set by id, {sum(R['sizes'])} of {48 * 512} expert-layers "
                  f"(mean {sum(R['sizes']) / 48:.1f} per layer), drop semantics", flush=True)
        else:
            print("[dm] d0_ops: the ops path, every expert resident", flush=True)
    return setup


for _b in (2, 3, 4):
    ABLATIONS[f"d{_b}_trunk"] = trunk(_b)
ABLATIONS["d0_resid"] = moe_ops(True)
ABLATIONS["d0_ops"] = moe_ops(False)
