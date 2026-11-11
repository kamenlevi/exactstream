"""PF-B kernel-level bit check (orchestrator Tue 14:0x; no fix built): PF-B runs the router and the shared expert ONCE per group of
chunks (M rows), PF-A once per 1,024-row chunk. For layer L's real weights (the model's own safetensors, MLX 0.32.2), compute
each op on M rows at once and on the same rows chunk by chunk, and compare bit for bit. M values: the failing 30K groups
(5,412 at TOK 8192; 4,096 and 1,316 at TOK 4096) and the passing ones (8,192; 1,796; 6,810; 3,619; 427). Control: the shared
expert's gate (N = 1), whose split-K depends on M (PF-B computes it per chunk since its fix).
Usage: kcheck.py <model dir> [layer]"""
import json, sys
import mlx.core as mx

ROOT, L = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1
cfg = json.load(open(f"{ROOT}/config.json"))
q = cfg.get("quantization") or cfg["text_config"]["quantization"]
wm = json.load(open(f"{ROOT}/model.safetensors.index.json"))["weight_map"]
pre = f"language_model.model.layers.{L}.mlp."
shards = {}


def mod(name):
    key = pre + name
    T = {}
    for s in (".weight", ".scales", ".biases"):
        f = wm.get(key + s)
        if f:
            if f not in shards:
                shards[f] = mx.load(f"{ROOT}/{f}")
            T[s] = shards[f][key + s]
    qq = q.get(key) or {"bits": q["bits"], "group_size": q["group_size"]}
    return T[".weight"], T[".scales"], T.get(".biases"), qq["group_size"], qq["bits"]


def qlin(x, m):
    w, s, b, gs, bits = m
    return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=gs, bits=bits)


gate, sgate = mod("gate"), mod("shared_expert_gate")
gp, up, dn = mod("shared_expert.gate_proj"), mod("shared_expert.up_proj"), mod("shared_expert.down_proj")
H = cfg.get("text_config", cfg)["hidden_size"]
print(f"layer {L}: hidden {H}; router bits {gate[4]} gs {gate[3]}; shared gate bits {sgate[4]}; shared expert bits {gp[4]} / {dn[4]}; "
      f"MLX {mx.__version__}, {mx.device_info().get('architecture')}")


def silu(x):
    return x * mx.sigmoid(x)


OPS = {"router (N=512)": lambda x: qlin(x, gate),
       "shared expert gate (N=1)": lambda x: qlin(x, sgate),
       "shared expert MLP": lambda x: qlin(silu(qlin(x, gp)) * qlin(x, up), dn)}
MS = [("5412 (30K, TOK 8192: FAIL)", 5412), ("4096 (30K, TOK 4096: FAIL)", 4096), ("1316 (30K, TOK 4096: FAIL)", 1316),
      ("8192 (10-25K: pass)", 8192), ("1796 (10K: pass)", 1796), ("6810 (15K: pass)", 6810), ("3619 (20K: pass)", 3619),
      ("427 (25K, one chunk)", 427)]
for label, M in MS:
    x = (mx.random.normal((1, M, H), key=mx.random.key(M)) * 0.5).astype(mx.bfloat16)
    sizes = [1024] * (M // 1024) + ([M % 1024] if M % 1024 else [])
    out = []
    for name, op in OPS.items():
        yg = op(x)
        o, parts = 0, []
        for n in sizes:
            parts.append(op(x[:, o:o + n])); o += n
        yc = mx.concatenate(parts, axis=1)
        diff = mx.any(yg != yc, axis=-1)
        mx.eval(yg, yc, diff)
        nd = int(mx.sum(diff).item())
        out.append(f"{name}: {'EQUAL' if nd == 0 else f'{nd} of {M} rows DIFFER'}")
    print(f"M {label} = {'+'.join(map(str, sizes))}: " + "; ".join(out), flush=True)
