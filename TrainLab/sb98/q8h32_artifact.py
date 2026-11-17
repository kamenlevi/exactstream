"""WEEK98 Q8H32 ENGINE ARTIFACT (MTP Head; orchestrator 11:0x): t30b in exactly the format MoonShots' M3 harness measured
as cut Q8H32 (~/MoonShots/week98/p3_dd_tree.py, read-only import): t30b's 10 trained Linears (q / o / k / v, se_gate /
se_up / se_down, fc_hidden / fc_embedding, router) and the head's 6 bf16 HC mixes, each as mx.quantize(w.astype(bf16),
group_size 64, bits 8); se_gatew, norms and the rest as stored in t30b; routed experts as stored in the 125B; top-10;
the vocab screen = Leap's frspec_rank.npy[:32768] + the session's first-appearance tokens (cap 4096), applied at run time.
VERIFY: MoonShots' own apply_cut(CUTS['Q8']) is run on a second in-memory copy, and every quantized tensor of the artifact,
dequantized, must equal apply_cut's fake-quantized weight BITWISE. Then the artifact's dequantized form IS the form
MoonShots measured, so its kept is theirs by construction (Q8H32, TRUE-STATE: HE+ t16 x0.9775 / t32 x0.9751, MBPP
x0.9981 / x1.0046). The packed form (QuantizedLinear layout: .weight uint32, .scales, .biases) is what an engine's
quantized matmul reads; its kept vs the dequantized form is the engine check's business.
Output: OUTDIR/t30b_q8h32.safetensors (+ .json with the recipe, sha256s and the verification).
Usage: BigMoE venv python -I -B q8h32_artifact.py OUTDIR    (MEM ~10-12 GB: the streamed 125B load, frac 0.03)"""
import os, sys, json, hashlib, copy
H = os.path.expanduser
sys.path.insert(0, H("~/TrainLab")); sys.path.insert(0, H("~/MoonShots/week98"))
import mlx.core as mx                                      # noqa: E402
from mlx.utils import tree_flatten                         # noqa: E402
import mtp125 as M                                         # noqa: E402
import p3_dd_tree as P3                                    # noqa: E402  (read-only: apply_cut, fq, CUTS, RANK_F, EXTRA_CAP)

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
CK = H("~/datasets/deepdraft/ckpt/dd_t30b.safetensors")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
LIN = ("q_proj", "o_proj", "k_proj", "v_proj", "se_gate", "se_up", "se_down", "fc_hidden", "fc_embedding", "router")
HC = (("attn_hc", lambda h: h.layers[0].attn_hyper_connection), ("mlp_hc", lambda h: h.layers[0].mlp_hyper_connection),
      ("mixer", lambda h: h.hyper_connection_mixer))


def build():
    rt, lm, head = M.load(0.03)
    P = M.Trainable(head)
    P.load_weights(CK); mx.eval(P.parameters())
    return rt, lm, head, P


# 1) the artifact, from the as-loaded weights
rt, lm, head, P = build()
mods = [(f"P.{n}", getattr(P, n)) for n in LIN] + [(f"head.{g}.{s}", getattr(f(head), s)) for g, f in HC
                                                   for s in ("input_mix_weight_down", "input_mix_weight_up")]
art, src = {}, {}
for key, lin in mods:
    w = lin.weight
    assert not hasattr(lin, "scales"), f"{key} is already quantized"
    q, s, b = mx.quantize(w.astype(mx.bfloat16), group_size=64, bits=8)
    art[f"{key}.weight"], art[f"{key}.scales"], art[f"{key}.biases"] = q, s, b
    src[key] = (w.dtype, mx.dequantize(q, s, b, group_size=64, bits=8).astype(w.dtype))
rest = {f"P.{k}": v for k, v in tree_flatten(P.parameters()) if k.split(".")[0] not in LIN}
mx.eval(art, rest)
# 2) MoonShots' own cut on the same in-memory model (in place), then the bitwise check
done = P3.apply_cut(mx, P3.CUTS["Q8"], P, head)
ver = {}
for key, lin in mods:
    ref = lin.weight
    ok = bool(mx.array_equal(src[key][1], ref).item()) and src[key][1].dtype == ref.dtype
    ver[key] = {"bitwise_equal_to_apply_cut": ok, "dtype": str(ref.dtype), "shape": list(ref.shape)}
allok = all(v["bitwise_equal_to_apply_cut"] for v in ver.values())
print(f"apply_cut Q8 changed: {done}", flush=True)
print(f"VERIFY: {sum(v['bitwise_equal_to_apply_cut'] for v in ver.values())} / {len(ver)} modules bitwise equal -> "
      + ("PASS" if allok else "FAIL"), flush=True)
fo = os.path.join(OUT, "t30b_q8h32.safetensors")
mx.save_safetensors(fo, {**art, **rest})
meta = {"source": {"dd_t30b.safetensors": sha(CK), "mtp125.py": sha(H("~/TrainLab/mtp125.py")),
                   "p3_dd_tree.py (MoonShots, the reference cut)": sha(H("~/MoonShots/week98/p3_dd_tree.py"))},
        "recipe": {"modules": [k for k, _ in mods], "quantize": "mx.quantize(w.astype(bfloat16), group_size=64, bits=8)",
                   "dequantize_to": "the module's stored dtype", "router_bits": 8, "routed_experts": "as stored (125B)",
                   "top_k": 10, "se_gatew": "as stored in t30b",
                   "screen": {"rank_file": P3.RANK_F, "rank_sha256": sha(P3.RANK_F), "N": 32768,
                              "session_extras": "prompt + committed answer up to the anchor, first appearance",
                              "extra_cap": getattr(P3, "EXTRA_CAP", 4096)}},
        "expected_kept_TRUE_STATE (MoonShots q8h32early_scores.txt, Q8H32 vs the full head)":
            {"HE+ tree-16": 0.9775, "HE+ tree-32": 0.9751, "MBPP tree-16": 0.9981, "MBPP tree-32": 1.0046},
        "verify": ver, "verify_pass": allok, "artifact_sha256": sha(fo)}
json.dump(meta, open(fo.replace(".safetensors", ".json"), "w"), indent=1)
print(f"-> {fo} sha256 {meta['artifact_sha256']}", flush=True)
sys.exit(0 if allok else 2)
