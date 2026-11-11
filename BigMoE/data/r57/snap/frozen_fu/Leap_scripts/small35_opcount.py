"""L1-15 follow-up (Leap, Sat 2026-10-24): how many MLX primitives one 16-row verify pass of the 35B dispatches (the loop's
graph: model forward + taps + the GDN capture + argmax), stock vs DFL35_GDNK=1. Counted from mx.export_to_dot (unevaluated
graph nodes that are primitives, i.e. ops; inputs and constants excluded).
Usage: [SpecDecode venv] small35_opcount.py  -> appends to ~/Leap/data/mlx_opcost.txt"""
import json, os, re, sys, tempfile
os.environ["HF_HUB_OFFLINE"] = "1"
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_fu/Leap_third_party/sd35"))
import dflash35_model_mlx as dfm
from mlx_lm.models.cache import make_prompt_cache
MODEL = os.path.expanduser("~/.cache/huggingface/hub/models--Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed/snapshots/"
                           "95460516fbfe34c48f577f67c9ed191f5d61fbaf")
model, tok = dfm.load(MODEL)
mx.eval(model.parameters())
ids = tok.apply_chat_template([{"role": "user", "content": "Write a Python function that reverses a string."}],
                              add_generation_prompt=True, enable_thinking=False)
ids = list(ids if isinstance(ids[0], int) else ids["input_ids"])
cache = make_prompt_cache(model)
mx.eval(model(mx.array([ids]), cache=cache))
dfm._patch_model(model, [1, 6, 11, 16, 22, 27, 32, 37])
inp = mx.array([ids[-16:]])
lines = []
for gdnk in ("0", "1"):
    os.environ["DFL35_GDNK"] = gdnk
    cap = dfm._GDNStateCapture()
    try:
        cap.clear()
        lg = model(inp, cache=cache)
        outs = [mx.argmax(lg, -1)] + [x for x in model._hidden_states if x is not None]
        f = tempfile.NamedTemporaryFile("w", suffix=".dot", delete=False).name
        mx.export_to_dot(f, *outs)
        txt = open(f).read(); os.unlink(f)
        nodes = re.findall(r'^\s*(\S+)\s*\[label\s*=\s*"([^"]*)"', txt, re.M)
        prims = [lab for _, lab in nodes if not re.match(r"^\s*(\{|\(|[0-9])", lab)]
        from collections import Counter
        top = Counter(re.split(r"[\s(]", lab.strip())[0] for lab in prims).most_common(12)
        lines.append(f"verify pass graph, GDN kernels {'on ' if gdnk == '1' else 'off'}: {len(nodes)} DOT nodes, ~{len(prims)} op nodes; "
                     f"top: {top}")
        print(lines[-1], flush=True)
    finally:
        cap.close()
    for c in cache:                                            # discard this pass's cache changes
        if hasattr(c, "trim") and c.is_trimmable():
            c.trim(16)
open(os.path.expanduser("~/Leap/data/mlx_opcost.txt"), "a").write("\n".join(lines) + "\n")
