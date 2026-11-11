"""L3 checks (CPU only): (1) every fine-tuned tensor reaches the eval's drafter (load_weights strict=False would skip a
mismatched name silently); (2) token overlap between the val and train windows (resumed sessions repeat content)."""
import json, os, sys
import numpy as np, mlx.core as mx
from mlx.utils import tree_flatten
mx.set_default_device(mx.cpu)
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_card/Leap_third_party"))
import dflash_model_mlx as D

ft = mx.load(os.path.expanduser("~/Leap/data/dfl2c_ft.safetensors"))
draft = D.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
orig = dict(tree_flatten(draft.parameters()))
trained = [k for k in orig if k.startswith(("layers.", "norm.", "candidate_selector."))]
miss_model = [k for k in ft if k not in orig]
miss_file = [k for k in trained if k not in ft]
bad_shape = [k for k in ft if k in orig and ft[k].shape != orig[k].shape]
print(f"file {len(ft)} tensors; trained modules in the model {len(trained)}; file keys not in model {len(miss_model)}; "
      f"trained keys not in file {len(miss_file)}; shape mismatches {len(bad_shape)}")
print("examples:", miss_model[:3], miss_file[:3], bad_shape[:3])
draft.load_weights(os.path.expanduser("~/Leap/data/dfl2c_ft.safetensors"), strict=False)
now = dict(tree_flatten(draft.parameters()))
changed = sum(1 for k in ft if k in now and bool(mx.any(now[k] != orig[k]).item()))
same_as_file = sum(1 for k in ft if k in now and bool(mx.all(now[k] == ft[k].astype(now[k].dtype)).item()))
rel = [float(mx.abs(ft[k].astype(mx.float32) - orig[k].astype(mx.float32)).mean() / (mx.abs(orig[k].astype(mx.float32)).mean() + 1e-12))
       for k in ft if k in orig and ft[k].ndim == 2]
print(f"after load: {changed}/{len(ft)} tensors differ from the original, {same_as_file}/{len(ft)} equal the file; "
      f"mean relative change of 2-D weights {np.mean(rel):.4f} (max {np.max(rel):.4f})")

wins = [json.loads(l) for l in open(os.path.expanduser("~/Leap/data/dfl2c_windows.jsonl"))]
def grams(ids, n=32):
    return {tuple(ids[i:i + n]) for i in range(0, len(ids) - n + 1)}
tr = set().union(*[grams(w["ids"]) for w in wins if w.get("split") == "train"])
for w in wins:
    if w.get("split") == "val":
        g = grams(w["ids"])
        print(f"val {w['key'][-12:]} ({w['project'][-20:]}): {len(w['ids'])} tokens, "
              f"{sum(x in tr for x in g) / max(1, len(g)):.3f} of its 32-grams also in train")
