"""PORT Kaggle upload folder (MTP Head, Wed 11-04 20:4x). Writes the weights dataset the Kaggle runner reads, with no
network use: config.json + model.safetensors (incoai DFlash2, from the local HF cache) and target_embed_head.safetensors
(the 125B's 4-bit embedding + LM head: the 6 tensors of model-00016-of-00017, ~0.72 GB, tags kept). Uploading it needs
Kamen's Kaggle account (created only when a probe passes) and his OK for the weights' licenses.
Usage: python make_upload.py OUT_DIR"""
import os, shutil, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import dflash2_torch as M
out = os.path.expanduser(sys.argv[1]); os.makedirs(out, exist_ok=True)
for f in ("config.json", "model.safetensors"):
    shutil.copyfile(os.path.join(M.DRAFT_DIR, f), os.path.join(out, f))
keys = {f"language_model.{p}.{s}" for p in ("model.embed_tokens", "lm_head") for s in ("weight", "scales", "biases")}
M.st_save_raw(os.path.join(out, "target_embed_head.safetensors"), M.st_load(M.TARGET_SHARD, keys))
print("wrote", out, sorted(os.listdir(out)), f"{sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)) / 1e9:.2f} GB")
