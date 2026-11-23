"""D-15a LANGPROBE packing (MTP Head, Thu 10-29; REPORT 'D-15a'): D-02a's 24 multilingual captures (data/d02a/
tf_multi_*.npz: f taps, tk, lp, ids, off 0, gen) -> the trainer's window format (dfl2c_extract.py's packing: h =
incoai's frozen hidden_norm(fc(taps)) as bf16 bits, n = len(ids) - 1 rows, tk / lp / ids / asst (positions >= gen) /
split). The third answer of each language (bg2 de5 fr8 es11 ru14 zh17 ja20 ar23) is held out: split "val", and the
walk set data/d15_held/ links its D-02a file. Output data/d15_multi/*.npz (16 train + 8 val).
Usage: [BigMoE venv] python sticky/d15_pack.py"""
import glob, os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Leap/third_party"))
import dflash_model_mlx as D

TD = os.path.expanduser("~/TrainLab/data")
HELD = {"bg2", "de5", "fr8", "es11", "ru14", "zh17", "ja20", "ar23"}
out, held = os.path.join(TD, "d15_multi"), os.path.join(TD, "d15_held")
os.makedirs(out, exist_ok=True); os.makedirs(held, exist_ok=True)
draft = D.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")          # only fc + hidden_norm are used (frozen in L1-20)
nt = nv = 0
for p in sorted(glob.glob(os.path.join(TD, "d02a", "tf_multi_*.npz"))):
    z = np.load(p); key = os.path.basename(p)[len("tf_multi_"):-4]
    assert int(z["off"]) == 0, p
    ids = z["ids"].astype(np.int32); n = len(ids) - 1; gen = int(z["gen"])
    taps = mx.array(z["f"][:n].reshape(n, -1))
    h = mx.concatenate([draft.hidden_norm(draft.fc(taps[i:i + 1024].astype(mx.bfloat16))) for i in range(0, n, 1024)])
    mx.eval(h)
    hb = (np.array(h.astype(mx.float32)).view(np.uint32) >> 16).astype(np.uint16)
    asst = (np.arange(len(ids)) >= gen).astype(np.uint8)
    split = "val" if key in HELD else "train"
    np.savez(os.path.join(out, os.path.basename(p)), h=hb, tk=z["tk"][:n], lp=z["lp"][:n], ids=ids, asst=asst, split=split)
    if split == "val":
        nv += 1
        dst = os.path.join(held, os.path.basename(p))
        if not os.path.exists(dst):
            os.symlink(p, dst)
    else:
        nt += 1
print(f"[d15-pack] {nt} train + {nv} held-out windows -> {out}; walk set {held}")
