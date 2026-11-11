"""R-78b installed check (Leap, Wed 2026-11-04 14:1x): draft_fused.install(forms=True) on the real L1-20 drafter; every
patched projection of all 5 layers called through its installed class at rows 8 / 16 / 17 / 32 == x @ W.T bitwise
(form C's W rebuilt from its kept Wt; form B's own weight); and which rows took the form (FORM_ROWS) vs the fallback.
Usage: [BigMoE venv] r78b_installed.py -> ~/Leap/data/r78b_rows.txt (appends; GPU ~1.5 GB, share lock, ~30 s)"""
import os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs4/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs4/Leap_third_party"))
import dflash_model_mlx as DF
import draft_fused as DFU
OUT = os.path.expanduser("~/Leap/data/r78b_rows.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
draft = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
draft.load_weights(os.path.expanduser("~/Leap/data/l120_ft.safetensors"), strict=False)
mx.eval(draft.parameters())
DFU.install(forms=True, draft=draft)
b16 = lambda x: np.array(x.view(mx.uint16))
say(f"\nR-78b installed {time.strftime('%a %H:%M:%S')}: FORM_ROWS {min(DFU.FORM_ROWS)}..{max(DFU.FORM_ROWS)} ({len(DFU.FORM_ROWS)}); "
    f"{len(DFU._FORMS)} patched projections")
ok, n = True, 0
mx.random.seed(17)
for r in (8, 16, 17, 32):
    bad = []
    for kind, lin, cls in DFU._FORMS:
        W = mx.contiguous(lin._wt.T) if kind == "C" else lin["weight"]
        x = mx.random.normal((1, r, W.shape[1])).astype(mx.bfloat16)
        y, ref = lin(x), x @ W.T
        mx.eval(y, ref)
        n += 1
        if not (b16(y) == b16(ref)).all():
            bad.append(f"{kind}:{tuple(W.shape)}")
    ok &= not bad
    say(f"  rows {r}: {'form' if r in DFU.FORM_ROWS else 'fallback'}; {'all bitwise' if not bad else 'DIFFERS ' + ', '.join(bad)}")
DFU.uninstall()
say(f"R-78b installed: {'PASS' if ok else 'FAIL'} ({n} calls)")
