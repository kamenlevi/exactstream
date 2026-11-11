"""RAM floor of the streamed 125B with no experts (TASKS.md dense diet, Breakthrough 2026-10-21): load at the minimum
cache (C = 16), apply dense changes, decode 32 tokens (fills the n-gram hot cache and the KV), then report MLX active and
peak memory, the process's resident set (ps rss) and the expert pools; floor = total - pools.
usage: floor_diet.py [MODS]   MODS: comma list of hc8 | hc4 | lm3 (env MTPLX_NGRAM_HOT_MB sets the n-gram hot cache)"""
import json, os, subprocess, sys, time
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
os.environ["BIGMOE_SPARE"] = "0"
os.environ["BIGMOE_PF_NOPUBLISH"] = "1"
sys.path.insert(0, os.path.expanduser("~/Breakthrough/fork125bt"))
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from bigmoe.loader import load_streamed
from bigmoe import decode as D

MODS = [m for m in (sys.argv[1] if len(sys.argv) > 1 else "").split(",") if m]
MD = os.path.expanduser("~/BigMoE/models/bare")
rt, st = load_streamed(MD, 16, mtp=False)
if "hc8" in MODS or "hc4" in MODS:
    sys.path.insert(0, os.path.expanduser("~/MoonShots/patches/hc8"))
    import hc8
    print("hc", hc8.install(rt.model, bits=8 if "hc8" in MODS else 4), flush=True)
if "lm3" in MODS:
    TM = getattr(rt.model, "language_model", rt.model)
    for name, mod in TM.named_modules():
        if name.endswith("lm_head") and isinstance(mod, nn.QuantizedLinear):
            W = mx.dequantize(mod.weight, mod.scales, mod.biases, mod.group_size, mod.bits)
            lin = nn.Linear(W.shape[1], W.shape[0], bias=False); lin.weight = W
            q = nn.QuantizedLinear.from_linear(lin, 64, 3)
            parent = TM
            for part in name.split(".")[:-1]:
                parent = getattr(parent, part)
            setattr(parent, name.split(".")[-1], q)
    mx.eval(TM.parameters())
mx.clear_cache()
ids = rt.tokenizer.apply_chat_template([{"role": "user", "content": "Explain briefly how a CPU cache works."}],
                                       add_generation_prompt=True, enable_thinking=False)
ids = list(ids if isinstance(ids[0], int) else ids["input_ids"])
D.greedy(rt.model, ids, 32)
mx.eval()
rss = int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout) / 1e6
act, peak, pools = mx.get_active_memory() / 1e9, mx.get_peak_memory() / 1e9, st.resident_gb()
print(f"floor [{','.join(MODS) or 'base'} hot={os.environ.get('MTPLX_NGRAM_HOT_MB', '1024')} MB]: MLX active {act:.2f} GB, "
      f"peak {peak:.2f} GB, process RSS {rss:.2f} GB, expert pools {pools:.2f} GB -> floor active {act - pools:.2f} / "
      f"RSS {rss - pools:.2f} GB", flush=True)
