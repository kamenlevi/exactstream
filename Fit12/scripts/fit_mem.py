"""Fit12 M1 (2026-10-23): where B12LL's non-MLX memory goes (peak footprint minus MLX peak: ~1.0 GB at 2.6K, ~1.4 GB at 8K).
Loads B12LL as fit_b12 (suite env + PF_GROUP + ASYNC_EVERY), prefills the ~2.6K-token prompt (decoupled 512/4096), decodes 64
tokens (MTP 3:3:0), then runs macOS `footprint` and `vmmap --summary` on itself and keeps their category tables.
Usage: [BigMoE venv] fit_mem.py OUT_DIR"""
import ctypes, os, subprocess, sys, time

ENV = {"MTPLX_PREFILL_CHUNK_SIZE": "512", "BIGMOE_TIER_PF_GROUP": "64", "BIGMOE_RESET_GEN": "0", "BIGMOE_PF_GROUP": "64",
       "B12_LATE3": "2,10", "B12_NORM": "mass", "BIGMOE_MTP_STREAM": "1", "BIGMOE_MTP_C": "64", "BIGMOE_MAXK": "80",
       "BIGMOE_FRSPEC": "1", "MTPLX_FAMILY_CAPTURE_COMMIT": "1", "MTPLX_DROP_EVENTS": "1", "MTPLX_LAZY_MTP_HISTORY_APPEND": "1",
       "MTPLX_DEFER_REPAIR_EVAL": "1", "BIGMOE_LOOP_GUARD": "0", "B12_CACHE_MB": "64", "BIGMOE_ASYNC_EVERY": "1"}
for k, v in ENV.items():
    os.environ.setdefault(k, v)
OUT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/Fit12/data/m1")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.expanduser("~/Breakthrough/scripts"))
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
import bt_b12 as B


class _RU(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(40)]


def fp():
    ru = _RU()
    ok = ctypes.CDLL(None).proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru)) == 0
    return (ru.u7 / 1e9, ru.u28 / 1e9) if ok else (float("nan"), float("nan"))


def snap(tag):
    import mlx.core as mx
    now, peak = fp()
    print(f"[m1] {tag}: footprint now {now:.2f} GB, lifetime peak {peak:.2f} GB; MLX active {mx.get_active_memory() / 1e9:.2f}, "
          f"peak {mx.get_peak_memory() / 1e9:.2f}, cache {mx.get_cache_memory() / 1e9:.2f} GB", flush=True)
    for tool, args in (("footprint", ["/usr/bin/footprint", str(os.getpid())]),
                       ("vmmap", ["/usr/bin/vmmap", "--summary", str(os.getpid())])):
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=120)
            open(os.path.join(OUT, f"{tag}_{tool}.txt"), "w").write(r.stdout + r.stderr)
        except Exception as e:                                      # noqa: BLE001 (diagnostic only)
            print(f"[m1] {tool} failed: {e}", flush=True)


rt, st = B.load_b12(0.10, mtp=True)
import mlx.core as mx
import spec125
from bigmoe_v3f.fuse import reset_generation_state
model, tok = rt.model, rt.tokenizer
snap("loaded")
text = ("Here is a Python script. Summarise what it does, section by section.\n\n```python\n"
        + open(os.path.expanduser("~/Breakthrough/scripts/tf_frugal_var.py")).read()[:7600] + "\n```")
ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
ids = list(ids if isinstance(ids[0], int) else ids["input_ids"])
reset_generation_state(model); B.STASH.clear(); B.PENDH.clear()
state = B.b12_prefill_state(model, ids, 512, 4096)
mx.eval(state["logits"])
snap("prefilled")
out, s = spec125.generate(model, ids, 64, k_mtp=3, max_draft=3, lookup=0, prefill_chunk=512, prefilled=state, stats={})
st.gate.check()
snap("decoded")
print(f"[m1] done: {len(ids) - 1} prompt tokens, {len(out)} decoded at {s['tok_s']:.1f} tok/s", flush=True)
