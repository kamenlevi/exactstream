"""One-command A/B for BigMoE: layer-major prefill (PF2) vs BigMoE's chunk-major greedy prefill on the streamed 125B.
usage (from any dir, BigMoE's venv):  BIGMOE_PF_PREFILL=1 ~/BigMoE/.venv/bin/python ~/MoonShots/patches/layer_major_prefill/bench_lmp125.py FRAC PROMPT_TOKENS [N]
  e.g. 0.5 4096 / 0.5 16384. Runs ref (bigmoe.decode.greedy) then lmp (lmp.greedy_lmp) in the same process (LMP_MODES=lmp,pf3,pf3
  for PF3 vs PF2 plus a PF3 repeat: PF3 batches each layer's MoE over PF3_MOE_TOKENS=8192 tokens), each from a fresh
  cache; prints prefill+first-token time, experts staged, and the token sha (must match). Read-only on ~/BigMoE (imports only)."""
import hashlib, os, sys, time
os.environ.setdefault("BIGMOE_PF_PREFILL", "1")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import mlx.core as mx
from bigmoe.loader import load_streamed
from bigmoe import decode as D
from bigmoe.stream import _STAGE
import lmp

frac = float(sys.argv[1]); PT = int(sys.argv[2]); N = int(sys.argv[3]) if len(sys.argv) > 3 else 32
MD = os.path.expanduser("~/BigMoE/models/bare")
cfg = json.load(open(os.path.join(MD, "config.json"))); cfg = cfg.get("text_config", cfg)
rt, st = load_streamed(MD, max(16, round(frac * cfg["num_experts"])), mtp=False)
text = ""
for p in ("~/SpecDecode/FINDINGS.md", "~/MoEPrefetch/FINDINGS.md", "~/BigMoE/FINDINGS.md", "~/SpecDecode/server.py",
          "~/SpecDecode/specdec.py", "~/BigMoE/bigmoe/stream.py", "~/MoonShots/IDEAS.md"):
    text += open(os.path.expanduser(p)).read() + "\n\n"
ids = rt.tokenizer.encode(text)[:PT]
print(f"prompt {len(ids)} tokens, C={round(frac*cfg['num_experts'])}/512", flush=True)
MODES = {"ref": D.greedy, "lmp": lmp.greedy_lmp, "pf3": lmp.greedy_pf3}
for mode, fn in [(m, MODES[m]) for m in os.environ.get("LMP_MODES", "ref,lmp").split(",")]:
    t0 = time.perf_counter(); first = []
    _STAGE["t"] = {}
    out = fn(rt.model, ids, N, on_token=lambda t: first or first.append(time.perf_counter()))
    t_first = first[0] - t0
    staged = _STAGE["t"].get("staged_experts", "n/a")
    sha = hashlib.sha1(str(out).encode()).hexdigest()[:10]
    print(f"{mode}: prefill+first token {t_first:6.1f} s ({len(ids)/t_first:5.0f} tok/s), {N} tokens sha {sha}, "
          f"peak {mx.get_peak_memory()/1e9:.1f} GB", flush=True)
    mx.clear_cache()
