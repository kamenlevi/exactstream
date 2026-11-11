"""MTP speculative decoding on Qwen3.6-35B-A3B 4-bit with streamed experts, reusing SpecDecode's
verify loop (GatedDeltaNet rollback) and MTP head (~/SpecDecode, read-only import).
usage: bench35_mtp.py ref|stream FRAC N K [ALLOC]   (K = MTP draft depth; 0 = plain; -n = lookup<=n else MTP k=2)
Compare streamed vs ref at the same K: tokens must be identical (same kernels).
-> data/runs35/mtp_<tag>.json + one summary line"""
import json, os, sys, time, hashlib
sys.path.insert(0, os.path.expanduser("~/BigMoE")); sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
import mlx.core as mx
import mlx_lm.utils as mu
import specdec, mtp as mtpmod
mtpmod.MTP_PATH = os.path.expanduser("~/SpecDecode/data/mtp/mtp.safetensors")
MD = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit")
mode, frac, N, K = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
alloc = sys.argv[5] if len(sys.argv) > 5 else "uniform"
E, NL = 256, 40
st = None
if mode == "stream":
    from bigmoe.stream import ExpertStreamer
    if alloc == "ends":
        full = set(range(0, 11)) | {35, 36, 38, 39}
        rest = round((frac * E * NL - E * len(full)) / (NL - len(full)))
        C = [E if L in full else rest for L in range(NL)]
    else:
        C = round(frac * E)
    st = ExpertStreamer(MD, C)
    orig = mu.load_model
    def lazy(path, *a, **kw):
        a = list(a)
        if a: a[0] = True
        else: kw["lazy"] = True
        m, cfg = orig(path, *a, **kw)
        st.install(m); mx.eval(m.parameters()); return m, cfg
    mu.load_model = lazy
m = specdec.Model(MD)
if st: mu.load_model = orig
drafter = None
if K != 0:
    m.cap = mtpmod.install_capture(m.model)
    head = mtpmod.load_mtp(m.model, bits=4)
    if K > 0:
        drafter = specdec.MTPDrafter(m, head, k=K)
    else:   # K = -n: prompt lookup (drafts <= n tokens, so a pass fits the gate's 40-id row) else MTP k=2
        drafter = specdec.LookupMTP(specdec.PromptLookup(3, 1, -K), specdec.MTPDrafter(m, head, k=2))
tok = m.tokenizer
prompt = open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read()
ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=False)
ids = list(ids)
t0 = time.perf_counter()
hook_of = (drafter.prefill_hook if hasattr(drafter, "prefill_hook") else drafter.mtpd.prefill_hook) if drafter else None
m.prefill(ids[:-1], on_chunk=hook_of(ids) if hook_of else None)
t_pre = time.perf_counter() - t0
from bigmoe.gated import ext
ext.reset_stats()
state = {}; out = []; t0 = time.perf_counter()
for new in specdec.generate_iter(m, ids, N, drafter, None, 1, state):
    out += new
dec = time.perf_counter() - t0
s = ext.stats(); passes = state["forwards"]
tag = f"{mode}_{frac:.2f}_{alloc}_K{K}_N{N}" if st else f"ref_K{K}_N{N}"
h = hashlib.sha1(str(out).encode()).hexdigest()[:10]
res = dict(mode=mode, frac=frac, alloc=alloc, K=K, N=N, tokens=out, prefill_s=t_pre, decode_tok_s=len(out) / dec,
           passes=passes, tok_per_pass=len(out) / passes, peak_gb=mx.get_peak_memory() / 1e9, stats=s)
os.makedirs(os.path.expanduser("~/BigMoE/data/runs35"), exist_ok=True)
json.dump(res, open(os.path.expanduser(f"~/BigMoE/data/runs35/mtp_{tag}.json"), "w"), indent=1)
extra = (f" | per pass: misses {s['misses']/passes:.1f}, stalled layers {s['stalled']/passes:.1f}, io {s['io_s']/passes*1e3:.2f} ms"
         if st else "")
if st and s.get("pf_issued"):
    print(f"   prefetch: issued {s['pf_issued']}, used {s['pf_used']}, waited {s['pf_waited_s']*1e3:.1f} ms total", flush=True)
print(f"{tag}: {len(out)/dec:.1f} tok/s, {len(out)/passes:.2f} tok/pass, prefill {t_pre:.1f}s, peak {mx.get_peak_memory()/1e9:.1f} GB, "
      f"sha {h}{extra}", flush=True)
