"""Fit12 P1 (2026-10-23) / T1 (2026-10-24): B12LL prefill and memory inside 12 GB. One mode per process (honest footprints):
  decoupled bt_b12.b12_prefill_state(s_attn 512, s_moe F12_SMOE) [+ D1I_STREAM=3 D1I_RING=R: TrainLab's fixed-order expert
  streaming], on a ~2.6K-token and a ~8K-token prompt (Python source to summarise, as bt_b12's 'long' / ctx32k prompts).
Per prompt: prefill seconds and tok/s, MLX peak, process peak footprint; then 32 decode tokens (spec125 3:3:0, B12LL) for a sha
(decode tokens must not depend on the prefill mode or the memory trims) and a rough decode speed.
A sampler thread (every 20 ms) keeps the peak physical footprint and MLX active memory per phase (load / <prompt> prefill /
<prompt> decode), so a transient can be placed. Trims (fit_trim.py): F12_EMBED=1, F12_HCPRE=1. Slots: B12_CM / B12_CL.
Env: the suite's b12ll env + BIGMOE_PF_GROUP=64 + ASYNC_EVERY=1 (as fit_b12). Usage: [BigMoE venv] fit_pf.py TAG"""
import ctypes, hashlib, os, sys, threading, time

ENV = {"MTPLX_PREFILL_CHUNK_SIZE": "512", "BIGMOE_TIER_PF_GROUP": "64", "BIGMOE_RESET_GEN": "0", "BIGMOE_PF_GROUP": "64",
       "B12_LATE3": "2,10", "B12_NORM": "mass", "BIGMOE_MTP_STREAM": "1", "BIGMOE_MTP_C": "64", "BIGMOE_MAXK": "80",
       "BIGMOE_FRSPEC": "1", "MTPLX_FAMILY_CAPTURE_COMMIT": "1", "MTPLX_DROP_EVENTS": "1", "MTPLX_LAZY_MTP_HISTORY_APPEND": "1",
       "MTPLX_DEFER_REPAIR_EVAL": "1", "BIGMOE_LOOP_GUARD": "0", "B12_CACHE_MB": "64", "BIGMOE_ASYNC_EVERY": "1"}
for k, v in ENV.items():
    os.environ.setdefault(k, v)
TAG = sys.argv[1] if len(sys.argv) > 1 else "pf"
SMOE = int(os.environ.get("F12_SMOE", "4096"))


class _RU(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(40)]


_LIBC = ctypes.CDLL(None)


def fp():                                            # (now, lifetime peak) physical footprint, GB
    ru = _RU()
    return (ru.u7 / 1e9, ru.u28 / 1e9) if _LIBC.proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru)) == 0 else (float("nan"),) * 2


def fp_peak():
    return fp()[1]


import mlx.core as mx
PH = {"name": "load"}
PK = {}                                              # phase -> [max footprint, max MLX active, samples]


SNAP = os.environ.get("F12_FPSNAP") == "1"       # footprint(1) category table near each prefill phase's peak (every ~2 s)
SNAPS = {}                                           # phase -> (footprint GB when taken, footprint(1) text)


def _snap(ph, f):
    import subprocess
    try:
        r = subprocess.run(["/usr/bin/footprint", str(os.getpid())], capture_output=True, text=True, timeout=30)
        if f >= SNAPS.get(ph, (0.0, ""))[0]:
            SNAPS[ph] = (f, r.stdout)
    except Exception:                                # noqa: BLE001 (diagnostic only)
        pass


def _sampler():
    last = 0.0
    while True:
        f = fp()[0]
        a = mx.get_active_memory() / 1e9
        ph = PH["name"]
        d = PK.setdefault(ph, [0.0, 0.0, 0])
        d[0], d[1], d[2] = max(d[0], f), max(d[1], a), d[2] + 1
        if SNAP and ph.endswith("prefill") and time.time() - last > 2.0 and f >= 0.97 * d[0]:
            last = time.time()
            threading.Thread(target=_snap, args=(ph, f), daemon=True).start()
        time.sleep(0.02)


threading.Thread(target=_sampler, daemon=True).start()
sys.path.insert(0, os.path.expanduser("~/Breakthrough/scripts"))
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
import bt_b12 as B
if os.environ.get("F12_HC8LW") == "1":         # layer-wise hc8 install (bitwise the same weights, smaller load transient)
    sys.path.insert(2, os.path.expanduser("~/Fit12/scripts"))
    import fit_hc8lw
    fit_hc8lw.apply(B.V3F)
sys.path.insert(2, os.path.expanduser("~/Fit12/scripts"))
import fit_trim                                # L2-02/03: F12_EMBED=1 (streamed embed), F12_HCPRE=1 (hc8 before the eval)
fit_trim.from_env(B)
rt, st = B.load_b12(float(os.environ.get("F12_FRAC", "0.10")), mtp=True)
import spec125
from bigmoe_v3f.fuse import reset_generation_state
model, tok = rt.model, rt.tokenizer
time.sleep(0.1)
PH["name"] = "idle"


def chat_ids(text):
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    return list(ids if isinstance(ids[0], int) else ids["input_ids"])


HEAD = "Here is a Python script. Summarise what it does, section by section.\n\n```python\n"
PROMPTS = {"p2k": lambda: HEAD + open(os.path.expanduser("~/Breakthrough/scripts/tf_frugal_var.py")).read()[:7600] + "\n```",
           "p8k": lambda: HEAD + open(os.path.expanduser("~/Breakthrough/data/prompts/ctx32k.txt")).read()[:26000] + "\n```"}
sha = lambda t: hashlib.sha1(str(list(t)).encode()).hexdigest()[:10]
n0, p0 = fp()
print(f"[{TAG}] mode: s_moe {SMOE}, stream {os.environ.get('D1I_STREAM', '0')} ring {os.environ.get('D1I_RING', '-')}, "
      f"C_m {B.CFG['cm']} C_l {B.CFG['cl']}, trims {fit_trim.STATS or 'none'} | after load: active "
      f"{mx.get_active_memory() / 1e9:.2f} GB, footprint now {n0:.2f}, lifetime peak {p0:.2f} GB", flush=True)
for P in os.environ.get("F12_PROMPTS", "p2k,p8k").split(","):
    ids = chat_ids(PROMPTS[P]())
    reset_generation_state(model); B.STASH.clear(); B.PENDH.clear()
    mx.reset_peak_memory()
    PH["name"] = f"{P} prefill"
    t0 = time.perf_counter()
    state = B.b12_prefill_state(model, ids, 512, SMOE)
    mx.eval(state["logits"])
    tp = time.perf_counter() - t0
    pk = mx.get_peak_memory() / 1e9
    PH["name"] = f"{P} decode"
    out, s = spec125.generate(model, ids, 32, k_mtp=3, max_draft=3, lookup=0, prefill_chunk=512, prefilled=state, stats={})
    st.gate.check()
    now = fp()[0]
    print(f"[{TAG}] {P}: prefill {len(ids) - 1} tokens in {tp:.2f} s = {(len(ids) - 1) / tp:.0f} tok/s | MLX peak {pk:.2f} GB, "
          f"process peak footprint {fp_peak():.2f} GB, now {now:.2f} GB | decode 32: {s['tok_s']:.1f} tok/s, sha {sha(out)}",
          flush=True)
    del state
    mx.clear_cache()
    PH["name"] = "idle"
    time.sleep(0.1)
    print(f"[{TAG}] {P}: after del state + clear_cache: footprint now {fp()[0]:.2f} GB", flush=True)
time.sleep(0.1)
for ph, (f, a, n) in PK.items():
    print(f"[{TAG}] phase {ph}: peak footprint {f:.2f} GB, peak MLX active {a:.2f} GB ({n} samples)", flush=True)
print(f"[{TAG}] embed gathers {fit_trim.fit_embed.STATS}", flush=True)
for ph, (f, txt) in SNAPS.items():                 # the category table taken nearest each prefill peak
    keep = [l for l in txt.splitlines() if ("MB" in l or "GB" in l) and not l.strip().startswith("0 B")][:14]
    print(f"[{TAG}] footprint(1) during {ph} at {f:.2f} GB:\n" + "\n".join(keep), flush=True)
print(f"[{TAG}] done", flush=True)
