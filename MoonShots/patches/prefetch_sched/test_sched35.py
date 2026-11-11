"""Exactness + mechanics check of the prefetch scheduler on the SSD-streamed Qwen3.6-35B-A3B (fits in RAM, cheap).
usage: test_sched35.py base|sched FRAC N K
  base   BigMoE's own package and ext (~/BigMoE, imports only)
  sched  the patched copy built by mk_tree.sh (bigmoe package + ext/build with this patch), same entry points
  K      MTP draft depth (0 = plain greedy; 3 = verify passes of 4 rows, the 125B's spec k=3 shape)
Flags come from the environment (BIGMOE_PREFETCH*, BIGMOE_PF_*). Every variant must print the same token sha as base.
-> ~/MoonShots/data/sched35/<tag>.json + one summary line. Nothing is written under ~/BigMoE."""
import hashlib, json, os, sys, time
code, frac, N, K = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
TREE = os.environ.get("SCHED_TREE", "/private/tmp/claude-501/-Users-kamen-MoonShots/3c6677ac-9553-4d46-9a7d-3f8008b167c0/scratchpad/bm")
sys.path.insert(0, TREE if code == "sched" else os.path.expanduser("~/BigMoE"))
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
import mlx.core as mx
import mlx_lm.utils as mu
import specdec, mtp as mtpmod
from bigmoe.stream import ExpertStreamer
from bigmoe.gated import ext
assert os.path.realpath(ext.__file__).startswith(os.path.realpath(TREE)) == (code == "sched"), ext.__file__
mtpmod.MTP_PATH = os.path.expanduser("~/SpecDecode/data/mtp/mtp.safetensors")
MD = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit")
st = ExpertStreamer(MD, round(frac * 256))
orig = mu.load_model


def lazy(path, *a, **kw):
    a = list(a)
    if a: a[0] = True
    else: kw["lazy"] = True
    m, cfg = orig(path, *a, **kw)
    st.install(m); mx.eval(m.parameters()); return m, cfg


mu.load_model = lazy
m = specdec.Model(MD)
mu.load_model = orig
drafter = None
if K > 0:
    m.cap = mtpmod.install_capture(m.model)
    drafter = specdec.MTPDrafter(m, mtpmod.load_mtp(m.model, bits=4), k=K)
tok = m.tokenizer
prompt = open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read()
ids = list(tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=False))
t0 = time.perf_counter()
m.prefill(ids[:-1], on_chunk=drafter.prefill_hook(ids) if drafter else None)
t_pre = time.perf_counter() - t0
if os.environ.get("POISON"):                      # LC1: fill MLX's free-buffer cache with NaN / 0xFF before the generation
    _w = {"nan": 0x7FC00000, "ff": 0xFFFFFFFF}[os.environ["POISON"]]; _keep = []; _tot = 0
    for _sz in [4096 << i for i in range(17)]:
        for _ in range(2):
            if _tot + _sz <= (900 << 20):
                _a = mx.full((_sz // 4,), _w, dtype=mx.uint32); mx.eval(_a); _keep.append(_a); _tot += _sz
    del _keep
    print(f"poisoned {_tot >> 20} MB of free buffers with {os.environ['POISON']}", flush=True)
ext.reset_stats()
state, out = {}, []
t0 = time.perf_counter()
for new in specdec.generate_iter(m, ids, N, drafter, None, 1, state):
    out += new
dec = time.perf_counter() - t0
s = ext.stats(); P = state["forwards"]
flags = " ".join(f"{k[7:]}={v}" for k, v in sorted(os.environ.items()) if k.startswith("BIGMOE_"))
tag = f"{code}_{frac:.2f}_K{K}_N{N}_" + hashlib.sha1(flags.encode()).hexdigest()[:6]
h = hashlib.sha1(str(out).encode()).hexdigest()[:10]
od = os.path.expanduser("~/MoonShots/data/sched35"); os.makedirs(od, exist_ok=True)
json.dump(dict(code=code, frac=frac, K=K, N=N, flags=flags, tokens=out, sha=h, tok_s=len(out) / dec, passes=P,
               prefill_s=t_pre, stats={k: v for k, v in s.items()}), open(f"{od}/{tag}.json", "w"), indent=1)
print(f"{code} [{flags}] K={K}: {len(out)/dec:.1f} tok/s, {len(out)/P:.2f} tok/pass, sha {h} | per pass: misses "
      f"{s['misses']/P:.1f}, stalled {s['stalled']/P:.1f}, io {s['io_s']/P*1e3:.2f} ms, pf issued {s['pf_issued']/P:.1f} "
      f"used {s['pf_used']/P:.1f} late {s['pf_late']/P:.1f}", flush=True)
if "pf_reserved" in s:
    print(f"   sched: reserved {s['pf_reserved']}, cancelled {s['pf_cancel']}, queued->demand {s['pf_qdemand']}, "
          f"gated off {s['pf_gated_off']}, no fit {s['pf_nofit']}, JIT tables {s['pub_jit']}, victims {s['pf_victims']}", flush=True)
    names = [f"d{d}b{b}" for d in (1, 2, 3) for b in (0, 1, 2)] + ["oracle", "slice"]
    print("   class precision: " + ", ".join(f"{n} {c[2]:.0f}/{c[1]:.0f}" for n, c in zip(names, s["pf_cls"]) if c[1] > 0), flush=True)
