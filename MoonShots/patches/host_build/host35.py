"""How much of the per-pass host graph build is exposed in real decode? (streamed Qwen3.6-35B-A3B, BigMoE's gate)
usage: host35.py FRAC N DELAY_US [K ...]     e.g.  host35.py 0.55 64 50 3 0
For each K (MTP draft depth; 0 = plain greedy) runs the real SpecDecode loop four times, alternating no delay / a busy-wait
of DELAY_US inside every streamed MoE call (= graph-build time added per layer). The wall-time slope per pass divided by the
added build time per pass is the exposed fraction of host build. In-loop timers split the build: trunk forward (lazy graph),
streamed-MoE glue (gate + look-ahead), SwitchGLU. Tokens must not change. Timing: run under a quiet lock."""
import hashlib, os, sys, time
frac, N, D = float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]) * 1e-6
Ks = [int(k) for k in sys.argv[4:]] or [3, 0]
sys.path.insert(0, os.environ.get("HOST_TREE", os.path.expanduser("~/BigMoE")))
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
import mlx.core as mx
import mlx_lm.utils as mu
import specdec, mtp as mtpmod
from bigmoe.stream import ExpertStreamer
import bigmoe.stream as S
from bigmoe.gated import ext
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
head = None
if any(k > 0 for k in Ks):
    m.cap = mtpmod.install_capture(m.model)
    head = mtpmod.load_mtp(m.model, bits=4)
tok = m.tokenizer
ids = list(tok.apply_chat_template([{"role": "user", "content": open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read()}],
                                   add_generation_prompt=True, enable_thinking=False))

# --- instrumentation: build-time accumulators (seconds) -------------------------------------------------------------
T = dict(trunk=0.0, moe=0.0, swiglu=0.0, trunk_calls=0, moe_calls=0)
DELAY = [0.0]
sw0 = next(l.mlp.switch_mlp for l in m.model.layers if hasattr(l.mlp, "switch_mlp") and not l.mlp.switch_mlp._bm["full"])
Streamed = type(sw0); Base = Streamed.__mro__[1]
s_call, b_call = Streamed.__call__, Base.__call__


def streamed_call(self, *a, **kw):
    t = time.perf_counter()
    if DELAY[0]:
        while time.perf_counter() - t < DELAY[0]: pass
    r = s_call(self, *a, **kw)
    T["moe"] += time.perf_counter() - t; T["moe_calls"] += 1
    return r


def base_call(self, *a, **kw):
    t = time.perf_counter(); r = b_call(self, *a, **kw); T["swiglu"] += time.perf_counter() - t; return r


Streamed.__call__, Base.__call__ = streamed_call, base_call
Trunk = type(m.model); t_call = Trunk.__call__


def trunk_call(self, *a, **kw):
    t = time.perf_counter(); r = t_call(self, *a, **kw); T["trunk"] += time.perf_counter() - t; T["trunk_calls"] += 1; return r


Trunk.__call__ = trunk_call


def run(K, delay, cglue=False):
    S._CGLUE = cglue and hasattr(ext, "gate_glue")
    drafter = specdec.MTPDrafter(m, head, k=K) if K > 0 else None
    m.prefill(ids[:-1], on_chunk=drafter.prefill_hook(ids) if drafter else None)
    for k in T: T[k] = 0
    DELAY[0] = delay
    state, out = {}, []
    t0 = time.perf_counter()
    for new in specdec.generate_iter(m, ids, N, drafter, None, 1, state):
        out += new
    wall = time.perf_counter() - t0
    DELAY[0] = 0.0
    P = state["forwards"]
    return dict(K=K, delay_us=delay * 1e6, cglue=S._CGLUE, sha=hashlib.sha1(str(out).encode()).hexdigest()[:10], tok_s=len(out) / wall,
                ms_pass=wall / P * 1e3, passes=P, trunk_ms=T["trunk"] / P * 1e3, moe_ms=T["moe"] / P * 1e3,
                swiglu_ms=T["swiglu"] / P * 1e3, moe_calls=T["moe_calls"] / P, trunk_calls=T["trunk_calls"] / P)


for K in Ks:
    run(K, 0.0)                                         # warm-up (kernels, pools)
    res = [run(K, d, c) for _ in range(3) for d, c in ((0.0, False), (D, False), (0.0, True))]
    for r in res:
        print(f"K={r['K']} delay {r['delay_us']:3.0f} us cglue {int(r['cglue'])}: {r['tok_s']:6.1f} tok/s, {r['ms_pass']:6.2f} ms/pass "
              f"({r['passes']} passes), sha {r['sha']} | build per pass: trunk {r['trunk_ms']:.2f} ms ({r['trunk_calls']:.1f} calls), "
              f"streamed MoE {r['moe_ms']:.2f} ms ({r['moe_calls']:.0f} calls) of which SwitchGLU {r['swiglu_ms']:.2f}", flush=True)
    med = lambda f: sorted(r["ms_pass"] for r in res if f(r))[1]
    base, slow, cg = med(lambda r: r["delay_us"] == 0 and not r["cglue"]), med(lambda r: r["delay_us"] > 0), med(lambda r: r["cglue"])
    added = res[1]["moe_calls"] * D * 1e3
    print(f"K={K}: median ms/pass {base:.2f}; +{added:.2f} ms build -> {slow:.2f} (exposed fraction {(slow - base) / added:.2f}); "
          f"C++ glue -> {cg:.2f} ({base - cg:+.2f} ms/pass saved)", flush=True)
