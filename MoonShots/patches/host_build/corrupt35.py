"""Isolate the sha change seen in host35.py (runs after the first C++-glue run in one process gave 844b0186b9).
usage: corrupt35.py SEQ   e.g. "p p p g p"   p = Python glue run, g = ext.gate_glue run, d = Python glue + 100 us build delay
Every run: fresh prefill, K=3 MTP spec, 64 tokens, prints sha + passes. Base flags from the environment."""
import hashlib, os, sys, time
sys.path.insert(0, os.environ.get("HOST_TREE", os.path.expanduser("~/BigMoE")))
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
import mlx.core as mx
import mlx_lm.utils as mu
import specdec, mtp as mtpmod
from bigmoe.stream import ExpertStreamer
import bigmoe.stream as S
mtpmod.MTP_PATH = os.path.expanduser("~/SpecDecode/data/mtp/mtp.safetensors")
MD = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit")
st = ExpertStreamer(MD, round(0.55 * 256))
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
m.cap = mtpmod.install_capture(m.model)
head = mtpmod.load_mtp(m.model, bits=4)
tok = m.tokenizer
ids = list(tok.apply_chat_template([{"role": "user", "content": open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read()}],
                                   add_generation_prompt=True, enable_thinking=False))
sw0 = next(l.mlp.switch_mlp for l in m.model.layers if hasattr(l.mlp, "switch_mlp"))
Streamed = type(sw0); s_call = Streamed.__call__
DELAY = [0.0]


def streamed_call(self, *a, **kw):
    if DELAY[0]:
        t = time.perf_counter()
        while time.perf_counter() - t < DELAY[0]: pass
    return s_call(self, *a, **kw)


Streamed.__call__ = streamed_call
for c in sys.argv[1].split():
    S._CGLUE = c == "g"; DELAY[0] = 100e-6 if c == "d" else 0.0
    drafter = specdec.MTPDrafter(m, head, k=3)
    m.prefill(ids[:-1], on_chunk=drafter.prefill_hook(ids))
    state, out = {}, []
    for new in specdec.generate_iter(m, ids, 64, drafter, None, 1, state):
        out += new
    print(f"{c}: sha {hashlib.sha1(str(out).encode()).hexdigest()[:10]} passes {state['forwards']} | {tok.decode(out)[:60]!r}", flush=True)
