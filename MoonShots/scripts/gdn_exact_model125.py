"""Model-level check of the exact GDN prework (patches/planes_engine/gdn_exact_patch.py) on the 125B: the same prompt,
prefill + a 1-row decode step + a 4-row verify forward (attention_phase decode_verify), logits compared bitwise between
  ref    the eager chain (MTPLX fusion flags off)
  exact  MTPLX_FUSED_GDN_CONVNORM=1 + MTPLX_FUSED_CONVNORM_VERIFY=1 routed to the exact kernel
  mtplx  the same flags with MTPLX's own kernel (expected to differ: fp32 throughout)
Each arm runs in its own process (flags are read at import). usage: gdn_exact_model125.py [FRAC]   (GPU, model memory)"""
import hashlib, json, os, subprocess, sys
ARM = os.environ.get("GDN_ARM")
FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
if ARM is None:                                             # driver: one process per arm
    res = {}
    for arm in ("ref", "exact", "mtplx"):
        env = dict(os.environ, GDN_ARM=arm)
        if arm != "ref":
            env.update(MTPLX_FUSED_GDN_CONVNORM="1", MTPLX_FUSED_CONVNORM_VERIFY="1")
        out = subprocess.run([sys.executable, __file__, str(FRAC)], env=env, capture_output=True, text=True)
        line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
        res[arm] = json.loads(line[-1][7:]) if line else {"error": out.stderr[-600:]}
        print(arm, res[arm], flush=True)
    same = lambda a, b: all(res[a].get(k) == res[b].get(k) for k in ("decode", "verify"))
    print(f"exact == ref: {same('exact', 'ref')} | mtplx == ref: {same('mtplx', 'ref')}", flush=True)
    sys.exit(0 if same("exact", "ref") else 1)
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("~/MoonShots/patches/planes_engine"))
import mlx.core as mx
from bigmoe.loader import load_streamed
if ARM == "exact":
    import gdn_exact_patch
    gdn_exact_patch.install()
rt, st = load_streamed(os.path.expanduser("~/BigMoE/models/bare"), round(FRAC * 512))
m = rt.model
ids = rt.tokenizer.encode("def parse(path):\n    rows = []\n    with open(path) as f:\n        for line in f:")
L0 = lambda o: o[0] if isinstance(o, tuple) else o
cache = m.make_cache()
mx.eval(L0(m(mx.array([ids[:-5]]), cache=cache)))           # prefill
d = L0(m(mx.array([[ids[-5]]]), cache=cache))                # 1-row decode step
mx.eval(d)
from mtplx.attention_context import attention_phase
with attention_phase("decode_verify"):
    v = L0(m(mx.array([ids[-4:]]), cache=cache))             # 4-row verify forward
mx.eval(v)
import numpy as np
h = lambda a: hashlib.sha256(np.array(a.astype(mx.float32)).tobytes()).hexdigest()[:16]
print("RESULT " + json.dumps({"decode": h(d), "verify": h(v)}), flush=True)
