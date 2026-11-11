"""Quality of the dh build that would ship (defer/v3d, tier.py's multi-row dh path), with the gating number's scorer: the Eval
suite's one-token-per-call step scorer (evalsuite.stages.step_forward + engine.var_head against its cached exact 4-bit
reference; read-only import), as Breakthrough's tf_b12.py / defer/tf_dense.py do on v3f. Env = the gating run's: plan
4433333333 tiers, 3c GLU + fused combine (+ OoO, bitwise), C_m 128 / C_l 51 (load_b12(0.25)'s pools), no JIT prefetch,
SPARE 0, PF_NOPUBLISH, none of the speed switches the suite refuses. Arms (env, as run.py): DH_ARM=v3 | dh (DH_W, default
4); BIGMOE_DH_D=1 for d1 (late part at the end of layer L+1). Sequences: each set's [OFF::EVERY]; arms pair by position
(defer/tf_dense_cmp.py NAME_A NAME_B eEoO reads data/defer/tfb12/NAME_eEoO.npz).
usage: step_dense.py NAME SETS EVERY [OFF=0]   e.g.  DH_ARM=dh BIGMOE_DH_D=1 step_dense.py my_d1w4 tf_ifeval 4"""
import os, sys, time
ENV = dict(BIGMOE_TIER="4433333333", BIGMOE_TIER_GLU="3c", BIGMOE_TIER_COMBINE="1", BIGMOE_OOO="1", BIGMOE_MAXK="80",
           BIGMOE_GDN_EXACT="1", BIGMOE_SPARE="0", BIGMOE_PF_NOPUBLISH="1", BIGMOE_TIER_CM="128", BIGMOE_TIER_CL="51",
           EVALSUITE_FRUGAL="1", MTPLX_NGRAM_PREWARM="off", MTPLX_COMPILED_VERIFY="0", MTPLX_QWEN4EXP_COMPILE="0",
           MTPLX_COMPILE_AR_FORWARD="0", MTPLX_COMPILED_TARGET_PREFIX="0")
ARM = os.environ.get("DH_ARM", "v3")
W = int(os.environ.get("DH_W", "4"))
assert ARM in ("v3", "dh"), ARM
if ARM == "dh":
    ENV.update(BIGMOE_DH=str(W), BIGMOE_DEFER="1", BIGMOE_DEFER_FULL="1")
os.environ.pop("BIGMOE_PF_JIT", None)                 # the loader turns JIT on if the variable exists at all
for k, v in ENV.items():
    os.environ.setdefault(k, v)
NAME, SETS, EVERY = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3])
OFF = int(sys.argv[4]) if len(sys.argv) > 4 else 0
sys.path.insert(0, os.path.expanduser("~/MoonShots/defer/v3d"))
import numpy as np
from bigmoe_v3d.loader import load_streamed           # before evalsuite puts ~/BigMoE on sys.path (bigmoe_ext name)
from bigmoe_v3d import tier as T
rt, st = load_streamed(os.path.expanduser("~/BigMoE/models/bare"), 128, mtp=False)
if ARM == "dh":
    T.install_dh(rt.model)
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
from evalsuite import stages as SG, engine as EN

TARGET = {"tf_code": 0.00253, "tf_gsm": 0.00449, "tf_ifeval": 0.0204, "tf_multi": 0.0206}   # 1.10 x ar4 (MoonShots)
items = SG.tf_items()
head = EN.var_head(SG.load_ref_tf(items), SG._targets(items))
seqs = [i for s in SETS for i in [j for j, it in enumerate(items) if it["set"] == s][OFF::EVERY]]
out = {s: [] for s in SETS}
keys = {s: [] for s in SETS}
t0 = time.time()
for i in seqs:
    it = items[i]
    r = SG.step_forward(rt, it["ids"], it["pos"], lambda pos, lg, i=i: head(i, pos, lg))
    st.gate.check()
    out[it["set"]].append(r)
    keys[it["set"]] += [i] * len(r["kld"])
cat = {s: np.concatenate([r["kld"] for r in v]) for s, v in out.items() if v}
agr = {s: np.concatenate([r["agree"] for r in v]) for s, v in out.items() if v}
allk = np.concatenate(list(cat.values()))
cfg = (f"v3d arm={ARM} DH={os.environ.get('BIGMOE_DH', 'off')} D={os.environ.get('BIGMOE_DH_D', 'h')} C_m=128 C_l=51 "
       f"plan={os.environ.get('BIGMOE_TIER')} dh_calls={T.DH['calls']}")
line = (f"{NAME} [{cfg}]: KLD {allk.mean():.4f} over {len(allk)} positions ({len(seqs)} seqs, every {EVERY} from {OFF}) | "
        + " ".join(f"{s} {cat[s].mean():.5f} (top-1 {agr[s].mean():.3f}"
                   + (f", {'PASS' if cat[s].mean() <= TARGET[s] else 'FAIL'} vs {TARGET[s]}" if s in TARGET else "") + ")"
                   for s in cat)
        + f" | {time.time() - t0:.0f}s")
print(line, flush=True)
d = os.path.expanduser("~/MoonShots/data/defer/tfb12")
os.makedirs(d, exist_ok=True)
with open(os.path.join(d, "results.txt"), "a") as f:
    f.write(line + "\n")
np.savez(os.path.join(d, f"{NAME}_e{EVERY}o{OFF}.npz"), **{f"kld_{s}": cat[s] for s in cat}, **{f"agree_{s}": agr[s] for s in agr},
         **{f"seq_{s}": np.array(keys[s]) for s in cat})
