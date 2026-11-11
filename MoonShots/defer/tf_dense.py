"""Denser quality re-check for the dh_w4 fast mode (NewIdeas 2026-10-22: "a denser IFEval re-check (every 4), since its CI
touches the target"). Same scorer as the gating number: Breakthrough's tf_b12.py logic (the Eval suite's one-token-per-call
step scorer, evalsuite.stages.step_forward + engine.var_head against its cached exact 4-bit reference) on the REAL v3f
decoder (bt_b12.load_b12, read-only imports from ~/Breakthrough and ~/EvalSuite); outputs go to ~/MoonShots/data/defer/tfb12
instead of Breakthrough's tree. Arms are bt_b12's env switches, as in the gating run (day_frpf.sh 07:42):
  v3 control:  B12_W=10 B12_HC=0          dh_w4:  B12_W=4 B12_LATE3=4,10 B12_HC=0          (FRAC 0.25: C_m 128, C_l 51)
Sequences: the SETS' sequences [OFF::EVERY] (per set list, not the combined list), so arms pair by position.
usage: tf_dense.py NAME FRAC SETS EVERY [OFF=0]   e.g.  B12_W=4 B12_LATE3=4,10 B12_HC=0 tf_dense.py dhw4 0.25 tf_ifeval 4"""
import os, sys, time

os.environ["EVALSUITE_FRUGAL"] = "1"                 # the suite's frugal-runtime env: no PF1 staging flags
sys.path.insert(0, os.path.expanduser("~/Breakthrough/scripts"))
NAME, frac, SETS, EVERY = sys.argv[1], float(sys.argv[2]), sys.argv[3].split(","), int(sys.argv[4])
OFF = int(sys.argv[5]) if len(sys.argv) > 5 else 0
from bt_b12 import load_b12, CFG
rt, st = load_b12(frac)                               # bigmoe_v3f is imported before evalsuite puts ~/BigMoE on sys.path
import numpy as np
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
    st.gate.check()                                   # a timed-out spin gate invalidates the pass
    out[it["set"]].append(r)
    keys[it["set"]] += [i] * len(r["kld"])
cat = {s: np.concatenate([r["kld"] for r in v]) for s, v in out.items() if v}
agr = {s: np.concatenate([r["agree"] for r in v]) for s, v in out.items() if v}
allk = np.concatenate(list(cat.values()))
cfg = (f"C_m={CFG['cm']} C_l={CFG['cl']} hc{CFG['hc']} w={CFG['w']} late3={CFG['l3']} norm={CFG.get('norm', 'used')} "
       f"plan={os.environ.get('BIGMOE_TIER')}")
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
