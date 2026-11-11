"""Teacher-forced step KLD of B12-family configs on the REAL v3 tier path (v3f: records3, plan 4433333333, B12's own
frugal / late / normalisation code), scored by the Eval suite's own one-token-per-call scorer (evalsuite.stages.step_forward
+ engine.var_head against its cached exact 4-bit reference; read-only import), as tf_frugal_var.py does on fork125bt
(exact 4-bit experts: its numbers do not include the 3-bit tier loss; these do).
Arms are bt_b12's env switches (set before the call), e.g.
  B12 as judged (C_m 51):                   (none)            / B12_NORM=mass
  v3 control (exact top-10 on the tiers):   B12_W=10
  dh_w4 (NewIdeas tf_defer125 'dh'):        B12_W=4 B12_LATE3=4,10   (ranks 0-3 waited, cold ranks 4-9 half a layer late)
usage: tf_b12.py NAME FRAC SETS EVERY   e.g.  tf_b12.py b12_mass 0.10 tf_ifeval,tf_multi 8
  -> a line in data/tfb12/results.txt (per-part KLD, top-1, vs the 1.10 x ar4 targets) + data/tfb12/NAME_e{EVERY}.npz"""
import os, sys, time

os.environ["EVALSUITE_FRUGAL"] = "1"                 # the suite's frugal-runtime env: no PF1 staging flags
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
NAME, frac, SETS, EVERY = sys.argv[1], float(sys.argv[2]), sys.argv[3].split(","), int(sys.argv[4])
from bt_b12 import load_b12, CFG
rt, st = load_b12(frac)                               # bigmoe_v3f is imported before evalsuite puts ~/BigMoE on sys.path
import numpy as np
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
from evalsuite import stages as SG, engine as EN

TARGET = {"tf_code": 0.00253, "tf_gsm": 0.00449, "tf_ifeval": 0.0204, "tf_multi": 0.0206}   # 1.10 x ar4 (MoonShots)
items = SG.tf_items()
head = EN.var_head(SG.load_ref_tf(items), SG._targets(items))
seqs = [i for i, it in enumerate(items) if it["set"] in SETS][::EVERY]
out = {s: [] for s in SETS}
t0 = time.time()
for i in seqs:
    it = items[i]
    r = SG.step_forward(rt, it["ids"], it["pos"], lambda pos, lg, i=i: head(i, pos, lg))
    st.gate.check()                                   # a timed-out spin gate invalidates the pass
    out[it["set"]].append(r)
cat = {s: np.concatenate([r["kld"] for r in v]) for s, v in out.items() if v}
agr = {s: np.concatenate([r["agree"] for r in v]) for s, v in out.items() if v}
allk = np.concatenate(list(cat.values()))
cfg = (f"C_m={CFG['cm']} C_l={CFG['cl']} hc{CFG['hc']} w={CFG['w']} late3={CFG['l3']} norm={CFG.get('norm', 'used')} "
       f"plan={os.environ.get('BIGMOE_TIER')}")
line = (f"{NAME} [{cfg}]: KLD {allk.mean():.4f} over {len(allk)} positions ({len(seqs)} seqs, every {EVERY}) | "
        + " ".join(f"{s} {cat[s].mean():.5f} (top-1 {agr[s].mean():.3f}"
                   + (f", {'PASS' if cat[s].mean() <= TARGET[s] else 'FAIL'} vs {TARGET[s]}" if s in TARGET else "") + ")"
                   for s in cat)
        + f" | {time.time() - t0:.0f}s")
print(line, flush=True)
d = os.path.expanduser("~/Breakthrough/data/tfb12")
os.makedirs(d, exist_ok=True)
with open(os.path.join(d, "results.txt"), "a") as f:
    f.write(line + "\n")
np.savez(os.path.join(d, f"{NAME}_e{EVERY}.npz"), **{f"kld_{s}": cat[s] for s in cat}, **{f"agree_{s}": agr[s] for s in agr})
