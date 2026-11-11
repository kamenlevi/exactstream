"""PORT data path over steps (MTP Head, Thu 11-05; diagnosis 3 of the steps-check FAIL). The MLX trainer's own data code
(sticky/d09_train.py's load_windows / groups_of / make_batch, taken from its source and run with a numpy stand-in for mx,
plus its pre-loop RNG use and its loop's item selection) vs port/data_port.py + train_port.py's loop, for the first
--steps steps of L1-20's recipe. Every batch array must be bitwise identical. No model, no GPU (~1.5 GB).
Usage: [TrainLab .venv-torch] python test_data_steps.py [--steps 5] [--S 4 --A 64 --bs 32 --pmask 0.5 --micro 1]"""
import argparse, glob, os, re, sys, types
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_port as DP
import dflash2_torch as M

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="~/Leap/data/l120"); ap.add_argument("--steps", type=int, default=5)
ap.add_argument("--S", type=int, default=4); ap.add_argument("--A", type=int, default=64)
ap.add_argument("--bs", type=int, default=32); ap.add_argument("--pmask", type=float, default=0.5)
ap.add_argument("--micro", type=int, default=1)
r = ap.parse_args()
MASK = int(M.draft_config()["dflash_config"]["mask_token_id"])

# ---- the MLX trainer's code, from its source
src = open(os.path.expanduser("~/TrainLab/sticky/d09_train.py")).read()
def grab(name):
    m_ = re.search(rf"^def {name}\(.*?(?=^\S)", src, re.S | re.M)
    return m_.group(0)
ns = dict(np=np, os=os, glob=glob, mx=types.SimpleNamespace(array=np.asarray), bf16=np.asarray,
          A=types.SimpleNamespace(A=r.A, data=r.data), W=2048, BS=r.bs, KD=r.bs - 1, MASK=MASK,
          PM_RNG=np.random.default_rng(20))
for f in ("load_windows", "groups_of", "make_batch"):
    exec(grab(f), ns)
wins = ns["load_windows"]()
tr = [g for w_ in wins if w_["split"] == "train" for g in ns["groups_of"](w_)]
va = [g for w_ in wins if w_["split"] == "val" for g in ns["groups_of"](w_)]
rng = np.random.default_rng(0)
va = [va[i] for i in sorted(rng.choice(len(va), min(len(va), 24), replace=False))] if va else []
order = []
mlx_steps = []
for s in range(r.steps):
    if not order:
        order = [int(x) for x in rng.permutation(len(tr))]
    items = [tr[order.pop()] for _ in range(min(r.S, len(order)))]
    parts = [ns["make_batch"](items[k:k + r.micro], r.pmask) for k in range(0, len(items), r.micro)]
    mlx_steps.append(([(w_["key"], list(g)) for w_, g in items], parts))

# ---- the port's
wp = DP.load_windows(r.data)
trp, vap, rngp = DP.split_groups(wp, r.A)
PM = np.random.default_rng(20)
orderp = []
keys = ("hc", "upos", "anc", "tok", "tkl", "lpl", "y", "pred", "w")
ok = len(tr) == len(trp) and len(va) == len(vap)
print(f"groups: mlx train {len(tr)} val {len(va)}; port train {len(trp)} val {len(vap)}", flush=True)
for s in range(r.steps):
    if not orderp:
        orderp = [int(x) for x in rngp.permutation(len(trp))]
    items = [trp[orderp.pop()] for _ in range(min(r.S, len(orderp)))]
    parts = [DP.make_batch(items[k:k + r.micro], r.A, r.bs, MASK, r.pmask, PM) for k in range(0, len(items), r.micro)]
    ids_m, parts_m = mlx_steps[s]
    same_items = ids_m == [(w_["key"], list(g)) for w_, g in items]
    diffs = []
    for i, (pm_, pp) in enumerate(zip(parts_m, parts)):
        for k, a in zip(keys, pm_):
            if not np.array_equal(np.asarray(a), pp[k]):
                diffs.append(f"part {i} {k}")
    wsum = [float(np.asarray(pm_[-1]).sum()) for pm_ in parts_m]
    print(f"step {s + 1}: items {'same' if same_items else 'DIFFERENT'} {[k_ for k_, _ in ids_m]}; "
          f"arrays {'identical' if not diffs else 'DIFFER: ' + ', '.join(diffs[:6])}; loss weight per part {wsum}", flush=True)
    ok = ok and same_items and not diffs
print(f"DATA PATH OVER {r.steps} STEPS: {'IDENTICAL' if ok else 'DIFFERENT'}", flush=True)
