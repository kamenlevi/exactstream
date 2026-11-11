"""Per-part prefill-KL screen for 2-bit-tail tier plans on the 125B (NewIdeas 2026-10-22, lever 1's quality gate), using the
Eval suite's own machinery READ-ONLY (~/EvalSuite: tf items, the cached 4-bit reference cache/ref/tf_ref.npz, var_head, the
vendored planes125 tier math) on the own-output tf sets only (the 4 x 64K documents are skipped: ~4 min per plan instead of
~15). Prints per set (gsm / ifeval / code / multi) mean KLD vs the reference and top-1 agreement, next to the same per-set
numbers of the suite's existing runs (floor = same weights, ar4 = accepted build, plan4433333333, plan4433332222) read from
~/EvalSuite/results/*/tf.npz. Writes ~/MoonShots/data/kl125/screen_<plan>.json (nothing is written into ~/EvalSuite).
usage: kl125_screen.py plan4322222222 [plan4222222222 ...]   (GPU, ~24 GB at C=128; quiet lock; no BIGMOE_* env flags)"""
import json, os, sys, time
import numpy as np
for k in [k for k in os.environ if k.startswith("BIGMOE_")]:
    os.environ.pop(k)                                         # evalsuite/env.py refuses BigMoE speed flags
EV = os.path.expanduser("~/EvalSuite")
sys.path.insert(0, EV)
from evalsuite import stages as SG, model as M, engine as E
OUT = os.path.expanduser("~/MoonShots/data/kl125")
os.makedirs(OUT, exist_ok=True)


def per_set(sets_by_item, seq, kld, agree):
    out = {}
    for s in sorted(set(sets_by_item)):
        if "doc" in s:
            continue
        m = np.isin(seq, [i for i, x in enumerate(sets_by_item) if x == s])
        out[s] = (float(kld[m].mean()), float(agree[m].mean()), int(m.sum()))
    return out


existing = {}
for name in ("floor", "ar4", "plan4433333333", "plan4433332222"):
    p = os.path.join(EV, "results", name, "tf.npz")
    if os.path.exists(p):
        z = np.load(p)
        existing[name] = per_set(list(z["sets"]), z["seq"], z["kld"], z["agree"])
names = sys.argv[1:]
vs = [M.variant(n) for n in names]
t0 = time.time()
rt, st = M.load(vs[0], C=128)
items_all = SG.tf_items()
ref_all = SG.load_ref_tf(items_all)
keep = [i for i, it in enumerate(items_all) if "doc" not in it["set"]]
items = [items_all[i] for i in keep]
ref = {j: ref_all[i] for j, i in enumerate(keep)}
tg = SG._targets(items)
print(f"loaded in {time.time() - t0:.0f}s; {len(items)} own-output sequences, {sum(len(it['pos']) for it in items)} positions", flush=True)
new = {}
for v in vs:
    M.configure(v)
    t = time.time()
    res = SG.run_batches(rt, items, E.var_head(ref, tg), capture=True, label=f"{v['name']} tf screen")
    seq = np.concatenate([np.full(len(res[i]["kld"]), i, np.int32) for i in range(len(items))])
    kld = np.concatenate([res[i]["kld"] for i in range(len(items))])
    agree = np.concatenate([res[i]["agree"] for i in range(len(items))])
    new[v["name"]] = per_set([it["set"] for it in items], seq, kld, agree)
    json.dump(dict(plan=v["name"], per_set=new[v["name"]], pooled=float(kld.mean()), seconds=time.time() - t),
              open(os.path.join(OUT, f"screen_{v['name']}.json"), "w"), indent=1)
    print(f"{v['name']}: pooled KLD {kld.mean():.5f}, agree {agree.mean():.4f} ({time.time() - t:.0f}s)", flush=True)
st.gate.check()
rows = {**existing, **new}
sets = sorted({s for r in rows.values() for s in r})
print("\nper-set mean KLD vs the 4-bit reference (top-1 agreement); targets = 1.10 x ar4 (gate_table.py)")
print("variant".ljust(16) + "".join(s.ljust(22) for s in sets))
for name, r in rows.items():
    print(name.ljust(16) + "".join((f"{r[s][0]:.5f} ({r[s][1]:.3f})" if s in r else "-").ljust(22) for s in sets))
if "ar4" in rows:
    print("target 1.10x ar4".ljust(16) + "".join((f"{1.10 * rows['ar4'][s][0]:.5f}" if s in rows["ar4"] else "-").ljust(22) for s in sets))
