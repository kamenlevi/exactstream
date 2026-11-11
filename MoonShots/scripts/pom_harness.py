"""M-03 POM confirmation (MoonShots Wed 16:1x; NON-exact; GPU in a quiet lock): measure, instead of interpolating, the quality of
precision on miss. The replay (pom24.py) put 20.0 % of the rank >= 4 gate mass (18.7 % of its energy) at 2 code planes for
R2 = 4. Here the Eval suite's own tf screen (kl125_screen.py's machinery, read-only) with its MoE block wrapped in THIS process:
every layer computes the plan 44 33 2..2 (R2) and plan 4433333333 expert outputs and keeps the 2-plane output for a random F
of the (token, rank >= R2) uses (seeded Bernoulli: random stands in for 'was a miss', which a teacher-forced prefill cannot
simulate), the 3-plane output elsewhere; same gate-weighted sum and shared expert as the suite's sorted block.
Per-part mean KLD vs the cached 4-bit reference, next to v3 and the gate 1.10 x ar4.
PASS (M-03 row): inside the gate on all four parts. KILL: over on >= 2 parts.
usage: [BigMoE venv] pom_harness.py [R2=4] [F=0.20]   (GPU, ~24 GB at C=128; quiet lock; no BIGMOE_* env flags)"""
import json, os, sys, time
import numpy as np
for k in [k for k in os.environ if k.startswith("BIGMOE_")]:
    os.environ.pop(k)
EV = os.path.expanduser("~/EvalSuite")
sys.path.insert(0, EV)
from evalsuite import stages as SG, model as M, engine as E
import mlx.core as mx
from mtplx.models import qwen4_exp

R2 = int(sys.argv[1]) if len(sys.argv) > 1 else 4
F = float(sys.argv[2]) if len(sys.argv) > 2 else 0.20
PLAN_LO, PLAN_HI = "44" + "3" * (R2 - 2) + "2" * (10 - R2), "4433333333"
OUT = os.path.expanduser("~/MoonShots/data/kl125")


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
v = M.variant("plan" + PLAN_LO)
t0 = time.time()
rt, st = M.load(v, C=128)
M.configure(v)
segs_lo, segs_hi = M.plan_segments(PLAN_LO), M.plan_segments(PLAN_HI)
their_block = qwen4_exp.SparseMoeBlock.__call__
KEY = [mx.random.key(1234)]
NCALL = [0]


def pom_block(self, x):
    if getattr(self.switch_mlp, "_bm", None) is None or not M.STATE.sorted:
        return their_block(self, x)
    gates = mx.softmax(self.gate(x), axis=-1, precise=True)
    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if self.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    order = mx.argsort(-scores, axis=-1)
    inds, scores = mx.take_along_axis(inds, order, axis=-1), mx.take_along_axis(scores, order, axis=-1)
    if M.STATE.capture is not None:
        M.STATE.capture.setdefault(self.switch_mlp._bm["L"], []).append(inds)
    M.STATE.segs = segs_lo
    y_lo = self.switch_mlp(x, inds)               # builds the layer's memo with tiers {3, 2}
    M.STATE.segs = segs_hi
    y_hi = self.switch_mlp(x, inds)               # same (layer, pass) memo: tier 3 reused
    M.STATE.segs = segs_lo
    KEY[0], sub = mx.random.split(KEY[0])
    m = (mx.random.uniform(shape=inds.shape, key=sub) < F) & (mx.arange(k) >= R2)
    NCALL[0] += 1
    y = (mx.where(m[..., None], y_lo, y_hi) * scores[..., None]).sum(axis=-2)
    return y + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)


qwen4_exp.SparseMoeBlock.__call__ = pom_block
items_all = SG.tf_items()
ref_all = SG.load_ref_tf(items_all)
keep = [i for i, it in enumerate(items_all) if "doc" not in it["set"]]
items = [items_all[i] for i in keep]
ref = {j: ref_all[i] for j, i in enumerate(keep)}
tg = SG._targets(items)
print(f"loaded in {time.time() - t0:.0f}s; {len(items)} own-output sequences; POM R2 {R2} F {F}: plan{PLAN_LO} on a random "
      f"{100 * F:.0f} % of rank >= {R2} uses, plan{PLAN_HI} elsewhere", flush=True)
t = time.time()
name = f"pom_R{R2}_F{int(round(100 * F))}"
res = SG.run_batches(rt, items, E.var_head(ref, tg), capture=True, label=f"{name} tf screen")
seq = np.concatenate([np.full(len(res[i]["kld"]), i, np.int32) for i in range(len(items))])
kld = np.concatenate([res[i]["kld"] for i in range(len(items))])
agree = np.concatenate([res[i]["agree"] for i in range(len(items))])
new = {name: per_set([it["set"] for it in items], seq, kld, agree)}
json.dump(dict(variant=name, per_set=new[name], pooled=float(kld.mean()), seconds=time.time() - t, moe_calls=NCALL[0]),
          open(os.path.join(OUT, f"screen_{name}.json"), "w"), indent=1)
print(f"{name}: pooled KLD {kld.mean():.5f}, agree {agree.mean():.4f} ({time.time() - t:.0f}s, {NCALL[0]} wrapped MoE calls)", flush=True)
st.gate.check()
rows = {**existing, **new}
sets = sorted({s for r in rows.values() for s in r})
print("\nper-set mean KLD vs the 4-bit reference (top-1 agreement); targets = 1.10 x ar4 (gate_table.py)")
print("variant".ljust(16) + "".join(s.ljust(22) for s in sets))
for nm, r in rows.items():
    print(nm.ljust(16) + "".join((f"{r[s][0]:.5f} ({r[s][1]:.3f})" if s in r else "-").ljust(22) for s in sets))
if "ar4" in rows:
    print("target 1.10x ar4".ljust(16) + "".join((f"{1.10 * rows['ar4'][s][0]:.5f}" if s in rows["ar4"] else "-").ljust(22) for s in sets))
