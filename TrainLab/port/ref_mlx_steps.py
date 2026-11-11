"""MLX's own trainer for the first --steps steps of L1-20's recipe (MTP Head, Thu 11-05; the full-scale confirmation of
REPORT 'PORT steps FAIL: ROOT CAUSE FOUND'). Executes sticky/d09_train.py (= Leap's dfl2c_train.py, which produced L1-20)
up to its main loop, so its windows, groups, rng use, make_batch, loss, step() (micro-batches, summed gradients, clip 1.0,
MLX AdamW) and lr schedule are the trainer's own, then runs its loop's item selection for --steps steps (no val: it
draws no random numbers; nothing is saved but the losses). Set MLX_ENABLE_TF32=0 in the environment for true fp32.
Usage: [BigMoE venv] python ref_mlx_steps.py OUT.json [--steps 5]"""
import argparse, json, os, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--steps", type=int, default=5)
r = ap.parse_args()
src = open(os.path.expanduser("~/TrainLab/sticky/d09_train.py")).read()
src = src[:src.index("if A.time:")]                 # everything up to the modes that run: setup, step(), lr_at, opt
sys.argv = ["d09_train.py", "--epochs", "2", "--lr", "2e-5", "--S", "4", "--A", "64", "--micro", "1", "--bs", "32",
            "--pmask", "0.5", "--data", "~/Leap/data/l120", "--out", "/dev/null"]          # L1-20's recipe
ns = {"__name__": "d09_steps"}
t0 = time.time()
exec(compile(src, "d09_train.py", "exec"), ns)
rng, tr, step, opt, lr_at, A, mx = ns["rng"], ns["tr"], ns["step"], ns["opt"], ns["lr_at"], ns["A"], ns["mx"]
tf32 = os.environ.get("MLX_ENABLE_TF32", "unset (default 1)")
print(f"[mlx-steps] loaded in {time.time() - t0:.0f}s; {len(tr)} train groups; {ns['steps']} steps in the schedule; "
      f"MLX_ENABLE_TF32={tf32}", flush=True)
order, losses = [], []
for s in range(r.steps):
    if not order:
        order = [int(x) for x in rng.permutation(len(tr))]
    items = [tr[order.pop()] for _ in range(min(A.S, len(order)))]
    opt.learning_rate = lr_at(s)
    t1 = time.time()
    l = step(items)
    losses.append(l)
    print(f"[mlx-steps] step {s + 1}: loss {l:.6f} lr {lr_at(s):.2e} {time.time() - t1:.1f}s {ns['mem']()}", flush=True)
json.dump(dict(loss=losses, tf32=tf32, peak_gb=mx.get_peak_memory() / 2 ** 30), open(os.path.expanduser(r.out), "w"))
print(f"[mlx-steps] saved {r.out}", flush=True)
