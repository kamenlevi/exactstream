"""F2a step 3 (MTP Head, Sun 11-08; GPU, the night's phase 2): continue the chat head (D-22 b5) on the captured ring.
D-22's trainer (sticky/d22_ttt.py: training-time-test chain, per-depth KL to the 125B's top-32 on assistant targets, AUF as
b5) over data/f2a/ring/w_*.npz (f2a_capture.py), one window per step, random order, F2A_EPOCHS passes (default 1).
Init = ckpt/d22_<F2A_INIT>.safetensors (default b5; stage 2 of the night: F2A_INIT=f2a --run f2a2); depth F2A_DEPTH (default 6: chat chains stop by depth ~4-6 under the 0.3 cut); lr =
F2A_LR (default b5's lr x 0.5), Adam, warm-up 20, cosine to 10 % over the planned steps; clip 1.0.
Writes ckpt/d22_f2a.safetensors (+ .json state) every 10 min and at the end; stops cleanly at F2A_UNTIL (epoch s) or
SIGTERM (checkpoint saved; same command resumes); waits while ~/.local/share/quiet/PAUSE exists.
Usage: [BigMoE venv] python3 -I sticky/f2a_train.py [--max-steps N]"""
import os, sys, json, time, glob, signal, argparse
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_card/TrainLab_sticky"))
import d22_ttt as T                                          # MTPLX env + mlx
M = T.M
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

ap = argparse.ArgumentParser()
ap.add_argument("--max-steps", type=int, default=10 ** 9)
ap.add_argument("--run", default="f2a")
a = ap.parse_args()
CK = os.path.expanduser("~/TrainLab/ckpt")
RING = os.path.expanduser("~/TrainLab/data/f2a/ring")
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
b5 = json.load(open(os.path.join(CK, "d22_b5.json")))["args"]
D = int(os.environ.get("F2A_DEPTH", "6"))
LR = float(os.environ.get("F2A_LR", str(float(b5["lr"]) * 0.5)))
EP = float(os.environ.get("F2A_EPOCHS", "1"))
INIT = os.environ.get("F2A_INIT", "b5")
WD, AUF = float(b5.get("wdecay", 1.0)), bool(b5.get("auf", True))
until = float(os.environ.get("F2A_UNTIL", "1e12"))
stop = {"now": False}
signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))


def load_w(path):
    z = np.load(path)
    W = mx.array((z["wd"].astype(np.uint32) << 16).view(np.float32)).astype(mx.bfloat16)[None]
    return {"W": W, "ids": np.asarray(z["ids"], np.int32), "tk": np.asarray(z["tk"], np.int32),
            "lp": np.asarray(z["lp"], np.float32), "asst": np.asarray(z["asst"], np.float32)}


t0 = time.time()
M.limit_memory()
rt, lm, head = M.load(0.03)
P = M.Trainable(head)
ckw, ckj = os.path.join(CK, f"d22_{a.run}.safetensors"), os.path.join(CK, f"d22_{a.run}.json")
files = sorted(glob.glob(os.path.join(RING, "w_*.npz")))
steps = min(a.max_steps, int(EP * len(files)))
state = {"step": 0, "order": [], "loss": [], "per": [], "files": len(files), "steps": steps, "status": "running",
         "args": {"depth": D, "lr": LR, "epochs": EP, "wdecay": WD, "auf": AUF, "init": f"d22_{INIT}"}}
if os.path.exists(ckj):
    state = json.load(open(ckj)); P.load_weights(ckw); steps = state["steps"]
    print(f"[f2a-train] resume at step {state['step']} of {steps}", flush=True)
else:
    P.load_weights(os.path.join(CK, f"d22_{INIT}.safetensors"))
mx.eval(P.parameters())
P.freeze(); P.unfreeze()
opt = optim.Adam(learning_rate=lambda s: LR * mx.minimum(1.0, (s + 1) / 20.0)
                 * (0.1 + 0.45 * (1 + mx.cos(np.pi * mx.minimum(1.0, s / max(1, steps - 1))))))
if state["step"]:
    opt.state["step"] = mx.array(state["step"], mx.uint64)
cur = {}
lossf = nn.value_and_grad(P, lambda: T.ttt_loss(lm, head, P, cur["x"], D, WD, AUF)[0])
rng = np.random.default_rng(5 + state["step"])
ntr = sum(v.size for _, v in tree_flatten(P.trainable_parameters()))
print(f"[f2a-train] {len(files)} ring windows, {steps} steps, depth {D}, lr {LR:g}, auf {AUF}, {ntr / 1e6:.1f}M trainable; "
      f"loaded in {time.time() - t0:.0f} s", flush=True)


def save(status):
    mx.save_safetensors(ckw + ".tmp.safetensors", dict(tree_flatten(P.parameters())))
    os.replace(ckw + ".tmp.safetensors", ckw)
    state["status"] = status
    json.dump(state, open(ckj + ".tmp", "w")); os.replace(ckj + ".tmp", ckj)


last, tok = time.time(), 0
while state["step"] < steps:
    while os.path.exists(PAUSE) and not stop["now"]:
        time.sleep(2)
    if stop["now"] or time.time() >= until:
        save("stopped"); print(f"[f2a-train] stopped at step {state['step']} ({'SIGTERM' if stop['now'] else 'deadline'}); saved", flush=True)
        sys.exit(3)
    if not state["order"]:
        state["order"] = [int(i) for i in rng.permutation(len(files))]
    f = files[state["order"].pop()]
    if not os.path.exists(f):
        continue
    cur["x"] = load_w(f)
    t1 = time.time()
    loss, g = lossf()
    g, _ = optim.clip_grad_norm(g, 1.0)
    opt.update(P, g)
    mx.eval(P.parameters(), opt.state, loss)
    state["step"] += 1; state["loss"].append(float(loss)); tok += len(cur["x"]["ids"])
    if state["step"] % 25 == 0 or state["step"] <= 2:
        print(f"[f2a-train] step {state['step']}/{steps} loss {float(loss):.4f} (mean of last 25 {np.mean(state['loss'][-25:]):.4f}); "
              f"{time.time() - t1:.1f} s/step, {tok / max(time.time() - t0, 1e-6):.0f} tok/s, peak {mx.get_peak_memory() / 2 ** 30:.1f} GB",
              flush=True)
    if time.time() - last > 600:
        save("running"); last = time.time()
save("done")
print(f"[f2a-train] DONE: {state['step']} steps in {time.time() - t0:.0f} s", flush=True)
