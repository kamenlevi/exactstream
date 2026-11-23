"""TTT training of the sandbox drafter (t30b's recipe, ported): the chain to depth --depth (30), accept-until-fail KL to the
target's top-32 at every depth, depth weight wdecay^(j-1) x kept share; one pass over the pinned training split in
--rows-anchor slices (seed --seed), Adam (bias-corrected), warm-up --warmup then cosine to 10 %, grad clip 1.0.

Levers plug in without touching this file: --lever NAME loads levers/NAME.py, which may define
  setup(D, args)                  -> called once after the start weights load (add parameters, change config)
  extra(j, h, n, m, x)            -> an added loss term per depth (weighted like the KL), e.g. a feature loss
  standin(j, Win, a0)             -> transform the carried stand-in before depth j's input (train AND eval)
  params(D)                       -> the parameter dict to train (default: everything in D)
A lever file can also set LR_MULT etc.; see levers/README in the top README.

Checkpoints: ckpt/<run>.safetensors (+ <run>.json: args, per-step loss / kl / extra). PAUSE-aware; SIGTERM saves.
usage: train.py --run s30 --start native --steps 600 [--rows 96] [--lr 3e-5] [--lever NAME] [--files base|pool]"""
import argparse, importlib.util, json, os, signal, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from sb import core as C


def load_lever(name):
    if not name:
        return None
    p = os.path.join(C.ROOT, "levers", name + ".py")
    spec = importlib.util.spec_from_file_location("lever_" + name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def split(which="base"):
    """base / pool / val (data/mc), or pool2 = pool + mc2 (22:3x: a broad screen pool, one pass without repeats)"""
    s = json.load(open(os.path.join(C.DATA, "split.json")))
    val = [os.path.join(C.DATA, "mc", k + ".npz") for k in s["val"]]
    if which == "pool2":
        return ([os.path.join(C.DATA, "mc", k + ".npz") for k in s["pool"]]
                + [os.path.join(C.DATA, "mc2", k + ".npz") for k in s["mc2"]]), val
    return [os.path.join(C.DATA, "mc", k + ".npz") for k in s[which]], val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--start", default="native")
    ap.add_argument("--steps", type=int, default=600); ap.add_argument("--rows", type=int, default=96)
    ap.add_argument("--depth", type=int, default=30); ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--wdecay", type=float, default=0.97); ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1114); ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--lever", default=""); ap.add_argument("--mem", type=float, default=4.5)
    ap.add_argument("--const", action="store_true", help="constant lr after the warm-up (screens)")
    ap.add_argument("--files", default="base", help="base (the base drafter's files) | pool (screens: new to the base)")
    ap.add_argument("--epochs", type=int, default=1, help="passes over the files (each pass reshuffled with seed + pass)")
    ap.add_argument("--save-opt", dest="save_opt", action="store_true", help="save Adam's state (moments + step) with the weights")
    ap.add_argument("--resume-opt", dest="resume_opt", default="", help="continue the Adam state saved by run NAME (--save-opt)")
    a = ap.parse_args()
    C.limit_memory(a.mem)
    t0 = time.time()
    os.makedirs(C.CKPT, exist_ok=True)
    C.load_target(); C.drop_trunk()
    D = C.load_drafter(a.start)
    L = load_lever(a.lever)
    if L is not None and hasattr(L, "setup"):
        L.setup(D, a)
    if L is not None and hasattr(L, "standin"):
        C.REPL["fn"] = L.standin
    tr, va = split(a.files)
    sl = []
    for e in range(a.epochs):                         # pass e reshuffled with seed + e (pass 0 = the one-pass order)
        sl += C.pass_slices(tr, a.rows, a.seed + e)
    sl = sl[a.skip:]
    D.freeze(); D.unfreeze()
    if L is not None and hasattr(L, "freeze"):
        L.freeze(D)
    steps = min(a.steps, len(sl))
    lr_mult = getattr(L, "LR_MULT", 1.0) if L is not None else 1.0
    s0 = {"v": 0}                                     # the optimizer step this run starts from (resume: the base's last)
    if a.const:
        sched = lambda s: a.lr * lr_mult * mx.minimum(1.0, (s - s0["v"] + 1) / a.warmup)
    else:
        sched = lambda s: a.lr * lr_mult * mx.minimum(1.0, (s - s0["v"] + 1) / a.warmup) * (       # from the run's own step 0
            0.1 + 0.45 * (1 + mx.cos(np.pi * mx.minimum(1.0, (s - s0["v"]) / max(1, steps - 1)))))
    opt = optim.Adam(learning_rate=sched, bias_correction=True)
    if a.resume_opt:                                  # continue the base's Adam (moments + step): no fresh-Adam drift
        sto = tree_unflatten(list(mx.load(os.path.join(C.CKPT, a.resume_opt + "_opt.safetensors")).items()))
        opt.init({k: v for k, v in D.trainable_parameters().items() if k in sto})
        opt.state = sto
        s0["v"] = int(sto["step"].item())
        print(f"[tr] Adam state of {a.resume_opt} restored (step {s0['v']})", flush=True)
    prefix = getattr(L, "PREFIX", None) if L is not None else None          # a lever's own modules (e.g. "tap_") ...
    mod_lr = getattr(L, "MODULE_LR", None) if L is not None else None       # ... train on their own Adam at MODULE_LR
    optx = optim.Adam(learning_rate=lambda s: mod_lr * mx.minimum(1.0, (s + 1) / a.warmup), bias_correction=True) \
        if prefix and mod_lr else None
    extra = getattr(L, "extra", None) if L is not None else None
    aux = getattr(L, "aux", None) if L is not None else None
    cur = {}
    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    st = {"args": vars(a), "loss": [], "kl": [], "ex": [], "slices": [], "status": "running"}
    ntr = sum(v.size for _, v in tree_flatten(D.trainable_parameters()))
    ckw, ckj = os.path.join(C.CKPT, a.run + ".safetensors"), os.path.join(C.CKPT, a.run + ".json")
    print(f"[tr] {a.run}: start {a.start}, lever {a.lever or '-'}, {len(tr)} train files ({len(sl)} slices of {a.rows}), "
          f"{steps} steps, depth {a.depth}, lr {a.lr}{' const' if a.const else ' cosine'}, {ntr / 1e6:.1f}M trainable"
          + (f"; lever modules {prefix}* on their own Adam at {mod_lr} (constant)" if optx is not None else "")
          + f"; ready in {time.time() - t0:.0f}s", flush=True)

    def save(status):
        mx.save_safetensors(ckw + ".tmp.safetensors", dict(tree_flatten(D.parameters())))
        os.replace(ckw + ".tmp.safetensors", ckw)
        if a.save_opt:
            cko = os.path.join(C.CKPT, a.run + "_opt.safetensors")
            mx.save_safetensors(cko + ".tmp.safetensors", dict(tree_flatten(opt.state)))
            os.replace(cko + ".tmp.safetensors", cko)
        st["status"] = status; st["wall_s"] = time.time() - t0
        json.dump(st, open(ckj + ".tmp", "w")); os.replace(ckj + ".tmp", ckj)
    t1 = time.time()
    for i in range(steps):
        C.wait_pause()
        f, b, r = sl[i]
        cur["x"] = C.window_at(f, b, r, a.depth)
        loss, kl, ex, g = C.ttt_step(D, cur["x"], a.depth, a.wdecay, extra, aux)   # exact two-pass gradient, ~3 GB
        g, _ = optim.clip_grad_norm(g, 1.0)
        if optx is not None:
            opt.update(D, {k: v for k, v in g.items() if not k.startswith(prefix)})
            optx.update(D, {k: v for k, v in g.items() if k.startswith(prefix)})
            mx.eval(D.parameters(), opt.state, optx.state)
        else:
            opt.update(D, g)
            mx.eval(D.parameters(), opt.state, loss, kl, ex)
        st["loss"].append(float(loss)); st["kl"].append(float(kl)); st["ex"].append(float(ex))
        st["slices"].append([os.path.basename(f), b, r])
        if (i + 1) % 20 == 0 or i < 2:
            print(f"[tr] step {i + 1}/{steps} loss {np.mean(st['loss'][-20:]):.4f} (kl {np.mean(st['kl'][-20:]):.4f}"
                  + (f", extra {np.mean(st['ex'][-20:]):.4f}" if extra or aux else "")
                  + f"); {(time.time() - t1) / (i + 1):.2f}s/step, peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
        if stop["now"]:
            save("stopped"); print(f"[tr] stopped at step {i + 1}: saved", flush=True); sys.exit(3)
    save("done")
    print(f"[tr] DONE {a.run}: {steps} steps in {time.time() - t0:.0f}s; peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
