> (a quote from the project owner, kept private)
ONE process on the same slices in the same order, each arm its own drafter copy and Adam (continued from the base's
saved state with --resume-opt, so no fresh-Adam drift). One data load per step, one model load per batch.
Defaults are the FAST screen: rollout depth 12 (~2x cheaper per step than 30; the read-out then goes to depth 15:
bands 0-14 and kept/16), lr 1e-5 continuing s30o's Adam. --save-at keeps copies for budget calibration.
Memory: ~1.5 GB shared (embedding + bf16 head) + ~0.75 GB per arm (fp32 weights + Adam) + one arm's step at a time
(~1.5 GB): 2 arms ~4.5 GB, 3 arms ~5.2 GB, 4 arms ~6 GB. By day keep a run under ~5 GB (RULES: one > 5 GB job at a time).
usage: train_multi.py --tag T --arms ctl:-,distill:distill [--start s30o --resume-opt s30o] [--steps 400] [--depth 12]
                      [--lr 1e-5] [--epochs 2] [--save-at 100,200,400]
-> ckpt/T_ARM.safetensors (+ T_ARM_sN copies), ckpt/T.json (per arm per-step loss / kl / extra, wall, s/step)"""
import argparse, json, os, signal, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from sb import core as C
from sb.train import load_lever, split


class Arm:
    def __init__(self, name, lever, a):
        self.name, self.lever = name, lever
        self.D = C.load_drafter(a.start)
        self.L = load_lever(lever) if lever and lever != "-" else None
        L = self.L
        if L is not None and hasattr(L, "setup"):
            L.setup(self.D, a)
        self.D.freeze(); self.D.unfreeze()
        if L is not None and hasattr(L, "freeze"):
            L.freeze(self.D)
        self.standin = getattr(L, "standin", None) if L is not None else None
        self.extra = getattr(L, "extra", None) if L is not None else None
        self.aux = getattr(L, "aux", None) if L is not None else None
        self.s0 = 0
        if a.const:
            sched = lambda s: a.lr * mx.minimum(1.0, (s - self.s0 + 1) / a.warmup)
        else:                                   # 22:3x: anneal like the base (cosine to 10 %), from this run's own step 0
            sched = lambda s: a.lr * mx.minimum(1.0, (s - self.s0 + 1) / a.warmup) * (
                0.1 + 0.45 * (1 + mx.cos(np.pi * mx.minimum(1.0, (s - self.s0) / max(1, a.steps - 1)))))
        self.opt = optim.Adam(learning_rate=sched, bias_correction=True)
        if a.resume_opt:
            sto = tree_unflatten(list(mx.load(os.path.join(C.CKPT, a.resume_opt + "_opt.safetensors")).items()))
            self.opt.init({k: v for k, v in self.D.trainable_parameters().items() if k in sto})
            self.opt.state = sto
            self.s0 = int(sto["step"].item())
        self.prefix = getattr(L, "PREFIX", None) if L is not None else None
        # SPLIT_CLIP (B98C, 11-17): the head's gradients are clipped (global norm 1.0) apart from the module's, so a new
        # module's gradient never shrinks the head's own steps (with the module's output at zero, the head's first step
        # is then bitwise ctl's). Off by default: every earlier lever shared one clip.
        self.split_clip = bool(getattr(L, "SPLIT_CLIP", False)) if L is not None else False
        mod_lr = getattr(L, "MODULE_LR", None) if L is not None else None
        self.optx = optim.Adam(learning_rate=lambda s: mod_lr * mx.minimum(1.0, (s + 1) / a.warmup), bias_correction=True) \
            if self.prefix and mod_lr else None
        self.log = {"loss": [], "kl": [], "ex": []}

    def step(self, x, a):
        C.REPL["fn"] = self.standin
        loss, kl, ex, g = C.ttt_step(self.D, x, a.depth, a.wdecay, self.extra, self.aux)
        if self.optx is not None and self.split_clip:
            gh, nh = optim.clip_grad_norm({k: v for k, v in g.items() if not k.startswith(self.prefix)}, 1.0)
            gx, nx = optim.clip_grad_norm({k: v for k, v in g.items() if k.startswith(self.prefix)}, 1.0)
            self.opt.update(self.D, gh)
            self.optx.update(self.D, gx)
            mx.eval(self.D.parameters(), self.opt.state, self.optx.state)
            self.log.setdefault("gnorm", []).append([float(nh), float(nx)])
            C.REPL["fn"] = None
            self.log["loss"].append(float(loss)); self.log["kl"].append(float(kl)); self.log["ex"].append(float(ex))
            return
        g, _ = optim.clip_grad_norm(g, 1.0)
        if self.optx is not None:
            self.opt.update(self.D, {k: v for k, v in g.items() if not k.startswith(self.prefix)})
            self.optx.update(self.D, {k: v for k, v in g.items() if k.startswith(self.prefix)})
            mx.eval(self.D.parameters(), self.opt.state, self.optx.state)
        else:
            self.opt.update(self.D, g)
            mx.eval(self.D.parameters(), self.opt.state)
        C.REPL["fn"] = None
        self.log["loss"].append(float(loss)); self.log["kl"].append(float(kl)); self.log["ex"].append(float(ex))

    def save(self, tag, suffix=""):
        p = os.path.join(C.CKPT, f"{tag}_{self.name}{suffix}.safetensors")
        mx.save_safetensors(p + ".tmp.safetensors", dict(tree_flatten(self.D.parameters())))
        os.replace(p + ".tmp.safetensors", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--arms", required=True)
    ap.add_argument("--start", default="s30o"); ap.add_argument("--resume-opt", dest="resume_opt", default="s30o")
    ap.add_argument("--steps", type=int, default=400); ap.add_argument("--rows", type=int, default=96)
    ap.add_argument("--depth", type=int, default=12); ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wdecay", type=float, default=0.97); ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1115); ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--files", default="pool2"); ap.add_argument("--save-at", dest="save_at", default="")
    ap.add_argument("--const", action="store_true", help="constant lr (the 21:4x format; drifted at 1e-5)")
    ap.add_argument("--mem", type=float, default=4.5)
    a = ap.parse_args()
    C.limit_memory(a.mem)
    t0 = time.time()
    C.load_target(); C.drop_trunk()
    tr, _ = split(a.files)
    sl = []
    for e in range(a.epochs):
        sl += C.pass_slices(tr, a.rows, a.seed + e)
    steps = a.steps = min(a.steps, len(sl))              # the arms' cosine spans exactly this run
    arms = [Arm(*spec.split(":"), a) for spec in a.arms.split(",")]
    save_at = {int(s) for s in a.save_at.split(",") if s}
    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    print(f"[tm] {a.tag}: arms {[(m.name, m.lever) for m in arms]}, start {a.start} (Adam {a.resume_opt or 'fresh'}), {steps} steps "
          f"of {a.rows} on {a.files} ({len(sl)} slices in {a.epochs} passes), depth {a.depth}, lr {a.lr} {'const' if a.const else 'cosine'}; ready in "
          f"{time.time() - t0:.0f}s", flush=True)
    st = {"args": vars(a), "arms": {}, "status": "running"}

    def dump(status):
        st["status"] = status; st["wall_s"] = time.time() - t0
        for m in arms:
            st["arms"][m.name] = {"lever": m.lever, **m.log}
        jf = os.path.join(C.CKPT, a.tag + ".json")
        json.dump(st, open(jf + ".tmp", "w")); os.replace(jf + ".tmp", jf)
    t1 = time.time()
    for i in range(steps):
        C.wait_pause()
        f, b, r = sl[i]
        x = C.window_at(f, b, r, a.depth)
        for m in arms:
            m.step(x, a)
        if (i + 1) in save_at:
            for m in arms:
                m.save(a.tag, f"_s{i + 1}")
        if (i + 1) % 20 == 0 or i < 2:
            print(f"[tm] step {i + 1}/{steps} " + " ".join(f"{m.name} {np.mean(m.log['loss'][-20:]):.4f}" for m in arms)
                  + f"; {(time.time() - t1) / (i + 1):.2f}s/step ({len(arms)} arms), peak {mx.get_peak_memory() / 1e9:.1f} GB",
                  flush=True)
        if stop["now"]:
            for m in arms:
                m.save(a.tag)
            dump("stopped"); print(f"[tm] stopped at step {i + 1}: saved", flush=True); sys.exit(3)
    for m in arms:
        m.save(a.tag)
    st["s_per_step"] = (time.time() - t1) / max(steps, 1)
    dump("done")
    print(f"[tm] DONE {a.tag}: {steps} steps x {len(arms)} arms in {time.time() - t0:.0f}s "
          f"({st['s_per_step']:.2f}s/step); peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
