"""WEEK98 S22 QATC2 trainer (MTP Head; REPORT 'WEEK98 S22 QATC2'): BigMoE's dd_ttt.py training loop, mirrored line for
line with its data / slices / chain loss imported read-only, plus MoonShots' C2 weight cuts IN THE FORWARD:
- P's 10 trained Linears are fake-quantized with a straight-through estimator: forward w + stop_gradient(fq(w) - w), with
  fq = MoonShots' p3_dd_tree.fq (bf16 -> mx.quantize g64 -> dequantize to the stored dtype) at 4 bits, the router at 8.
  The saved weights are the latent fp32 weights, under t30b's keys. M3's harness at cut C2 applies the same fq to them,
  so the read-out sees exactly the trained forward.
- The head's 6 frozen bf16 HC mixes are quantized once at 4 bits (apply_cut's way). The routed experts' top-k is 4.
- The 8K screen is NOT in the loss (S18: training does not recover a screen); it is applied at read-out only.
--nocut runs the same pass without the cuts (the control). Start: t30b's weights. One pass of the manifest (default:
manifest A, 755 unseen n1112 files, 1,479 slices of 192 rows), pass seed 1117, fresh Adam, dd_ttt's schedule (20-step
warm-up, cosine from 1e-5 to 1e-6), depth 30, wdecay 0.97, grad clip 1.0.
Stops on SIGTERM / PAUSE like dd_ttt: it saves weights + Adam + state and exits 3, and a re-run resumes. An MLX peak above
--peak-max saves and exits 4. Checkpoints: ~/datasets/deepdraft/ckpt/dd_<run>{.safetensors,_opt.safetensors,.json}.
--smoke N: N steps on a throwaway name, nothing saved. The bits come from P3.CUTS['C2'] through apply_cut's own
module mapping, so the trained forward equals the read-out's cut by construction.
Usage: BigMoE venv python -I -B dd_qat.py --run t30qa [--nocut] [--smoke 5]"""
import argparse, json, os, signal, sys, time
H = os.path.expanduser
sys.path.insert(0, H("~/BigMoE/scripts")); sys.path.insert(0, H("~/MoonShots/week98"))
import numpy as np                                         # noqa: E402
import dd_ttt as DD                                        # noqa: E402  (read-only: cap_files, pass_slices, cap_window_at, CKD, T, M)
import p3_dd_tree as P3                                    # noqa: E402  (read-only: fq)
import mlx.core as mx                                      # noqa: E402
import mlx.nn as nn                                        # noqa: E402
import mlx.optimizers as optim                             # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten         # noqa: E402

T, M, CKD = DD.T, DD.M, DD.CKD
LIN = ("q_proj", "o_proj", "k_proj", "v_proj", "se_gate", "se_up", "se_down", "fc_hidden", "fc_embedding", "router")


_EFF = {}                                                  # id(FQ) -> (its weight array, the STE weight), outside the module


class FQ(nn.Module):
    """a Linear whose forward reads fq(weight) with a straight-through gradient; the parameter is 'weight' as before"""

    def __init__(self, lin, bits):
        super().__init__()
        self.weight = lin.weight
        self._bits = int(bits)

    def __call__(self, x):
        w = self.weight
        c = _EFF.get(id(self))
        if c is None or c[0] is not w:                 # once per step: the 30 depths share one STE weight (smoke 14:38: a
            c = (w, w + mx.stop_gradient(P3.fq(mx, w, self._bits) - w))   # per-depth copy each took the peak to 18.2 GB)
            _EFF[id(self)] = c
        return x @ c[1].T


C2 = P3.CUTS["C2"]                                         # MoonShots' cut; bits per module by apply_cut's own mapping
KEYS = (("q_proj", "qo"), ("o_proj", "qo"), ("k_proj", "kv"), ("v_proj", "kv"), ("se_gate", "se"), ("se_up", "se"),
        ("se_down", "se"), ("fc_hidden", "fc"), ("fc_embedding", "fc"), ("router", "rt"))
BITS = {n: (8 if k == "rt" else C2[k]) for n, k in KEYS}
assert sorted(BITS) == sorted(LIN)


def cut_head(P, head):
    for n in LIN:
        setattr(P, n, FQ(getattr(P, n), BITS[n]))
    L0 = head.layers[0]
    for g in (L0.attn_hyper_connection, L0.mlp_hyper_connection, head.hyper_connection_mixer):
        for sub in ("input_mix_weight_down", "input_mix_weight_up"):
            lin = getattr(g, sub)
            assert not hasattr(lin, "scales")
            lin.weight = P3.fq(mx, lin.weight, C2["hc"])
    L0.mlp.top_k = int(C2["top"])
    mx.eval(head.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="t30qa"); ap.add_argument("--nocut", action="store_true")
    ap.add_argument("--manifest", default=H("~/TrainLab/data/t30c/split_A.json")); ap.add_argument("--pass-seed", type=int, default=1117)
    ap.add_argument("--depth", type=int, default=30); ap.add_argument("--rows", type=int, default=192)
    ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--wdecay", type=float, default=0.97)
    ap.add_argument("--peak-max", type=float, default=18.0); ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    if a.smoke:
        a.run = "smk_" + a.run
    t0 = time.time()
    rt, lm, head = M.load(0.03)
    P = M.Trainable(head)
    ckw, ckj, cko = (os.path.join(CKD, f"dd_{a.run}{s}") for s in (".safetensors", ".json", "_opt.safetensors"))
    tr, va = DD.cap_files(a.manifest)
    sl = DD.pass_slices(tr, a.rows, a.pass_seed)
    state = {"step": 0, "loss": [], "args": vars(a), "status": "running"}
    if os.path.exists(ckj) and not a.smoke:
        state = json.load(open(ckj)); P.load_weights(ckw)
        print(f"[qat] resume {a.run} at step {state['step']}", flush=True)
    else:
        P.load_weights(os.path.join(CKD, "dd_t30b.safetensors"))
        print("[qat] start from t30b", flush=True)
    if not a.nocut:
        cut_head(P, head)
    P.freeze(); P.unfreeze()
    steps = a.smoke or len(sl)
    sched = lambda s: a.lr * mx.minimum(1.0, (s + 1) / 20.0) * (0.1 + 0.45 * (1 + mx.cos(np.pi * mx.minimum(1.0, s / max(1, steps - 1)))))
    opt = optim.Adam(learning_rate=sched)
    if state["step"]:
        opt.state["step"] = mx.array(state["step"], mx.uint64)
        if os.path.exists(cko):
            opt.init(P.trainable_parameters()); opt.state = tree_unflatten(list(mx.load(cko).items()))
            print(f"[qat] optimizer state restored ({cko})", flush=True)
    cur = {}
    lossf = nn.value_and_grad(P, lambda: T.ttt_loss(lm, head, P, cur["x"], a.depth, a.wdecay, True)[0])
    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    ntr = sum(v.size for _, v in tree_flatten(P.trainable_parameters()))
    print(f"[qat] {a.run}: {'NO cuts (control)' if a.nocut else 'C2 cuts (4-bit STE, router 8, HC mixes 4, top-4)'}; "
          f"{len(tr)} files, one pass = {len(sl)} slices, {steps} steps, depth {a.depth}, rows {a.rows}, lr {a.lr}; "
          f"{ntr / 1e6:.1f}M trainable; loaded in {time.time() - t0:.0f}s", flush=True)

    def save(status):
        if a.smoke:                                      # a smoke saves nothing (no scratch files)
            return
        mx.save_safetensors(ckw + ".tmp.safetensors", dict(tree_flatten(P.parameters())))
        os.replace(ckw + ".tmp.safetensors", ckw)
        mx.save_safetensors(cko + ".tmp.safetensors", dict(tree_flatten(opt.state)))
        os.replace(cko + ".tmp.safetensors", cko)
        state["status"] = status
        json.dump(state, open(ckj + ".tmp", "w")); os.replace(ckj + ".tmp", ckj)
    last = time.time()
    while state["step"] < steps:
        while os.path.exists(T.PAUSE) and not stop["now"]:
            time.sleep(1)
        if stop["now"]:
            save("stopped"); print(f"[qat] stopped at step {state['step']}: saved", flush=True); sys.exit(3)
        f_, b_, r_ = sl[state["step"] % len(sl)]
        cur["x"] = DD.cap_window_at(f_, b_, r_, a.depth)
        t1 = time.time()
        loss, g = lossf()
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(P, g)
        mx.eval(P.parameters(), opt.state, loss)
        state["step"] += 1; state["loss"].append(float(loss))
        pk = mx.get_peak_memory() / 1e9
        if state["step"] % 20 == 0 or state["step"] <= 3:
            print(f"[qat] step {state['step']}/{steps} loss {float(loss):.4f} (mean of last 20 {np.mean(state['loss'][-20:]):.4f}); "
                  f"{time.time() - t1:.1f}s/step, peak {pk:.1f} GB", flush=True)
        if a.peak_max and pk > a.peak_max:
            save("stopped: peak"); print(f"[qat] MLX peak {pk:.1f} GB > {a.peak_max}: stopped and saved", flush=True); sys.exit(4)
        if stop["now"]:
            save("stopped"); print(f"[qat] stopped at step {state['step']}: saved", flush=True); sys.exit(3)
        if time.time() - last > 300:
            save("running"); last = time.time()
    save("done")
    print(f"[qat] DONE {a.run}: {state['step']} steps in {time.time() - t0:.0f}s; peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
    if a.smoke:
        print(f"[qat] SMOKE: C2 bits {BITS}, HC mixes {C2['hc']}-bit, top_k {head.layers[0].mlp.top_k}; loss first / last {state['loss'][0]:.4f} / {state['loss'][-1]:.4f}; nothing saved", flush=True)

if __name__ == "__main__":
    main()
