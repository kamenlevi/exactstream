"""TRL layer-18 check (Leap, Wed 2026-11-04 20:2x; the orchestrator 20:0x: the Metal encoder must match Fit12's C encoder bit
for bit on layer 18, Fit12's store being the reference): Fit12's own pipeline (fit_trl_store.py's per-expert steps, its
modules imported read-only, MLX default device CPU as there, VECLIB / OMP / B3 threads 4 as trl_chunks2.sh) with ONE change:
fit_b3.vit (the C Viterbi) -> trl_metal.recon (the Metal Viterbi on the GPU). Per (expert, projection): the 8-bit view
(w, s, b) == Fit12's stored file, bit for bit; the in-sample error ratio as in the store's log; time split GPU / CPU.
Also: H_x recomputed vs Fit12's cached _hx.npz (bitwise?), which the reference store used.
Pre-registered (20:2x): every view bitwise == the store (else a bug in the kernel, the tie rules or the setup).
usage: env VECLIB_MAXIMUM_THREADS=4 OMP_NUM_THREADS=4 B3_THREADS=4 [BigMoE venv] trl_l18_check.py E0-E1[,E..] [step]
       -> ~/Leap/data/trl_l18_check.txt (GPU tiny + CPU 4 threads, MEM ~3 GB; share lock)"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser("~/Fit12/scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel1/Leap_scripts"))
import fit_q as F
import fit_q2 as Q2
import fit_b3 as B
import mlx.core as mx
mx.set_default_device(mx.cpu)                          # as fit_trl_store.py (q8 and any MLX op on the CPU)
import trl_metal as TM

L = 18
CAP = os.path.expanduser("~/Fit12/data/b10rcap")
REF = os.path.expanduser(f"~/Fit12/data/trl/L{L}")
DAMP = {"gate": 0.3, "up": 0.3, "down": 0.1}
KS = {"gate": 2, "up": 2, "down": 3}                   # the mix
if os.environ.get("TRL_FORM") == "trl3":               # TRL-3: no reference files exist; compare against Fit12's C path
    KS = {"gate": 3, "up": 3, "down": 3}               # in-process instead (TRL_VS_C=1)
OUT = os.path.expanduser("~/Leap/data/trl_l18_check.txt")
TV = [0.0]


def say(s=""):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")


def silu(z):
    return z / (1.0 + np.exp(-z))


def q8(W):                                             # = fit_trl_store.q8 (on the CPU default device)
    a = mx.array(W.astype(np.float32)).astype(mx.bfloat16)
    wq, s, b = mx.quantize(a, group_size=64, bits=8)
    deq = mx.dequantize(wq, s, b, group_size=64, bits=8).astype(mx.float32)
    mx.eval(wq, s, b, deq)
    return {"w": wq, "s": s, "b": b}, np.array(deq).astype(np.float64)


def vit_metal(t, k, lut=None):
    t0 = time.perf_counter()
    r = TM.recon(t, k, lut)
    TV[0] += time.perf_counter() - t0
    return r


def main():
    spec = sys.argv[1] if len(sys.argv) > 1 else "0-7"
    es = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        es += list(range(int(a), int(b or a) + 1, int(sys.argv[2]) if len(sys.argv) > 2 else 1))
    say(f"\nTRL L{L} check {time.strftime('%a %H:%M:%S')}: experts {spec} ({len(es)}); Fit12's pipeline with fit_b3.vit -> "
        f"trl_metal.recon; threads VECLIB {os.environ.get('VECLIB_MAXIMUM_THREADS')} B3 {os.environ.get('B3_THREADS')}")
    assert np.array_equal(B.LUT3.view(np.uint32), TM.LUT3.view(np.uint32))
    global _C_VIT
    _C_VIT = B.vit
    B.vit = vit_metal
    t0 = time.time()
    XF = np.concatenate([B.dedupe(np.load(os.path.join(CAP, f"R{i}", f"x_L{L}.npy")))[0] for i in range(8)])
    XF32 = XF.astype(np.float32)
    Rx = B.rot(2560, L)
    z = np.load(os.path.join(REF, "_hx.npz"))
    Hx, Ux = z["Hx"], z["Ux"]
    Hx2 = XF.T @ XF
    eqh = np.array_equal(Hx2.view(np.uint64), Hx.view(np.uint64))
    Ux2, dead = Q2.prep(Rx.T @ Hx @ Rx, DAMP["gate"])
    equ = np.array_equal(Ux2.view(np.uint64), Ux.view(np.uint64))
    say(f"L{L}: fit rows {XF.shape[0]}; H_x recomputed == cached bitwise {eqh} (max rel diff "
        f"{float(np.abs(Hx2 - Hx).max() / np.abs(Hx).max()):.1e}); U_x from cached H_x == cached U_x {equ}; "
        f"{time.time() - t0:.1f} s (the reference store used the cached H_x / U_x; so does this check)")
    nb = nok = 0
    tcpu = tgpu = 0.0
    for e in es:
        te, TV[0] = time.time(), 0.0
        rec = F.read(L, e)
        W4, W3 = {}, {}
        for p in F.PROJ:
            r = rec[p]
            s3, b3, bb = F.bf16(r["s3u"]), F.bf16(r["b3u"]), F.bf16(r["bu"])
            W4[p] = (np.repeat(s3 / 2, 64, 1) * (2 * r["m"] + r["l"]) + np.repeat(bb, 64, 1)).astype(np.float64)
            W3[p] = (np.repeat(s3, 64, 1) * r["m"] + np.repeat(b3, 64, 1)).astype(np.float64)
        if os.environ.get("TRL_HH64") == "1":          # B10-r's original H_h in fp64 (before fit_trl_store's 14:3x fp32 change)
            hF = silu(XF @ W4["gate"].T) * (XF @ W4["up"].T)
            Hh = hF.T @ hF
        else:
            hF = silu(XF32 @ W4["gate"].T.astype(np.float32)) * (XF32 @ W4["up"].T.astype(np.float32))
            Hh = (hF.T @ hF).astype(np.float64)
        Rh = B.rot(640, 1000 * L + e)
        Uh, dead = Q2.prep(Rh.T @ Hh @ Rh, DAMP["down"])
        assert not dead.any()
        line = []
        for p in F.PROJ:
            k = KS[p]
            R, U, H = (Rh, Uh, Hh) if p == "down" else (Rx, Ux, Hx)
            Wr = W4[p] @ R
            sc = F.rbf16(np.sqrt((Wr * Wr).mean(1)).astype(np.float32)).astype(np.float64)
            Q = B.trl(Wr, U, k, sc, B.LUT3)
            Wd = Q @ R.T
            r_trl = Q2.err(Wd, W4[p], H) / Q2.err(W3[p], W4[p], H)
            view, _ = q8(Wd)
            if os.environ.get("TRL_VS_C") == "1":       # the reference = the same pipeline with Fit12's C Viterbi
                B.vit = _C_VIT
                Qc = B.trl(Wr, U, k, sc, B.LUT3)
                B.vit = vit_metal
                ref, _ = q8(Qc @ R.T)
                ref["qeq"] = np.array_equal(Qc, Q)
            else:
                ref = mx.load(os.path.join(REF, f"e{e:03d}_{p}_k{k}.safetensors"))
            same = all(np.array_equal(np.array(view[a].view(mx.uint8) if view[a].dtype != mx.uint32 else view[a]),
                                      np.array(ref[a].view(mx.uint8) if ref[a].dtype != mx.uint32 else ref[a]))
                       for a in ("w", "s", "b"))
            nb += 1
            nok += same
            line.append(f"{p}k{k} {r_trl:.3f}x {'==' if same else 'DIFF'}")
        dt = time.time() - te
        tgpu += TV[0]
        tcpu += dt - TV[0]
        say(f"L{L} e{e}: {dt:.2f} s (Viterbi calls {TV[0]:.2f} s) | " + " | ".join(line))
    say(f"TRL L{L} check: {nok} / {nb} matrices bitwise == Fit12's store; per expert {tgpu / len(es):.2f} s in Viterbi "
        f"calls (GPU, 160 / 160 / 40 sequential calls), {tcpu / len(es):.2f} s CPU rest (incl. this check's error "
        f"metric and 8-bit view) -> {'PASS' if nok == nb else 'FAIL'}")


if __name__ == "__main__":
    main()
