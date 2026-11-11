> (a quote from the project owner, kept private)
have tested it"; map screen 7 with NewIdeas). B10-r's data-aware trellis codes (QTIP-style: RHT + bf16 row RMS + BlockLDLQ +
bitshift trellis L 12 on 16 x 16 tiles, the 3INST table = Leap's decodable code), encoded for EVERY expert of the chosen
layers, from B10-r's own capture (/Users/kamen/Fit12/data/b10rcap: layers 2 7 12 18 24 30 36 41 46; fit rows R0..R7 = the
same broad calibration as B10-r; CPU only, no GPU).
  Per layer: H_x = X^T X over the fit rows, rotated once (gate / up share it). Per expert: H_h = h^T h with
  h = silu(X Wg^T) * (X Wu^T) through the expert's own W4 (down), rotated by the expert's own RHT (seed 1000 L + e).
  Damping FIXED per projection = B10-r's majority validation-best (gate 31/36 and up 32/36 at 0.3; down 18/36 at 0.1, 17 at
  0.3): a deviation from B10-r's per-matrix choice (which needs 3 encodes per matrix).
  FORMS: mix = gate / up k 2 + down k 3 (2.40 b); trl2 = 2 / 2 / 2 (2.06 b); trl3 = 3 / 3 / 3 (3.07 b); all = every k.
Output per (layer, expert, projection, k): the decoded weight W = Q R^T as an MLX 8-bit affine view (group 64, bf16 scales
and biases) for the map harness's gather_qmm, in OUT/L{L}/e{EEE}_{proj}_k{k}.safetensors (keys w, s, b). 8-bit rounding
adds ~0.4 % of W4's own error (~0.1-0.3 % of TRL's); measured per matrix below.
Self-checks: R orthogonal per layer; per matrix the in-sample H-weighted error vs W4 in MSB-3 units (B10-r's metric on the
fit rows; held-out levels were B10-r's 1.74-1.85x for TRL-2, ~0.47-0.5x for TRL-3) and the 8-bit view's added error.
Guards: disk >= 112 GB free (the floor is 110), swap growth < 2 GB, MLX on the CPU. Resumable: finished files are skipped.
Usage: [BigMoE venv] taskpolicy -b python -B fit_trl_store.py OUT LAYERS [E0-E1] [FORM]
       e.g. fit_trl_store.py /Users/kamen/Fit12/data/trl 18,30 0-511 mix   (B3_THREADS / VECLIB_MAXIMUM_THREADS = 4)"""
import json, os, subprocess, sys, time
import numpy as np
os.environ.setdefault("B3_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_q as F
import fit_q2 as Q2
import fit_b3 as B
import mlx.core as mx
mx.set_default_device(mx.cpu)

CAP = os.environ.get("F12_TRL_CAP", os.path.expanduser("~/Fit12/data/b10rcap"))
CAP_LAYERS = (2, 7, 12, 18, 24, 30, 36, 41, 46)
DAMP = {"gate": 0.3, "up": 0.3, "down": 0.1}
FORMS = {"mix": {"gate": (2,), "up": (2,), "down": (3,)}, "trl2": {"gate": (2,), "up": (2,), "down": (2,)},
         "trl3": {"gate": (3,), "up": (3,), "down": (3,)}, "all": {"gate": (2, 3), "up": (2, 3), "down": (2, 3)}}
DISK_FLOOR_GB, SWAP_GROW_MB = 112.0, 2048


def silu(z):
    return z / (1.0 + np.exp(-z))


def free_gb(p):
    st = os.statvfs(p)
    return st.f_bavail * st.f_frsize / 1e9


def swap_mb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(out.split("used = ")[1].split("M")[0])


def q8(W):
    """dense W [o, K] (float64) -> MLX 8-bit affine view (group 64, bf16) and its dequantized values (float64)"""
    a = mx.array(W.astype(np.float32)).astype(mx.bfloat16)
    wq, s, b = mx.quantize(a, group_size=64, bits=8)
    deq = mx.dequantize(wq, s, b, group_size=64, bits=8).astype(mx.float32)
    mx.eval(wq, s, b, deq)
    return {"w": wq, "s": s, "b": b}, np.array(deq).astype(np.float64)


def path(out, L, e, p, k):
    return os.path.join(out, f"L{L}", f"e{e:03d}_{p}_k{k}.safetensors")


def main():
    out, layers = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
    e0, e1 = (int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0-511").split("-"))
    form = sys.argv[4] if len(sys.argv) > 4 else "mix"
    ks = FORMS[form]
    assert all(L in CAP_LAYERS for L in layers), f"layers must be captured ones {CAP_LAYERS}"
    os.makedirs(out, exist_ok=True)
    sw0, t00 = swap_mb(), time.time()
    print(f"[trl] start {time.strftime('%H:%M:%S')}: layers {layers}, experts {e0}-{e1}, form {form} {ks}, damping {DAMP}, "
          f"B3_THREADS {B.NTHR}, disk {free_gb(out):.1f} GB free, swap {sw0:.0f} MB, cap {CAP}", flush=True)
    for L in layers:
        os.makedirs(os.path.join(out, f"L{L}"), exist_ok=True)
        t0 = time.time()
        XF = np.concatenate([B.dedupe(np.load(os.path.join(CAP, f"R{i}", f"x_L{L}.npy")))[0] for i in range(8)])
        XF32 = XF.astype(np.float32)              # (Wed 14:3x) H_h in fp32: ~4x faster than B10-r's fp64, same H to ~1e-6
        Rx = B.rot(2560, L)
        assert np.allclose(Rx @ Rx.T, np.eye(2560), atol=1e-9), "Rx not orthogonal"
        hxf = os.path.join(out, f"L{L}", "_hx.npz")   # cached per layer (chunked runs): H_x and its factor at DAMP['gate']
        if os.path.exists(hxf):
            z = np.load(hxf)
            Hx, Ux = z["Hx"], z["Ux"]
        else:
            Hx = XF.T @ XF
            Ux, dead = Q2.prep(Rx.T @ Hx @ Rx, DAMP["gate"])
            assert not dead.any()
            np.savez(hxf[:-4] + ".tmp.npz", Hx=Hx, Ux=Ux)
            os.replace(hxf[:-4] + ".tmp.npz", hxf)
        print(f"[trl] L{L}: fit rows {XF.shape[0]}, H_x + prep {time.time() - t0:.1f} s", flush=True)
        acc = {}
        for e in range(e0, e1 + 1):
            todo = [(p, k) for p in F.PROJ for k in ks[p] if not os.path.exists(path(out, L, e, p, k))]
            if not todo:
                continue
            if free_gb(out) < DISK_FLOOR_GB:
                print(f"[trl] STOP: disk {free_gb(out):.1f} GB free < {DISK_FLOOR_GB}", flush=True)
                sys.exit(3)
            if swap_mb() - sw0 > SWAP_GROW_MB:
                print(f"[trl] STOP: swap grew {swap_mb() - sw0:.0f} MB", flush=True)
                sys.exit(4)
            te = time.time()
            rec = F.read(L, e)
            W4, W3 = {}, {}
            for p in F.PROJ:
                r = rec[p]
                s3, b3, bb = F.bf16(r["s3u"]), F.bf16(r["b3u"]), F.bf16(r["bu"])
                W4[p] = (np.repeat(s3 / 2, 64, 1) * (2 * r["m"] + r["l"]) + np.repeat(bb, 64, 1)).astype(np.float64)
                W3[p] = (np.repeat(s3, 64, 1) * r["m"] + np.repeat(b3, 64, 1)).astype(np.float64)
            th = time.time()
            hF = silu(XF32 @ W4["gate"].T.astype(np.float32)) * (XF32 @ W4["up"].T.astype(np.float32))
            Hh = (hF.T @ hF).astype(np.float64)
            Rh = B.rot(640, 1000 * L + e)
            Uh, dead = Q2.prep(Rh.T @ Hh @ Rh, DAMP["down"])
            assert not dead.any()
            th = time.time() - th
            line = []
            for p, k in todo:
                R, U, H = (Rh, Uh, Hh) if p == "down" else (Rx, Ux, Hx)
                Wr = W4[p] @ R
                sc = F.rbf16(np.sqrt((Wr * Wr).mean(1)).astype(np.float32)).astype(np.float64)
                t1 = time.time()
                Q = B.trl(Wr, U, k, sc, B.LUT3)
                tq = time.time() - t1
                Wd = Q @ R.T
                r_trl = Q2.err(Wd, W4[p], H) / Q2.err(W3[p], W4[p], H)
                view, deq = q8(Wd)
                r_8 = Q2.err(deq, W4[p], H) / Q2.err(Wd, W4[p], H) - 1.0
                f = path(out, L, e, p, k)
                tmp = f[:-len(".safetensors")] + ".tmp.safetensors"
                mx.save_safetensors(tmp, view, metadata={"layer": str(L), "expert": str(e), "proj": p, "k": str(k),
                                                         "damp": str(DAMP[p]), "code": "TRL 3INST L12 BlockLDLQ RHT"})
                os.replace(tmp, f)
                acc.setdefault((p, k), []).append((r_trl, r_8, tq))
                line.append(f"{p}k{k} {r_trl:.3f}x ({tq:.1f} s, 8-bit +{100 * r_8:.2f} %)")
            print(f"[trl] L{L} e{e}: {time.time() - te:.1f} s (H_h {th:.1f} s) | " + " | ".join(line), flush=True)
        for (p, k), v in sorted(acc.items()):
            a = np.array(v)
            print(f"[trl] L{L} summary {p} k{k}: {len(v)} experts, in-sample err vs MSB-3 mean {a[:, 0].mean():.3f}x "
                  f"(min {a[:, 0].min():.3f}, max {a[:, 0].max():.3f}), 8-bit view +{100 * a[:, 1].mean():.2f} %, "
                  f"encode {np.median(a[:, 2]):.2f} s per matrix", flush=True)
    print(f"[trl] done {time.strftime('%H:%M:%S')} ({time.time() - t00:.0f} s), disk {free_gb(out):.1f} GB free, "
          f"swap {swap_mb():.0f} MB", flush=True)


if __name__ == "__main__":
    main()
