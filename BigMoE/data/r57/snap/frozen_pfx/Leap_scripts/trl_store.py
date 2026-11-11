"""TRL store format (Leap, Wed 2026-11-04 20:5x; TRELLIS FULL-STORE prep, engine wiring): one file per layer of fixed-size
expert records (16 KB-aligned, so the engine's reader preads whole records into pool slots like records3) + index.json.
Record pieces, in order (byte offsets in index.json):
  gate_w uint32 [40, 160, TWg]   up_w uint32 [40, 160, TWg]   down_w uint32 [160, 40, TWd]   (tiles (r, c), bitstreams
  as trl_encode.py: s_0's 12 bits then k bits per value, MSB first; TW = 17 words at k 2, 25 at k 3)
  gate_s / up_s bf16 [640], down_s bf16 [2560]   (row RMS sc; weight = inst3(state) / RMS3 x sc)
  down_sg int8 [640]   (expert e's h signs, = numpy default_rng(1000 L + e).choice([-1, 1], 640), fit_b3.rot's)
Layer constants (index.json): k per projection, x-sign seed L (gate / up), Hadamard blocks 512 / 128 (Sylvester / sqrt n).
API: write_layer(out, L, recs) (recs: e -> trl_encode's arrays), load_pools(out, L, experts=None) -> tierglu_trl pools.
NO full-store writes without Kamen's OK (the orchestrator 20:0x); test layers only.
usage: [BigMoE venv] trl_store.py test -> encodes 16 L18 experts in-process, writes a 16-expert test layer under
       ~/Leap/data/trl_store_test/, loads it, and runs TRL-G vs the float64 forward of the same codes (GPU small)"""
import json, os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfx/Leap_scripts"))

PAGE = 16384
TWK = {2: 17, 3: 25}
SHP = {"gate": (640, 2560), "up": (640, 2560), "down": (2560, 640)}


def layout(ks):
    """piece table for k per projection: [(name, dtype, shape)]"""
    out = []
    for p in ("gate", "up", "down"):
        o, i = SHP[p]
        out.append((f"{p}_w", "uint32", (o // 16, i // 16, TWK[ks[p]])))
    out += [("gate_s", "uint16", (640,)), ("up_s", "uint16", (640,)), ("down_s", "uint16", (2560,)), ("down_sg", "int8", (640,))]
    return out


def entry(L, ks, E=512, fname=None):
    pieces, off = {}, 0
    for nm, dt, sh in layout(ks):
        n = int(np.prod(sh)) * np.dtype(dt).itemsize
        pieces[nm] = {"off": off, "size": n, "dtype": dt, "shape": list(sh)}
        off += n
    rec = -(-off // PAGE) * PAGE
    return {"file": fname or f"L{L}.trl", "E": E, "recsz": rec, "payload": off, "k": dict(ks), "x_sign_seed": L,
            "h_sign_seed": f"1000*{L}+e", "hadamard": {"x": 512, "h": 128}, "pieces": pieces, "layout": "trl-v1"}


def down_signs(L, e):
    return np.random.default_rng(1000 * L + e).choice(np.array([-1.0, 1.0]), 640).astype(np.int8)


def write_layer(out, L, recs, ks, E=512):
    """recs: e -> {gate_w, up_w, down_w, gate_s, up_s, down_s} (trl_encode.py's arrays); missing experts stay zero.
    Atomic per layer file (temp + os.replace); index.json merged atomically."""
    os.makedirs(out, exist_ok=True)
    en = entry(L, ks, E)
    f = os.path.join(out, en["file"])
    tmp = f + ".tmp"
    with open(tmp, "wb") as fh:
        fh.truncate(en["recsz"] * E)
        for e, a in sorted(recs.items()):
            buf = bytearray(en["recsz"])
            for nm, pc in en["pieces"].items():
                v = down_signs(L, e) if nm == "down_sg" else np.asarray(a[nm])
                v = np.ascontiguousarray(v.astype(pc["dtype"]))
                assert list(v.shape) == pc["shape"], (nm, v.shape, pc["shape"])
                buf[pc["off"]:pc["off"] + pc["size"]] = v.tobytes()
            fh.seek(e * en["recsz"])
            fh.write(buf)
    os.replace(tmp, f)
    ip = os.path.join(out, "index.json")
    ix = json.load(open(ip)) if os.path.exists(ip) else {"format": "trl-v1", "layers": {}}
    ix["layers"][str(L)] = en
    json.dump(ix, open(ip + ".tmp", "w"), indent=1)
    os.replace(ip + ".tmp", ip)
    return en


def load_pools(out, L, experts=None):
    """the layer (or the listed experts, in that order: slot i = experts[i]) as tierglu_trl pools"""
    import tierglu_trl as TR
    en = json.load(open(os.path.join(out, "index.json")))["layers"][str(L)]
    assert en["layout"] == "trl-v1", en["layout"]
    es = list(range(en["E"])) if experts is None else list(experts)
    raw = np.memmap(os.path.join(out, en["file"]), np.uint8, "r").reshape(en["E"], en["recsz"])
    sub = np.ascontiguousarray(raw[es])
    get = lambda nm: sub[:, en["pieces"][nm]["off"]:en["pieces"][nm]["off"] + en["pieces"][nm]["size"]].copy().view(
        en["pieces"][nm]["dtype"]).reshape([len(es)] + en["pieces"][nm]["shape"])
    P = {"kg": en["k"]["gate"], "kd": en["k"]["down"]}
    assert en["k"]["up"] == en["k"]["gate"]
    for p, nm in (("gate", "g"), ("up", "u"), ("down", "d")):
        P["w" + nm] = mx.array(get(f"{p}_w"))
        P["s" + nm] = mx.array(get(f"{p}_s")).view(mx.bfloat16)
    sx = np.random.default_rng(en["x_sign_seed"]).choice(np.array([-1.0, 1.0]), 2560).astype(np.float32)
    P["sx"], P["sdn"] = mx.array(sx), mx.array(get("down_sg").astype(np.float32))
    mx.eval([v for v in P.values() if isinstance(v, mx.array)])
    return P


def test():
    """16 real L18 experts: encode (bitwise pipeline) -> write a test layer -> load -> TRL-G vs float64 forward of the codes,
    and that forward vs the dense W = Q R^T path (fit_b3.rot) on the same x"""
    sys.argv = ["trl_encode.py", "18", "0-15"]
    import trl_encode as TE
    import tierglu_trl as TR
    out = os.path.expanduser("~/Leap/data/trl_store_test")
    L, es, ks = 18, list(range(16)), {"gate": 2, "up": 2, "down": 3}
    XF = np.concatenate([TE.B.dedupe(np.load(os.path.join(os.path.expanduser("~/Fit12/data/b10rcap"), f"R{i}",
                                                           f"x_L{L}.npy")))[0] for i in range(8)])
    z = np.load(os.path.expanduser("~/Fit12/data/trl/L18/_hx.npz"))
    Ux = z["Ux"]
    Rx = TE.B.rot(2560, L)
    XF32 = XF.astype(np.float32)
    recs, dense = {}, {}
    jobs = []
    for e in es:
        rec = TE.read_record(L, e)
        W4 = {}
        for p in TE.F.PROJ:
            r = rec[p]
            s3, bb = TE.F.bf16(r["s3u"]), TE.F.bf16(r["bu"])
            W4[p] = (np.repeat(s3 / 2, 64, 1) * (2 * r["m"] + r["l"]) + np.repeat(bb, 64, 1)).astype(np.float64)
        hF = TE.silu(XF32 @ W4["gate"].T.astype(np.float32)) * (XF32 @ W4["up"].T.astype(np.float32))
        Hh = (hF.T @ hF).astype(np.float64)
        Rh = TE.B.rot(640, 1000 * L + e)
        Uh, dead = TE.Q2.prep(Rh.T @ Hh @ Rh, TE.DAMP["down"])
        for p in TE.F.PROJ:
            R, U = (Rh, Uh) if p == "down" else (Rx, Ux)
            Wr = W4[p] @ R
            sc = TE.F.rbf16(np.sqrt((Wr * Wr).mean(1)).astype(np.float32)).astype(np.float64)
            jobs.append(TE.Job(e, p, ks[p], Wr, U, sc))
        dense[e] = {"Rh": Rh}
    TE.run_jobs(jobs)
    mx.set_default_device(mx.gpu)                         # trl_encode set the CPU default (Fit12's q8); TRL-G runs on the GPU
    for e in es:
        a = {}
        for j in (j for j in jobs if j.e == e):
            o, K = SHP[j.p]
            a[f"{j.p}_w"] = TE.TM.pack_bits(j.st.reshape(-1, 256), j.k).reshape(o // 16, K // 16, TWK[j.k])
            a[f"{j.p}_s"] = (j.sc.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
            R = dense[e]["Rh"] if j.p == "down" else Rx
            dense[e][j.p] = j.Q @ R.T                          # the dense weight these codes stand for (float64)
        recs[e] = a
    t0 = time.time()
    en = write_layer(out, L, recs, ks, E=16)
    P = load_pools(out, L)
    print(f"test layer: {en['recsz']} B per record ({en['payload']} payload), 16 experts written + loaded in "
          f"{time.time() - t0:.1f} s", flush=True)
    rng = np.random.default_rng(3)
    M = 4
    ids = np.stack([rng.choice(16, 10, replace=False) for _ in range(M)]).astype(np.int32)
    x = mx.array(XF[rng.choice(XF.shape[0], M, replace=False)].astype(np.float32)).astype(mx.bfloat16)
    sco = mx.softmax(mx.array(rng.standard_normal((M, 10)).astype(np.float32)), axis=-1).astype(mx.bfloat16)
    y32 = TR.tier_glu_trl(x, P, mx.array(ids), out=mx.float32)
    mx.eval(y32)
    yk = np.array(y32).reshape(M, 10, -1)
    xs = np.array(x.astype(mx.float32)).astype(np.float64)
    worst = worst_d = 0.0
    for m in range(M):
        for r in range(10):
            e = int(ids[m, r])
            g, u = dense[e]["gate"] @ xs[m], dense[e]["up"] @ xs[m]
            yd = dense[e]["down"] @ (g / (1 + np.exp(-g)) * u)
            worst = max(worst, float(np.abs(yk[m, r] - yd).max() / np.abs(yd).max()))
    yr = TR.ref_forward(x, P, ids, sco, pairs=True).reshape(M, 10, -1)
    worst_d = float(np.abs(yk - yr).max() / np.abs(yr).max())
    ynax = TR.tier_glu_trl_nax(x, P, mx.array(ids), out=mx.float32)      # the serving kernel (half operands)
    mx.eval(ynax)
    yn = np.array(ynax).reshape(M, 10, -1)
    worst_n = max(float(np.abs(yn[m, r] - (dense[int(ids[m, r])]["down"] @ (lambda g, u: g / (1 + np.exp(-g)) * u)(
        dense[int(ids[m, r])]["gate"] @ xs[m], dense[int(ids[m, r])]["up"] @ xs[m]))).max() / np.abs(yr[m, r]).max())
        for m in range(M) for r in range(10))
    print(f"TRL-NAX on the same real codes vs the dense path (per pair, relative to the pair's max |y|): {worst_n:.2e}", flush=True)
    ok = worst <= 1e-4 and worst_d <= 1e-4 and worst_n <= 5e-3
    print(f"TRL store test: real L18 codes through the store -> TRL-G (fp32) vs the dense W = Q R^T path: max |dy| / max |y| "
          f"{worst:.2e}; vs the float64 forward of the loaded pools {worst_d:.2e} -> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


if __name__ == "__main__":
    sys.exit(0 if test() else 1)
