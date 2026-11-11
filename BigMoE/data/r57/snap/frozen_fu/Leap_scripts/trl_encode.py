"""TRL full-store encoder (Leap, Wed 2026-11-04 20:3x; the orchestrator 20:0x, TRELLIS FULL-STORE prep): Fit12's TRL recipe
(fit_trl_store.py: RHT + bf16 row RMS + BlockLDLQ + bitshift trellis L 12 on 16 x 16 tiles, 3INST table, damping fixed per
projection) for many experts at once, with the Viterbi on the GPU (trl_metal.py, bitwise == Fit12's C) and everything else
in Fit12's own float64 numpy code (the same calls with the same shapes per expert), so the codes are bit for bit those of
Fit12's pipeline (checked on layer 18: trl_l18_check.py; here --check decodes the codes and compares the 8-bit views).
  Lockstep: B experts x 3 projections run BlockLDLQ as generators (one 16-column block per round); each round gathers
  every pending block of the same k into ONE Viterbi call; two halves ping-pong (GPU on one half while the CPU feeds back
  the other).
  Output per (layer, expert): OUT/L{L}/e{EEE}.npz: for each projection p: {p}_w = uint32 [o/16, K/16, nw] (tile (r, c) =
  rows 16r.., cols 16c.. of W' = W R, row-major in the tile, B10(b)'s bitstream: s_0's 12 bits then q_1..q_255, k bits
  each, MSB first, padded to nw = 17 (k 2) / 25 (k 3) words), {p}_s = uint16 [o] (bf16 row RMS of W' = Fit12's sc; the
  decode multiplies by inst3(s) / RMS3), {p}_k.  W = Q R^T with Q = LUT3[s] x sc, R = diag(signs(seed)) x blockdiag(H)
  (seed L for gate / up, 1000 L + e for down; H = 512 / 128 Hadamard blocks = fit_q.HB).
  NO full-store writes without Kamen's OK (the orchestrator 20:0x): --out is for test experts; the full run waits.
usage: env VECLIB_MAXIMUM_THREADS=4 OMP_NUM_THREADS=4 [BigMoE venv] trl_encode.py LAYER E0-E1 [--form mix|trl3]
       [--batch 16] [--out DIR] [--check] [--cap DIR]  (GPU + CPU; MEM ~4 GB at batch 16; share lock)"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser("~/Fit12/scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_fu/Leap_scripts"))
import fit_q as F
import fit_q2 as Q2
import fit_b3 as B
import mlx.core as mx
mx.set_default_device(mx.cpu)                          # as fit_trl_store.py; the Viterbi kernel runs on mx.gpu
import trl_metal as TM

TT = 16
DAMP = {"gate": 0.3, "up": 0.3, "down": 0.1}
FORMS = {"mix": {"gate": 2, "up": 2, "down": 3}, "trl2": {"gate": 2, "up": 2, "down": 2},
         "trl3": {"gate": 3, "up": 3, "down": 3}}
NW = {2: 17, 3: 25}
TIM = {"vit_wait": 0.0, "feedback": 0.0, "prep": 0.0, "pack": 0.0}


def silu(z):
    return z / (1.0 + np.exp(-z))


# ---- records3 after Thursday's in-place conversion (native+p11+pad): BigMoE's ext rebuilds the row-major record
# (ext.record_rowmajor, tested by BigMoE's scripts/test_layout.py), parsed exactly as fit_q.read parses records3 today.
SH3 = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}
ROWS3 = [SH3[p][0] for p in SH3]
PERS4 = [n for o, i in SH3.values() for n in (o * i // 2, o * i // 64 * 2, o * i // 64 * 2)]
LAYCODE = {"native": 1, "native+p11": 2, "native+p11+pad": 2}


def parse_rowmajor(buf, rm):
    """= fit_q.read's parse of one row-major record (rm = the row-major index entry)"""
    mo, ms, lo, ls = rm["msb_piece_off"], rm["msb_piece_size"], rm["lsb_piece_off"], rm["lsb_piece_size"]
    out = {}
    for i, p in enumerate(F.PROJ):
        o, K = F.SH[p]
        w3 = np.frombuffer(buf, np.uint8, ms[3 * i], mo[3 * i])
        bits = np.unpackbits(w3, bitorder="little").reshape(o, K, 3)
        m = bits[..., 0] + 2 * bits[..., 1] + 4 * bits[..., 2]
        l = np.unpackbits(np.frombuffer(buf, np.uint8, ls[2 * i], lo[2 * i]), bitorder="little").reshape(o, K)
        g = lambda off, sz: np.frombuffer(buf, np.uint16, sz // 2, off).reshape(o, K // 64)
        out[p] = dict(m=m.astype(np.int16), l=l.astype(np.int16), s3u=g(mo[3 * i + 1], ms[3 * i + 1]),
                      b3u=g(mo[3 * i + 2], ms[3 * i + 2]), bu=g(lo[2 * i + 1], ls[2 * i + 1]))
    return out


_EXT = {}


def read_record(L, e, root=None, rm=None):
    """records3 (or another store dir `root`) layer L expert e as fit_q.read returns it, for any layout"""
    import json
    root = root or os.path.expanduser("~/BigMoE/models/bare/records3")
    if "ix" not in _EXT or _EXT.get("root") != root:
        _EXT["ix"], _EXT["root"] = json.load(open(os.path.join(root, "index.json")))["layers"], root
    li = _EXT["ix"][str(L)]
    lay = li.get("layout", "row-major")
    if lay == "row-major":
        assert root == os.path.expanduser("~/BigMoE/models/bare/records3"), "row-major reads go through fit_q.read"
        return F.read(L, e)
    rm = rm or li["rowmajor"]                          # Leap's converter keeps the row-major entry under "rowmajor"
    if "ext" not in _EXT:
        sys.path.insert(0, os.path.expanduser("~/BigMoE/ext/build_nat"))
        import bigmoe_ext
        _EXT["ext"] = bigmoe_ext
    out = mx.zeros((rm["recsz"],), mx.uint8)
    mx.eval(out)
    n = _EXT["ext"].record_rowmajor(os.path.join(root, li["file"]), li["recsz"], li["msb_sz"], PERS4, LAYCODE[lay], ROWS3,
                                    list(li["msb_piece_off"]) + list(li["lsb_piece_off"]), e, out)
    assert n == rm["recsz"], (n, rm["recsz"])
    return parse_rowmajor(np.array(out).tobytes(), rm)


def trl_gen(W, U, k, sc):
    """fit_b3.trl as a generator: yields each block's tiles (float64 [o/16, 256]); receives LUT3[states] (float64)"""
    W = W.copy()
    o, K = W.shape
    Q = np.empty_like(W)
    for i1 in range(0, K, TT):
        i2 = i1 + TT
        WB = W[:, i1:i2]
        rec = yield (WB / sc[:, None]).reshape(o // TT, TT * TT)
        t0 = time.perf_counter()
        QB = rec.reshape(o, TT) * sc[:, None]
        Q[:, i1:i2] = QB
        if i2 < K:
            E = np.linalg.solve(U[i1:i2, i1:i2].T, (WB - QB).T).T
            W[:, i2:] -= E @ U[i1:i2, i2:]
        TIM["feedback"] += time.perf_counter() - t0
    return Q


class Job:
    """one (expert, projection): its generator, the states per tile, and the result"""
    def __init__(self, e, p, k, W, U, sc):
        self.e, self.p, self.k, self.sc = e, p, k, sc
        o, K = W.shape
        self.st = np.zeros((o // TT, K // TT, TT * TT), np.uint16)
        self.c, self.g = 0, trl_gen(W, U, k, sc)
        self.tiles, self.Q = next(self.g), None

    def feed(self, st):
        self.st[:, self.c] = st
        self.c += 1
        try:
            self.tiles = self.g.send(TM.LUT3[st.astype(np.int64)].astype(np.float64))
        except StopIteration as fin:
            self.tiles, self.Q = None, fin.value


def run_jobs(jobs):
    """lockstep over two halves: the GPU runs one half's Viterbi while the CPU feeds the other half back"""
    halves = [jobs[0::2], jobs[1::2]]

    def launch(h):
        out = []
        for k in (2, 3):
            js = [j for j in h if j.tiles is not None and j.k == k]
            if js:
                x = np.concatenate([j.tiles for j in js]).astype(np.float32)
                st = TM.viterbi(x, k)
                mx.async_eval(st)
                out.append((js, st))
        return out

    pend = [launch(halves[0]), launch(halves[1])]
    while pend[0] or pend[1]:
        for i in (0, 1):
            if not pend[i]:
                continue
            t0 = time.perf_counter()
            res = [(js, np.array(st)) for js, st in pend[i]]       # waits for this half; the other half stays queued
            TIM["vit_wait"] += time.perf_counter() - t0
            for js, st in res:
                n0 = 0
                for j in js:
                    n = j.tiles.shape[0]
                    j.feed(st[n0:n0 + n])
                    n0 += n
            pend[i] = launch(halves[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layer", type=int)
    ap.add_argument("experts")
    ap.add_argument("--form", default="mix")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--cap", default=os.path.expanduser("~/Fit12/data/b10rcap"))
    ap.add_argument("--hx", default=None, help="cached _hx.npz (Hx, Ux) to use, e.g. Fit12's L18 cache")
    ap.add_argument("--fit12-out", default=None, help="also write Fit12's store format (fit_trl_store.py: the 8-bit view of "
                    "W = Q R^T, keys w / s / b, same metadata) as DIR/e{EEE}_{proj}_k{k}.safetensors (temp + os.replace)")
    ap.add_argument("--store", default=None, help="write trl-v1 (trl_store.py) layer files here: DIR/L{L}.trl.part while "
                    "running, os.replace to L{L}.trl + index.json once experts 0-511 are all in (E0-E1 must be 0-511)")
    a = ap.parse_args()
    L, ks = a.layer, FORMS[a.form]
    e0, e1 = (int(x) for x in a.experts.split("-"))
    t00 = time.time()
    XF = np.concatenate([B.dedupe(np.load(os.path.join(a.cap, f"R{i}", f"x_L{L}.npy")))[0] for i in range(8)])
    XF32 = XF.astype(np.float32)
    Rx = B.rot(2560, L)
    if a.hx:
        z = np.load(a.hx)
        Hx, Ux = z["Hx"], z["Ux"]
    else:
        Hx = XF.T @ XF
        Ux, dead = Q2.prep(Rx.T @ Hx @ Rx, DAMP["gate"])
        assert not dead.any()
    print(f"[trl_encode] L{L} experts {e0}-{e1} form {a.form} {ks} batch {a.batch}: fit rows {XF.shape[0]}, H_x "
          f"{'cached' if a.hx else 'computed'} {time.time() - t00:.1f} s", flush=True)
    if a.out:
        os.makedirs(os.path.join(a.out, f"L{L}"), exist_ok=True)
    stf = None
    if a.store:                                         # trl-v1: records written in place as experts finish
        import trl_store as TS
        os.makedirs(a.store, exist_ok=True)
        sen = TS.entry(L, {p: ks[p] for p in ("gate", "up", "down")})
        part = os.path.join(a.store, sen["file"] + ".part")
        stf = open(part, "r+b" if os.path.exists(part) else "w+b")
        stf.truncate(sen["recsz"] * sen["E"])
    nchk = nok = 0
    tl = time.time()
    for b0 in range(e0, e1 + 1, a.batch):
        tb = time.time()
        es = list(range(b0, min(b0 + a.batch, e1 + 1)))
        jobs, W4s, Rs, Hs = [], {}, {}, {}
        tp = time.perf_counter()
        for e in es:
            rec = read_record(L, e)
            W4 = {}
            for p in F.PROJ:
                r = rec[p]
                s3, bb = F.bf16(r["s3u"]), F.bf16(r["bu"])
                W4[p] = (np.repeat(s3 / 2, 64, 1) * (2 * r["m"] + r["l"]) + np.repeat(bb, 64, 1)).astype(np.float64)
            hF = silu(XF32 @ W4["gate"].T.astype(np.float32)) * (XF32 @ W4["up"].T.astype(np.float32))
            Hh = (hF.T @ hF).astype(np.float64)
            Rh = B.rot(640, 1000 * L + e)
            Uh, dead = Q2.prep(Rh.T @ Hh @ Rh, DAMP["down"])
            assert not dead.any()
            for p in F.PROJ:
                R, U = (Rh, Uh) if p == "down" else (Rx, Ux)
                Wr = W4[p] @ R
                sc = F.rbf16(np.sqrt((Wr * Wr).mean(1)).astype(np.float32)).astype(np.float64)
                jobs.append(Job(e, p, ks[p], Wr, U, sc))
            W4s[e], Rs[e], Hs[e] = W4, Rh, Hh
        TIM["prep"] += time.perf_counter() - tp
        run_jobs(jobs)
        for e in es:
            js = {j.p: j for j in jobs if j.e == e}
            tq = time.perf_counter()
            arrs = {}
            for p, j in js.items():
                o, K = F.SH[p]
                w = TM.pack_bits(j.st.reshape(-1, TT * TT), j.k).reshape(o // TT, K // TT, NW[j.k])
                arrs[f"{p}_w"], arrs[f"{p}_k"] = w, np.array(j.k)            # (keys gate_w / up_w / down_w = trl-v1's)
                arrs[f"{p}_s"] = (j.sc.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
            TIM["pack"] += time.perf_counter() - tq
            if a.fit12_out:                             # Fit12's file, byte for byte as fit_trl_store.py writes it
                os.makedirs(a.fit12_out, exist_ok=True)
                for p, j in js.items():
                    R = Rs[e] if p == "down" else Rx
                    a8 = mx.array((j.Q @ R.T).astype(np.float32)).astype(mx.bfloat16)
                    wq, s8, b8 = mx.quantize(a8, group_size=64, bits=8)
                    mx.eval(wq, s8, b8)
                    f = os.path.join(a.fit12_out, f"e{e:03d}_{p}_k{j.k}.safetensors")
                    tmp = f[:-len(".safetensors")] + ".tmp.safetensors"
                    mx.save_safetensors(tmp, {"w": wq, "s": s8, "b": b8},
                                        metadata={"layer": str(L), "expert": str(e), "proj": p, "k": str(j.k),
                                                  "damp": str(DAMP[p]), "code": "TRL 3INST L12 BlockLDLQ RHT"})
                    os.replace(tmp, f)
            if stf is not None:
                buf = bytearray(sen["recsz"])
                for nm, pc in sen["pieces"].items():
                    v = TS.down_signs(L, e) if nm == "down_sg" else arrs[nm]
                    v = np.ascontiguousarray(np.asarray(v).astype(pc["dtype"]))
                    assert list(v.shape) == pc["shape"], (nm, v.shape)
                    buf[pc["off"]:pc["off"] + pc["size"]] = v.tobytes()
                stf.seek(e * sen["recsz"])
                stf.write(buf)
            if a.out:
                f = os.path.join(a.out, f"L{L}", f"e{e:03d}.npz")
                np.savez(f[:-4] + ".tmp.npz", **arrs)
                os.replace(f[:-4] + ".tmp.npz", f)
            if a.check:                                 # decode the CODES (not Q) and compare the 8-bit views
                ref_dir = os.path.expanduser(f"~/Fit12/data/trl/L{L}")
                for p, j in js.items():
                    o, K = F.SH[p]
                    st = TM.unpack_states(arrs[f"{p}_w"].reshape(-1, NW[j.k]), j.k).reshape(o // TT, K // TT, TT, TT)
                    sc = (arrs[f"{p}_s"].astype(np.uint32) << 16).view(np.float32).astype(np.float64)
                    Qd = TM.LUT3[st.astype(np.int64)].astype(np.float64).transpose(0, 2, 1, 3).reshape(o, K) * sc[:, None]
                    assert np.array_equal(Qd, j.Q), "codes do not decode to the encoder's Q"
                    R = Rs[e] if p == "down" else Rx
                    Wd = Qd @ R.T
                    a8 = mx.array(Wd.astype(np.float32)).astype(mx.bfloat16)
                    wq, s8, b8 = mx.quantize(a8, group_size=64, bits=8)
                    mx.eval(wq, s8, b8)
                    fr = os.path.join(ref_dir, f"e{e:03d}_{p}_k{j.k}.safetensors")
                    if os.path.exists(fr):
                        ref = mx.load(fr)
                        same = (np.array_equal(np.array(wq), np.array(ref["w"]))
                                and np.array_equal(np.array(s8.view(mx.uint16)), np.array(ref["s"].view(mx.uint16)))
                                and np.array_equal(np.array(b8.view(mx.uint16)), np.array(ref["b"].view(mx.uint16))))
                        nchk += 1
                        nok += same
        n = len(es)
        print(f"[trl_encode] L{L} e{es[0]}-{es[-1]}: {time.time() - tb:.1f} s ({(time.time() - tb) / n:.2f} s per expert); "
              f"cum prep {TIM['prep']:.1f} s, feedback {TIM['feedback']:.1f} s, Viterbi wait {TIM['vit_wait']:.1f} s, "
              f"pack {TIM['pack']:.1f} s" + (f"; check {nok} / {nchk} views bitwise == Fit12's store" if a.check else ""),
              flush=True)
    if stf is not None:
        stf.flush(); os.fsync(stf.fileno()); stf.close()
        if (e0, e1) == (0, 511):                        # a whole layer: publish it (atomic rename + index merge)
            import json
            os.replace(part, os.path.join(a.store, sen["file"]))
            ip = os.path.join(a.store, "index.json")
            ix = json.load(open(ip)) if os.path.exists(ip) else {"format": "trl-v1", "layers": {}}
            ix["layers"][str(L)] = sen
            json.dump(ix, open(ip + ".tmp", "w"), indent=1)
            os.replace(ip + ".tmp", ip)
            print(f"[trl_encode] L{L} published: {os.path.join(a.store, sen['file'])}", flush=True)
    ne = e1 - e0 + 1
    print(f"[trl_encode] done L{L} {ne} experts in {time.time() - tl:.1f} s = {(time.time() - tl) / ne:.3f} s per expert "
          f"(full store at this rate: {(time.time() - tl) / ne * 24576 / 3600:.2f} h)" +
          (f"; CHECK {nok} / {nchk} {'PASS' if nok == nchk else 'FAIL'}" if a.check else ""), flush=True)


if __name__ == "__main__":
    main()
