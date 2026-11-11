"""R-44 (b) check before the POM A/B (NewIdeas 10-28): on a few experts, the 2-plane slot must reproduce exactly the tier-2
values MoonShots measured M-03 with, i.e. the Eval suite's tier math (evalsuite/model.py tier_weights, n = 2) on the
4-bit safetensors pieces: codes q4 & 12 at scale s4 and bias bf16(b4 + s4 * mean(q4 & 3)). The engine's 2-plane slot is
(codes 2 m2 + 0 in MLX 3-bit, scale s3 = 2 s4, bias2 from records3p seg1): weight = s3 * 2 m2 + bias2. Checked bitwise:
bias2 == the suite's bias; 4 m2 == the suite's codes; s3 == 2 s4 (so s3 * 2 m2 == s4 * 4 m2 exactly). Also the full slot.
usage: [BigMoE venv] m03_suite_check.py [layers=0,23,47] [n_experts=8]   (CPU, read-only)"""
import fcntl, json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
import mlx.core as mx
from bigmoe.stream import read_headers, PIECES, DT
from evalsuite.model import tier_weights

mx.set_default_device(mx.cpu)
MD = os.path.expanduser("~/BigMoE/models/bare")
LAYERS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0,23,47").split(",")]
NE = int(sys.argv[2]) if len(sys.argv) > 2 else 8
ixp = json.load(open(os.path.join(MD, "records3p", "index.json")))["layers"]
hdr = read_headers(MD)
ok = True


def unpack2(b):
    return ((b[..., None] >> (2 * np.arange(4, dtype=np.uint8))) & 3).reshape(-1)


for L in LAYERS:
    rp = ixp[str(L)]
    fd = os.open(os.path.join(MD, "records3p", rp["file"]), os.O_RDONLY); fcntl.fcntl(fd, 48, 1)
    pre = f"language_model.model.layers.{L}.mlp.switch_mlp"
    ids = sorted(np.random.default_rng(7 + L).choice(rp["E"], NE, replace=False).tolist())
    for e in ids:
        rec = np.frombuffer(os.pread(fd, rp["seg1"], e * rp["recsz"]), np.uint8)
        c1, NWs, Gs = 0, [], []
        for p, proj in enumerate(("gate_proj", "up_proj", "down_proj")):
            arrs = []
            for field in ("weight", "scales", "biases"):
                path, off, dt, shape = hdr[f"{pre}.{proj}.{field}"]
                per = int(np.prod(shape[1:])) * np.dtype(DT[dt][0]).itemsize
                with open(path, "rb") as g:
                    fcntl.fcntl(g.fileno(), 48, 1)
                    g.seek(off + e * per); raw = g.read(per)
                arrs.append(mx.array(np.frombuffer(raw, DT[dt][0]).reshape(shape[1:])).view(DT[dt][1]))
            w4, s4, b4 = arrs
            wm, bias = tier_weights(w4, s4, b4, 64, 2)            # the suite's tier-2 (codes, bias)
            mx.eval(wm, bias)
            q4m = np.array(wm).view(np.uint32).reshape(-1)
            codes_suite = ((q4m[:, None] >> (4 * np.arange(8, dtype=np.uint32))) & 15).reshape(-1)
            NWs.append(codes_suite.size); Gs.append(int(np.prod(s4.shape)))
            m2 = unpack2(rec[c1:c1 + codes_suite.size // 4]); c1 += codes_suite.size // 4
            good_codes = np.array_equal(codes_suite, 4 * m2.astype(np.uint32))
            # seg1 fields of projection p
            ok &= good_codes
            arrs_np = (np.array(s4.view(mx.uint16)).reshape(-1), np.array(bias.view(mx.uint16)).reshape(-1))
            if p == 0:
                keep = []
            keep.append((arrs_np, good_codes))
        # unpack the seg1 fields (s3, bias2) as the ext does and compare with 2 s4 and the suite's bias
        o1 = sum(n // 4 for n in NWs)
        h = rec[o1 + sum(2 * g + g // 2 for g in Gs):][:8]
        assert bytes(h[:2]) == b"P2", h[:2]
        for p in range(3):
            g = Gs[p]
            A, B = rec[o1:o1 + g].astype(np.uint32), rec[o1 + g:o1 + 2 * g].astype(np.uint32)
            N = rec[o1 + 2 * g:o1 + 2 * g + g // 2].astype(np.uint32)
            o1 += 2 * g + g // 2
            nb = np.stack([N & 0xF, N >> 4], 1).reshape(-1)
            es = int(h[5 + p]) + ((B >> 7) | ((nb & 3) << 1))
            s3 = (((A & 0x80) << 8) | (es << 7) | (A & 0x7F)).astype(np.uint16)
            b2 = ((((A & 0x80) ^ 0x80) << 8) | (((es + int(np.int8(h[2 + p])) + (nb >> 2)) & 0xFF) << 7) | (B & 0x7F)).astype(np.uint16)
            (s4u, bias_suite), good_codes = keep[p]
            s4_from_s3 = np.where((s3 & 0x7FFF) == 0, s3, s3 - 0x80).astype(np.uint16)
            gs, gb = np.array_equal(s4_from_s3, s4u), np.array_equal(b2, bias_suite)
            ok &= gs and gb
            if not (gs and gb and good_codes):
                print(f"L{L} e{e} p{p}: codes {good_codes} scale {gs} bias {gb} (bias mismatches {int((b2 != bias_suite).sum())})")
    os.close(fd)
    print(f"L{L}: {NE} experts x 3 projections: 2-plane slot == the suite's tier-2 (codes, scale, bias): "
          f"{'all bitwise' if ok else 'MISMATCH'}", flush=True)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
