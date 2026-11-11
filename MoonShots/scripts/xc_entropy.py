"""24 GB wall, new exact ideas (MoonShots Tue 20:0x, CPU only, one screen): can the MSB-3 codes (2.72 b order-0, R-16) or
the LSB bits be coded below their order-0 entropy with context a decoder would have? Real records3 experts of one layer.
  W24-xc  cross-expert: H(m_e | m_f) at the same (projection, row, column) for the best reference f among the other
          sampled experts of the layer (exact conditional coding against a resident expert; pays only if experts share
          structure, e.g. an upcycled model). PASS median best-reference <= 2.40 b; KILL >= 2.65 b.
  W24-gu  within an expert: H(m_up | m_gate) at the same position (gate / up rows feed the same SwiGLU neuron).
          PASS <= 2.60 b; KILL >= 2.69 b.
  W24-gc  per-group context: each group of 64 codes gets a class (8 classes by the group's code variance, quantiles over
          the layer; class index costs 3 b / 64 weights); H(m | class) + 3 / 64. PASS <= 2.62 b; KILL >= 2.68 b.
  W24-lc  LSB sidecar: H(l | m) for the 4th code bit given the MSB-3 code (rank 0-1 misses read it). PASS <= 0.90 b;
          KILL >= 0.97 b.
usage: [BigMoE venv] taskpolicy -b python xc_entropy.py [layer=24] [experts=32] -> stdout + data/xc_entropy.txt"""
import fcntl, json, os, sys, time
import numpy as np

D3 = os.path.expanduser("~/BigMoE/models/bare/records3/")
IX = json.load(open(D3 + "index.json"))["layers"]
L = int(sys.argv[1]) if len(sys.argv) > 1 else 24
NE = int(sys.argv[2]) if len(sys.argv) > 2 else 32
PIECE, W3, L1 = 640 * 2560, 614400, 204800
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def H(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def unpack3(p):
    t = p.reshape(-1, 3).astype(np.uint32)
    w = t[:, 0] | (t[:, 1] << 8) | (t[:, 2] << 16)
    return ((w[:, None] >> (3 * np.arange(8, dtype=np.uint32))) & 7).astype(np.uint8).reshape(-1)


def main():
    t0 = time.time()
    ix = IX[str(L)]
    ex = sorted(np.random.default_rng(5).choice(512, NE, replace=False).tolist())
    fd = os.open(D3 + ix["file"], os.O_RDONLY)
    fcntl.fcntl(fd, 48, 1)
    m = np.empty((NE, 3, PIECE), np.uint8)
    lb = np.empty((NE, 3, PIECE), np.uint8)
    for a, e in enumerate(ex):
        for p in range(3):
            o = e * ix["recsz"]
            m[a, p] = unpack3(np.frombuffer(os.pread(fd, W3, o + ix["msb_piece_off"][3 * p]), np.uint8))
            lb[a, p] = np.unpackbits(np.frombuffer(os.pread(fd, L1, o + ix["lsb_piece_off"][2 * p]), np.uint8), bitorder="little")
    os.close(fd)
    say(f"layer {L}, {NE} experts {ex[:8]}..., read in {time.time() - t0:.1f} s")
    h0 = [H(np.bincount(m[a].ravel(), minlength=8)) for a in range(NE)]
    say(f"order-0 H(m) per expert: median {np.median(h0):.4f} b (min {min(h0):.4f}, max {max(h0):.4f})")
    # W24-xc: best reference per expert
    flat = m.reshape(NE, -1).astype(np.uint16)
    best = np.full(NE, 9.0)
    for a in range(NE):
        for b in range(a + 1, NE):
            j = np.bincount(flat[a] * 8 + flat[b], minlength=64).reshape(8, 8)
            hj = H(j.ravel())
            best[a] = min(best[a], hj - H(j.sum(0)))       # H(a | b)
            best[b] = min(best[b], hj - H(j.sum(1)))       # H(b | a)
    v = float(np.median(best))
    say(f"W24-xc: best-reference H(m_e | m_f) over {NE - 1} references: median {v:.4f} b (min {best.min():.4f}, max "
        f"{best.max():.4f}); gain vs order-0 {np.median(h0) - v:.4f} b -> "
        f"{'PASS' if v <= 2.40 else ('KILL' if v >= 2.65 else 'BETWEEN')}")
    # W24-gu: up given gate at the same position
    hgu = []
    for a in range(NE):
        j = np.bincount(m[a, 1].astype(np.uint16) * 8 + m[a, 0], minlength=64).reshape(8, 8)
        hgu.append(H(j.ravel()) - H(j.sum(0)))
    v = float(np.median(hgu))
    say(f"W24-gu: H(m_up | m_gate) median {v:.4f} b (H(m_up) {np.median([H(np.bincount(m[a, 1], minlength=8)) for a in range(NE)]):.4f}) -> "
        f"{'PASS' if v <= 2.60 else ('KILL' if v >= 2.69 else 'BETWEEN')}")
    # W24-gc: 8 variance classes per group of 64 (gate / up rows: 40 groups; down rows: 10 groups, all contiguous in K)
    g = m.reshape(-1, 64).astype(np.float32)
    var = g.var(1)
    edges = np.quantile(var, np.linspace(0, 1, 9)[1:-1])
    cls = np.searchsorted(edges, var)
    hc = 0.0
    for c in range(8):
        sel = g[cls == c].astype(np.uint8).ravel()
        hc += len(sel) / g.size * H(np.bincount(sel, minlength=8))
    v = hc + 3 / 64
    say(f"W24-gc: H(m | 8 variance classes) {hc:.4f} + class cost {3 / 64:.4f} = {v:.4f} b (pooled order-0 "
        f"{H(np.bincount(m.ravel(), minlength=8)):.4f}) -> {'PASS' if v <= 2.62 else ('KILL' if v >= 2.68 else 'BETWEEN')}")
    # W24-lc: LSB bit given the MSB-3 code
    j = np.bincount(m.ravel().astype(np.uint16) * 2 + lb.ravel(), minlength=16).reshape(8, 2)
    v = H(j.ravel()) - H(j.sum(1))
    say(f"W24-lc: H(l) {H(j.sum(0)):.4f} b, H(l | m) {v:.4f} b; P(l = 1 | m) {np.round(j[:, 1] / j.sum(1), 3).tolist()} -> "
        f"{'PASS' if v <= 0.90 else ('KILL' if v >= 0.97 else 'BETWEEN')}")
    say(f"({time.time() - t0:.0f} s)")
    open(os.path.expanduser(f"~/MoonShots/data/xc_entropy_L{L}.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
