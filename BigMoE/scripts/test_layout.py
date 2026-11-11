"""Test of the ext's converted-records readers (ext/build_nat; docs/PLAN_records3_convert.md section 3.6), CPU only, on
scratch copies written by scripts/r3layout_ref.py (never on records3). Per scratch layer:
  (a) record_rowmajor (the ext's native / native+p11 -> today's row-major record) == the records3 record, every byte,
      for all 512 experts;
  (b) read_rebuilt_layout on the scratch == read_rebuilt on records3: the 9 4-bit prefill pieces of 64 experts (shuffled
      ids, scattered rows), bitwise;
  (c) the guard: native+p11 offsets on a row-major file abort (a p11 piece without its header);
  (d) the tier pools (base MSB part + LSB sidecar, piece sizes from the converted index) load 24 experts through the
      loader (ext.ensure): every slot holds its record's part bytes, piece by piece.
usage: .venv/bin/python scripts/test_layout.py [scratch dirs...]   (default scratch/r3p11 scratch/r3nat; native+p11+pad
copies too, e.g. scratch/r3p11pad: the pool check takes each part's pieces in record order)"""
import json, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ext", "build_nat"))
import mlx.core as mx
import bigmoe_ext as ext

R3 = os.path.join(ROOT, "models", "bare", "records3")
SH = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}
ROWS3 = [SH[p][0] for p in ("gate_proj", "up_proj", "down_proj")]
PERS4 = [n for o, i in SH.values() for n in (o * i // 2, o * i // 64 * 2, o * i // 64 * 2)]
CODE = {"row-major": 0, "native": 1, "native+p11": 2, "native+p11+pad": 2}
ix0 = json.load(open(os.path.join(R3, "index.json")))["layers"]


def offs(e):
    return list(e["msb_piece_off"]) + list(e["lsb_piece_off"])


def rebuilt(path, e, ids, layout):
    dst = [mx.zeros((len(ids), p), mx.uint8) for p in PERS4]
    mx.eval(*dst)
    rows = list(range(len(ids)))[::-1]                   # scattered rows: expert ids[k] into row len - 1 - k
    if layout:
        ext.read_rebuilt_layout(path, e["recsz"], e["msb_sz"], PERS4, ids, rows, dst, layout, ROWS3, offs(e))
    else:
        ext.read_rebuilt(path, e["recsz"], e["msb_sz"], PERS4, ids, rows, dst)
    return [np.array(d) for d in dst]


def main(dirs):
    assert ext.layout_max() == 2
    for D in dirs:
        ix = json.load(open(os.path.join(ROOT, D, "index.json")))["layers"]
        for L, e in sorted(ix.items(), key=lambda kv: int(kv[0])):
            r, lay = ix0[L], CODE[e["layout"]]
            p_new, p_old = os.path.join(ROOT, D, e["file"]), os.path.join(R3, r["file"])
            out = mx.zeros((r["recsz"],), mx.uint8)
            mx.eval(out)
            bad = []
            with open(p_old, "rb") as f:
                for x in range(e["E"]):
                    n = ext.record_rowmajor(p_new, e["recsz"], e["msb_sz"], PERS4, lay, ROWS3, offs(e), x, out)
                    assert n == r["recsz"], n
                    if not np.array_equal(np.array(out), np.frombuffer(f.read(r["recsz"]), np.uint8)):
                        bad.append(x)
            print(f"{D} L{L} {e['layout']}: (a) {e['E'] - len(bad)} / {e['E']} records == records3 (bad {bad[:8]})", flush=True)
            assert not bad
            ids = np.random.default_rng(int(L)).choice(e["E"], 64, replace=False).astype(np.int32).tolist()
            a, b = rebuilt(p_new, e, ids, lay), rebuilt(p_old, r, ids, 0)
            same = [np.array_equal(u, v) for u, v in zip(a, b)]
            print(f"{D} L{L} {e['layout']}: (b) 4-bit prefill pieces of 64 experts equal: {same}", flush=True)
            assert all(same)
    # (c) the guard: native+p11 offsets on today's row-major file must abort, not decode garbage
    D = dirs[0]
    ix = json.load(open(os.path.join(ROOT, D, "index.json")))["layers"]
    L, e = next(iter(ix.items()))
    if e["layout"] == "native+p11":
        code = (f"import sys; sys.path.insert(0, {os.path.join(ROOT, 'ext', 'build_nat')!r}); import mlx.core as mx, "
                f"bigmoe_ext as ext; out = mx.zeros(({ix0[L]['recsz']},), mx.uint8); mx.eval(out); "
                f"ext.record_rowmajor({os.path.join(R3, ix0[L]['file'])!r}, {ix0[L]['recsz']}, {ix0[L]['msb_sz']}, "
                f"{PERS4}, 2, {ROWS3}, {offs(e)}, 3, out)")
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        print(f"(c) p11 offsets on a row-major file: rc {p.returncode}, {p.stderr.strip()[-90:]!r}")
        assert p.returncode != 0 and "without its header" in p.stderr
    # (d) the pool path: parts copied piece by piece into slots, sizes from the index (contiguous pieces from each part's start)
    for D in dirs:
        ix = json.load(open(os.path.join(ROOT, D, "index.json")))["layers"]
        L, e = next(iter(ix.items()))
        path = os.path.join(ROOT, D, e["file"])
        for part, (po, ps, a, b) in (("MSB", (e["msb_piece_off"], e["msb_piece_size"], 0, e["msb_sz"])),
                                     ("LSB", (e["lsb_piece_off"], e["lsb_piece_size"], e["msb_sz"], e["recsz"]))):
            o_ = sorted(range(len(po)), key=lambda i: po[i])    # record order (native+p11+pad: the l1 planes first)
            po, ps = [po[i] for i in o_], [ps[i] for i in o_]
            assert all(po[i + 1] == po[i] + ps[i] for i in range(len(po) - 1)) and po[0] == a, (part, po, ps)
            C = 8
            pools = [mx.zeros((C, s), mx.uint8) for s in ps]
            mx.eval(*pools)
            Lx = ext.add_layer(C, e["E"], [-1] * len(ps), [0] * len(ps), list(ps), pools)
            ext.set_records_part(Lx, path, e["recsz"], a, b - a)
            rng, n = np.random.default_rng(5), 0
            with open(path, "rb") as f:
                for _ in range(3):
                    ids = rng.choice(e["E"], C, replace=False).astype(np.int32)
                    ext.ensure(Lx, mx.array(ids))
                    host = [np.array(p) for p in pools]
                    for x, s in zip(ids.tolist(), ext.lookup(Lx, ids.tolist())):
                        f.seek(x * e["recsz"])
                        rec = np.frombuffer(f.read(e["recsz"]), np.uint8)
                        assert s >= 0 and all(np.array_equal(h[s], rec[o:o + z]) for h, o, z in zip(host, po, ps)), (part, x)
                        n += 1
            print(f"{D} L{L} {e['layout']}: (d) {part} pool, pieces {list(ps)}: {n} loads, every slot == its record's part")
    print("PASS")


if __name__ == "__main__":
    main(sys.argv[1:] or ["scratch/r3p11", "scratch/r3nat"])
