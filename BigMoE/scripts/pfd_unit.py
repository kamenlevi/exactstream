"""PF-D unit test (Sun 8 Nov 17:xx, no model load): random experts of three layers, their records3 bytes split into the MSB-3 /
LSB-1 parts a pool slot holds verbatim, rebuilt by pflm.rebuild_parts on the GPU, against the ext's own rebuild
(read_rebuilt_layout, the code tier.prefill_call runs today). Prints mismatching words per piece; exit 0 only if all equal.
usage: ~/BigMoE/.venv/bin/python scripts/pfd_unit.py [n_experts_per_layer]"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser(f"~/BigMoE/ext/{os.environ.get('BIGMOE_EXT_BUILD', 'build_qom')}"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx  # noqa: E402
import bigmoe_ext as ext  # noqa: E402
import pflm  # noqa: E402

D = os.path.expanduser("~/BigMoE/models/bare/records3")
cfg = json.load(open(os.path.expanduser("~/BigMoE/models/bare/config.json")))
tc = cfg.get("text_config", cfg)
H, I = int(tc["hidden_size"]), int(tc["moe_intermediate_size"])
ix = json.load(open(os.path.join(D, "index.json")))
rows3, ins = [I, I, H], [H, H, I]
shapes4 = []
for O, K in zip(rows3, ins):
    shapes4 += [((O, K // 8), mx.uint32), ((O, K // 64), mx.bfloat16), ((O, K // 64), mx.bfloat16)]
pers4 = [int(np.prod(sh)) * (4 if dt == mx.uint32 else 2) for sh, dt in shapes4]
n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
rng = np.random.default_rng(1108)
bad = 0
for L in (0, 23, 47):
    r = ix["layers"][str(L)]
    assert r.get("layout") == "native+p11+pad", r.get("layout")
    t = dict(r=r, rows3=rows3, shapes4=shapes4, offs=list(r["msb_piece_off"]) + list(r["lsb_piece_off"]),
             msb_sz=r["msb_sz"], recsz=r["recsz"], path=os.path.join(D, r["file"]), layout=2)
    ids = np.sort(rng.choice(int(r["E"]), n, replace=False))
    buf = pflm.read_parts(t, ids, 0, int(r["recsz"]))
    w3s, s3s = pflm.msb_pieces(t, buf[:, :r["msb_sz"]])
    l1s, bs = pflm.lsb_pieces(t, buf[:, r["msb_sz"]:])
    reb = pflm.rebuild_parts(t, w3s, s3s, l1s, bs)
    ref = [mx.zeros((n, *sh), dt) for sh, dt in shapes4]
    mx.eval(*ref)
    ext.read_rebuilt_layout(t["path"], int(r["recsz"]), int(r["msb_sz"]), pers4, ids.astype(np.int32).tolist(),
                            list(range(n)), ref, 2, rows3, t["offs"])
    for i, (g, rr) in enumerate(zip(reb, ref)):
        gv = np.array(g.view(mx.uint16) if g.dtype == mx.bfloat16 else g)
        rv = np.array(rr.view(mx.uint16) if rr.dtype == mx.bfloat16 else rr)
        assert gv.shape == rv.shape, (L, i, gv.shape, rv.shape)
        nb = int((gv != rv).sum())
        bad += nb
        print(f"layer {L:2d} piece {i} {str(gv.shape):18s} {'==' if nb == 0 else f'{nb} words differ'}"
              f"{'' if rv.any() else '  (reference all zero!)'}")
print(f"PFD UNIT ({'kernel' if os.environ.get('R57_PFD_KERNEL', '1') != '0' else 'ops'}): {'PASS' if bad == 0 else 'FAIL'} "
      f"({bad} differing words; {3 * n} experts, 9 pieces each)")
# v3 ASSEMBLE: 24 experts of layer 23 as 8 A (base + side 'pooled'), 8 B (base pooled, side 'read'), 8 C (ext-rebuilt rows),
# scratch rows in a shuffled order, against read_rebuilt_layout of the same 24 in that order
r = ix["layers"]["23"]
t = dict(r=r, rows3=rows3, shapes4=shapes4, offs=list(r["msb_piece_off"]) + list(r["lsb_piece_off"]),
         msb_sz=r["msb_sz"], recsz=r["recsz"], path=os.path.join(D, r["file"]), layout=2)
ids = np.sort(rng.choice(int(r["E"]), 24, replace=False))
buf = pflm.read_parts(t, ids, 0, int(r["recsz"]))
w3s, s3s = pflm.msb_pieces(t, buf[:, :r["msb_sz"]])
l1s, bs = pflm.lsb_pieces(t, buf[:, r["msb_sz"]:])
P3 = ("gate_proj", "up_proj", "down_proj")
kind = np.array([0] * 8 + [1] * 8 + [2] * 8)
perm = rng.permutation(24)                                   # scratch row k holds expert ids[perm[k]]
Ai, Bi, Ci = np.arange(8), np.arange(8, 16), np.arange(16, 24)
base = {p_: {"w3": w3s[p], "s3": s3s[p]} for p, p_ in enumerate(P3)}   # base 'pool': slot j = expert ids[j]
side = {p_: {"l1": l1s[p][mx.array(Ai)], "b": bs[p][mx.array(Ai)]} for p, p_ in enumerate(P3)}   # side 'pool': slot j = ids[j]
bl1s = [l1s[p][mx.array(Bi)] for p in range(3)]
bbs = [bs[p][mx.array(Bi)] for p in range(3)]
scrC = [mx.zeros((8, *sh), dt) for sh, dt in shapes4]
mx.eval(*scrC)
ext.read_rebuilt_layout(t["path"], int(r["recsz"]), int(r["msb_sz"]), pers4, ids[Ci].astype(np.int32).tolist(), list(range(8)),
                        scrC, 2, rows3, t["offs"])
rm = np.zeros((24, 3), np.int32)
for k in range(24):
    j = perm[k]
    rm[k] = (0, j, j) if kind[j] == 0 else (1, j, j - 8) if kind[j] == 1 else (2, j - 16, 0)
scr = pflm.assemble(t, rm, base, side, bl1s, bbs, scrC)
ref = [mx.zeros((24, *sh), dt) for sh, dt in shapes4]
mx.eval(*ref)
ext.read_rebuilt_layout(t["path"], int(r["recsz"]), int(r["msb_sz"]), pers4, ids[perm].astype(np.int32).tolist(), list(range(24)),
                        ref, 2, rows3, t["offs"])
bad3 = 0
for i, (g, rr) in enumerate(zip(scr, ref)):
    gv = np.array(g.view(mx.uint16) if g.dtype == mx.bfloat16 else g)
    rv = np.array(rr.view(mx.uint16) if rr.dtype == mx.bfloat16 else rr)
    bad3 += int((gv != rv).sum())
print(f"PFD UNIT v3 assemble (8 A / 8 B / 8 C, shuffled rows): {'PASS' if bad3 == 0 else 'FAIL'} ({bad3} differing words)")
bad += bad3
# v4: kinds 0 / 1 / 3 (raw full record) exact vs the ext; kinds 4 (base pool) / 5 (raw MSB part) = the 3-bit base-only pieces,
# vs v1's MLX-op rebuild with zero low bits and the b3 bias (q = 2m, s = s3 / 2, b = b3); 6 experts each, shuffled rows
ids = np.sort(rng.choice(int(r["E"]), 30, replace=False))
buf = pflm.read_parts(t, ids, 0, int(r["recsz"]))
w3s, s3s = pflm.msb_pieces(t, buf[:, :r["msb_sz"]])
l1s, bs = pflm.lsb_pieces(t, buf[:, r["msb_sz"]:])
b3s = [mx.array(np.ascontiguousarray(buf[:, int(t["offs"][3 * p + 2]):int(t["offs"][3 * p + 2]) + int(r["msb_piece_size"][3 * p + 2])])
                .view(np.uint32)) for p in range(3)]
kind = np.repeat([0, 1, 3, 4, 5], 6)
perm = rng.permutation(30)
sl = lambda a, idx: a[mx.array(np.array(idx, np.int32))]
base = {p_: {"w3": w3s[p], "s3": s3s[p], "b3": b3s[p]} for p, p_ in enumerate(P3)}   # base 'pool' slot j = ids[j]
side = {p_: {"l1": sl(l1s[p], range(0, 6)), "b": sl(bs[p], range(0, 6))} for p, p_ in enumerate(P3)}   # side slot j = ids[j]
rl = mx.array(np.ascontiguousarray(buf[6:12, r["msb_sz"]:]).view(np.uint32))      # kind 1: raw LSB parts of ids[6:12]
rr = mx.array(np.ascontiguousarray(buf[12:18]).view(np.uint32))                   # kind 3: full records of ids[12:18]
rm5 = mx.array(np.ascontiguousarray(buf[24:30, :r["msb_sz"]]).view(np.uint32))   # kind 5: MSB parts of ids[24:30]
scrC = [mx.zeros((1, *sh), dt) for sh, dt in shapes4]
rm = np.zeros((30, 3), np.int32)
for k in range(30):
    j = perm[k]
    rm[k] = {0: (0, j, j), 1: (1, j, j - 6), 3: (3, j - 12, 0), 4: (4, j, 0), 5: (5, j - 24, 0)}[kind[j]]
got = pflm.assemble_v4(t, rm, base, side, rl, scrC, rr, rm5)
ref = [mx.zeros((30, *sh), dt) for sh, dt in shapes4]
mx.eval(*ref)
ext.read_rebuilt_layout(t["path"], int(r["recsz"]), int(r["msb_sz"]), pers4, ids[perm].astype(np.int32).tolist(), list(range(30)),
                        ref, 2, rows3, t["offs"])
three = pflm.rebuild_parts_ops(t, w3s, s3s, [mx.zeros_like(x) for x in l1s], b3s)   # 3-bit pieces of all 30 (expert order)
pi = mx.array(perm.astype(np.int32))
three = [x[pi] for x in three]                                                     # -> scratch row order
fast = mx.array((kind[perm] >= 4).reshape(-1, 1, 1))
bad4 = {0: 0, 1: 0, 3: 0, 4: 0, 5: 0}
for g, rr_, th in zip(got, ref, three):
    exp = mx.where(fast, th, rr_)
    gv = np.array(g.view(mx.uint16) if g.dtype == mx.bfloat16 else g).reshape(30, -1)
    ev = np.array(exp.view(mx.uint16) if exp.dtype == mx.bfloat16 else exp).reshape(30, -1)
    rowbad = (gv != ev).any(axis=1)
    for kk in bad4:
        bad4[kk] += int(rowbad[kind[perm] == kk].sum())
ok4 = sum(bad4.values()) == 0
print(f"PFD UNIT v4 assemble (6 each of kinds 0 / 1 / 3 exact, 4 / 5 at 3 bits; shuffled): {'PASS' if ok4 else 'FAIL'} "
      f"(differing rows by kind {bad4})")
bad += 0 if ok4 else 1
# v4 with the engine's placeholders (Sun 23:5x: the in-run v4 / FAST arms died compiling: MLX passes inputs of fewer than 8
# elements in the 'constant' address space, the kernels take 'device' pointers, and every input above was a real array).
# Exact v4 calls without kind 1 / 5 rows (rl, rm5 = placeholder) and FAST calls without kind 1 / 3 rows (rl, rr = placeholder).
ph = pflm.v4_placeholder()
bad4p = 0
for subk, (rl_, rr_, rm5_) in (((0, 3), (ph, rr, ph)), ((0, 4, 5), (ph, ph, rm5))):
    sub = [k for k in range(30) if kind[perm[k]] in subk]
    gotp = pflm.assemble_v4(t, rm[sub], base, side, rl_, scrC, rr_, rm5_)
    si = mx.array(np.array(sub, np.int32))
    for g, rf, th in zip(gotp, ref, three):
        exp = mx.where(fast, th, rf)[si]
        gv = np.array(g.view(mx.uint16) if g.dtype == mx.bfloat16 else g).reshape(len(sub), -1)
        ev = np.array(exp.view(mx.uint16) if exp.dtype == mx.bfloat16 else exp).reshape(len(sub), -1)
        bad4p += int((gv != ev).any(axis=1).sum())
print(f"PFD UNIT v4 with placeholders (kinds 0 / 3 exact; 0 / 4 / 5 FAST): {'PASS' if bad4p == 0 else 'FAIL'} "
      f"({bad4p} differing rows)")
bad += 0 if bad4p == 0 else 1
# v4 staging plumbing (no model): zero-copy stage_reads / gather vs plain read_parts on real records; a partial stage
# (prefetched subset + misses into spare rows) and an overflowing one (a second array, concatenated)
import types
sys.path.insert(0, os.path.expanduser('~/BigMoE'))   # install_pfd imports bigmoe.stream's PIECES
fake = types.SimpleNamespace(prefill_call=lambda *a, **k: None, STATS={"prefill_calls": 0})
pflm.install_pfd(fake)
dbg = fake._pfd_dbg
ids = [int(e) for e in np.sort(rng.choice(int(r["E"]), 20, replace=False))]
want = pflm.read_parts(t, np.array(ids), int(r["msb_sz"]), int(r["recsz"]) - int(r["msb_sz"]))   # LSB parts
okS = True
for case, pre, spare in (("none", None, 0), ("partial", ids[:12], 4), ("overflow", ids[:5], 2)):
    st = None if pre is None else dbg["stage_reads"](t, pre, int(r["msb_sz"]), int(r["recsz"]) - int(r["msb_sz"]), spare)
    arr, rows = dbg["gather"](st, t, ids, int(r["msb_sz"]), int(r["recsz"]) - int(r["msb_sz"]), "B")
    got_b = np.array(arr).view(np.uint8)[rows]
    ok = bool(np.array_equal(got_b, want))
    okS &= ok
    print(f"  staging {case:8s}: {arr.shape[0]} rows, {'==' if ok else 'DIFFERENT'}")
print(f"PFD UNIT v4 staging (zero-copy reads, spare rows, overflow): {'PASS' if okS else 'FAIL'}; counters "
      f"h {pflm.PFD['hB']} / m {pflm.PFD['mB']}")
bad += 0 if okS else 1
if os.environ.get("PFD_UNIT_TIME", "1") == "1" and bad == 0:   # speed: 96 experts of layer 23, kernel vs ops (same bytes)
    import time
    r = ix["layers"]["23"]
    t = dict(r=r, rows3=rows3, shapes4=shapes4, offs=list(r["msb_piece_off"]) + list(r["lsb_piece_off"]),
             msb_sz=r["msb_sz"], recsz=r["recsz"], path=os.path.join(D, r["file"]), layout=2)
    ids = np.sort(rng.choice(int(r["E"]), 96, replace=False))
    buf = pflm.read_parts(t, ids, 0, int(r["recsz"]))
    w3s, s3s = pflm.msb_pieces(t, buf[:, :r["msb_sz"]])
    l1s, bs = pflm.lsb_pieces(t, buf[:, r["msb_sz"]:])
    mx.eval(*w3s, *s3s, *l1s, *bs)
    res = {}
    for name, fn in (("kernel", pflm.rebuild_parts), ("ops", pflm.rebuild_parts_ops)):
        os.environ["R57_PFD_KERNEL"] = "1" if name == "kernel" else "0"
        best = 1e9
        for _ in range(3):
            t0 = time.perf_counter()
            o = fn(t, w3s, s3s, l1s, bs)
            mx.eval(*o)
            best = min(best, time.perf_counter() - t0)
        res[name] = o
        print(f"  {name}: {best * 1e3:.1f} ms for 96 experts = {best * 1e3 / 96:.3f} ms per expert")
    same = all(np.array_equal(np.array(a.view(mx.uint16) if a.dtype == mx.bfloat16 else a),
                              np.array(b.view(mx.uint16) if b.dtype == mx.bfloat16 else b)) for a, b in zip(res["kernel"], res["ops"]))
    print(f"  kernel == ops on the 96: {same}")
    bad += 0 if same else 1
sys.exit(0 if bad == 0 else 1)
