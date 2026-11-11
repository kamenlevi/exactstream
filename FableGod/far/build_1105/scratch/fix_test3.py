import sys
p = sys.argv[1]; s = open(p).read()
def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (old[:70], s.count(old)); s = s.replace(old, new)
rep('DT, NPDT = getattr(mx, DTN), getattr(np, DTN)                   # noise ~5,000 x, which would blur "harness == reference"\n',
    'HALF = DTN in ("bfloat16", "float16")                           # noise ~5,000 x, which would blur "harness == reference"\n'
    'DT, NPDT = getattr(mx, DTN), (np.float32 if HALF else getattr(np, DTN))\n'
    '# NW_TEST_DTYPE=bfloat16 (the 125B\'s dtype; the compiled residual write engages at >= 32 rows): a smoke run. The exact\n'
    '# checks stay hard (bit-equalities, counts, guards, measure()); the comparisons with the reference are printed as notes,\n'
    '# since 8-bit mantissas over 48 layers do not allow a tolerance.\n'
    'def A_(a): return mx.array(a).astype(DT)\n'
    'def N_(x): return np.array(x.astype(mx.float32)) if HALF else np.array(x)\n')
rep('        x = fwd(mx.tile(mx.array(h[p:p + S])[None], (1, 1, HC)))\n        mx.eval(x); outs.append(np.array(x[0])); p += S\n',
    '        x = fwd(mx.tile(A_(h[p:p + S])[None], (1, 1, HC)))\n        mx.eval(x); outs.append(N_(x[0])); p += S\n')
rep('        x = mx.tile(mx.array(h[p:p + S])[None], (1, 1, HC)); pend = {}\n', '        x = mx.tile(A_(h[p:p + S])[None], (1, 1, HC)); pend = {}\n')
rep('        mx.eval(x); outs.append(np.array(x[0])); p += S\n    return np.concatenate(outs)\n\ndef rel(x, y)',
    '        mx.eval(x); outs.append(N_(x[0])); p += S\n    return np.concatenate(outs)\n\ndef rel(x, y)')
rep('TOL = 1e-9 if DTN == "float64" else 1e-4; LPT = max(10 * TOL, 5e-6)   # log-probs are stored as float32\n',
    'TOL = 1e-9 if DTN == "float64" else 1e-4; LPT = max(10 * TOL, 5e-6)   # log-probs are stored as float32\n'
    'def soft(c, m):                                                 # a comparison with the reference: hard, but a note in half precision\n'
    '    if HALF: print(("  note " if c else "  NOTE (not within the float tolerance, expected in half precision) ") + m, flush=True)\n'
    '    else: ok(c, m)\n')
for old in ('ok(e_ref < TOL, ', 'ok(e0 < TOL, ', 'ok(e1 < TOL, ', 'ok(eff > 50 * max(e1, 1e-7), ', '    ok(rel(o_1, w) > 50 * max(e1, 1e-7), ', 'ok(e2 < TOL, ', 'ok(w > 50 * max(e2, 1e-7), '):
    rep(old, old.replace('ok(', 'soft(', 1))
rep('    ok(rel(o_, run_ref(h, chunks_, keep, late1, 1)) < TOL and C.NW["terms"] == int(late1.sum()), f"nw1 with {nm_} {chunks_} == the reference")\n',
    '    ok(C.NW["terms"] == int(late1.sum()) and not C.NW["stash"], f"nw1 with {nm_} {chunks_}: every late flag used once, nothing left stashed")\n'
    '    soft(rel(o_, run_ref(h, chunks_, keep, late1, 1)) < TOL, f"nw1 with {nm_} {chunks_} == the reference")\n')
rep('HEAD = mx.array(np.random.RandomState(6).randn(HC * D, 50).astype(NPDT) * 0.05)', 'HEAD = A_(np.random.RandomState(6).randn(HC * D, 50).astype(NPDT) * 0.05)')
rep('        lg = x @ HEAD; top = mx.argsort(-lg, axis=-1)[:, :8]\n',
    '        lg = x.astype(HEAD.dtype) @ HEAD; lg = lg.astype(mx.float32) if HALF else lg   # the real hook casts the logits to float32\n'
    '        top = mx.argsort(-lg, axis=-1)[:, :8]\n')
rep('            x0 = mx.tile(mx.array(hh[p:p + S_])[None], (1, 1, HC))\n', '            x0 = mx.tile(A_(hh[p:p + S_])[None], (1, 1, HC))\n')
rep('    ok(all(R0[w]["tk"] == T[w]["tk"] for w in T) and d0 < LPT and "WARNING" not in out_0, f"measure(): nw0\'s top-8 ids == the twin\'s, max |d log p| {d0:.1e}")\n',
    '    ok("WARNING" not in out_0 and all("nw" in R0[w] and R0[w]["nw"]["d"] == 0 for w in R0), "measure(): nw0 runs with its record and no WARNING")\n'
    '    soft(all(R0[w]["tk"] == T[w]["tk"] for w in T) and d0 < LPT, f"measure(): nw0\'s top-8 ids == the twin\'s, max |d log p| {d0:.1e}")\n')
rep('    good = True; dref = 0.0\n', '    good = True; gref = True; dref = 0.0\n')
rep('        good &= np.array_equal(t[:, 0], np.array(R1[nm]["tk"])[:, 0]) and R1[nm]["a"] == a_\n',
    '        gref &= np.array_equal(t[:, 0], np.array(R1[nm]["tk"])[:, 0]); good &= R1[nm]["a"] == a_\n')
rep('    ok(good and dref < LPT, f"measure(): nw1\'s top-1 == the reference run\'s on both windows (max |d log p| {dref:.1e}); res[\'nw\'] holds d, late, terms, writes; no warn")\n',
    '    ok(good, "measure(): nw1\'s res[\'nw\'] holds d, late, terms (== the mask\'s count), writes (== the chunk-layer pairs); no warn")\n'
    '    soft(gref and dref < LPT, f"measure(): nw1\'s top-1 == the reference run\'s on both windows (max |d log p| {dref:.1e})")\n')
rep('    ok(dt > 20 * max(dref, LPT), ', '    soft(dt > 20 * max(dref, LPT), ')
rep('__call__, Q.GatedResidual, Q.SparseMoeBlock, Q._hyper_residual_write) at 32 dims, float64, on MLX\'s CPU device, run',
    '__call__, Q.GatedResidual, Q.SparseMoeBlock, Q._hyper_residual_write) at 32 dims, float64, on MLX\'s CPU device, run')
rep('about 5,000 x, which would blur "harness == reference"; NW_TEST_DTYPE=float32 NW_TEST_LAYERS=6 runs the small form.',
    'about 5,000 x, which would blur "harness == reference"; NW_TEST_DTYPE=float32 NW_TEST_LAYERS=6 runs the small form, and\n'
    'NW_TEST_DTYPE=bfloat16 a smoke run in the 125B\'s own dtype (exact checks hard, reference comparisons as notes).')
open(p, "w").write(s); print("ok")
