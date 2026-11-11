"""eprep_k (Leap, Sat 2026-11-07; PLAN 220 #3e, from Fable B's DENSE_REBUILD v2 kernels/eprep.py): the expert id glue of
ref2._ooo_1c_call as ONE dispatch, for the no-split path v1' runs (QOM on, QOM-SPLIT off, U3's nolsb rows optional).
Today, per MoE layer, the glue is mx ops: ids.astype(int32) twice, | (1 << 21), where (ids_p, the loader's ids with QOM's
optional uses marked) and, with U3's nolsb rows, a where over the 4-bit ranks' slice (ids_s: a sure row posts the first
unsure row's rank-0 id). Integer selects only, so the kernel is bitwise == those ops, dtypes included (ids_p int32,
ids_s the ids' own dtype). The posts, their slices and pre_l's -2 select stay where they are.
glue(ids, qopt, nl, R4) -> (ids_p, ids_s); ids [..., K] (uint32 or int32), qopt bool [M*K] (or None: no QOM), nl bool
[M, 1] (or None: no U3 rows), R4 the 4-bit ranks (0: no ids_s). reference(...) = today's ops; check() compares them."""
import mlx.core as mx

_SRC = r"""
    const uint i = thread_position_in_grid.x;
    if (i >= (uint)(M_C * K_C)) return;
    const uint m = i / K_C, r = i % K_C;
    const int id = int(ids[i]);
    ids_p[i] = (HAS_Q && qopt[i]) ? (id | (1 << 21)) : id;
    if (r < (uint)R4) {
        IdT v = ids[i];
        NL_BLOCK
        ids_s[m * R4 + r] = v;
    }
"""
_NL = r"""if (nl[m]) {
            uint u = 0;
            for (uint j = 0; j < (uint)M_C; j++) if (!nl[j]) { u = j; break; }
            v = ids[u * K_C];
        }"""
_K = {}


def _kern(has_q, has_nl):
    key = (has_q, has_nl)
    if key not in _K:
        ins = ["ids"] + (["qopt"] if has_q else []) + (["nl"] if has_nl else [])
        src = _SRC.replace("NL_BLOCK", _NL if has_nl else "")
        if not has_q:
            src = src.replace("(HAS_Q && qopt[i])", "(false)")
        else:
            src = src.replace("(HAS_Q && qopt[i])", "(qopt[i])")
        _K[key] = mx.fast.metal_kernel(name=f"leap_eprep_q{int(has_q)}_n{int(has_nl)}", input_names=ins,
                                       output_names=["ids_p", "ids_s"], source=src)
    return _K[key]


def glue(ids, qopt, nl, R4):
    """ids_p (int32, ids.shape) and ids_s (ids.dtype, ids.shape[:-1] + (R4,)) in one dispatch; R4 == 0 -> ids_s None"""
    K = ids.shape[-1]
    M = ids.size // K
    has_q, has_nl = qopt is not None, nl is not None and R4 > 0
    ins = [ids.reshape(M, K)] + ([qopt.reshape(M, K)] if has_q else []) + ([nl.reshape(M)] if has_nl else [])
    r4 = max(int(R4), 1)
    out = _kern(has_q, has_nl)(inputs=ins, template=[("IdT", ids.dtype), ("M_C", M), ("K_C", K), ("R4", int(R4)),
                                                     ("HAS_Q", int(has_q)), ("HAS_NL", int(has_nl))],
                               grid=(((M * K + 31) // 32) * 32, 1, 1), threadgroup=(32, 1, 1),
                               output_shapes=[(M, K), (M, r4)], output_dtypes=[mx.int32, ids.dtype])
    ids_p = out[0].reshape(ids.shape) if has_q else ids   # no QOM: the loader's ids are the ids, as today
    ids_s = out[1].reshape(*ids.shape[:-1], int(R4)) if R4 else None
    return ids_p, ids_s


def reference(ids, qopt, nl, nl_np, R4):
    """ref2._ooo_1c_call's own expressions (no split), for the comparison"""
    import numpy as np
    K = ids.shape[-1]
    M = ids.size // K
    ids_p = ids if qopt is None else mx.where(qopt.reshape(ids.shape), ids.astype(mx.int32) | (1 << 21), ids.astype(mx.int32))
    if not R4:
        return ids_p, None
    if nl is not None:
        nl2 = nl.reshape(M, 1)
        un = np.where(~np.asarray(nl_np))[0]
        dup = ids.reshape(M, K)[int(un[0]) if len(un) else 0, 0]
        ids_s = mx.where(nl2, dup, ids.reshape(M, K)[:, :R4]).reshape(*ids.shape[:-1], R4)
    else:
        ids_s = ids[..., :R4]
    return ids_p, ids_s


def check(n=400, seed=0):
    """random cases at the engine's shapes (M 1-16 rows, K 10, R4 0-10, ids uint32 < 512, QOM / U3 on and off): the
    kernel vs reference, values and dtypes; returns the number that differ"""
    import numpy as np
    rng = np.random.default_rng(seed)
    bad = 0
    for c in range(n):
        M, K = int(rng.integers(1, 17)), 10
        R4 = int(rng.choice([0, 4, 10]))
        ids = mx.array(rng.integers(0, 512, (1, M, K)).astype(np.uint32))
        qopt = mx.array(rng.random((1, M, K)) < 0.3) if c % 3 else None
        nl_np = (rng.random(M) < 0.4) if c % 2 else None
        if nl_np is not None and c % 7 == 0:
            nl_np[:] = True                              # every row sure: the reference falls back to row 0
        nl = mx.array(nl_np) if nl_np is not None else None
        a = reference(ids, qopt, nl, nl_np, R4)
        b = glue(ids, qopt, nl, R4)
        mx.eval([x for x in a + b if x is not None])
        for x, y in zip(a, b):
            if (x is None) != (y is None):
                bad += 1
            elif x is not None and (x.dtype != y.dtype or x.shape != y.shape or not bool(mx.array_equal(x, y))):
                bad += 1
    return bad


if __name__ == "__main__":
    b = check()
    print(f"EPREP_K check: {b} of 400 cases differ ->", "PASS" if b == 0 else "FAIL")
