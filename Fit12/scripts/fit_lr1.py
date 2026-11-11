"""Fit12 LR1 (2026-10-24): Leap's lazy-replay finding on B12LL. MTPLX's commit_verified_window is lazy by design ("the next
round's eval pulls the replay"), so the replayed GDN recurrences and the PLE layer forward of a partial accept run inside
the NEXT verify graph, on its critical path. F12_LR1=1 wraps Model.commit_verified_window (this process only): after a
landed commit, mx.async_eval the ArraysCache entries' arrays (GDN / PLE states; evaluated ones cost nothing), so the GPU
runs the replay right away, beside the CPU's drafting bookkeeping. Scheduling only: same arrays, same values -> shas must
not move. CALLS = [commits kicked]."""
import os

ON = os.environ.get("F12_LR1") == "1"
CALLS = [0]


def apply(Q, mx):
    if not ON:
        return False
    cls = Q.Model
    orig = cls.commit_verified_window

    def cvw(self, cache, *a, **kw):
        ok = orig(self, cache, *a, **kw)
        if ok:
            arrs = []
            for e in cache:
                if e is not None and type(e).__name__ == "ArraysCache":
                    st = e.state
                    for x in (st if isinstance(st, (list, tuple)) else [st]):
                        if x is not None and hasattr(x, "shape"):
                            arrs.append(x)
            if arrs:
                CALLS[0] += 1
                mx.async_eval(arrs)
        return ok
    cls.commit_verified_window = cvw
    return True
