"""Fit12 (Tue 3 Nov 19:3x): stand-ins for fit_t3.py's harness dry test (F12_FAKE=1: no model, no GPU, CPU only, seconds).
generate() returns T3.generate's shapes: tokens, the per-check log (S, path, k, cf), the phase marks (name, check, t, seq,
bytes, misses at start and at end) and the stats dict. Deterministic per prompt and arm: K1U arms leave K2's text at 3/4 of
the cap, M3 arms at 1/2; K1U runs 10 % faster per token than K2 (F12_FAKE_DT s per K2 check of 8 tokens); PF2 arms = their
base arm (tokens + lsha), so GATE PF2 passes. Used by /Users/kamen/Fit12/scripts/fit_t3_dry.sh."""
import os, time, types

DT = float(os.environ.get("F12_FAKE_DT", "0.01"))


class Ext:
    @staticmethod
    def defer_stats():
        return [0, 0, 0, 0]


B = types.SimpleNamespace(CFG={}, V3F="fake")
T3 = types.SimpleNamespace(_K2=[False], _K1U=[0], _M3=[False], _PF2=[0], PHLOG=[None], _TOPK=[0], CAL=[None], CALLS=[None], W1=["4"], W1AW=[True])   # _TOPK / CAL: Thu 5 Nov fields (added Fri 6 Nov)


def generate(B_, model, draft, DF, ids, cap, th, stop_ids=None, log=None, **kw):
    k1u, m3, pf2 = T3._K1U[0], T3._M3[0], T3._PF2[0]
    base = sum(ids) % 997
    div = cap // 2 if m3 else (3 * cap) // 4 if k1u else cap
    toks = [base + i if i < div else 5000 + base + i for i in range(cap)]
    dt = DT / 1.1 if k1u else DT
    ph = T3.PHLOG[0]
    t_dec = time.perf_counter()
    n = 0
    for c in range(cap // 8):
        t0 = time.perf_counter()
        time.sleep(dt)
        t1 = time.perf_counter()
        if log is not None:
            log.append([16, "p1", 8, 0.95])
        if ph is not None:
            ph.append(["p1", c, t0, 0, 0, 0, t1, 1, 1 << 20, 3])
        if T3.CALLS[0] is not None:                # (Sun 8 Nov, T2) one fake exact call per check
            T3.CALLS[0].append([c, 0, 1 + c % 4, 1, t0, 0, 0, 0, t1, 1, 1 << 20, 3])
        n += 1
    dec = time.perf_counter() - t_dec
    s = dict(lsha=("k1u" if k1u else "k2") + ("m3" if m3 else ""), checks=n, kept=8.0, pass2=0, exact=0, cfgate=0, skipped=0,
             rows1=16 * n, rows2=0, calls2=0, refused=0, pf_s=0.0, dec_s=dec, tok_s=cap / dec, t_draft=0.0, t_p1=dec, t_p2=0.0,
             t_commit=0.0, miss1=3 * n, miss2=0, pf_mb=0.5 * len(ids))   # pf_mb: (Sun 8 Nov) the PREFILL line's field
    if k1u:
        s.update(k1u_kept=25 * n, k1u_rl=10 * n, k1u_over=n)
    if pf2:
        s.update(pf2_ids=6 * n, pf2_calls=n)
    return toks, s


T3.generate = generate
