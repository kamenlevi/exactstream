"""350 at 45 GB, repriced from measured kernels (NewIdeas, Mon 2026-10-26; Leap asked for D32's 22.2 ms).
The non-exact 45 GB stack (top-8, routed experts ~2.7 b, dense diet 2.1 GB, all resident), one stream, one check of
16 or 32 rows. Tokens per check needed for 350 tok/s = 350 x check time.
Inputs (all measured unless marked):
  dense      D32 (Leap Mon 10:02): exact dense 17.96 ms (16 rows) / 22.19 ms (32 rows), bandwidth floor 12.3 ms.
             The diet's own time is unmeasured: bounded by the exact time (32-row kernels are ALU-bound) and by the exact
             time scaled to the diet's bytes (2.1 GB -> floor 7.5 ms).
  experts    L1-30c (Sun 03:13): per expert at 16 rows 4-bit 11.9 us, 3-bit 11.1, 2-bit K-paired 8.5 (floor 5.5);
             ~2.7 b taken as 10.0 us (assumed). 32 rows (~6 real rows per expert) x1.1 (assumed).
             Distinct experts per layer: top-8 54.6 at 16 rows (L1-32); 32 rows x 100.5 / 66.1 (par32, top-10).
             Byte floor: 4.9 GB per 16-row check (NewIdeas' '! 350' figure) at 280 GB/s.
  GDN state  113 MB read + written per stream (par45).
  other      attention, GDN core, routing, glue: 10-20 ms (Leap TABLE_350, NOT measured); 5 ms at the floor (assumed).
  drafting   MEASURED (Leap CA, Mon ~14:18): DFlash2 1-step draft 12.5 ms at block 16 (forward 7.5 + head top-16 over 248K
             4.5 + selector 0.8), 15.5 at block 32; 'fast' = an 8-bit drafter + fused head top-k, ~4.5 / 5.5 (Leap's
             estimate). The first version of this file assumed 3.5 / 4.5.
  engine     MEASURED (Leap CA interim): the engine's own resident exact 16-row check (MTPLX kernels, 4-bit, top-10) is
             ~141-147 ms eager / ~130-136 overlapped, drafting not included.
Drafters (code, HumanEval held-out): L1-20 9.54 (b16) / 10.51 (b32 tail 0.9); lattice oracle top-64 14.11 / 20.29;
perfect drafter 15.59 / 30.50.  usage: python price350.py  -> stdout"""
BW, E16_GB, GDN_MS = 280.0, 4.9, 2 * 0.113 / 280.0 * 1e3
U16, U32 = 54.6, 54.6 * 100.5 / 66.1
DRAFT = {16: (9.54, 14.11, 15.59), 32: (10.51, 20.29, 30.50)}


def check(rows, dense, us_exp, other, draft_ms, floor=False):
    u = U16 if rows == 16 else U32
    exp_ms = (E16_GB * u / U16) / BW * 1e3 if floor else u * 48 * us_exp * (1.0 if rows == 16 else 1.1) / 1e3
    return dense + exp_ms + GDN_MS + other + draft_ms, exp_ms


cases = [
    ("today's kernels, high (exact dense, other 20)", {16: 17.96, 32: 22.19}, 10.0, 20, False),
    ("today's kernels, low (diet dense by bytes, other 10)", {16: 17.96 * 7.5 / 12.3, 32: 22.19 * 7.5 / 12.3}, 10.0, 10, False),
    ("90 % kernels (diet dense, experts, other 7)", {16: 7.5 / 0.9, 32: 7.5 / 0.9}, None, 7, "90"),
    ("every part at its floor (other 5)", {16: 7.5, 32: 7.5}, None, 5, True),
]
for dname, dms in (("drafting as measured", {16: 12.5, 32: 15.5}), ("fast drafting (8-bit + fused head)", {16: 4.5, 32: 5.5})):
    print(f"== {dname}")
    for name, dense, us, other, fl in cases:
        print(" " + name)
        for rows in (16, 32):
            if fl == "90":
                t, e = check(rows, dense[rows], 0, other, dms[rows], floor=True)
                t += e / 0.9 - e; e /= 0.9
            else:
                t, e = check(rows, dense[rows], us, other, dms[rows], floor=bool(fl))
            need = 350 * t / 1e3
            l120, lat, perf = DRAFT[rows]
            print(f"   {rows} rows: {t:5.1f} ms per check (dense {dense[rows]:4.1f}, experts {e:4.1f}, drafting {dms[rows]:4.1f}) -> 350 needs "
                  f"{need:4.1f} tokens = L1-20 x{need / l120:.2f}, {need / lat:.0%} of the lattice oracle, {need / perf:.0%} of perfect; "
                  f"L1-20 gives {1e3 * l120 / t:3.0f} tok/s")
for lo, hi, how in ((141, 147, "eager"), (130, 136, "overlapped")):
    t0, t1 = lo + 12.5, hi + 12.5
    print(f"engine today, exact 4-bit top-10 resident, {how}: {t0:.0f}-{t1:.0f} ms per 16-row check + draft -> L1-20 gives "
          f"{1e3 * 9.54 / t1:.0f}-{1e3 * 9.54 / t0:.0f} tok/s; 350 needs {350 * t0 / 1e3:.0f}-{350 * t1 / 1e3:.0f} tokens per check")
