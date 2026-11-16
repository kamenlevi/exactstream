"""pipe_price.py (Leap, Sun 15 Nov 2026 00:4x; WEEK98 pillar 3, the orchestrator 00:3x): price a pipelined
draft/check loop on our engine from measured parts. Desk only, no GPU.

Pipelined loop: while the target checks block N, the drafter drafts block N+1 as if all of block N is kept.
- That draft is usable only if the whole block is kept AND the drafter's guess for the bonus token is right:
  q = p(1) ... p(G) x p(G+1). At a flat 98 % and G = 31 guesses (32 rows), q = 0.98^32 = 0.52.
- Otherwise (1 - q) the next block is drafted after the check, serially, as today.
- The speculative block is drafted from a state one block old: its guesses sit at depths G+2 .. 2G+1 (33-63), not
  1-31. The drafter must hold its accuracy that deep, or the speculative blocks keep less.
- Cycle (pipelined)  = T_check + kappa x D + (1 - q) x D + overhead
  Cycle (sequential) = T_check + D + overhead
  kappa = the share of the draft's own time that the overlapped draft adds to the check. So pipelining pays iff
  kappa < q. At the check's byte floor, kappa x D >= the draft's bytes / bandwidth (bandwidth is shared).
Two-state chain for depth-dependent accuracy: F = fresh block (depths 1..G), S = speculative block (depths G+2..2G+1;
assumes the drafter re-anchors on the newest true state each cycle).
Sources: see data/week98/pipe_price.txt."""
import sys

BW = 280.0                      # GB/s, the floor convention (cheapcheck.md); bw_probe 285, loads 287-305
DENSE = 3.44                    # GB per check (D32)
EXP = 2.135e-3                  # GB per expert, 4-bit
UNION = {1: 10.0, 16: 72.5, 32: 108.5}   # distinct experts per layer: BigMoE step 0 (64c9d2d) means 71-74 / 107-110
UNION_OLD = {1: 10.0, 16: 66.1, 32: 100.5}  # cheapcheck.md's earlier estimates (floors 15.9 / 36.5 / 49.1)


def check_bytes(rows, u=UNION):
    return DENSE + 48 * u[rows] * EXP


def floor_ms(rows, bw=BW, u=UNION):
    return check_bytes(rows, u) / bw * 1e3


def tau(acc, G):
    """tokens kept per check incl. the bonus, for per-guess accuracies acc(depth)."""
    s, run = 1.0, 1.0
    for j in range(1, G + 1):
        run *= acc(j)
        s += run
    return s


def pipe(Tc, D, Bd, kappa, G, accF, accS, over, Bc=None, bw=BW):
    """steady state of the pipelined loop. Tc check ms, D serial draft ms, Bd draft GB, kappa overlap cost share.
    accF(j) / accS(j): accuracy of guess j of a fresh / speculative block. Returns (tok/s, q_F, pi_S, Tc')."""
    add = kappa * D
    if Bc is not None:          # the bandwidth floor: check + draft bytes share the bus
        add = max(add, (Bc + Bd) / bw * 1e3 - Tc)
    tF, tS = tau(accF, G), tau(accS, G)
    qF = 1.0
    for j in range(1, G + 1):
        qF *= accF(j)
    qF *= accS(0)               # the bonus guess, at depth G+1
    qS = 1.0
    for j in range(1, G + 1):
        qS *= accS(j)
    qS *= accS(0)
    piS = qF / (1.0 - qS + qF)
    kept = (1 - piS) * tF + piS * tS
    cyc = Tc + add + (1 - piS) * D + over
    return kept / cyc * 1e3, qF, piS, Tc + add


def seq(Tc, D, G, acc, over):
    return tau(acc, G) / (Tc + D + over) * 1e3


def main():
    G = 31
    out = []
    P = out.append
    P("pipe_price.py (Leap, Sun 15 Nov 2026 00:4x): a pipelined draft / check loop, priced from measured parts")
    P("")
    P("== Floors (exact 4-bit bytes / 280 GB/s) ==")
    for r in (1, 16, 32):
        P(f"  {r:2d} rows: {check_bytes(r):5.2f} GB -> {floor_ms(r):5.1f} ms (bw_probe 285: {floor_ms(r, 285):5.1f}); "
          f"earlier estimate {floor_ms(r, u=UNION_OLD):5.1f} ms (union {UNION_OLD[r]} vs measured {UNION[r]})")
    flat = lambda p: (lambda j: p)
    for p in (0.98, 0.95, 0.90):
        P(f"  tau at flat {p:.2f}: 32 rows (31 guesses) {tau(flat(p), 31):5.2f}; 30 guesses {tau(flat(p), 30):5.2f}")
    t98 = tau(flat(0.98), G)
    P(f"  450 tok/s exact at tau {t98:.2f} needs a cycle <= {t98 / 450 * 1e3:.1f} ms: "
      f"{t98 / 450 * 1e3 - floor_ms(32):.1f} ms over the 32-row floor at 280 GB/s, "
      f"{t98 / 450 * 1e3 - floor_ms(32, 300):.1f} ms at 300 GB/s (the loads' best, 287-305)")
    P(f"  max at the floor, zero drafting and overhead: {t98 / floor_ms(32) * 1e3:.0f} tok/s (280 GB/s), "
      f"{t98 / floor_ms(32, 300) * 1e3:.0f} (300 GB/s)")
    P("")
    # drafters: (name, serial ms per block of G+1 = 32 steps, GB per block)
    DR = [
        ("chain head today (MTP step 1.3 ms, ~170 MB)", 32 * 1.3, 32 * 0.170),
        ("chain head, bytes cut (0.5 ms, ~60 MB) [P]", 32 * 0.5, 32 * 0.060),
        ("one-pass block drafter (DFlash2 4-bit 9.4 ms / 0.52 GB at 16 rows; 10 ms at 32 [P])", 10.0, 0.52),
    ]
    CK = [("32 rows today, single pass (est. 90-97)", 93.0, None),
          ("32 rows after kernel work [P]", 65.0, None),
          ("32 rows at the byte floor", floor_ms(32), check_bytes(32))]
    OVER = 5.0   # commit replay (R-12: 3.9-5.4 ms at keep 1-15, ~6 at 31 [P]) + GPU idle seams (2.5-3.6 at SB 2) - overlap
    P(f"== tok/s at a flat 98 % (tau {t98:.2f}, q = 0.98^32 = {0.98 ** 32:.3f}); overhead {OVER} ms per cycle ==")
    P("  kappa 1.0 = the overlapped draft costs the check its own full time (same GPU, NAX-OCC-like: two streams at S 32")
    P("  only 1.2-2.4 % faster than serial); 0.25 = good overlap; 'bytes' = perfect overlap, only the shared bus counts.")
    for cn, Tc, Bc in CK:
        P(f"  check: {cn}: {Tc:.1f} ms")
        for dn, D, Bd in DR:
            s = seq(Tc, D, G, flat(0.98), OVER)
            r = []
            for k in (1.0, 0.25, 0.0):
                v = pipe(Tc, D, Bd, k, G, flat(0.98), flat(0.98), OVER, Bc=Bc if Bc else check_bytes(32))
                r.append(f"k {k:.2f}: {v[0]:4.0f} (check {v[3]:5.1f})")
            P(f"    {dn}: draft {D:4.1f} ms, {Bd:4.2f} GB | sequential {s:4.0f} | pipelined " + "; ".join(r))
    P("")
    P("== The depth the speculative block needs (G = 31; chain head today, check at the floor, kappa = bytes only) ==")
    for pS in (0.98, 0.95, 0.90, 0.80):
        v = pipe(floor_ms(32), DR[0][1], DR[0][2], 0.0, G, flat(0.98), flat(pS), OVER, Bc=check_bytes(32))
        v2 = pipe(floor_ms(32), DR[2][1], DR[2][2], 0.0, G, flat(0.98), flat(pS), OVER, Bc=check_bytes(32))
        P(f"  accuracy at depths 32-63 {pS:.2f}: q_F {v[1]:.3f}, share of speculative blocks {v[2]:.3f}, "
          f"tau_S {tau(flat(pS), G):5.2f}; tok/s chain {v[0]:4.0f}, one-pass block {v2[0]:4.0f} "
          f"(sequential: chain {seq(floor_ms(32), DR[0][1], G, flat(0.98), OVER):4.0f}, block "
          f"{seq(floor_ms(32), DR[2][1], G, flat(0.98), OVER):4.0f})")
    P("")
    P("== Deeper checks at the floor, IF the drafter holds 98 % that deep (union beyond 32 rows extrapolated [P]) ==")
    P("  Union growth per added row: 3.38 (8-16 rows), 2.25 (16-32), so ~1.75 (32-48) and ~1.5 (32-64) [P].")
    P("  Above 256 distinct (C_main at 45 GB) a check splits in two; step 0's 32-row max is already 242.")
    UX = dict(UNION)
    UX[48] = 108.5 + 16 * 1.75
    UX[64] = 108.5 + 32 * 1.5
    for r in (32, 48, 64):
        tb = check_bytes(r, UX)
        fm = tb / BW * 1e3
        tr = tau(flat(0.98), r - 1)
        P(f"  {r} rows: union {UX[r]:5.1f}, {tb:5.2f} GB, floor {fm:5.1f} ms, tau {tr:5.2f} -> {tr / fm * 1e3:4.0f} tok/s with "
          f"free drafting; with a one-pass block drafter (10 ms, 0.52 GB) + 5 ms overhead, sequential "
          f"{tr / (fm + 10 + 5) * 1e3:4.0f}")
    P("")
    P("== What 450 exact asks of drafting (check at the 32-row floor, 300 GB/s) ==")
    Tc3 = floor_ms(32, 300)
    for over in (5.0, 2.5, 0.0):
        room = t98 / 450 * 1e3 - Tc3 - over
        P(f"  overhead {over:.1f} ms: room for drafting {room:5.2f} ms per cycle = bytes / bw + (1 - q) x serial draft; "
          f"e.g. serial <= {max(room, 0) / (1 - 0.98 ** 32):4.1f} ms with zero bytes")
    print("\n".join(out))


if __name__ == "__main__":
    main()
