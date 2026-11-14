"""WEEK98 pricing (Leap, Sat 14 Nov 16:5x): a partial trunk pass (layers 1..L of the 125B) on drafted rows, ONE
drafted token at a time, sequential over draft depth 1..30, at 45 and 24 GB. Existing engine numbers only (no run).
Anchors (all measured, sources in README.md next to this file):
  C1    0.62 ms per layer at 1 row, today's engine path: the 1-row check is 31.4-31.9 ms for 48 layers + head (BigMoE
        CHS5, Sun 11-08 08:12, with SB1), CPU-bound (graph build ~31.5 of ~33 ms); minus ~1.5 ms head / final / glue.
  FAST  0.39 ms per layer at 1 row, a compiled 1-row trunk (NOT BUILT): bytes floor 87 MB per layer (dense ~66 MB from
        D32's 32-row dense floor 12.3 ms at 280 GB/s; 10 experts x 2.135 MB) = 0.31 ms at 280 GB/s, at 80 %.
  IO24  +0.035 ms per layer at 24 GB: a drafted row's misses (w13 HumanEval: 246 MB SSD per 13-row check = ~0.18 misses
        per layer per row, ~0.19 ms each). Moved earlier from the verify, not new bytes.
  FIX   per pass: embed + ridge + hand-off + one sync: 0.6 ms (engine path), 0.3 ms (fast path). Estimate.
  T0    today's per check (t30b frame: draft + 32-row verify), ESTIMATED: 45 GB 100 ms = verify ~97 (the measured 16-row
        HumanEval check 61.4 ms at C 256 + rows 17-32: dense +4.2 (D32), experts +22.6 (NAX6 +1.41 ms per row), misses
        ~+4, GDN row by row ~+5) + a ~5 ms chain (as dfl's measured 5.8). 24 GB 145 ms = the 13-row HumanEval cycle
        78 ms (W24d) + 19 rows x ~3.3 ms (experts 1.41 + SSD ~1.9) + dense ~4; the 24 GB line runs <= 13 rows today.
  REUSE the verify skips layers 1..L on rows a partial pass already did (the engine's rows are batch-invariant, so this
        stays exact; needs GDN-state rollback): saves (L / 48) x the verify."""
K0 = {"HumanEval+": 13.83, "MBPP": 6.92}
LINES = {45: dict(T0=100.0, TV=97.0, io=0.0), 24: dict(T0=145.0, TV=140.0, io=0.035)}
PATHS = {"engine": (0.62, 0.6), "fast": (0.39, 0.3)}
D = 30
print("| line | L | path | ms / drafted token | ms / 30-deep draft | net per check (after reuse) | x today's check | "
      "break-even kept / check: HumanEval+ (13.83) | MBPP (6.92) |")
print("|---|---|---|---|---|---|---|---|---|")
for line, p in LINES.items():
    for L in (5, 15, 25, 45):
        for path, (c, fx) in PATHS.items():
            t = (c + p["io"]) * L + fx
            draft = D * t
            net = draft - L / 48 * p["TV"]
            f = (p["T0"] + net) / p["T0"]
            be = {k: v * f for k, v in K0.items()}
            mark = lambda x: f"{x:.1f}" + (" (> 31: impossible)" if x > 31 else "")
            print(f"| {line} | {L} | {path} | {t:.1f} | {draft:.0f} | +{net:.0f} ms | {f:.2f} | {mark(be['HumanEval+'])} | {mark(be['MBPP'])} |")

# Block refresh (the cheap option the numbers point at): draft k tokens on the carried stand-in, then ONE k-row partial
# pass refreshes all k (lag 1..k inside a block). The dense part is flat from 1 to 16 rows (C0 timeline); each added row
# costs experts: the measured 16-row HumanEval check at C 256 (61.4 ms, rli/rlh) vs the 1-row check (~31.5 ms) gives
# 2.0 ms per added row per 48 layers = 0.042 ms per layer per row; at 24 GB plus the row's SSD share ~1.9 ms per 48
# layers = 0.04 ms per layer per row. Engine path only. +-1 pass: full verify reuse needs the last block (or token) passed.
ROW = {45: 0.042, 24: 0.042 + 0.04}
print()
print("| line | L | refresh every k | passes / draft | ms / draft | net per check (after reuse) | break-even HumanEval+ | MBPP |")
print("|---|---|---|---|---|---|---|---|")
for line, p in LINES.items():
    for L in (5, 15, 25, 45):
        for k in (1, 2, 4, 8, 16):
            c, fx = PATHS["engine"]
            n = -(-D // k)
            draft = n * ((c + p["io"]) * L + fx + (k - 1) * ROW[line] * L)
            net = draft - L / 48 * p["TV"]
            f = (p["T0"] + net) / p["T0"]
            be = {kk: v * f for kk, v in K0.items()}
            mark = lambda x: f"{x:.1f}" + (" (> 31)" if x > 31 else "")
            print(f"| {line} | {L} | {k} | {n} | {draft:.0f} | +{net:.0f} ms | {mark(be['HumanEval+'])} | {mark(be['MBPP'])} |")
