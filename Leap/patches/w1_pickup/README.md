# W1-PICKUP patch for BigMoE's ext (Leap, Tue 2026-11-03; TABLE_350 'W1-PICKUP')

`w1_pickup.patch` applies to BigMoE's ext/bigmoe_ext.cpp at ec08cec (the same ext as a0a5fbe):
`cd ~/BigMoE && git apply ~/Leap/patches/w1_pickup/w1_pickup.patch` (BigMoE's call; its ext/build_w1).
It touches no other file. gated.py needs no change: the flags are read in the ext, and gated.py's Python `_spin`
kernel has no caller.

## Flags (both OFF by default; today's kernels and release are the fallback)

- **BIGMOE_SPIN_COHERENT=1.** SpinWait runs one simdgroup that polls ONE row (pair 0 of its (layer, pass) region).
  - It uses `volatile coherent(system)` loads with seq_cst `thread_scope_system` fences (MLX's MLX_METAL_FAST_SYNCH
    pattern, `#pragma METAL internals`).
  - The loader's release writes that one row instead of 128 pairs.
  - Same stamps, same decode (the -1 / -2 codes), the same fast path (status[16]) and the same timeout behaviour
    (status[0]++, out = 0). A stale word polls again; status[17] counts those re-polls.
  - BIGMOE_SPIN_COH_CAP = polls before the timeout (default 1000000, ~0.25 s at ~0.25 us per poll; today's 128
    back-off pairs end after ~0.2 s).
- **BIGMOE_POST_SYS=1.** post_ids writes the routing row through a `coherent(system)` pointer with system-scope fences
  (plus one fence after the last tag).
  - The words and their order are unchanged.
  - With the flag off, the macros (ROW_T, RST, RSCOPE) preprocess to today's exact statements.

## Do the base and sidecar spins share a row?

No. Each spin polls its own row:
- the region is keyed by (seq, L): `(seq * g_nl + L) % g_reg_n`;
- JOINT's sidecar spin is `gate.spin(..., t["side"], ...)`, so it carries its own L (side0 + L);
- the loader serves and releases the sidecar row under that same L.

So pair 0 of a region belongs to exactly one (pass, row), and nothing is shared. A region is reused only after
g_reg_n = 128 posts; the stale tags hold an older seq, and the stamps carry seq & 0x3FF, so a reused row can't pass.

Optional: under SPIN_COHERENT only pair 0 is used, so BIGMOE_SPIN_PAIRS=1 would shrink the region ring from
~12.6 MB to ~96 KB. Never set it without SPIN_COHERENT: today's spin needs its pairs.

## Checks

| Check | Result |
|---|---|
| `git apply` on a copy of ec08cec's file | clean; the result equals `new/` |
| `c++ -fsyntax-only` with BigMoE's ext/build flags (flags.make) | rc 0 |
| `test/kcheck` (the exact kernel texts, extracted from the patched file) | KCHECK PASS, test/kcheck.txt |

What kcheck covered, on the GPU (beside Breakthrough's shared job):
- **Coherent spin, pickup path:** 400 trials with the engine's region ring and the one-row release; the slots are
  exact, including the -1 / -2 codes. Pickup median 5.9 us at w 0 and 3.5 us at w 1 ms (shared GPU).
- **Fast path:** out = dep[1..n], status[16] = 1.
- **A stale word fixed 200 us late:** it polled again (status[17] = 131), and the final slots are exact.
- **Timeout** (COH_CAP 1000, no release): status[0] = 1, out = 0.
- **post_ids:** today's kernel, the patched default and POST_SYS write bitwise-identical rows (224 ints) and dep
  (161 ints), with and without a residency table.

The timing comes from the harness (native/w1_pickup.swift, quiet lock, 300 trials per cell):

| Path | Release | Pickup |
|---|---|---|
| Today | 6.5 us | 16.7 us at the floor, +~10 % of the wait on top (73.5 us at 1 ms) |
| Coherent | 0.2 us | 2.9 us flat |
| MTLSharedEvent | | 77-360 us |

Post visibility: a device-scope post is seen ~20-23 us after the next real dispatch, and practically never while
the GPU only spins. A system-scope post is seen in 7.4 us.

## For S0

- Today's spin, pair p -> elapsed time on this Mac (quiet): p0 17 us, p6 22, p26 73, p41 236, p52 623, p58 1080,
  p65 2074.
- Under SPIN_COHERENT the pair index is always 0. A histogram of the poll count `it` (~0.25 us per poll) would time
  the GPU-side wait directly; S0 owns that design, and the kernel has the `it` counter in its loop.

## Engine A/B and line (agreed with BigMoE)

Thu's traced arms: default / + JOINT / + JOINT + keepalive / + SPIN_COHERENT / + POST_SYS. The line: stall_report's
cadence a minus burst a falls >= 0.03 ms AND tokens identical.

## Risks

- `#pragma METAL internals` is undocumented. MLX uses it in production behind MLX_METAL_FAST_SYNCH, and an MLX issue
  reports a fast-synch deadlock in a distributed pipeline.
- A failure here shows as a timeout (status[0], the pass reported invalid), exactly as today's spin. Keep the flags
  off until the A/B passes.
