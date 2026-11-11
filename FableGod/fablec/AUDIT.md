# AUDIT — three claims against the code and the recorded data (Fable C, mission C1 task 1, Mon 2 Nov 2026, 23:0x EET)

[K] measured (file cited) · [P] arithmetic on [K] · [G] judgement. CPU only.

## (a) Fable B's floors, EXACT ~19-21 / FAST ~16-17 ms per 16-row check — CONFIRMED, two soft spots

- **Base recounted [K]:** `~/Leap/data/c0/c0_c160_probes.json` (15 probes, `synced_16`, one 0.222 ms sync per record removed) gives GDN 2.06 build / 10.33 eval, HC 4.60 + 4.68, attention 5.82, experts 50.91 — every number in `fableb/GDN_PASS.md` §2.1 and `fableb/BOUNDARIES.md` §2.2 within 0.02 ms. My sum is 32.3 ms; Fable B's 34.0 books ~1.7 ms more as "router / shared / tail". Immaterial.
- **14 ms bytes floor [K/P]:** GDN 1,516 MB → 5.41 ms at 280 GB/s (`fableb/gdn_bytes.out`, shapes from `~/BigMoE/cfg/config.json`); HC 96 × 13.2 MB → 4.5 (`TABLE_350.md:4703`); attention + head (318 MB) + router/shared ~3.3; sum 13.2-14.0 at 258-294 GB/s (`TABLE_350.md:4479`). D1-B's measured −0.73 / −1.03 ms (`TABLE_350.md:4703-4760`) is what §5 uses.
- **Soft spots:** the FAST ledger (14 + ~1.3 + ~1 + ~1) sums to **17.3**, printed "~16-17" [P]; ~3 ms of "sequential structure" is unmeasured in situ (GDN_PASS §2.4's trace has not run) [G]. Nothing contradicted.

## (b) Leap: "NAX6 can't be beaten bitwise", "the ALU hybrid can't be bitwise" — CONFIRMED on every measured row

- **Hybrid [K]:** `scripts/nax_order_probe.py` runs the 16×32×32 op in NAX6's form on 1,024 × 512 adversarial outputs against eight fp32 orders: best 44.96 % bitwise, integers 100 % (`data/nax_order_probe.txt`); ~100 block-fp models top out at 45.6 % (`data/nax_order_fit.txt`). An inference from failure to fit, not a proof — but any nonzero mismatch kills bitwise. H-1 on FAST: +1-3 % KILL, post-hoc +10 % (`TABLE_350.md:4948`).
- **NAX6 [K]:** N7-2 +11 %, N7-5 +35 % (bitwise, slower); N7-3 / 4 / 8 rule out threadgroup memory, operand form, added registers; kernel A 0.575 ms vs max(bytes, ops) ~0.37. The last exact lever, the four small dispatches (N7-9), **died at 22:41: N7-10 bits 64/64, +1.2 %, KILL** (`data/n7f_screen.txt`).
- **Flag [G]:** N7-8 only *added* registers. N7-6's linear ~7 µs per group with no co-residency gain, and N7-1's knockouts summing to 60 %, read as register-limited residency of ~1-2 threadgroups per core: a trim crossing the next residency step (`tierglu_nax6.py:134-141`) is the one untested kernel-A lever.

## (c) Cloud ladder: "the SSD is critical; in exact mode only hideable; 2 ms at target" — CRITICAL yes; "ONLY HIDDEN" and "2 ms" REFUTED

- **The default [K]** (`~/BigMoE/data/c50/c45l_256.txt`): 62.4 MB, 39.0 misses per kept token. HumanEval/68: demand SSD 124.4 ms of a 170.8 ms check + commit, 94.2 MB per kept token → 1,452 MB in 124 ms = **11.7 GB/s** [P]: bandwidth-bound bursts (peak 14.6). The warm check is 87 ms of GPU (`c0_live.out`): the SSD is already the longer pole; hiding leaves max(124, 87) (K-62L: `TABLE_350.md:3669`). The read path (`ext/bigmoe_ext.cpp:347-375`, `:533-571`; `bigmoe/tier.py:152-158`) wastes no latency on that scale; the bytes do it.
- **Bytes at the target [K/P]:** `~/MoonShots/data/cachesize45.txt` line 8: C 320 + M-17 → **37.9 MB per kept token**, ceiling 343 tok/s at 13 GB/s, 309 at 11.7 (my replay 36.9, `fablec/replay_c1.out`). Per check at kept 12.6: 478 MB = **37-41 ms = the ladder's whole 38.5 ms cycle**. "SSD 2 ms" is a ~40 ms byte floor; `TRIED.md:721` already states the 350 budget (37.1 MB, C ~323 = 50.8 GB, zero margin). Exact chat at C 320 caps at 77 / 69 tok/s (ifeval / multi) on bytes alone (`~/FableGod/floor_bytes.out` §2).
- **"Only hidden" [K]:** this week's byte cuts were residency and size (M-17 −15.7 %, `TRIED.md:11`; Q1 ×0.60, `TRIED.md:33`); hiding gave OOO +13 % (`TRIED.md:17`) and R-57's KILL (`HANDOVER.md:492-500`), whose predictor recalls **0.665** of the next layer (`replay_c1.out` §1).
- **Net:** exact at C 320 is SSD-capped at ~310-340 tok/s on code with the drive at 100 %; 350 needs ≲ 33 MB per kept token (RAM, Q1 or P-frame), look-ahead after.
