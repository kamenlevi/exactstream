# The lossy 32-row check costs ~50-55 ms on the measured engine, not 21-25 ms (Fit12, Mon 16 Nov 2026 08:3x)

WEEK98 pillar 2 (lossy check), desk only. Pre-registered and recorded in /Users/kamen/Fit12/PLAN_12GB_90_fit12.md
("WEEK98 PILLAR 2 LOSSY" + "LOSSY32 RESULT"). Script /Users/kamen/Fit12/scripts/lossy32_price.py, output
/Users/kamen/Fit12/data/lossy32/.

## What was priced
Candidate lossy 32-row checks on the lossy-45 base, at 45 GB on code: GB per check, byte floor, and a realistic time.
- The realistic time is per class, from Breakthrough's measured engine profile (a082f8c) and lossy-45's measured
  16-row dense prices (Leap TABLE_350), grown to 32 rows by Leap's per-tile factors.
- It adds the CPU build as a bound, and lossy-45's measured 45 GB misses.
- Validated at 16 rows: the model gives 61.4 ms for lossy-45's HumanEval check, against 61.2 measured (BigMoE's release
  rows, C 256).

## Findings
1. **Realistic price:**
   - The 1c lossy check family costs ~50-55 ms per 32-row check, 2.3-2.4x its byte floor.
   - Breakthrough's "5.86 GB" plan (1c tiers + 4-bit mixers + a 16K head list) is 49.8-53.3 ms, against the "24.9 ms at
     84 % bandwidth" used for the 900 tok/s framing.
   - The dense side (~24-29 ms, about 3x its floor, with ~5 ms of attention / GDN cores) and the CPU build (27.9 ms) do
     not shrink with bytes.
2. **lossy-45 as shipped, run at 32 rows, is ~90-92 ms**, the same as exact-45's measured 90.5 ms. Its faster dense side
   (-14 ms) is eaten by 45 GB SSD misses (+13 ms).
3. **The plan's bytes were all-domain.** Its 99.9 experts per layer and 36.4 four-bit-equivalent are the means over code,
   IFEval and multi windows.
   - On code, the union is 114.2 experts per layer and 1c keeps 39.2 four-bit-equivalent.
   - So the code-scoped plan is 6.12 GB with a 21.9 ms floor, not 5.86 GB with 20.9 ms.
4. **The exact 32-row floor on code is 47.3 ms, not 52-54.** REF2-T's rank tiers read the low bit plane for only 31 of
   the 114 union experts per layer (the ones some row ranks first or second).
   - The engine's experts run at 1.43x that real floor.
5. **The cheapest lossy check EvalSuite could gate this week is 1c tiers on code (CODE-ONLY).**
   - It carries ~92 % of the plan's gain, and its rule is built and self-checked.
   - Still missing: the engine wiring (a small ref2 change, then BigMoE's wiring and counter) and a new base run on its
     snapshot.
   - This week's gate would test it at 16-row checks.
6. **What 900 tok/s lossy would need.** At the lossy K* (kept 16.91 per 32-row check), 900 tok/s needs a ~15 ms check.
   - Even with every kernel near its floor and no misses, the plan's bytes give ~30 ms.
   - The lossy check shares its binding terms with the exact check work: one-pass forms above 16 rows, dense-side kernel
     efficiency and cores, and the CPU build.
