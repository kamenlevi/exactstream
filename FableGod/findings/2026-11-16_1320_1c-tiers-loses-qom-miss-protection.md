# 1c-tiers alone gives back QOM's miss protection; a resident-only rule for its 3-bit tier is cheap (Fit12, Mon 16 Nov 2026 13:2x)

WEEK98 pillar 2 (lossy check), desk only. Pre-registered and recorded in /Users/kamen/Fit12/PLAN_12GB_90_fit12.md
("RES-1c PRICE + SCOREBOARD v3", "RES-1c PRICE RESULT"). Scripts res1c_price.py, res1c_16.py and lossy32_board.py in
/Users/kamen/Fit12/scripts; output in /Users/kamen/Fit12/data/lossy32/.

## Findings
1. **1c-tiers gives back QOM's miss protection.** On the code sets, 1c-tiers replaces lossy-45's QOM rule, which skips
   low-ranked experts that are not in memory. Without it, every expert 1c keeps is read, in memory or not.
   - On the map's residency masks (the 45 GB pool), that leaves about as many SSD misses as before: 29 per 16-row check
     under 1c vs QOM's lower bound of 20.
   - At 32 rows it is 60 misses (~15 ms), where the earlier price had assumed ~5 ms.
2. **A resident-only rule for the 3-bit tier fixes it.** 1c-tiers keeps the 4-bit tier always, and reads its 3-bit
   tier only when the expert is in memory.
   - Misses drop to ~8 per 32-row check (~2 ms).
   - Its predicted quality cost is small: +0.05 pt code flips (all rows) and +0.03 (non-tie). That comes from
     Breakthrough's calibrated flip law, and the 1c self-check reproduces that law's input within 0.1 %.
   - Only 3.6 % of the 3-bit-tier experts are dropped.
   - A real map arm is next. Its masks are built.
3. **The realistic lossy 32-row check:**
   - the plan (6.12 GB on code): 61-66 ms, or 48-52 ms with the rule;
   - this month's best rung of the ladder: ~42-44 ms.
   - Kamen's 900 tok/s needs ~15 ms.
4. **A falsifiable prediction for today's 1c-tiers gate rows:** the variant's misses per check are at least 0.8x the
   base's on the QOM sets (humaneval, mbpp), and at most 0.25x on humanevalx.
