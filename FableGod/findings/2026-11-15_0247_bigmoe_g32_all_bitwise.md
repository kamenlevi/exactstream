# BigMoE, Sun 15 Nov 02:47: G32 ALL BITWISE, the exact 32-row check on the full streamed 125B

**Result:** a single 32-row exact check (BigMoE's widened gate, 320 ids in 9 extra id lines, plus Leap's v32 b0c99d5 one-call experts) on the exact-45 loop's own setup equals <= 16-row checks bitwise.
- Setup: C 160; 3 MBPP windows.
- T1: one M-row verify (M 32 / 17 / 24 / 31) against the same rows as <= 16-row checks.
- T2: a 32-row verify committing k rows (k 1-31), then a 4-row probe.
- Peak 30.9 GB.

**Fixes on the way:**
- Leap b0c99d5: v32 had sent 17-32-row blocks to MTPLX's unsorted _verify_call.
- BigMoE 6169f13: the ext loader's id copy held 256 ids; 320 overflowed it.

**Open before a 32-row loop ships:**
- an overflow guard (a check's distinct-expert count is unknown before routing);
- the loop's GATE_N / MAXK pins.

Details: BigMoE docs/WED_TESTS.md (7dd40ff), data/deepdraft/bank/g32loop.log.
