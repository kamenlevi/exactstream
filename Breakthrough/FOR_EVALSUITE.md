# For the Eval suite: repetition guard on the B12 path (2026-10-22 10:37)

Breakthrough -> Eval suite: the repetition guard is now on for B12 / B8 / B8-fast (orchestrator 09:04). Re-run the capped rows when ready.
WHY IT WAS OFF: spec125.generate imports BigMoE's guard as `from bigmoe import repguard`. With ~/BigMoE off sys.path (your setup, and mine) that import failed silently, so no guard.
THE FIX (~/Breakthrough HEAD 0418c6c or later): load_b12 registers v3f's copy of bigmoe/repguard.py, byte-identical to the main track's (MTPLX LoopGuard, on by default; BIGMOE_LOOP_GUARD=0 turns it off), under that name. It does not import BigMoE's package. A new process picks it up at load_b12; no env change is needed. Your MTP setup is otherwise unchanged and matches what ships (MTP_C 64 vs my 32 changes RAM only).
CHECKED (10:32-10:36; your exact B12 + MTP setup with mass weights, one row per fresh process, guard off vs on):
- Synthetic loop ("write this sentence 100 times"): off = loops to the 1,024 cap; on = token-identical for 257 tokens, then the guard arms and the model ends with EOS at 258.
- ifeval/1248 and ifeval/2304 replayed cold: identical with the guard off and on (470 and 587 tokens, EOS); the guard never armed.
- math500 intermediate_algebra/1422: the guard armed on the long solution's repeated checks (61 steered steps); the row still hits 1,536, as in your run.
CAVEAT for the re-run: frugal decode depends on which experts the previous rows left resident. Cold replays of ifeval/1248 and 2304 do not loop even WITHOUT the guard. So re-running only the 14 capped rows in a new process changes them for reasons besides the guard. The like-for-like test is the affected set(s) in your original order with the guard on, if time allows before 11:30. Otherwise, re-run the capped rows and note the caveat.
