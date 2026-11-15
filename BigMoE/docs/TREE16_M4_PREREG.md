# TREE16 M4 engine A/B: pre-registration (BigMoE, Sun 15 Nov 2026 17:17, before any run)

**Question.** On the 125B at exact-45, does the exact 16-node EAGLE-2 tree check decode faster than t30b's chain-16, with
identical tokens? The tree check is snapshot t16b, arm R57_TREE16=1; the chain arm is the engine as it runs today.

**Slot.** The orchestrator, Sun 17:1x: Monday night 16 -> 17 Nov, FIRST in the order.
- 23:05 (after the 23:00 apps quit) to a HARD END at 00:25, stopping at a process boundary.
- One quiet ticket, NO-SWAP >= 36 GB.
- Gated on Sunday night's M1 (t16a: the tree's main chain == the greedy chain on the real model). If M1 fails, the slot
  is dropped.
- Runner: scripts/m4_night.sh (head-tested: all four runs reach the model load). End marker
  data/deepdraft/bank/M4_1116_BIGMOE_DONE on any end.

**Snapshot.** t16b (7f4b4e0 + 72b5f46): a frozen copy of Leap de7fcc0 (week98/tree16.py, route (b)), Breakthrough 35aacf4
(B1 tree_meta, B2 ple_tree) and TrainLab a98ed8c. If Leap or Breakthrough change the tree pieces before the run, it is
re-frozen (t16c) and this file says so before the run. runs.json "t16" records every file's hash.

**Config (both arms).**
- x45rpl1's env line + R57_PRESET=45 R57_REPS=1 R57_WARM=1 R57_CHATSW=1 R57_MTP_ALL=1 R57_MTP_CKPT=dd_t30b.safetensors.
- Args: C 160, 256 new tokens.
- Keys (16): the first 8 of EvalSuite's dec_humaneval (HumanEval/68, 16, 91, 6, 79, 19, 161, 17) and of dec_mbpp
  (mbpp/2, 13, 16, 17, 19, 28, 29, 30). Prompts are 81-367 tokens, so every check stays inside the dense QSA budget
  (a + T <= 2044); no chain fallback is expected, and runs.json reports any.

**Arms.**
- **C**, chain-16: MTPHead.draft_cut, MTP_K 15, confidence cut 0.3 (the engine today).
- **T**, tree-16: root + 15 best-first pops, top-8 children, no cut; tree_verify, walk, tree_commit; the accepted path's
  pairs go into t30b's MTP history.
- **Order:** C1, T1, T2, C2, four processes. Back-to-back arms favour the second; ABBA balances that. A run starts only
  if its expected length (the last run's + 10 %) ends by 00:25.

**Pass rule (EXACT), decided first.**
- Every key's tokens are identical across all runs that ran (C1 = T1 = T2 = C2), and no run is cut or flagged bad.
- Any difference means M4 FAILS: no speed claim, and a bug hunt comes next.

**Then speed** (headline = whole decode: runs.json "tps" = n / decode wall per key):
- R_set = the geometric mean over the set's 8 keys of mean(T1, T2 tok/s) / mean(C1, C2 tok/s).
- K_set = kept per check pooled over checks, T / C.
- **Prediction:** S9c (MTP Head, TrainLab fa3ce76) priced kept per check at x1.060 HE+ / x1.136 MBPP vs chain-16. The tree
  check costs +2.5 % (Leap TREE16-TIME v2). So R_humaneval ~ 1.034 and R_mbpp ~ 1.108.
- **Known differences from that pricing:**
  - Arm C cuts at 0.3, so it verifies fewer rows on unsure checks; S9c's chain-16 is the full chain.
  - The tree's drafting cost is in the wall: 14 head forwards with a top-8 readback per pop. Sunday night's M1 measures
    tree vs chain drafting ms.
- **Reading:**
  - PAYS if R_mbpp >= 1.03 and R_humaneval >= 0.99.
  - COSTS if R_mbpp < 0.98 or R_humaneval < 0.97.
  - NEUTRAL otherwise.
- **Also reported:**
  - per-key ratios, and the C2/C1 and T2/T1 ratios (run-to-run noise);
  - T's side-path accepts, main-chain-end accepts, chain fallbacks, and draft / commit ms;
  - K_set against S9c's. A K more than 0.05 below S9c's means investigating the tree drafter in the loop (MTP history,
    PathCache) before any speed reading.

**Partial window.**
- If the hard end stops the runs early, EXACT is judged on what ran (it needs at least one C and one T run).
- A speed reading from fewer than two runs per arm is flagged ORDER-BIASED and is not a verdict.

**Analysis.** scripts/tree16_m4ab.py (committed with this file, tested on fake runs) -> data/deepdraft/tree16/m4ab.json.
The runner calls it on any end.

## Amendment 1 (Sun 15 Nov 21:0x, before any run; the orchestrator 20:5x)

The orchestrator's HOLD at 18:52 (Kamen closing the laptop) stopped Sunday night's runner, so the M1 gate may have no
verdict by Monday 23:05. The runner (scripts/m4_night.sh) now applies an **M1 condition** instead of dropping the slot:
- **If data/r57/night1115.log has "#3 M1 rc 0: PASS":** C1 T1 T2 C2 start at once, as booked.
- **Otherwise:** M1 runs FIRST under the same ticket, as Sunday night's #3 verbatim: t16a, keys HumanEval/1, mbpp/16,
  ifeval/1219, C 160, 128 tokens, tag t16am1w.
  - On PASS (every check's tree main chain == the greedy chain, rc 0): C1 T1 T2 C2 follow.
  - On FAIL or rc != 0: the runner stops (no A/B) and writes the end marker.
- **Hard end:** unchanged at 00:25, so Leap's slot starts at 00:30 on the marker.

**A window cut short** (by the hard end, heat, HOLD / PAUSE or the swap guard, e.g. only C1 T1 T2):
- The token-identity pass rule applies to the runs that ran (it needs at least one C and one T run).
- Timing from an unbalanced set (fewer than two runs per arm) is information only, flagged ORDER-BIASED, and not a
  verdict.
- This was already the rule above. The amendment states it for heat cuts too.

**Tested before arming:**
- Head test: M1 + the four runs reach the model load.
- Stub simulations of the three branches:
  - S1: PASS line present, straight to C1 T1 T2 C2;
  - S2: no line, M1 PASS, then the A/B;
  - S3: no line, M1 FAIL, stop with the marker, no A/B.
