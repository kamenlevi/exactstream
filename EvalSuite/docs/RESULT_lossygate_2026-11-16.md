# RESULT: the lossy gate's first night (Mon 2026-11-16 03:28-06:28) — the tier-0a calibration is INCOMPLETE (exact self-check PASS; all three step-3 arms crashed, so there are no A / F_hi lines); the lossy-45 base lx45cc1 is cached, PASS in both orders (information)

Pre-registered in docs/PREREG_lossygate_2026-11-15.md: amendment 2 (the tier-0a calibration plan, BigMoE's envs 03be8b3)
and amendment 3 (the close-call variant's base, lx45cc1 on the cc1 snapshot). Runner 64 (tools/day_runner64.sh, sha256
5cab78942ae9501c, as registered) ran every step. The chain was tools/night1116_chain.sh at 2535085 (sha256
767fc369180f23f7), armed 03:22:39 as pid 68891. Its log is results/night1116_chain.log.

## How it ran [K]
- **The start:** 03:28:10, on Leap's a45 marker (a45 ended 03:27:19, rc 0, lifetime peak 42.71 GB), written after
  a45's own start line and read twice, 30 s apart. NO-SWAP was 49 GB at the start; the gate was 43 GB per process (47
  for m1b).
- **No yields:** Kamen was away all night (away_secs.sh >= 900), with no HOLD, PAUSE or shut lid. The swap total stayed
  at 8 GB, so the +2 GB guard never fired.
- **Timeline:**

  | Step | Run | Time | Outcome |
  |---|---|---|---|
  | S1 | x45rpl1 own rows (card_x45rpl1_t0code), 2 processes | 03:28-03:35 | rc 0, tier 0 GO |
  | S2 | x45rpl1 CARD_REPLAY=exact (card_x45rpl1_t0code_rpl), 2 processes | 03:35-03:40 | rc 0, self-check PASS |
  | S2 check | tools/replay_unset_check.py | 03:40 | UNSET-PATH PASS |
  | S3 | p2x45rpl1 on the merged margins | 03:40-03:47 | rc 3: 30 processes, every one rc 1, no tokens |
  | S3 | lossycalx45rpl1 on the merged margins | 03:47-03:56 | rc 3: 30 processes, every one rc 1, no tokens |
  | S3 | m1bx45rpl1 on the merged margins (gate 47) | 03:56-04:08 | rc 3: 30 processes, every one rc 1, no tokens |
  | B1 | lx45cc1 forward, tier 1 code scope | 04:08-05:18 | rc 0, PASS |
  | B2 | lx45cc1 reverse (CARD_ORDER=rev) | 05:18-06:28 | rc 0, PASS |

- The end marker (results/CHAIN_1116_END) was written at 06:28:43.

## The tier-0a calibration: INCOMPLETE [K]
- **Step 1, exact-45's own rows:**
  - tier 0 GO (n 40: arm only 0 / reference only 0, net loss 0; no empty rows);
  - **the unset-path check: PASS.** All 40 keys are token-identical to the REF2-path references (ref2tnax,
    ref2tnaxn.r3, ref2tnaxn.r2), so there were no divergences to check against the near-tie rule. Identity was not
    required: the exact paths may differ at ties.
- **Step 2, the exact REPLAY self-check: PASS** (results/card_x45rpl1_replay.txt):
  - A 1.0000 on 10,315 / 10,315 drafted rows, and 11,023 / 11,023 rows in all;
  - 0 flips in every margin bin, so F_hi 0.00 per 1,000;
  - the prefill picks are all equal;
  - margins were written fresh by both processes for all 40 keys (results/card/replay_margins_x45rpl1.json).
  - This repeats BigMoE's certification (abe7772: 724 / 724 drafted rows at C 160), now on 40 keys at the registered
    C 256.
- **Step 3: no rows.** Each of the three runners stopped after 30 processes ("30 processes without all rows", exit 3).
  All 30 logs of each arm carry the same traceback (grep 30 / 30). Two different faults:
  - **p2x45rpl1 and m1bx45rpl1: IndexError in the REPLAY row gate.**
    - Where: `relax_core.gate_rows` (`cf[j]`, ~/BigMoE/scripts/relax_core.py line 28), called from
      data/r57/snap/rpl1/r57_screen_rpl1.py line 1324. That line is R57_FAST's sure-row gate (the U3 branch).
    - Why, from the code: under R57_REPLAY, a check's block is the reference's next tokens,
      `rp_["ref"][len(out):len(out) + len(dr)]`. The comment there says a block cut by the reference's end has no 16th
      row. So near a key's end, the block and its confidence list hold fewer than 15 entries. `gate_rows` reads
      `cf[0]` to `cf[14]` with no length check.
    - The other branch (`keep_mask` with the copy grams) calls the same `gate_rows`, so a short block fails there too:
      the IndexError when a threshold is set, otherwise the assert on a full 15-draft window.
    - When: after the three warm-ups (REPLAY engages only once the loop sets a key), at the first scored key. So every
      process died on the same key.
    - Why the self-check passed: exact-45 has no R57_FAST, so it never calls `gate_rows`. BigMoE's certification was
      also on exact-45.
  - **lossycalx45rpl1: the engine refuses the config.**
    - Where: data/r57/snap/frozen_rpl1/Leap_scripts/ref2.py, `_ooo_sel`: "REF2: QOM on the NAX path, or on native
      records without ooo_1c, is not built (those two phases would ignore t['qom']); QOM runs on 3c (BigMoE's two phases
      or ref2's one-call) and 3c-N (one-call)".
    - Why: the env is exact-45's flags (R57_NAX=1, with no REF2_OOO_1C or REF2_NAX_1CQ) plus R57_FAST=QOM and R57_RCUT.
      The QOM configs that run (p2x45rpl1, lx45cc1) carry REF2_OOO_1C=1 and REF2_NAX_1CQ=1.
    - When: the first verify of the first scored key, after the dfl and mtp warm-ups (gsm8k is outside its QOM sets).
  - **Why no check caught either fault before arming:**
    - tools/r57_headtest.py runs each env to the model load, and the dry runs stop at the pins.
    - Both faults come after the load, at the first scored key.
  - **Cost:** 90 crashed processes, about 28 minutes at ~42 GB (03:40-04:08), and no rows. The runner retries a
    process until 30 have run, even when every one crashes the same way.
- **Outcome:**
  - With no passing arm, A_line and F_line cannot be computed; amendment 2 computes them "from these runs only". So
    there is no adoption check either.
  - Tier 0 stays the mini-panel KILL line, and a REPLAY pass stays information (amendment 2a).
  - A re-run needs BigMoE's fix in a new snapshot, a lossycal env on a path where QOM runs, and a dated amendment with
    the new pins. Before arming, each env should decode one scored key on the real loop.

## The lossy-45 base lx45cc1, cached in both orders: PASS (information) [K]
lossy-45 as shipped (env blob 216c8e20d35494ad32c855eb83c2712e86c94d18) on the cc1 snapshot, tier 1 code scope,
against the exact references (ref2tnax, ref2tnaxn.r3, ref2tnaxn.r2). It is Monday's close-call reference
(CARD_REF=card_lx45cc1_t1code / card_lx45cc1_t1code_rev). The base's own verdict is information.

| Line | Forward (card_lx45cc1_t1code) | Reverse (card_lx45cc1_t1code_rev) | Reference |
|---|---|---|---|
| HumanEval+ (n 164) | 92.1 %, -0.61 pt, p 1.0000 (2 / 3) | 93.9 %, +1.22 pt, p 0.6875 (4 / 2) | 92.7 % |
| MBPP (n 100) | 77.0 %, -2.00 pt, p 0.7266 (3 / 5) | 77.0 %, -2.00 pt, p 0.7266 (3 / 5) | 79.0 % |
| **Pooled code (n 264), line > -3 pt** | **-1.14 pt, p 0.5811 (5 / 8): PASS** | **+0.00 pt, p 1.0000 (7 / 7): PASS** | |
| Runaway guard (pooled code) | arm only 8 / reference only 4, p 0.3877: PASS | 8 / 5, p 0.5811: PASS | |
| IFEval (n 100) | 88.0 %, +1.00 pt, p 1.0000 (4 / 3) | 88.0 %, +1.00 pt, p 1.0000 (4 / 3) | 87.0 % |
| Interim (n 132 of 264) | z -0.71: CONTINUE | z -0.82: CONTINUE | |
| **Verdict** | **LOSSY tier1 (CODE-ONLY mode): PASS** | **PASS** | |

Counts in brackets are arm only / reference only (exact McNemar p).

- **Validity, each order:**
  - the env blob, the trees (cc1 85a88a46, frozen_cc1 ec97bdbb, bigmoe d92cdb4b), the screen efc1f9b6f531 and the .so
    ff69fa24a745 matched the pins at the start;
  - all 19 processes ended rc 0;
  - the 40 outside modules were equal to the first process's manifest on all 18 later processes;
  - QOM was engaged on every run in its QOM sets (humaneval, mbpp);
  - no engagement field, as a base arm should have none (20 / 20 kept runs per process);
  - 364 rows (264 code + 100 IFEval), 364 unique keys, the same key set in both orders.
- **Memory:** the lifetime peak was 42.2 GB in both orders, under the 43 GB gate.
- **Information, forward vs reverse token identity:**
  - identical: humanevalx 104 / 104 and ifeval 100 / 100, the sets outside the config's QOM sets;
  - differing: humaneval 32 / 60 and mbpp 35 / 100, its QOM sets.
  - So the order effect sits in the QOM sets. This is consistent with QOM's state-dependent masks, and it is why the
    variant gate pools both orders.
- **Speed:** no read-out. tools/lossy_speed.py compares an arm with its base, so it runs after each close-call arm.

## Data
- results/night1116_chain.log, results/CHAIN_1116_END, and results/day64_<tag>_runner.log for x45rpl1, p2x45rpl1,
  lossycalx45rpl1, m1bx45rpl1, lx45cc1 and lx45cc1_rev.
- results/CARD_x45rpl1_DONE(.step1), CARD_lx45cc1_DONE, CARD_lx45cc1_rev_DONE.
- The verdict files: card_x45rpl1_{tier0code,unset,replay}.txt and card_lx45cc1{,_rev}_{tier1code,interimcode}.txt.
- The rows: card_x45rpl1_t0code, card_x45rpl1_t0code_rpl, card_lx45cc1_t1code and card_lx45cc1_t1code_rev.
- Per process: results/card/es_cd<tag>_<start>_* (logs, manifests, checks, runs, tokens), with replay_rows_*,
  replay_margins_x45rpl1.json and the key lists.
