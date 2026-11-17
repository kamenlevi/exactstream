# DRAFT, not filed: PREREG_lossygate amendment 6, the GATED, MEASURED lossy stack for Saturday (lossy-45 + CC(1) + the t30b Q8H32 tree-16 drafter + the exact-safe engine levers)

Prepared Tue 2026-11-17 09:4x from the orchestrator's decisions of 09:3x and 09:4x (relayed by NewIdeas). Rewritten at
15:0x for the orchestrator's 15:0x composition: G1 is skipped, tgl16eu is out, and tgl16e and GDN16 are in.
- It is filed into docs/PREREG_lossygate_2026-11-15.md when BigMoE sends the stack's blob (Thu ~midday). That is
  BEFORE the smoke and before any stack data.
- Placeholders are in <angle brackets>. The open points are listed at the end.

## Amendment 6 (Thu 2026-11-19 <HH:Mx>, before any stack data): the lossy stack's gate (tier 1 in both orders vs exact, the card gate (A), the headline timing run)

### Why
- **The orchestrator, 09:3x:** a GATED, MEASURED lossy number by Saturday on a NEW config, on BigMoE's r57 path
  (plumbed from Wednesday).
  - It supersedes the chain-16 stk / stk0 plan. tools/stk_chain.sh (ede7aa9) was never armed and has no data.
- **The orchestrator, 09:4x:**
  - t30b runs at Q8 + the 32K screen if MoonShots' Wed ~07:30 Q8 read-out holds kept within its pre-registered line.
    Otherwise it runs at bf16 + the 32K screen. It never runs as stored.
  - Sync-free is BigMoE's default if MoonShots' bit test on BigMoE's ops passes by Wed evening. fused2 is out.
  - In lossy mode, tier 1 and the card gate run ON the chosen format, so the gate covers it. This amendment therefore
    pins the format and the screen, alongside the tree flags.
- **The orchestrator, 15:0x (Tue): the composition.** The stack is lossy-45 + CC(1) + the exact-safe engine levers:
  GDN16, tgl16e, sync-free, and the Q8H32 drafter.
  - G1 is SKIPPED this week: every G1 env carries the 1c tiers that failed V2 on HumanEval+ (PREREG 5b, dated 15:0x).
  - **tgl16eu is CONDITIONAL** (the orchestrator 15:3x, correcting 15:0x's premise). Breakthrough showed it does not need
    the 1c tiers: without them, one-call segments shrink 13.9 % at 16 rows, and the unchanged pairs stay bitwise
    (f09e1e6).
    - **The rule:** if Leap's slot #3 (Wed 07:15) prices tgl16eu at >= ~1 ms on the no-tier base path (F16nx vs the same
      arm without it), it goes BACK into this amendment before Thursday's tier 1, as a FAST lever judged inside the
      stack's gate. Otherwise it stays out, and tgl16e alone is in.
    - BigMoE's live key test gives the real ON / OFF number (information, beside the rule).
    - <Slot #3's reading, with its file: IN (R57_XRW=tgl16eu) or OUT (R57_XRW=tgl16e).>

### 1. The pins (from BigMoE's blob; the runner checks them at every start and per process)
- **The env:** <path>, blob sha256 <...>, tag <...>. The snapshot trees, the screen and the .so are <...> (bm_pin).
- **The base: lossy-45 + CC(1).** <BigMoE's base flags>.
  - The CC(1) chain arm (ccx45cc1) PASSED tier 1 in both orders on Monday night (RESULT_lossygate_2026-11-17).
  - CC(1) uses amendment 3's flags unchanged: R57_RELAX_DELTA=1, R57_RELAX_DELTA_SETS=humaneval,humanevalx,mbpp.
- **The drafter** (BigMoE's flags, 10:56 Tue; final unless a test forces a change, and BigMoE says so first):
  - **Code:** R57_DRAFT=t30b, i.e. code drafting by t30b over the shipped MTP layer (unset = DFlash, as today).
    - **The format (Q8H32):** R57_T30B_FMT=<q8h32 | b16h32>.
      - q8h32 is the default: MTP Head's t30b_q8h32.safetensors (b5388cc4...) + Leap's ScreenedHead 32K + session
        extras.
      - b16h32 is the fallback.
      - "stored" is CERTIFY98's count check only, never this stack.
    - **Tree-16:** R57_TREE_NODES=15 (best-first pops; check rows = nodes + 1 = 16) and R57_TREE_K=8 (top-8 children
      pushed per pop, as in the T8 dumps and tree_cc32).
    - **Sync-free:** R57_TREE_SF=<1 | 0> (MoonShots' sync-free frontier; 0 = the host-synced heapq loop).
  - **Chat:** unchanged. R57_DRAFT changes code drafting only.
- **CC(1) on the tree** follows BigMoE's rule, per check:
  - The 125B verifies the current token + the 15 popped nodes in one forward (tree mask), so each node's row is
    conditioned on its own path. The walk starts at the current token's row. At node u, with the 125B's logits z_u:
    1. if a selected child of u carries argmax z_u, it is accepted (exact);
    2. else, with CC on for this set: take the selected child c of u with the largest z_u[c], and accept it if
       z_u[c] >= max z_u - 1 nat (a close call, counted in close_call.accepted);
    3. else the walk stops, and the check emits argmax z_u (the bonus token).
  - Close calls may occur at any depth and any number of times per check, exactly as in the gated chain rule
    (relax_delta.relaxed_k: no per-check cap; amendment 3's d1 = delta 1 nat, not a count). On a chain (k = 1) the rule
    reduces to the gated one.
  - <R57_TREE_CC_MAX: unset (no cap), unless NewIdeas asks for a cap by Wed midday.>
- **The exact-safe engine levers.** Each is bitwise against the path it replaces, by its owner's test, so it changes
  time, not tokens. The stack's gate still covers them.
  - **GDN16** (Leap's 16-row GDN module; code ~/Leap/scripts/week98/m16.py, the GDN16 switch, install()): <its r57
    flag>. Its bitwise records (Leap 15:02, checked here 15:0x):
    - Kernel level: ~/Leap/data/week98/check25/m16_check.txt (Leap 65a85da). The 14 GDN16 cases (the hooked
      gated_delta_kernel, gdn_fast C 2 / U 4, vs mlx_lm's kernel) match in y AND state, bit for bit, at T 2 / 3 / 5 / 8
      / 12 / 15 / 16 with 2 seeds each, engagement counted. Verdict PASS.
    - In the real 16-row check: ~/Leap/data/week98/c25/wed3smoke_run.txt line 21 (Leap ae2363a):
      'F16gd_vs_F16g': (16, 16, 16, 0.0), i.e. all 16 rows' logits bitwise with GDN16 on vs off (216 engaged calls).
      Tonight's slot #2 repeats it over 3 windows (~/Leap/data/week98/c25/n2_run.txt).
  - **tgl16e** (Breakthrough's expert rewrite, the exact part only): R57_XRW=<tgl16e, the value BigMoE wires>; xrw.mode
    reads "tgl16e" (Breakthrough 15:0x).
    - The expert one-call's exact part is bitwise == today. There is no tier-4 upgrade.
    - **The code: knob_v2, Breakthrough commit 20eaaf3** (~/Breakthrough/data/xrw/knob_v2; Breakthrough 14:0x, and
      BigMoE's wiring matches). Checked here Tue 14:0x: the files equal 20eaaf3 (HEAD differs only in README.md,
      70eda7a).

      | file | md5 (knob_v2.md5) | sha256 |
      |---|---|---|
      | xknob.py | 2f7170d952e303c56d7abaff6f6b370e | 6474eab8d8f57921... |
      | tgl16v.py | 272df4d93a7b22caed9b2da9a99815ae | ae1f9e4875a741e6... |
      | xgrp.py | 6f7b3be9bafef860a7b75b138954e61d | 42ddeac34e962659... |
      | tgl16.py | e5747cebc18fcad18a54ab481d987a20 | 31d61092fbc002c2... |
      | w4a8.py | 75c9c3b29d42ca3c4de0ed70ec945ffe | 5cca62e4fbc5025d... |

    - runs.json carries the per-run delta of xknob's counter: "xrw": {"mode", "cand" (calls run by the rewrite),
      "orig" (calls sent to today's kernel), "fallback_not_p11" (calls on a non-p11 store)}.
    - **Bitwise records**, tgl16e == today's one-call (ooo1cq) in y, rs_m and rs_l (all in ~/Breakthrough; doc
      docs/EXPERT_REWRITE.md; Breakthrough 15:0x):
      1. Real blocks: data/xrw/xrw_check_var2.txt (87e2edd). 1,280 of 1,280 pairs on L20 / L44, 2 real 32-row blocks
         each.
      2. The knob_v1 release check: scripts/xrw/knob_check.py (ef23c19). Every module comes from the frozen dir,
         asserted by __file__.
      3. Robustness through knob_v1: scripts/xrw/xrw_robust.py (a02a173). 18 cases: M 1-32, L20 / L44, QOM drops,
         30 % nolsb marks.
      4. knob_v2 (20eaaf3): tgl16e's code is byte-identical to v1 (tgl16.py has the same md5). The tgl16v.py diff
         touches only the upgrade's grouping call and a bench-only function.
      5. **The stack's exact form** (no 1c tiers, so no nolsb marks), through the frozen knob_v2:
         data/xrw/robust_knob_v2_p0.txt (f09e1e6), 18 of 18 bitwise. The 30 % rerun (_p0.3.txt) is also 18 of 18.
         Checked here 15:0x: 18 "True" lines in each, none "False".
      6. In the engine: Leap's 11:18 smoke (knob_v1, on the G1 base) had F16gk == F16g on 16 / 16 rows and
         F32c6k == F32c6i on 32 / 32 rows, bitwise.
      7. Late slots / posts / late release: Leap's m32 scenarios passed 24 of 24 on tgl16 and on tgl16eu, which runs
         tgl16e's kernels (data/xrw/m32_tgl16.txt, m32_tgl16eu.txt). tgl16e alone was not run through m32.
  - **Sync-free** and **the Q8H32 drafter:** above.
- **Engagement (validity):** CARD_ENGAGE = close_call:accepted@run, tree:sib_accepted@run, xrw:cand@run <+ GDN16's
  field, if it has one>.
  - If tgl16eu is back in, the tgl16e checks below apply with xrw.mode == "tgl16eu". Its upgrade changes outputs
    (toward the 4-bit tier), so it is a FAST lever: the stack's tier 1 and card gate judge it, with no separate pass.
  - runs.json's "tree" block is {"nodes", "k", "checks", "sib_accepted"}. sib_accepted counts accepted nodes that are
    not their parent's head-top-1 child, i.e. what a chain could not have kept.
  - Runner 69's parser reads these as is. A term is field[:key+key][@run]; a dict field's listed keys are summed over
    the run's kept code answers. The DRY on BigMoE's env confirms it.
  - **tgl16e:** it is also VOID unless all of these hold. EvalSuite checks them from runs.json and the per-process
    modules manifests before the verdict:
    - every process's modules manifest shows xknob loaded from .../knob_v2/xknob.py, with the sha256 above;
    - every run has xrw.mode == "tgl16e", ON for the whole run;
    - summed over the run, xrw.orig == 0 and xrw.fallback_not_p11 == 0.
    - xrw.cand > 0 (the engagement term) also proves the one-call path is on (REF2_OOO_1C=1 + REF2_NAX_1CQ=1). The 1c
      tiers are a separate flag (R57_TIERS=1c), so dropping them keeps the knob active (Breakthrough 15:0x).
  - The QOM sets are <...>.
- **The format's evidence:**
  - MoonShots' Q8 PASS on Tue was the basis of the orchestrator's 09:4x choice (~/MoonShots/week98/draft/
    DRAFT_PLAN_1117.md, STATUS Tue 11:1x). The registered read-out is Wed ~07:30, against its line: <result, file>.
  - Sync-free: MoonShots' bit test on BigMoE's ops, bitwise on 176 + 120 trees: <file>.
- **The scope:** CARD_SCOPE=code. CC and t30b act on the code sets, while IFEval runs r57's unchanged chat path; tier 1
  still reads IFEval by decision 4 against exact.

### 2. The smoke (first: Thursday by day, after BigMoE's key test)
- **What runs:** runner 69's card tier on tools/stk_smoke_keys.txt (gpqad/rec2ZTsqazBZ3LTWZ, ifbench/124, lcb6/3692),
  both orders, CARD_RUNSUF=_smoke.
- **The check:** a function check, not a verdict. It needs rc 0 and a "(tier 2" DONE line that is not VOID or
  INCOMPLETE.
- **The memory rule:** gate = max(43, the smoke's lifetime peak + 0.5 GB, rounded up to a whole GB), and CARD_MEM =
  gate - 4.
  - The smoke reads 3 answers per process; full processes read 20. That is information, and the +2 GB swap guard is
    the backstop.

### 3. Tier 1 (Thursday by day; the AWAY rule may push part into the evening)
- **The comparison:** the stack against the EXACT references, results/ref2tnax, ref2tnaxn.r3 and ref2tnaxn.r2
  (CARD_REF=ref2tnax,ref2tnaxn.r3,ref2tnaxn.r2; NewIdeas 09:3x). It runs both orders, forward then reverse, 364 keys
  each.
- **The lines:** the registered tier 1, on the two orders pooled (lossy_d4.py's pooled pair "EX+EX"):
  - pooled: delta above -3 pt AND McNemar p >= 0.05 (the screening line);
  - per set (HumanEval+, MBPP, IFEval): decision 4, delta above -5 pt AND p >= 0.05. IFEval uses decision 4 against
    exact, not identity;
  - the runaway guard (amendment 3a);
  - the interim after the forward order's first 182 keys: STOP-FAIL if z <= -2.8.
- **The stops:** a forward FAIL, or an interim STOP-FAIL, ends the chain. There is then no reverse order and no card
  gate.
- **Attribution:** a FAIL cannot be pinned on one lever. That is acceptable for a headline gate, whose question is the
  stack as a whole.
- **Tiers 0 and 0a** are not run. They only kill, the smoke checks function, and tier 1 decides.

### 4. The headline timing run (i) (Thursday night, first in the night's order)
- **What runs:** runner 69, CARD_TIMING=1, with the stack's pinned env on the chosen format.
  - One queued quiet lock (90 min) and no share locks.
  - HumanEval+ 164 + MBPP 100 (the code keys; IFEval dropped), in the panel's forward order, one order. No verdict.
- **The number:** whole decode, n / decode wall (sum n / sum(n / tps)), per set and pooled.
  - The config line goes beside it: format, screen, tree flags, CC(1), sync-free, the exact-safe levers.
  - Tokens per second with prefill are reported beside it, never as the headline.
- **Heat:** a heat pause inside a timed decode (the fanwatch log) taints that answer, and the answer is re-run (twice at
  most).
- **The night's order (NewIdeas):**
  1. this run, ~45 min, quiet;
  2. BigMoE's exact timing (ii): t30b + tree-16, HE+ 164 + MBPP 100, token identity 264 / 264;
  3. CERTIFY98's count check, ~30 min (its own PREREG: t30b as stored, chain-32, exact acceptance).

### 5. The card gate (A) (Friday night)
- **Tier 2:** IFBench 150, GPQA 100 and LCB v6 131, both orders, against NAX6-N (ref2tnaxn.r6).
  - Decision 4 per benchmark and per order.
  - Amendment 1's PENDING / extension rule applies (CARD_EXTEND). The identity sets are <CARD_IDENTSETS, if any>.
- **The scope:** the full (A) scope, with lcb6 in every lever's sets.
- **Memory:** CARD_MEM comes from the smoke (section 2).
- **Typing (information):** IFBench chat 141 / math 6 / code 2 / agent 1; GPQA chat 67 / math 33; LCB code 131.

### 6. The reading (~Sat 07:00)
- **GATED, MEASURED:** tier 1 PASSES (both orders, pooled), the card gate PASSES (every benchmark and order), and the
  timing run gives the number.
- **A step that slips:** if the card gate has not finished, the reading is "measured, card gate pending". If tier 1
  has not finished, it is "measured, not gated".
- **A tier-1 FAIL:** the stack is not adopted, and any number is information only.
  - A bisect is a NEW config. It needs its own dated amendment before its data, and its own tier 1 and card gate. This
    gate's runs are not reused as its reference.

### 7. Carried over
- **BREAK98_HOLD:** nothing starts while it exists. A running step yields at once and resumes when it is gone.
- **The guards:**
  - the AWAY rule (away_secs < 900: no start, and a running step yields);
  - HOLD, PAUSE and the lid;
  - one > 16 GB job at a time;
  - the +2 GB swap guard;
  - the yield to another session's run > 5 GB.
- **Validity as registered:** the pins; engagement > 0 on every kept run; QOM; every key has exactly one uncut run.

## Open (to settle before filing)
- **BigMoE:**
  - the blob (Thu midday) and the lossy-45 base flags in it;
  - the format after MoonShots' Wed ~07:30 read-out;
  - sync-free on or off;
  - tgl16e's R57_XRW value, and GDN16's flag;
  - any flag change forced by a test.
  - The tree flags, the CC-on-tree rule, the engagement field and chat's path are named (10:56 Tue).
- **Leap (Wed 07:15, slot #3):** tgl16eu's price on the no-tier base path (F16nx vs the same arm without it). At >= ~1 ms
  it is back in, by the orchestrator's 15:3x rule.
- **Leap / Breakthrough:** DONE 15:0x, both bitwise records are cited above. Still open: GDN16's runs.json field, if
  r57 writes one (BigMoE).
- **NewIdeas:** a per-check cap on close calls (R57_TREE_CC_MAX) or none, by Wed midday.
- **EvalSuite:**
  - DONE Tue 11:1x (bd19455): runner 69's timing summary reads the tree flags (the shape and the draft flags). Its
    flag list also needs GDN16's flag once named.
  - DRY passes of runner 69 on the stack's env.
  - A chain for Thursday (smoke -> tier 1 forward -> tier 1 reverse) and the night's follower (the timing run, then the
    card gate on Friday night). Both are built from tools/stk_chain.sh, with the exact references in place of G1's
    base runs.
