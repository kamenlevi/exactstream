# DRAFT, not filed: PREREG_lossygate amendment 6, the GATED, MEASURED lossy stack for Saturday (t30b + tree-16 + CC(1) + G1's pick)

Prepared Tue 2026-11-17 09:4x from the orchestrator's decisions of 09:3x and 09:4x (relayed by NewIdeas).
- It is filed into docs/PREREG_lossygate_2026-11-15.md when BigMoE sends the stack's blob (Thu ~midday). That is
  BEFORE the smoke and before any stack data.
- Placeholders are in <angle brackets>. The open points are listed at the end.

## Amendment 6 (Thu 2026-11-19 <HH:Mx>, before any stack data): the lossy stack's gate (tier 1 in both orders vs exact, the card gate (A), the headline timing run)

### Why
- **The orchestrator, 09:3x:** a GATED, MEASURED lossy number by Saturday on a NEW config: t30b + tree-16 + CC(1) +
  G1's pick, on BigMoE's r57 path (plumbed from Wednesday).
  - It supersedes the chain-16 stk / stk0 plan. tools/stk_chain.sh (ede7aa9) was never armed and has no data.
- **The orchestrator, 09:4x:**
  - t30b runs at Q8 + the 32K screen if MoonShots' Wed ~07:30 Q8 read-out holds kept within its pre-registered line.
    Otherwise it runs at bf16 + the 32K screen. It never runs as stored.
  - Sync-free is BigMoE's default if MoonShots' bit test on BigMoE's ops passes by Wed evening. fused2 is out.
  - In lossy mode, tier 1 and the card gate run ON the chosen format, so the gate covers it. This amendment therefore
    pins the format and the screen, alongside the tree flags.

### 1. The pins (from BigMoE's blob; the runner checks them at every start and per process)
- **The env:** <path>, blob sha256 <...>, tag <...>. The snapshot trees, the screen and the .so are <...> (bm_pin).
- **The draft policy:**
  - **Code:** t30b (~/datasets/deepdraft/ckpt/dd_t30b.safetensors, 790c42db..., step 4206)
    - at <Q8 | bf16> + the 32K screen (<the screen file, sha256>);
    - tree-16 (<the tree flags BigMoE names>: top-<k> children, 16 nodes = 15 best-first pops + the current token);
    - sync-free <on | off>.
  - **Chat:** <BigMoE's: the MTP head as today, or ...>.
  - **Acceptance:** CC(1), the close call at delta 1 nat (amendment 3), on the tree's acceptance (<BigMoE's flag>).
  - **G1's pick:** <the env NewIdeas picks at V2's verdict (tools/g1_pick.sh), or none ("comments-only")>.
- **Engagement (validity):** CARD_ENGAGE = <one term per lever: close_call:accepted@run, the tree's field, G1's fields>.
  Each must be > 0 on every kept run. The QOM sets are <...>.
- **The format's evidence:** MoonShots' Q8 read-out (Wed ~07:30) against its line: <result, file>. The sync-free bit
  test: <result, file>.
- **The scope:** CARD_SCOPE = <code | all, per BigMoE's env>.

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
  - The config line goes beside it: format, screen, tree flags, CC(1), G1's pick, sync-free.
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
  - the blob and the tree flags;
  - CC(1) on the tree's acceptance: its flag and engagement field;
  - chat's drafter;
  - the format + screen after MoonShots' read-out;
  - sync-free.
- **NewIdeas:** G1's pick at V2's verdict.
- **EvalSuite:**
  - Runner 69's timing summary hard-codes "a chain (the r57 screen has no tree path)". It must read the tree flags,
    then be re-tested.
  - DRY passes of runner 69 on the stack's env.
  - A chain for Thursday (smoke -> tier 1 forward -> tier 1 reverse) and the night's follower (the timing run, then the
    card gate on Friday night). Both are built from tools/stk_chain.sh, with the exact references in place of G1's
    base runs.
