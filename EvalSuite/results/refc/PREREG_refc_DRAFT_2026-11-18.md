# DRAFT, not filed: PREREG: the reference contract (REFC) as the 45 GB EXACT reference for CHAIN-16: NAX6-N + Leap's contract vs NAX6 (Wed 2026-11-18 <HH:MM>; before any row)

Drafted Wed 11:5x on NAX6's template (docs/PREREG_nax6_2026-11-04.md) and EvalSuite's RE-CERT PRICE, Part A
(HANDOVER). It is filed into docs/PREREG_refc_2026-11-18.md once BigMoE names the contract snapshot and its flags.
Placeholders are in <angle brackets>.

Written before any REFC row.
- **Decision:** Kamen said YES to adopting the reference contract into exact. Verbatim (the orchestrator, relayed by
  NewIdeas, received Wed 11:1x): "Yes you can adopt the reference contract into exact."
  - The condition is decision 4's, as for NAX6: the contract's reference must score the same as NAX6 on math, code and
    instructions. Its loop must also equal its own AR.
  - Until this test certifies it, NAX6-N stays the reference. Exact numbers under the contract are quoted "exact vs its
    own AR; the new reference is not yet re-certified" (the orchestrator, received 11:4x).
- **The scope: CHAIN-16** (the orchestrator and NewIdeas, received 11:4x).
  - M4C FAILED 15 / 16 with the contract on (BigMoE 11:4x): at mbpp/30's token 95, a near-tie flips between tree-16
    and chain-16, while chain-16 == AR on 16 / 16.
  - So trees are out of this certificate. They get their own M4C and exactness piece later, only if BigMoE's diagnostic
    puts the first tree-vs-chain divergence in a GDN layer and Leap's tree-invariant GDN lands.
- **The contract** (Leap; CHECK55_LEVERS #6; ~/Leap/scripts/week98/refc.py, Leap 3ec85e8): it fixes the K-chunks and
  the summation order of every dense GEMM, the HC read and the head. Their outputs then do not depend on the row
  count.
  - **Why it is needed:** today's kernels do arithmetic that depends on the row count. So a 16-row verify cannot be
    bitwise with AR at every near-tie (M4, BigMoE Wed 02:10: EXACT FAIL by batch variance; mbpp/16's token 57 is a
    1-bf16-step near-tie). Under the contract, M4C's chain-16 == AR on 16 / 16, mbpp/16 included.
  - **The switch:** REF2_REFC=1 runs refc.install_from_env(ref2, model). Unset, nothing is patched: today's build,
    bitwise. The scope guards: ref2 on, and only the target's own 4-bit g64 weight arrays. The drafter's Q8H32 / c2 /
    PK arrays are never in the target set.
  - **The exact set under the contract** (Leap, via NewIdeas, received 11:2x; ~/Leap/data/week98/check25/
    REFC_CONTRACT_RECIPE.md): the contract ON; REF2_BOF inert; REF2_HC_B OFF; GDN16 and tgl16e ride; GDN_LAZY off.
  - **Its kernel-level self-test:** refc_rowinv.py, 18 / 18 PASS (Leap 3ec85e8): 7 dense shapes row-invariant at 1-16
    rows plus tree-shaped row sets; HC S4 permutations and tails; the guards fall through bitwise.
  - **Price [P]:** ~-2.6 ms per 16-row exact check vs today's BOF + HC_B build (Leap). E14: A16c - A16 = -5.76 ms.

## Arms
| variant | build | rows |
|---|---|---|
| `ref2tnax` (stored, not re-run) | NAX6 (PREREG_nax6) | results/ref2tnax (the 210) |
| `<ctl>`, the control | the contract snapshot, REF2_REFC unset | step 0 |
| `<refc>`, the contract | the contract snapshot, REF2_REFC=1 + the exact set | the 210; stage 2 if triggered |

- **The snapshot:** s98n (BigMoE, received 12:0x): the s98 screen + Leap ec06e43 (refc scope ref2, all 7 shapes;
  refc_rowinv 18 / 18 PASS on its own files with REFC_ROOT). <Its trees, screen and .so sha256s, BigMoE's commit, and
  the env files' blobs: from BigMoE's manifest, ~19:00>.
  - `<refc>`'s env: ~/BigMoE/data/stack/c98kar_env.txt (x45rpl1 + REF2_REFC=1 + R57_XRW=tgl16e + t30b's mtp125
    install on every prompt + R57_MTP_K=0 + R57_LOGHASH_ROWS=1). The exactness loop's: c98kch_env.txt (the same with
    R57_MTP_K=15). `<ctl>`'s: c98kar with REF2_REFC=1 removed and nothing else changed.
  - **The registered env files are EvalSuite's** (tools/refc_envs.sh): results/refc/c98kctl_env.txt, c98kar_env.txt and
    c98kch_env.txt. Each env line is BigMoE's byte for byte (the control's minus REF2_REFC=1). Each comment names the
    snapshot's screen (data/r57/snap/s98n/r57_screen_s98n.py) and the run's tag, which runner 70 reads: it stops on a
    comment that names no screen, or not its tag. BigMoE's own comments name neither. <Their blobs, and BigMoE's
    sources' blobs.>
  - The runs: card_c98kctl_ar (step 0), card_c98kar_ar (the 210), card_c98kch_ar (exactness).
- **Where the rows come from** (BigMoE, received 11:5x): BigMoE's r57 screen, which wires refc, tgl16e and GDN16 at
  load. ref2.py itself never reads REF2_REFC, so EvalSuite's own loader would not install the contract.
  - AR = the screen's AR mode, R57_MTP_K=0 (a chain with 0 drafts; M4C used it: chain-16 == it on 16 / 16).
  - The rows are written in r57_wrap's tokens.json form and turned into results/<run>/decode.jsonl by
    tools/r57rows.py (35913cec...), as runner 69's rows are.
  - **EvalSuite runs them** (BigMoE, received 11:59: its main account is at 90 %): tools/day_runner70.sh (d3ac75be...)
    = runner 69 + CARD_AR=1. That mode runs CARD_KEYSFILE's keys in one order under the share lock, with no verdict;
    CARD_C sets C, and every process's runs.json goes into the run's ledger. 30 lines differ from runner 69. DRY Wed
    12:0x: the 210 and R-56d's 36 build and pin. Key files: results/refc/keys_core210.txt and keys_r56.txt (R-56d's 36
    all lie inside the 210).
  - **The per-run checks:** tools/refc_check.py (48f1d44f...): `--on` (the 7 shapes, s4 + s4g, ref2_off_moves == 0),
    `--off` (every form 0), and `exact <chain run> <AR run>` (tokens, and every kept row's hash vs the AR's at the same
    answer index). Selftest PASS.
  - Each piece writes runs.json (with r57's per-run 'refc' forms-taken deltas), its env and its modules manifest, and a
    marker on any end. EvalSuite freezes every output's sha256 before grading.
- Everything else as NAX6's Part 1: greedy, non-thinking, each row from a fresh state, the items' max_tokens. AR at C
  192, under a share lock (no timing).
- **Rows:** the core 210 (math 50 = math500 25 + gsm8k 25, IFEval 100, HumanEval+ 60), as decision 4: data/items
  dec_math, dec_ifeval and dec_humaneval.

## Step 0 (VOID gates)
1. **NAX6-N reproduced on the contract snapshot, switch off:** `<ctl>` decodes ifeval/3409, ifeval/2439 and
   humaneval/HumanEval/27, which must be token-identical to results/ref2tnax.
2. **The contract engaged** (Leap and BigMoE, received 11:5x-12:0x). The contract's scope is "while ref2 is on" (Leap
   ec06e43, refc.py md5 1a70f599: all 7 dense shapes engage; the earlier 3-shape build was safe but not the intended
   contract). BigMoE's screen writes runs.json 'refc' per run: the forms taken {c16, c32, s4, s4x2, s4g}, 'shapes' (the
   dense forms taken per shape) and 'ref2_off_moves'.
   - In `<refc>`, on every run: the dense forms taken are > 0 for ALL 7 shapes, and s4 + s4g > 0.
   - **The drafter guarantee** (Leap): refc's counters do not move while ref2 is off (between ref2.remove() and the next
     apply(), i.e. the drafts): 'ref2_off_moves' == 0 on every run. So the drafter's kernels are untouched.
   - In `<ctl>`: every form 0.
   - REF2_REFC=1 alone does nothing: the loader must call refc.install_from_env(ref2, model) after ref2.apply(), with
     the loaded target model. Once installed and ON, the contract covers every eligible call while ref2 is on, AR
     included: AR pads to 16 rows and runs the same contract forms (AR == verify by construction).
3. **Row invariance:** Leap's self-test on the certified snapshot's own files, right before the arms:
   `REFC_ROOT=<the snapshot's scripts dir> [BigMoE venv] python <snapshot>/week98/refc_rowinv.py` (Leap e1620ca or
   later). It imports refc / qmm16 / ck / leapk / one32 from that dir, asserts it, and prints their files. It must
   read 18 / 18 PASS with every module from the snapshot (Leap's run: 913250f, data/week98/check25/refc_rowinv.txt).
- A failed gate is VOID: nothing is judged; the cause is fixed and the test reruns.

## The verdict (decision 4)
- `<refc>` vs `ref2tnax` on the 210: `tools/d4x.py <refc run> ref2tnax [<stage-2 reference run>]` (5ddf9afd...;
  NAX6's design, with its stage 2 built in; last line `D4X: PASS | FAIL | DIFFERS (HIGHER) | STAGE2`).
- Per set (math; IFEval strict, prompt level; HumanEval+ pass@1): the paired delta is above -5 points AND the exact
  two-sided McNemar p is at least 0.05. Judged on unrounded numbers.
- **PASS:** all three sets pass, and the exactness piece passes. The contract's chain-16 then becomes the EXACT
  reference.
- **FAIL:** a set fails (IFEval only after stage 2), or the exactness piece fails. NAX6-N stays the reference.
- **DIFFERS (HIGHER):** a set gains with p < 0.05. **VOID:** a step 0 gate fails.
- **Stage 2,** as NAX6's: it runs only if IFEval alone fails on the core 100 while math and HumanEval+ pass.
  - `<refc>` decodes IFX's 220 stage-1 prompts (results/ifx/stage1_keys.txt). NAX6-N has no IFX rows, so the NAX6-N
    arm decodes them too (~1.6 h each). IFEval's line becomes decision 4 on those 220.
  - The triggering core 100 become information.

## Exactness (NAX6's Part 2, on the loop that runs the exact headline)
- r57's exact chain-16 under the contract (exact (ii)'s loop) vs `<refc>`'s own AR (R57_MTP_K=0), the same flags, on
  R-56d's 36 prompts (results/r56/keys.txt) at full length, with R57_LOGHASH_ROWS=1 (one hash per kept row; BigMoE adds
  it today, ~15 lines). Every token, and every kept row's logits hash == the AR's hash at the same position.
- Chain-32 is not run: r57 has no plain MTP chain-32 (BigMoE 11:5x), and the certificate is for chain-16.
- **Lines:**
  1. The loop == its own AR on 36 / 36 prompts.
  2. Engaged: the contract and the exact set ran on every prompt.
- **Verdict:** PASS = 1 + 2. FAIL = 1 fails. NOT ENGAGED = 2 fails. ~35 min.

## The runtime path at C 256: dropped
- The RE-CERT PRICE had runner 64's tier 0 on the new exact flags (REPLAY margins, then replay_unset_check.py). The
  exact headline runs on r57, whose own checks (the exactness piece above, and (ii)'s token identity 264 / 264) carry
  the runtime. So it is not run (BigMoE, received 11:5x; EvalSuite's call).

## Information (no lines)
- Token-identical rows and first-divergence positions, against NAX6.
- The standard table: answer agreement, length, hesitation, truncations and loops.
- The largest lifetime footprint at a row.
- Per set: the discordant items.
- Tree-16 vs chain-16 under the contract: M4C's record (15 / 16), and BigMoE's diagnostic.

## Predictions (mine; information)
- **Tokens:** the contract moves dense / HC / head sums by ulps at most rows, so most rows diverge somewhere.
  Token-identical rows ~20-40 % (NAX6's 1-8 ulp change kept 73 / 210 = 35 % identical to REF2-T; RESULT_nax6).
- **Scores:** inside the same-weights spread: math and HumanEval+ +-3 points, IFEval +-4. No lean expected.
- P(PASS) ~85 % (stage 2 included), FAIL ~10 %, VOID ~5 %; P(stage 2 runs) ~10-15 %.

## Runs
- **When:** Wednesday night after blob 1's tier 1 #1, if this PREREG is filed by ~22:00 and there is room. Else
  Thursday night after (i), (ii) and (ii-b) (lossy amendment 6, section 8's placement). Stage 2, if triggered, runs
  Friday by day.
- **The chain:** tools/refc_chain.sh (DRY Wed 12:1x): STEP0 (the switch-off control on results/refc/keys_step0.txt:
  tokens == ref2tnax, refc off) -> ROWINV -> AR210 (~2 h, share; refc engaged) -> D4X -> EXACT (~35 min; refc engaged,
  tokens and kept-row hashes == AR210's). A failed gate is VOID and ends it. Every end writes results/REFC_CHAIN_END
  with the verdict.
- **Code pin:** the snapshot, tools/d4x.py (5ddf9afd...), tools/r57rows.py (35913cec...), the key files and MLX /
  mtplx.
- **Report:** results/refc_d4.txt and a RESULT doc.
  - To the orchestrator at once: the verdict is Kamen's decision.
  - Also to NewIdeas, Leap and BigMoE.
