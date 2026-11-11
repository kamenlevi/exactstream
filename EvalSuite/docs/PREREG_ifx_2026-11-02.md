# REF2c-T IFX: REF2c-T vs REF2-T on IFEval's other 441 prompts, in two stages (pre-registered Mon 2026-11-02 07:4x)

Written before any IFX row.
- **Request:** NewIdeas 07:3x, verbatim: "yes to the larger IFEval test of REF2c-T vs REF2-T, tonight, in two
  pre-registered stages so it can stop early: Stage 1: ~220 of the 441 prompts, both arms, in your 60-min chunks. Stop if
  decision 4's paired interval is already clear either way, by a stopping rule you fix in the PREREG before the run.
  Otherwise run the rest. Queue it behind BigMoE's Monday-night items (warm per-row split, then one P arm, ~1 h)."
- **Why:** REF2c-T Part 1 failed decision 4 on IFEval exactly at the line: 85.0 vs 90.0 %, −5.0, McNemar p 0.0625
  (0/5); math and HumanEval+ passed (`docs/RESULT_ref2ct_2026-11-02.md`).
  - Same-weights arms spread 85-90 on those 100 items. REF2c-T has 4 lone misses there, REF2b 3.
  - NewIdeas: the stake is ~6 ms per check, ~7 % of today's 84 ms check at C 256.

## What runs
- **Items:** `data/items/dec_ifevalx.jsonl` (sha256 f0006582e92a). IFEval's 541 prompts minus the suite's 100 core
  prompts (data.picks(), seed 77) = 441, keys `ifevalx/<key>`.
  - Built by `tools/ifx.py build` with dec_ifeval's chat template and max_tokens 1024. The build checks that the
    template still gives all 100 core prompts' saved ids.
  - Built with BigMoE's venv, the GPU stages' Python (EvalSuite's venv has no jinja2).
- **Stages:**
  - stage 1 = 220 keys, `random.Random(1102).sample` (`results/ifx/stage1_keys.txt`, sha256 61feac00e176);
  - stage 2 = the other 221 (`results/ifx/stage2_keys.txt`, sha256 3df32181d644).
- **Arms:** each as its core rows were made: C 192, greedy, non-thinking, up to 1024 tokens, each row from a fresh state.
  - `ref2ct.r2` = variant ref2ct as Part 1 ran it: `snap/leap-91cd316` with ref2_cfg att_single + gdn_inproj, on
    BigMoE's tiers 4433333333 / 3c / C_l 112 (`snap/bigmoe-6c64148`).
  - `ref2t.r2` = variant ref2t as R-58 ran it: `snap/leap-faff8e5`, the same tiers.
  - Scratch runs (`results/<arm>.r2/`) keep `results/ref2ct` and `results/ref2t` (the 210-row references) as they are.
- **Reproduction (VOID check):** each arm's first process decodes ifeval/3409 and ifeval/2439 first
  (`results/ifx/repro_keys.txt`, sha256 f106f4c0b913). Both must be token-identical to that arm's saved rows (Part 1,
  R-58). Otherwise VOID: the runner stops and nothing is judged.

## The verdict: decision 4's rule on the new rows
- **The verdict set is the new prompts only: a fresh sample.** The 100 core rows triggered this test (Part 1's 0/5), so
  they are information.
  - Pooling them in would FAIL a REF2c-T with no real difference ~10 % of the time (simulation, 4,000 runs), twice the
    rule's nominal rate.
- **PASS:** the paired delta REF2c-T − REF2-T (IFEval strict, prompt level) is above −5 points AND the exact two-sided
  McNemar p is at least 0.05.
  - Judged on unrounded numbers. tools/d4.py reads the report's rounded values; that only matters within 0.005 of a line.
- **FAIL** otherwise. A gain with p < 0.05 is DIFFERS (HIGHER) (NewIdeas' call).
- At n 441 the p part flags real drops of ~3-4 points; Part 1's n 100 flagged ~5.
- **What a result means:** math and HumanEval+ passed in Part 1.
  - A PASS here lets NewIdeas adopt REF2c-T as the 45 GB exact mode; Part 1's FAIL stays on the record beside it.
  - A FAIL keeps REF2-T.
  - Either way R-56e (REF2c-T's live loop) would come after an adoption, as its own test.

## The stopping rule after stage 1 (`tools/ifx.py stage1`)
- On the 220 stage-1 rows, b = items only REF2c-T gets right, c = items only REF2-T gets right.
- **P** = the predictive probability that decision 4's rule on all 441 does not say FAIL.
  - Prior: Jeffreys Dirichlet(½, ½, ½) over (only REF2c-T right, only REF2-T right, both the same).
  - Posterior: from the 220 stage-1 rows.
  - The other 221 rows' counts follow the Dirichlet-multinomial. P is an exact sum.
- **STOP** when either holds:
  - P ≥ 0.95 and decision 4's rule on the 220 does not say FAIL;
  - P ≤ 0.05 and it says FAIL.
  - The verdict is then decision 4's rule on the 220 rows.
- Otherwise stage 2 runs, and the verdict is decision 4's rule on all 441.
- In plain words: stop when finishing has less than a 5 % chance to turn FAIL into not FAIL, or back.
- **The calls** (`tools/ifx.py table`), by how many rows differ (b + c) and REF2c-T's deficit c − b:

  | b + c | stop, not FAIL | run stage 2 | stop, FAIL |
  |---|---|---|---|
  | 8 | c − b <= 0 | 2 to 6 | >= 8 |
  | 12 | <= 0 | 2 to 8 | >= 10 |
  | 16 | <= 0 | 2 to 10 | >= 12 |
  | 20 | <= 0 | 2 to 10 | >= 12 |
  | 24 | <= 0 | 2 to 12 | >= 14 |
  | 30 | <= 0 | 2 to 14 | not tabled (16+) |

  With REF2c-T ahead by 8+ on few differing rows (8/0, 10/2), the stop is DIFFERS (HIGHER).
- **Its behaviour** (`tools/ifx.py oc`; 4,000 runs each; new rows 7 % discordant, as same-weights arms):

  | true REF2c-T − REF2-T gap | stops after stage 1 | verdict not FAIL |
  |---|---|---|
  | 0 | 65 % | 98 % |
  | −2 points | 28 % | 72 % |
  | −4 points | 39 % | 12 % |

## Information (no pass line)
- The same line pooled with the 100 core rows (541), and the core rows alone.
- On the new rows:
  - token-identical rows;
  - truncations, loops and generated tokens per arm;
  - instruction misses by group;
  - rough tok/s per arm (not an A/B) and the largest lifetime footprint at a row.

## Runner 38 (`tools/day_runner38.sh`)
- **Start:** queues from Mon 23:00 once `results/ifx/GO` exists, or from Tue 02:00 without it. GO means BigMoE's
  Monday-night items are done; BigMoE, NewIdeas or EvalSuite touch it.
- **Chunks:** one quiet ticket per 60-min chunk (`--chunk-minutes 60 --minutes 80`), arms alternating, re-queued at once.
  - `--need 34`: Part 1's peak was 32.4 GB; the cli gates on need + 4 GB NO-SWAP after the grant.
  - PAUSE, lid and soft_end gates; swap guard (+2 GB: wait 2 min, 6 trips stop it); three chunks without a new row
    stop it.
- **Code pin:** both Leap snapshots (+ ms_gdn), `snap/bigmoe-6c64148`, MoonShots' planes_engine, evalsuite/,
  tools/ifx.py, the item and key files, MLX / mtplx. A change stops it (exit 7).
- **Deadline** Tue 2026-11-03 08:30; no chunk starts in its last 90 min.
  - What is left runs another night with the same keys, pins and rule: a relaunch, noted as an amendment.
- **Time:** ~23 s per IFEval row (Part 1). Stage 1 is 2 × 222 rows, ~3 h with loads; stage 2 is ~3 h more.
- **Output:** `results/ifx_stage1.txt` (the stage-1 call), `results/ifx.txt` (the verdict), `results/IFX_DONE`;
  log `results/day38_runner.log`.
- **Reporting:** the stage-1 call to NewIdeas at once; the verdict to NewIdeas (adoption), Leap and BigMoE.

## Amendment 1 (Mon 10:39, before any IFX row)
- Runner 38 was stopped in its wait loop (pid 68893, 10:37) and relaunched (pid 21519, code pin 125db5ca272b, 10:39).
- Why: `evalsuite/model.py` gained the HC8 variant `ref2th8` for Kamen's day-9 order (Q3), which changed the pin.
- The arms are unchanged. The new code runs only for a variant that sets `hc8`. `ref2ct`, `ref2t` and their `.r2` scratch
  names resolve to the same settings as before (checked at 10:39), and each arm's reproduction rows still guard it
  (VOID unless token-identical to its saved rows).

## Amendment 2 (Mon 20:47, before any IFX row): schedule only
- Runner 38 was stopped in its wait loop (pid 21519, 20:47) and relaunched (pid 69420, 20:47:14). Its code pin is
  unchanged (125db5ca272b).
- Why: Kamen's laptop schedule (Mon 19:3x, RULES.md): big tests on weekdays 00:30-15:00 only. NewIdeas (20:4x): "not
  from 23:00. BigMoE holds 01:00-05:30 ... IFX fits after it, both stages by 15:00 if needed."
- New times:
  - It queues from Tue 00:30 once results/ifx/GO exists, or from Tue 05:30 without it (it was Mon 23:00 / Tue 02:00).
  - The deadline is Tue 15:00, and no chunk starts in the last 90 min (it was 08:30).
- Unchanged: the arms, the keys, the items, the stages, the stopping rule, decision 4's rule, the reproduction rows and
  C 192.
