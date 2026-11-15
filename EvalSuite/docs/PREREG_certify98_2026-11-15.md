# PREREG: certifying GOAL A, ">= 98 % per guess at every depth up to 30" (Sun 2026-11-15 01:3x, before any proof run)

- **The design** is ~/FableGod/deepdraft/CERTIFY98.md (copy: docs/CERTIFY98_2026-11-15.md). This file registers its
  rule, numbers and checks.
- **Sizing:** tools/certify98_power.py (CPU, ~20 s), from t30b's stored cells. Its output is quoted in CERTIFY98
  section 3.
- **The proof run is Thu 19 - Fri 20.** It gets a dated amendment before it starts, naming:
  - the candidate's pins (checkpoint sha256, code commit, settings);
  - the scorer's sha256;
  - the fresh-set manifest (ids + capture sha256s);
  - the everyday prompt pool.

## The metric (CERTIFY98 section 1)
- **Per guess:** teacher-forced per-depth accuracy with true prefix tokens, against the target's runtime top-1.
  - The bands are dd_ttt.py's BANDS (0, 1-2, 3-6, 7-10, 11-14, 15-22, 23-30). Each band's accuracy is the ratio of
    right cells to cells, pooled over windows.
  - Each depth 0-30's own accuracy is kept as a guard.
- **kept/32 and kept/16:** dd_ttt.kept_walk (the decode-path replay; caps 31 and 15).
- **Margin bins (required):** each band split by the target's margin, sure (m >= 4 nats) vs unsure (m < 4).
- **The engine's tokens per check:** on >= 20 fresh HumanEval+ windows, within +-3 % of kept_walk on the same windows.
- **Scorer reproduction** (validity 1): t30b from the stored cells, to 0.01 pt.
  - MBPP 100 (cells_t30b_mbpp100.npz): band 0 94.01 and 23-30 72.13.
  - HumanEval+ (cells_t30b_heplus.npz): band 0 97.95 and 23-30 89.65.

## The sets (CERTIFY98 section 2)
- **Dev (reported, never gated):** MBPP 100 (d1102mbpp), the 24 HumanEval+ d16w windows, the BANK val 40, and
  Sandbox98's MBPP-50 / HE-30.
- **Fresh test sets** (runtime path; manifest frozen before the run):
  - HumanEval+ fresh: the 140 HumanEval+ problems outside the 24 d16w windows.
  - MBPP fresh: a seeded draw of 300 from MBPP's test split (task_ids 11-510). The draw excludes every id in
    d1102mbpp's 100, EvalSuite's dec_mbpp 100 and Sandbox98's MBPP-50.
    - Source: ~/TrainLab/data/mbpp/mbpp.jsonl (974 problems, local; no download).
  - Everyday text: >= 150 chat and agent answers from a pool no drafter trained on.
- **Contamination:**
  - source tags;
  - any shared 50-token span with the candidate's training data drops the window;
  - more than 5 % dropped in a set makes it VOID (re-draw);
  - the 13-gram overlap rate is reported.
- **Looks:** one pinned candidate per look. A second candidate on the same sets uses a one-sided 97.5 % bound.

## The rule (CERTIFY98 section 4)
- **The lower bound:** the ratio estimate minus t(K-1, 0.95) x the window-clustered CR1 SE (one-sided 95 %).
- **HE_LINE = 97.0.** HumanEval+ fresh is CERTIFIED when all of these hold:
  - every band's lower bound >= HE_LINE;
  - every depth 0-30's point estimate >= 96.5 %;
  - the run is valid.
  - The label adds "98 met" when every band's point estimate is >= 98.0.
- **MBPP_LINE = none:** MBPP is reported against its own computed exact ceiling (NewIdeas, ~94-95 %). Setting a number
  L gates MBPP with the same rule at L.
- **Everyday text:** reported, never gated.
- **Last line:** `CERTIFY98 HE+: CERTIFIED [98 met] | NOT CERTIFIED (bands ...) | VOID (reason)`.

## Why HE_LINE is 97.0, not 97.5 (measured, 01:2x)
- **HumanEval+ errors cluster by window:** a design effect of 15.7 at band 0 (97.95 %) and 131 at 23-30 (t30b, 24
  windows, CR1).
- **The power at a flat 98 % truth,** every band, with HumanEval+'s own spread (sd 2.96 pt at 98 %):
  - the 97.5 line: 0.56 at the 140 fresh windows, 0.64 at all 164, 0.80 at 300;
  - the 97.0 line: 0.94 at 140.
- **False certification:**
  - a flat 97.5 % truth against the 97.5 line (K 140): 1.0 %;
  - a flat 97.0 % truth against the 97.0 line: 0.5 %.
- **To register 97.5 instead,** an amendment before Wed 18 must add ~150 HumanEval+-like code windows from a second set.
  That is a download, so it needs Kamen's OK.
- **MBPP-like windows:** ~150-200 give 0.8 power at the 97.5 line (~300 in the stress case with the spread doubled).
  Everyday-text-like windows: ~100 (~200 in the stress case).

## If only the METHOD is certified (CERTIFY98 section 6)
- METHOD CERTIFIED (HE+) needs all three:
  - **(a) The 125B oracle** (the true fresh state at every depth): every band's lower bound >= HE_LINE on HumanEval+
    fresh.
  - **(b) The 2B sandbox at full budget:** within 1.0 pt of its own oracle at 23-30. The test is window-paired (the
    one-sided 95 % lower bound of the difference >= -1.0). Band 0 must stay within 0.3 pt of the 2B baseline.
  - **(c) A 125B partial run at matched steps:**
    - it passes its stop line on the dev sets;
    - it beats the baseline at the same steps at 23-30 on HumanEval+ fresh (window-paired, one-sided p < 0.05; one
      look);
    - its closed share of the chain-to-oracle gap is >= 0.8 x the 2B's at the same relative step.
- The projection to the full budget is reported, never certified.

## Validity (otherwise VOID: fix, then a new run)
- The scorer reproduces t30b.
- The manifest was frozen before the run, and the held-out and contamination checks pass.
- Runtime-path features: the target's top-1 equals the next token on every answer row.
- The pins match the amendment's.
- One candidate per look.
- The engine check is within +-3 %. Failing it voids only the kept/32 claim.

## Timeline
- **Mon 16 - Tue 17 (CPU):** tools/certify98.py and the contamination checker; reproduce t30b.
- **Wed 18 night (45 GB, ~2.1 h):** capture ~590 fresh files (the BANK's ~12.6 s per file); freeze the manifest; file
  the proof-run amendment.
- **Thu 19 - Fri 20:** score the one candidate; run the engine check (~30 min of GPU).
- **Sat 21:** the verdict.

## Appendix A: the v1' code confirm (a plan only; it runs under its own amendment)
- **The question:** is the shipped 45 GB v1' stack below exact on code? The answer tells MoonShots, as papers editor,
  whether the public texts need a caveat. EvalSuite changes no public text.
- **What is known, both orders** (CERT's published table; the orchestrator 01:3x, from MoonShots'
  ~/Papers/CERT_CODE_CAVEAT_2026-11-15.md):
  - forward: -2.65 pt on HumanEval+ + MBPP;
  - reverse: +0.76 pt (HumanEval+ 0.00, MBPP +2.00);
  - both orders: -0.95 pt (net -5 of 528 pairs).
  - That was a post-hoc subset, so it is information only. CERT's pre-registered PASS stands.
- **The panel, fresh for this question:**
  - **MBPP-x:** a seeded 300 from MBPP's test split, outside dec_mbpp's 100 (local source; ~202 tokens per answer).
    It may share problems with CERTIFY98's MBPP fresh set: decode quality trains and selects nothing for the drafter.
  - **LiveCodeBench v6:** the card window's 131 problems (dec_lcb6, graded by LCB's tests; ~1,570 tokens per answer).
- **The arms:**
  - v1' (CERT's pins) in BOTH orders, forward and reverse, each in its own processes from the same cold start.
  - Exact NAX6-N: the cached ref2tnaxn.r6 for LCB (131 / 131), plus one exact pass on MBPP-x (exact is
    order-independent).
- **The reading:**
  - Pairs from both orders are pooled: (300 + 131) x 2 = 862 pairs against exact.
  - Lines: decision 4 per set; the pooled code screening line (delta > -3 pt AND McNemar p >= 0.05); and the pooled
    delta's 95 % interval (clustered by problem, since both orders pair the same problem).
  - "Caveat needed" if the interval lies wholly below 0. Otherwise "no measurable code loss", quoting the interval.
- **The run time** at 45 GB:
  - v1' is 2 x (61K + 206K) = ~534K tokens, plus exact MBPP-x at ~61K: ~595K tokens.
  - That is ~4.9 h with the fans up (2,045 tokens/min) and ~9 h in heavy heat (1,084).
  - **A smaller cut:** MBPP-x 300 alone is ~182K tokens, ~1.5 h with the fans up.
    - Power against exact at ~3.4 % discordance: ~95 % for a true -2.65 pt, ~20 % for -0.95 pt.
    - LCB's higher discordance (card sets 5-15 %) is what buys power for the small effect.
- **When:**
  - The night into Tue 17 (00:30-05:30, fans up), under a share lock, MEM 45.
  - Not tonight (share-only, and a 4.9 h job), not Wed 18 (CERTIFY98's capture), not Thu-Fri (the proof run).
  - If the night is short, the smaller MBPP-x cut first; LCB's two orders the next night.

## Amendment 1 (Sun 2026-11-15 01:5x, before any proof run): the scorer and the contamination checker are built
- **tools/certify98.py** (`score <cells.npz> [--label L] [--line X | --report]`, `reproduce`, `selftest`). It does:
  - the bands and their CR1 lower bounds;
  - the per-depth guard;
  - the margin bins;
  - kept/32 and kept/16 with window-clustered 95 % intervals;
  - the verdict line.
- **Validity 1 PASSES** (`reproduce`): t30b's stored cells give the published read-outs exactly.
  - MBPP 100 against tapsq/t1_chain.json: all seven bands at 0.0000 pt; kept/32 6.923860 and kept/16 6.588458 to 1e-6.
  - HumanEval+ (bank features) against tapsq/mh_t30b_heplus_bankctx.json: the same; kept/32 13.768898, kept/16
    10.625000.
- **Two layout facts were found while reproducing** (both now match dd_taps.py's eval):
  - The decode-path walk starts at row 1 of the cells. The first anchor is the prompt's last row, and the prefill
    makes the first answer token.
  - Each window's matrix has one more row than its depth-0 cells: the answer's last row has no stored target.
  - The bands do not depend on either; kept/32 and kept/16 do.
- **The dev-set score** (`score cells_t30b_heplus.npz --line 97.0`): t30b on the 24 HumanEval+ dev windows is NOT
  CERTIFIED, as expected.
  - Band 0 is 97.95 with a lower bound of 96.74 (SE 0.703); 23-30 is 89.65 with 86.87.
  - kept/32 is 13.77 (95 % CI 10.38-17.16), kept/16 10.63, a ratio of 1.296.
- **tools/certify98_contam.py:**
  - source tags, with forbidden patterns;
  - 50-token shared runs, hash-matched and then verified token by token;
  - 13-gram overlap rates.
  - **Registered here:** a run found in >= 5 % of the training files is template boilerplate, not content. It never
    flags, and its count is reported.
  - The toy test catches a planted 60-token copy, ignores a 40-token one and a template shared by every file, and VOIDs
    the set at 1 of 3 flagged.
- **The dev sets against all 3,072 n1112 captures** (results/certify98/contam_dev_2026-11-15.txt; a superset of t30b's
  training files):
  - every training src is dlog-magicoder, with none forbidden;
  - HumanEval+ dev: CLEAN, 0 of 24; 13-gram overlap mean 2.16 %, max 8.48 %;
  - MBPP dev: CLEAN, 0 of 100; 13-gram overlap mean 1.52 %, max 10.79 %;
  - peak 0.33 GB; 31 s and 111 s.

## Amendment 2 (Sun 2026-11-15 01:5x, before any candidate): the fresh sets are drawn and fixed
- **tools/certify98_manifest.py** writes results/certify98/fresh_manifest_2026-11-15.json (sha256 0bc0fe527d22f89e...).
  `--check` rebuilds it and compares.
- **Two dev overlaps found while drawing,** which the rule now covers:
  - **Sandbox98's 60 HumanEval windows** (the 2B's dev set; its screens default to 30 of them) all lie OUTSIDE the 24
    d16w windows. The 140 fresh problems therefore include problems the method was screened on, through the 2B's own
    answers.
    - The gated set stays the 140. Those 60 are a different model's answers, and the 125B's drafter was never scored
      on them.
    - **A sensitivity line is registered:** the same verdict on the strict 80 that also drop the 2B's 60, reported
      beside the gated one. If the two disagree, both are quoted and the certificate says so.
    - Power at a true flat 98 %: the strict 80 at 0.79 for the 97.0 line (0.36 for 97.5); the 140 at 0.94 (0.56).
  - **TrainLab captured 375 MBPP problems** (data/dlog/d1102mbpp), not 100: ids 2-973, 190 of them in the test split.
    The fresh MBPP draw now excludes all 375. Those 375 include EvalSuite's dec_mbpp 100, which are d1102mbpp's 100,
    and Sandbox98's 100.
    - The pool is the test split (task_ids 11-510) minus those 375: 310 problems.
    - The draw is 300 (seed 20261115).
- **The sets:**
  - HumanEval+ fresh: 140, strict 80.
  - MBPP fresh: 300.
  - Everyday text: still to be named.
- **The capture (Wed 18 night, BigMoE's bank path)** takes exactly these ids, in the d1102mbpp / d16w prompt formats.
  BigMoE adds each file's sha256 to a copy of the manifest, and the proof-run amendment freezes that copy.
- **The cells format every candidate must write:** dd_ceiling.py's npz.
  - Required: depth, right and m; fi, an and row recommended.
  - Layout: depth-major per window, with the first anchor on the prompt's last row.
  - certify98.py checks that fi agrees with the depth resets. When `an` is stored, it checks that each depth's cells sit
    on depth 0's first anchors, which the decode walk relies on. Both checks pass on t30b's HumanEval+ cells.
  - Cells without `m` lose only the margin bins.

## Amendment 3 (Sun 2026-11-15 02:1x, before any candidate): the everyday pool, named and drawn (the orchestrator 02:0x: local sources only)
- **The pool:** all from EvalSuite's own item files.
  - Agent: all 29 windows (data/items/dec_agent.jsonl). These are Claude Code-style turns: prompts of 1.6K-12K tokens,
    answers of 61-128.
  - Chat: a seeded 60 of IFEval's 100 (seed [20261115, 0]) and 61 of the 150 IFBench prompts that have exact rows
    (seed [20261115, 1]). These are EvalSuite's instruction-following chat prompts.
  - Together: 150 windows.
- **Why these:**
  - No drafter trained on them: t30b's and Sandbox98's training data is magicoder.
  - No drafter screen used them.
  - They are local, with no download.
  - Exact 45 GB answers already exist for all of them (ref2tnax; ref2tnaxn.r6; ref2tnaxn + .r4).
- **Not in the pool:** MT-Bench-style multi-turn chat. There is no such set on disk. StrataEval's head-to-head prompts
  (h2h/prompts) are subsets of EvalSuite's own HumanEval, MBPP, IFEval and math items, so they add no new pool.
  dec_multi and dec_mgsm are kept back for a later amendment if a wider everyday curve is wanted.
- **The manifest** (tools/certify98_manifest.py; results/certify98/fresh_manifest_2026-11-15.json, sha256 acdf3e62d4fd8f74...)
  adds everyday_fresh. The HumanEval+, strict and MBPP lists are unchanged from amendment 2 (checked against its
  commit).
- **Contamination** (results/certify98/contam_everyday_2026-11-15.txt; prompt_ids + the exact answer, against all 3,072
  n1112 captures): CLEAN, 0 of 150 share a 50-token run. 13-gram overlap: mean 0.00 %, max 0.10 %.
- **The capture decodes every fresh answer on the runtime path itself**, then teacher-forces it.
  - Validity needs the target's top-1 to equal the next token on every row. Answers made in another exact mode
    (NAX6-N) are only a cross-check: their tokens are compared and the match share reported.
  - The estimate grows to ~3-4 h at 45 GB:
    - decoding ~121K tokens: HumanEval+ ~35K, MBPP ~61K, everyday ~25K; ~1 h with the fans up, ~2 h in heat;
    - the BANK pass on ~590 files: ~2.1 h, plus the long agent prompts.

## Amendment 4 (Sun 2026-11-15 02:3x, before any candidate): the gate moves to the STRICT accuracy on the engine's path
- **The orchestrator's fix (02:2x):** dd_ttt's bands count every row at every depth, including rows past an earlier miss
  that the engine never drafts.
  - MTP Head's ALLROWS read +8.58 pt at MBPP-50 23-30 while kept/32 stayed flat (4.076 -> 4.066).
  - In the engine, "98 % per guess" means P(right at k | right at 0..k-1).
- **A second population effect, found while building it** (certify98.py; t30b's stored cells): the strict accuracy
  depends on WHERE a check starts.
  - At every anchor it reads high: HumanEval+ 97.94 / 96.87 / 96.68 / 97.04 / 97.45 / 97.48 / 97.85 by band, which
    implies kept/32 21.25.
  - At the decode walk's own check starts, which come right after a miss, it is far lower. HumanEval+ depth 0 is
    91.58 % (MBPP 89.46 %, against 93.97 at every anchor). This version reproduces kept_walk exactly: 13.768898 /
    6.923860, identity off by 1.8e-15.
  - Misses cluster, so "strict at every anchor" overstates the engine just as the bands do.
- **The rule from now on (certify98.py sha256 6392887880703160...):**
  - **PRIMARY:** the strict accuracy at the decode walk's check starts, in the gate bands 0-2, 3-6, 7-10, 11-14, 15-22
    and 23-30 (0-2 merged for power).
    - Every band's window-clustered (CR1) one-sided 95 % lower bound must be >= HE_LINE (97.0).
    - The per-depth guard: a point estimate >= 96.5 at every depth with >= 200 rows.
  - **CO-PRIMARY:** kept/32 (kept_walk), with its one-sided 95 % lower bound >= the flat line's kept: 1 + sum_{k=1..31}
    0.97^k = 20.76 at HE_LINE 97.0. Plus the engine match within +-3 % (validity).
  - "98 met": every walk-strict band's point estimate >= 98.0.
  - **INFORMATION only:** the strict accuracy at every anchor, dd_ttt's bands and the margin bins.
- **Power (tools/certify98_power_strict.py sha256 fac6401389a89d76...):**
  - The model: HumanEval+'s measured per-window spread, and a geometric walk per window. It is calibrated so that the
    POOLED walk-strict depth-0 accuracy equals the truth, since check starts gather in the windows that miss.
  - At a true 98 % (depth 0 at the check starts; deeper depths sit higher, as in t30b's own walk), with 140 HumanEval+
    windows:
    - every strict band at the 97.0 line: ~0.7 (0.69-0.74 across two 300-rep runs);
    - kept/32: 1.00;
    - both: 0.68-0.73.
    - With 300 windows: 0.93-0.94.
    - The committed output is results/certify98/power_strict_2026-11-15.txt.
  - Without merging 0-2, band 0 alone would have 0.47 power at 140 windows (a 200-rep run).
  - At a 97 % depth-0 truth: both pass 0.16-0.20 (the merged band's own truth sits above 97 there).
  - The 97.5 line would have 0.31-0.46 power for the bands. HE_LINE stays 97.0. A 300-window panel (a second code set)
    is still the way to 0.93.
- **t30b on the HumanEval+ dev windows** (information): walk-strict 91.39 / 91.93 / 93.78 / 96.50 / 96.22 / 97.46 by
  gate band, which rises with depth; kept/32 13.77, lower bound 10.96. NOT CERTIFIED.
- **Where kept/32 is lost** (the walk-strict curve with some depths set to 98 %):
  - HumanEval+: 13.77 as is. Depths 0-2 at 98 % give 16.53, 0-6 give 19.83, 0-14 give 22.63; only 15-30 at 98 % gives
    14.38.
  - MBPP: 6.92 as is. 0-2 give 9.02, 0-6 11.96, 0-14 17.74; only 15-30 gives 7.63.
  - The deep guesses the engine actually reaches are already ~97 % (they are survivors). The gap sits in the first ~15
    guesses after each check.
  - This is optimistic for the deep part: better early guesses carry harder stretches deeper.
