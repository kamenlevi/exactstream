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

## Amendment 5 (Sun 2026-11-15 14:5x, before any capture or candidate): one pass of the candidate, BigMoE's TOPK + TREE dump
- **Why:** the orchestrator (14:3x) asked BigMoE for a head-only TOPK + TREE dump on the fresh files. MTP Head can then
  price tree-16 kept/32 under the walk, so the certificate also covers the trees that ship.
- **The scoring input is BigMoE's TOPK dump** (scripts/dd_topk.py, the layout of 9f49e03). Checked today on both dev sets
  with tools/certify98_topk_check.py (results/certify98/topk_vs_cells_2026-11-15.txt):
  - fi / depth / right equal the cells files array for array;
  - certify98.py scores both the same: kept/32 13.768898 on HumanEval+ 24 and 6.923860 on MBPP 100; walk-strict band
    0-2 91.3928 / 87.2266;
  - TOPK's walk_start flags equal certify98.py's decode walk on 124 / 124 windows;
  - the checker FAILs on a wrong set and on one flipped cell.
  - So one pass of the candidate gives both the certificate's cells and the tree pricing's inputs.
- **That pass IS the single look** (section 2: the sets open once, for one pinned candidate). TOPK holds `right`, the metric
  itself.
  - It runs only after the proof-run amendment is filed. That amendment names the candidate's checkpoint sha256, code
    commit and settings, the scorer's sha256 and the frozen file hashes.
  - Before then, no drafter runs on the fresh files, not even a smoke test. Smoke tests use the dev bank.
  - TOPK and TREE come from the same sitting; their sha256s go in the RESULT.
- **Tree-16 kept/32 under the walk is INFORMATION, with no line.**
  - It is MTP Head's pricing from BigMoE's TREE dump: 32 best-first nodes, top-8 children, at every anchor. It is
    reported per set with the same window-clustered standard error as kept/32.
  - TREE's own checks must pass (roots == TOPK band 0; true-path nodes == TOPK's teacher-forced cells). Otherwise the
    tree read-out is VOID, and the certificate stands without it.
  - A tree LINE would need its own line and power, filed before the proof-run amendment.
- **Layout (BigMoE's capture, outside git):** ~/datasets/deepdraft/certify98/<set>/<name>.npz.
  - set = heplus (140), mbpp (300) or everyday (150). The strict 80 are a subset of heplus.
  - name = HumanEval_<id>, mbpp_<task_id>, or the everyday key with "/" replaced by "_" (agent_bash17, ifeval_1107,
    ifbench_101).
  - Beside them, the capture writes files_sha256.json: the manifest's sha256 (acdf3e62d4fd8f74...) and one sha256 per
    file.
  - EvalSuite copies it to results/certify98/ and commits it, and the proof-run amendment freezes its sha256. BigMoE's
    dumps refuse any file whose hash differs.

## Amendment 6 (Sun 2026-11-15 15:2x, before any capture or candidate): the TREE LINE
- **Why:** the orchestrator (14:5x) said yes to a tree LINE, because trees are the exact method Kamen chose. NewIdeas
  drafted it (~/NewIdeas/research/week98/TREE_LINE.md v1, 15:0x); this amendment adapts it to this PREREG's form.
- **The candidate** is named in the proof-run amendment. NewIdeas proposes t30b + EAGLE-2-16: best-first by path
  probability, top-8 children, 15 nodes plus the bonus, one exact 16-row tree check. A better drafter replaces it if one
  lands by Thursday. The same single pass (amendment 5) serves the chain gate and this line.
- **The statistic, per fresh set:** T = tree-16 kept per check / chain-16 kept per check.
  - Each runs under its OWN decode walk (certify98's walk, R0 = 1) over the same windows, and is pooled over windows.
  - An N-row check drafts the first N - 1 pops of the tree, or N - 1 depths of the chain.
  - accepted = the deepest on-true node + 1 (tree), or the run of right cells from depth 0 (chain). kept = accepted + 1.
- **The lower bound LB(T):** one-sided 95 %, by the delta method on log T with window-clustered (CR1) influences and
  t(K - 1). This is the same family as certify98.py's bounds, and it is deterministic.
  - NewIdeas' window bootstrap gives the same bound on the dev sets (1.119 / 1.032 vs 1.119 / 1.031). It is reported
    beside LB(T) as information.
- **The line, per gated set (HumanEval+ 140 and MBPP 300; everyday is reported, never gated):**
  - **PASS:** LB(T) >= 1.021 (Leap's tree-verify overhead c16) AND T >= 1.052 (S9c's 3 % net margin over it);
  - **BETWEEN:** T >= 1.052 but LB(T) < 1.021;
  - **FAIL:** T < 1.052.
  - **Overall:** PASS if both sets PASS; FAIL if either set FAILs; otherwise BETWEEN.
  - The verdict line: `CERTIFY98 TREE-16: PASS | BETWEEN | FAIL (HE+ T [LB], MBPP T [LB])`.
- **Power,** using the dev standard errors scaled to the fresh windows (log SE 0.0162 -> 0.0067 on HumanEval+, 0.0093 ->
  0.0054 on MBPP); NewIdeas' table, checked here:
  - MBPP: 1.00 at the dev T 1.136.
  - HumanEval+: 1.00 on the bound and 0.87 on the point at the dev T 1.060.
  - HumanEval+ at the dev's low end, T 1.027: 0.22 on the bound and 0.00 on the point.
  - HumanEval+'s 3 % net margin is the binding part.
- **Coverage, information only (no line), per band 0, 1-2, 3-6, 7-10, 11-14** (a 16-row check reaches depth 14):
  - **Definition:** walk-strict COVERAGE = P(the 125B's token is among the verified tree's nodes at depth d | depths
    0..d-1 covered), at the tree-16 walk's check starts.
  - It uses certify98's walk convention: a missing cell, or the walk's last start at the answer's end, ends the run. So
    a chain's coverage equals certify98's walk-strict accuracy exactly, per depth. The scorer checks this on every run;
    it holds on both dev sets.
  - Dev numbers, chain-16 -> tree-16:
    - HumanEval+ 24: 93.33 -> 98.23 / 93.46 -> 96.16 / 93.99 -> 93.20 / 95.36 -> 94.60 / 96.94 -> 96.08.
    - MBPP 100: 89.72 -> 97.61 / 86.27 -> 91.92 / 85.64 -> 85.73 / 83.51 -> 80.07 / 85.13 -> 78.52.
  - NewIdeas' table drops the walk's last start and missing cells, so it reads 0.2-0.5 pt higher (98.58 / 97.79 at
    depth 0).
  - Read: trees reach ~98 % where the wall is, the first guess after each check. Deeper, the 16-node budget leaves few
    siblings, and coverage equals or trails the chain's top-1.
- **The scorer:** tools/certify98_tree.py; its sha256 goes in the proof-run amendment.
  - It reproduces MTP Head's S9c on both dev sets: TREE-REPRODUCE: PASS, max |diff| 0.0005
    (results/certify98/tree_repro_2026-11-15.txt).
  - Per window, it checks the tree's anchors, a_walk_start == the chain-32 walk, chain-32 == certify98's kept/32, and
    the coverage nesting.
  - TREE's own checks (amendment 5) gate the read-out: if any fails, the line is VOID and the chain certificate stands.
- **The statement, if PASS** (NewIdeas' wording): "Exact. The verified 16-node tree covers the 125B's token at x % per
  position (per band), and keeps T x more tokens per check than the chain, at a check cost at most 2.1 % higher."

## Amendment 6a (Sun 2026-11-15 15:2x, before any capture data): the tree line's bars follow the MEASURED tree-check cost
- **Why (the orchestrator 15:1x):** Leap measured the 16-node tree check's real cost (TREE16-TIME v2, a968cfe; quiet
  ticket, 24 rotated rounds, paired against chain-16 within each run).
  - Its batched-heads form costs +0.129 ms [-0.008, +0.249] on a 4-layer slice. Engine-scaled with the per-path tail,
    that is +1.55 ms, or +2.5 % of a ~62 ms chain-16 check (v1 measured +4.0 %).
  - So c16 = 1.025, not the 1.021 that amendment 6 used. Leap's interval scales to about -0.2 % .. +4.8 %.
- **The change:** amendment 6 defines both bars from c16: a net win at 95 % confidence, plus a 3 % net margin over it.
  This amendment keeps that rule and puts in the measured cost:
  - **PASS:** LB(T) >= 1.025 AND T >= 1.03 x 1.025 = 1.05575;
  - **BETWEEN:** T >= 1.05575 but LB(T) < 1.025;
  - **FAIL:** T < 1.05575.
  - Overall, and per set, as in amendment 6. Judged on unrounded numbers.
  - c16 is fixed here, at the last measurement before any data. A later engine trim does not move the bars; the
    certificate states the cost at the time beside T.
- **Power** (the dev standard errors scaled to the fresh windows, as in amendment 6):
  - HumanEval+: 0.72 at the dev T 1.060 (it was 0.87 at the old bars); 0.00 if the truth is the dev's low end, 1.027.
  - MBPP: 1.00, including at its dev low end 1.116.
  - Keeping the point bar at 1.052 would hold 0.87, but the "3 % net" claim would quietly become 2.6 %. The bars follow
    the cost instead, so the claim keeps its meaning.
- **The scorer now prints the verdict:**
  - `tools/certify98_tree.py verdict <HE+ topk> <HE+ tree> <MBPP topk> <MBPP tree> [<everyday topk> <everyday tree>]`;
  - constants C16 = 1.025, NET = 1.03;
  - last line `CERTIFY98 TREE-16: PASS | BETWEEN | FAIL | VOID (...)`.
  - On the dev sets (information): PASS on both (HE+ T 1.060 [LB 1.031]; MBPP 1.136 [1.119]).
  - Edge tests: T 1.0557 FAILs, 1.0558 passes, LB 1.0249 gives BETWEEN. TREE-REPRODUCE still passes.

## Amendment 7 (Sun 2026-11-15 22:1x, before any capture or candidate): bigger trees; a 127-node, top-16 TREE pass
> (a quote from the project owner, kept private)
  dump spec, asked for a fresh-set TREE dump of 127 nodes per anchor, so tree-32 / 64 / 128 are measured on fresh data.
  Fresh data is one-shot. MTP Head also asked for top-16 children: K98 puts the top-8 ceiling at 98-100 % of reached
  positions, and a tree-64 / 128 can reach a 9th-16th child near the root.
- **The TREE passes** (scripts/dd_tree.py, best-first by path probability; anchors are TOPK's depth-0 cells; TOPK's
  layout stays frozen at 9f49e03):
  - **T8:** top-8 children, the tree registered in amendments 5, 6 and 6a. It is the tree line's only input; tree-16
    reads its first 15 pops.
    - 32 nodes, as in amendment 5. If the tree-K line (NewIdeas, due Wed ~18:00) reads top-8 trees beyond 32 nodes,
      T8 becomes 127 nodes.
    - A best-first tree's first n pops do not depend on the budget. BigMoE checks this bitwise on Tuesday's bank
      pass: a 127-node top-8 pass on heplus24 must equal today's 32-node dump in its first 32 pops.
  - **T16:** top-16 children, 127 nodes (MTP Head's spec). The per-node top_ids / top_p are 16 wide. Its check against
    TOPK compares the first 8.
    - INFORMATION: tree-N at k = 16, and MTP Head's top-8 truncation count.
    - A top-16 tree's first 15 pops can differ from the top-8 tree's, since a 9th-16th child near the root can
      outrank a deep node. So the tree-16 line never reads T16.
    - The tree-K line may read T16 if it is filed on top-16 trees before the proof-run amendment.
  - Both passes come from the same sitting as TOPK, after the proof-run amendment (amendment 5's single look). That
    amendment lists the final passes and their arguments.
- **The scorer:** tools/certify98_tree.py reads tree-N for N = 16, 32, 64 and 128 wherever the dump holds the first
  N - 1 pops (`nodes` >= N - 1). It also checks that every anchor holds exactly `nodes` nodes.
  - `verdict` VOIDs the tree line if a gated dump's k is not 8.
  - Chain-N stays at 16 and 32, since TOPK holds 32 depths.
  - Tests: TREE-REPRODUCE still passes (max |diff| 0.0005), and the dev verdict is unchanged (PASS; HE+ T 1.060
    [LB 1.031], MBPP 1.136 [1.119]).
  - A dump relabelled k = 16 VOIDs the verdict. A dump with one node missing fails the node-count check.
  - A 127-node dump padded with off-path nodes reads tree-64 = tree-128 = all 32 original nodes.
- **The capture's generation length:**
  - Each answer is decoded to EOS on the runtime path, capped at the dev captures' 512 tokens: d1102mbpp ran with
    DLOG_MAXTOK=512 (TrainLab turns/e2_cap.sh). Its 375 answers averaged 139 tokens, max 524; the 8 longest end at
    516-524, the cap plus the last check's overshoot.
  - The anchors are about the answer tokens: ~105-120K (HumanEval+ ~35-37K, MBPP ~42-47K at the bank's density,
    everyday ~25-30K).
- **The dump manifest:** dumps_sha256.json beside the outputs. It is separate from files_sha256.json, which the
  proof-run amendment freezes before any dump exists.
  - It holds files_sha256.json's sha256; the candidate checkpoint's sha256; BigMoE's commit, dd_topk.py's and
    dd_tree.py's sha256s and their arguments (k, nodes, parts); and one sha256 per output file.
  - EvalSuite copies it to results/certify98/ and commits it before scoring. The scorer runs only on files whose sha256
    match it, and the RESULT quotes it.
- **The window** (the orchestrator decides): TOPK + T8 + T16 take ~2.2-2.5 h head-only at ~12.5 GB peak (BigMoE's
  estimate; Tuesday's bank passes measure it), in --part slices.
  - Proposed: Thursday by day, right after the proof-run amendment, as the one job above 4 GB. Scoring is then done
    Thursday evening, before the night's proof run.

## Amendment 8 (Mon 2026-11-16 08:3x, before any capture or candidate): the pre-capture lines (LAB, C1, C2, tree-K) and the targets-only convention
- **Why:** NewIdeas, the Goal A lead, sent the lines the orchestrator approved (Sun 21:5x; K* and T8 22:0x-22:1x), with
  its Monday 08:1x updates: ~/NewIdeas/research/week98/AMENDMENT_INPUT_1116.md at 64d8a29. This amendment adapts them
  to this PREREG's form.
  - Amendment 8a (Tue evening, before the capture) pins c32 and R32/R16 from Leap's measured E2 and names K* per set.
- **Every dev number below was recomputed here** with the scorer this amendment registers
  (results/certify98/lines_dev_2026-11-16.txt). Dev numbers are information; only the fresh sets gate.

### 0. The convention: walk-strict bands count targets only
- A walk cell with no target (past the answer's end) still ends the check, but it is not a miss. This is MoonShots'
  K98 definition, checked in 151d997.
- It applies to every walk-strict count in this PREREG: amendment 4's gate, C2's bands, and tree coverage (amendment
  6's information and C1).
- kept_walk and kept/32 do not change. certify98.py keeps amendment 4's counts for its strict-to-kept identity, as a
  code check.
- **Dev effect on t30b's gate bands (HE+ 24):** 91.39 / 91.93 / 93.78 / 96.50 / 96.22 / 97.46 become 91.68 / 92.38 /
  94.30 / 96.63 / 96.71 / 97.68. Still NOT CERTIFIED.
- Amendment 4's power model still counts the end as a miss, which is now slightly conservative.

### 1. LAB: the knowledge line (TRUE STATE; not deployable)
- **What:** the head's guess with the 125B's true state at every depth.
  - At depth d from anchor a, the head reads the trunk at row a + d, so its guess is the depth-0 guess of anchor a + d.
  - lab[a, d] = the depth-0 cell of anchor a + d, wherever cell (a, d) exists. It comes from the capture's depth-0
    cells, with no extra run.
- **Metric:** MARGINAL accuracy per band over every cell (bands 0 / 1-2 / 3-6 / 7-10 / 11-14 / 15-22 / 23-30), with
  certify98's lower bound.
- **Line (HE+ 140):** every band's point >= 97.5 AND its lower bound >= 97.0. Each point is quoted against Kamen's 97.8.
- **MBPP:** reported (dev 94.0-93.6, flat: knowledge-limited).
- **Dev (HE+ 24):** 97.95 / 97.93 / 97.91 / 97.88 / 97.84 / 97.79 / 97.72. This matches NewIdeas' numbers to 0.00 pt.
- **The REPLACE cross-check is mixed.** The input says the shortcut reproduces BigMoE's REPLACE oracle. NewIdeas' own
  check (lab_truestate.txt) agrees on MBPP (kept/32 12.72 vs 12.76) but prints DIFFERS on HE+: 20.77 vs 20.90, 0.13
  apart at its 0.05 tolerance.
  - The line is defined by the shortcut above. REPLACE is information.
  - Every LAB sentence says TRUE STATE (lab), never real decoding.

### 2. C1: walk-strict coverage at band 0 (targets only)
- **Candidates:** tree-16 (t30b + EAGLE-2-16, amendments 6 / 6a) and tree-32 (T8's first 31 pops).
- **Line (HE+), per tree:** band 0's point >= 97.8 AND its lower bound >= 97.0. MBPP is reported.
- **Dev:** HE+ tree-16 98.58 [LB 97.73 at 24 windows], tree-32 98.47 [96.54]; MBPP 97.79 / 98.61.
- Section 4's "band-0 coverage at K*" is this line for the K* tree.

### 3. C2: t30b + the frozen 2B audit (exact)
- **Auditor:** mlx-community/Qwen3.5-2B-MLX-4bit, revision 93760be4f1f69842a46bc13dbdc0f19e291392a3, model.safetensors
  sha256 713fe7e5d3c3965f7106b0d0ee17615f7869c23c8d327996df8c1196fbcf07d5. Checked here against the local copy, which
  is already on disk, so there is no download. No training.
- **Head:** dd_t30b.safetensors, sha256 790c42db393ad158f0f61e64962c868967995268babb816b27cbf8f4e215d502.
- **Rule (MoonShots', frozen Sun 01:3x):**
  - The 2B runs one teacher-forced pass over the bonus + drafts.
  - At the first row where its top-1 differs from the draft and its margin >= tau(depth), its token replaces the draft,
    and the head re-drafts the rest of the block. tau = 4 nats at depth 0, 2 at 1-2, 1 at 3-6 and 0.5 at 7-30.
  - P audit passes per check: P = 2 is the headline; P = 1 is reported.
  - Per cell this is the same as: where the 2B's margin >= tau, its outcome replaces the head's (a repair if it is
    right, a break if it is wrong). The scorer walks that, with at most P repairs per check.
- **Scored against the head's own label:** the 2B is right when its top-1 equals the dump's target y.
  - The 2B rows join the dump on (file, row), and every cell must find its row.
  - On the fresh files, the rows' 125B top-1 (tk0) must equal y on every row (the runtime-path validity), or C2 is VOID.
  - On the dev HE+ answers, which were not decoded on the runtime path, tk0 and y differ on 35 near-tie rows (125B
    margin <= 0.75 nats). MBPP has none. MoonShots scored against tk0 there.
- **PRIMARY (HE+, P = 2):** walk-strict 15-22 and 23-30, each point >= 97.8 AND its lower bound >= 97.0.
- **CO-PRIMARY (HE+, P = 2):** kept/32's paired difference, C2 minus t30b over the same windows. Its window-clustered
  one-sided 95 % lower bound must be > 0.
- **Reported:** P = 1, bands 0-14, MBPP, and the C2-4B arm.
- **Dev, as registered (HE+ 24):**
  - P = 2: 98.41 [LB 97.74] / 98.11 [97.42]; kept/32 15.07 vs 13.77 (+9.5 %; paired LB +0.33).
  - P = 1: 97.65 / 98.22.
  - In MoonShots' tk0 convention: 98.32 [97.62] / 98.28 [97.62], kept/32 15.00 (+8.9 %). That reproduces
    walkstrict_bands.json (MoonShots d9d3148) to 0.0002 pt.
  - MoonShots' quoted bounds (97.49 / 97.45) are bootstrap 2.5 % points. The line uses certify98's bound.
- **The 2B rows on the fresh files:** MoonShots writes them in rows2b's layout (fname, row, top1, margin2b, tk0), by day,
  after the proof-run amendment (GPU minutes, 2-4 GB).
  - That amendment pins the script's sha256 and arguments.
  - The dump manifest (amendment 7) lists the rows files.
- **C2-4B (information, no gate):** the same rule at P = 2 with mlx-community/Qwen3.5-4B-MLX-4bit, revision
  32f3e8ecf65426fc3306969496342d504bfa13f3, sha256 5fb9acd0246866381cf8c5c354c6db1019f6498eec4ccb4f5edcc71ffeacb2db
  (checked locally).
- **No speed claim:** the 2B pass is untimed.
- **C2b is DROPPED:** Sandbox98's L1CF is a KILL (07:09; HE+ +0.82 < +1.0).

### 4. The tree-K line: T8 at K* = 32 (31 best-first pops + the bonus)
- **T32/32** = tree-32 kept per check / chain-32 kept per check, each under its own walk over the same windows (the same
  32 rows).
  - PASS: lower bound >= c32 AND point >= 1.03 x c32. BETWEEN and FAIL as in amendment 6.
  - **Gated on HE+ AND MBPP**; overall as in amendment 6.
  - Dev: HE+ 1.181 [LB 1.126]; MBPP 1.304 [1.283].
- **T32/16** = tree-32 / chain-16 (amendment 6's statistic at K*).
  - PASS: lower bound >= rho AND point >= 1.03 x rho, where rho = R32/R16 x c32.
  - **Gated on HE+ only.** MBPP is information (dev 1.371 [1.351]).
  - Dev HE+: 1.531 [1.470].
- **c32 and R32/R16** come from Leap's measured E2 (Tue 00:30). Amendment 8a pins them Tue evening, before the capture.
  No bar is set before then.
  - The priors are for orientation only: c32 1.03-1.05 (first cut); R32/R16 1.480 (Breakthrough a082f8c: C32one 90.46
    / A16 61.11 ms). The orchestrator corrected its earlier 1.445 at 08:2x.
  - At 1.480 x 1.04, the T32/16 bar is 1.539 (point 1.585), against dev 1.531. Tree-32 would lose to chain-16 on HE+,
    and K* would be 16 there.
- **K*:** amendment 8a names it per set: 32 or 16 for HE+-like code, by the measured bar; 16 for MBPP-like. The
  sentence names K*.
- **T8's node count:** the line reads T8's first 31 pops, so amendment 7's 32-node T8 suffices. Tree-64 and tree-128 from
  a 127-node T8 would be information. T16 (top-16) stays information.
- TREE's own checks gate the read-out (amendment 5). If one fails, the line is VOID.

### 5. Conditional: C3, t30c
- It runs only if MTP Head's Tue-night run pays on dev by Wed ~15:00: HE+ 23-30 marginal >= t30b + 1.0, and no band
  below t30b - 0.3. Then it is a separate, named arm in C1, C2 and tree-K. t30b's lines stand regardless.
- **It is a second candidate on the same sets.** Under section 2 (Looks), its lines use the one-sided 97.5 % bound,
  t(K-1, 0.975). Its pins, and its power at that bound, are filed before the proof-run amendment.

### 6. The family, and how it is reported
- **There are 8 gated claims:** the chain line (amendment 4); TREE-16 (amendments 6 / 6a); LAB; C1 tree-16; C1 tree-32;
  C2; T32/32; T32/16.
  - All are read-outs of the one candidate's single pass (amendment 5), plus the 2B's rows, named before any data.
  - So each uses the one-sided 95 % bound of the approved input. There is no family-wise correction.
- **The RESULT quotes all eight verdicts,** failures included. A sentence that quotes one line links the others' verdicts.
- **Every HE+ verdict is also computed on the strict 80** (amendment 2's sensitivity line), reported beside it. If they
  disagree, both are quoted.

### 7. Power
tools/certify98_lines.py power: 2,000 draws of windows with replacement from the dev windows, at the fresh counts (HE+
140, with its strict 80 in brackets; MBPP 300). The bounds are the registered ones, counted targets only
(results/certify98/power_lines_2026-11-16.txt).
- **LAB:** 0.71 [0.54]. All points >= 97.8: 0.41.
- **C1:** tree-16 1.00 [1.00]; tree-32 0.92 [0.76]. MBPP (reported): 0.48 / 1.00.
- **C2, P = 2:**
  - 0.97 [0.92] at the dev truth.
  - If both deep bands' truth is 98.0: 0.81; at 97.9: 0.60; at 97.8: 0.34.
  - CO-PRIMARY alone: 1.00. P = 1: 0.24.
  - This is the re-power under targets only promised in 151d997. The registered bound gives 0.97, against NewIdeas'
    bootstrap 1.00.
- **T32/32:** 1.00 on both sets at c32 1.03, 1.04 and 1.05.
- **T32/16 (HE+):**
  - 0.00 at the prior 1.480 x 1.04;
  - 0.45 at 1.445 x 1.03;
  - 0.84 at 1.43 x 1.03;
  - 1.00 at 1.40 x 1.03.
  - So it passes only if E2 measures R32/R16 at or below ~1.43.
- These are draws from 24 HE+ windows, with the dev point taken as the truth. The shifted rows are the sensitivity.

### 8. The scorer (pinned again in the proof-run amendment)
- tools/certify98_lines.py (new; sha256 579a60ac9ed4e329...): `lab`, `c1`, `c2`, `treek ... --c32 X --r3216 Y`,
  `reproduce` and `power`.
- tools/certify98.py (targets only; sha256 c0528ad765fd5e59...).
- tools/certify98_tree.py (coverage counts targets only; the nesting check runs under both counts; sha256
  4a0ba21625497be1...).
- Tests:
  - certify98 selftest and reproduce: PASS;
  - TREE-REPRODUCE: PASS (max |diff| 0.0005);
  - LINES-REPRODUCE: PASS (NewIdeas' LAB and C1, MoonShots' C2, and 151d997's tree-K).

### 9. What the certificate can and cannot say (NewIdeas' section 6, if the lines pass)
- **Can:** "Exact. The first guess after a check is covered ~98.5 % (the tree at K*); deep guesses 15-30 are right ~98 %
  (2B audit, P = 2). The head's knowledge, given the true state, is ~97.8 % flat (LAB, TRUE STATE). Tree-32 keeps x1.18
  more tokens per check than the chain at the same 32 rows."
- **Cannot:**
  - 97.8 % at walk-strict bands 1-14: every deployable lever is at 90.6-97.1, and even the true state is at 93.2-96.4
    for bands 0-6.
  - 450 exact: that needs a flat 98.0-98.4 % with the check at its byte floor (TARGETS_450_900.md).

## Amendment 7a (Mon 2026-11-16 11:2x, before any capture or candidate): Thursday's fresh dumps are TOPK + T8 at 32 nodes; the fresh T16 is dropped
- **Why:** amendment 7 said the bank passes would measure the window. They did:
  - BigMoE's 127-node bank on the dev sets (scripts/n127_dry.sh 1622030f, Mon 09:05-11:03) costs 124-180 ms per anchor.
    TOPK and the 32-node TREE cost 12-14 ms, so a 127-node anchor is ~10x, not ~4-4.5x (~55 ms).
  - On ~105-120K fresh anchors, TOPK + T8 + T16 would take ~4.3-5.5 h, not 2.2-2.5 h. T16 alone is 3.6-4.7 h.
  - Read in EvalSuite 11ffa06 / 696a937 (docs/RESULT_bank127_2026-11-16.md): the dump manifest check PASSES 4 / 4, and
    the 127-node k = 8 dumps' tree-16 / tree-32 are IDENTICAL to the 32-node dumps.
  - The order is NewIdeas' (11:1x). The window is the orchestrator's (11:1x; it reconciled its note with NewIdeas'
    conditional, so point 3 stands).
- **1. The fresh T16 is DROPPED.** It was information and never gated. On dev, k = 16 vs k = 8 at the same tree size:
  - HE+ 24: x1.000 at tree-16 / 32 / 64, and x1.011 at tree-128;
  - MBPP 100: x1.000 / 1.001 / 0.999 / 1.003, with tree-16's band-0 coverage 97.84 vs 97.79.
  - MTP Head's truncation count can read the bank's k = 16 files (dev).
  - NewIdeas may ask for T16 back, with a reason, by Wed 18:00 (the orchestrator).
- **2. Thursday's window: TOPK + T8 at 32 nodes ONLY**, by day after the proof-run amendment as in amendment 7.
  - It takes ~0.8 h by the bank's price: TOPK ~21-24 min and T8 ~22-27 min on ~105-120K anchors.
  - Amendment 7's "32 nodes, or 127" is settled at 32. Every gated line reads T8's first 15 or 31 pops (tree-16,
    tree-32, the tree-K line at K* = 32, C1).
- **3. CONDITIONAL, information only, never gated: a fresh T8 at 127 nodes on HE+ fresh only** (~35-37K anchors,
  ~1.2-1.4 h). It runs ONLY if both hold:
  - (a) Wednesday's early-death probe PASSES (CPU, on BigMoE's Tuesday-night layer 8 / 16 / 24 dev pass);
  - (b) a by-day window is left after the gated TOPK + T8-32 that does not touch the proofs (the orchestrator's call).
  - Its file is separate (tree_*_n127k8), with its own manifest entries. Its first 32 pops must match the gated T8 file
    BITWISE (BigMoE's nest check), or its read is VOID. It never changes the gated T8.
  - **Wording settled with BigMoE (11:2x; the orchestrator 11:2x asked us to settle it):**
    - HE+ T8(127) has k 8 and 127 nodes, and uses the same capture files, checkpoint and dd_tree as T8(32).
    - It is INFORMATION (tree-64 / tree-128). It runs only BESIDE the gated HE+ T8(32), never instead of it.
    - The gated tree-32 read stays on the T8(32) dump, as amendment 7 says.
    - The T8(127) read-out is valid only if its first 32 pops equal the T8(32) dump BITWISE, on every node field, in
      every window: dd_tree's nest check. On dev it gave 124 / 124 windows BITWISE (BigMoE 3e239ce8; EvalSuite 11ffa06).
      Otherwise it is VOID, and the gated line is untouched.
    - Both dumps and their args are in the manifest.
    - Taking tree-32 from T8(127) itself would still need T8(32) on a named subset as the nest base, and it saves under
      ~8 min (HE+ T8(32) is ~35-37K anchors x 12-14 ms). So it is not done.
  - If either condition fails, no fresh 127-node dump is made. The dev K-curve stands as the record: EvalSuite 11ffa06
    and NewIdeas' BIGGER_TREES.md.
- **4. The dump manifest check** runs before any scoring:
  - the command is `tools/certify98_dumpcheck.py <dumps_sha256.json> <files> --gated <files_sha256.json>` (sha256
    ef6cadbeae099f07...);
  - it needs amendment 7's run fields under `_run`: files_sha256_json, checkpoint_sha256, bigmoe_commit,
    dd_topk_sha256, dd_tree_sha256 and args;
  - the field names were proposed to BigMoE 11:1x. The proof-run amendment pins the final names and the tool's sha256.
  - **Updated 11:4x on BigMoE's sample manifest** (37744f27: the dev bank heplus24, not the capture). The tool is now
    sha256 92410754282d977d..., replacing ef6cadbeae099f07. `--gated` now also needs:
    - checkpoint_path;
    - `modules` ({path: sha256}) ALWAYS, because the dumpers import ~/TrainLab/mtp125.py, which is outside BigMoE's git;
    - bigmoe_dirty as a bool;
    - on each file, `script` + `args` that appear in _run.args.
    - `--pins <json>` checks the proof-run amendment's values (modules as an exact map). `--fresh` checks that
      files_sha256.json's manifest_sha256 is the frozen fresh manifest's (acdf3e62...).
    - Checks: 26 / 26 as intended (results/certify98/dumpcheck_checks_2026-11-16.txt). The sample passes `--gated`.
    - The proof-run amendment pins mtp125.py's sha256 with the other modules.
- **Unchanged:** every line of amendments 6 / 6a / 8, and amendment 8's scorer pins. certify98_tree.py is still
  4a0ba21625497be1....
